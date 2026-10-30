import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.dataset import build_dataset
from models.losses import build_loss
from models.tdc_deblur_net import build_deblur_model
from utils.reproducibility import set_global_seed
from utils.model_checkpoint import fusion_description


def report_peak(label, device):
    torch.cuda.synchronize(device)
    allocated = torch.cuda.max_memory_allocated(device) / 2**30
    reserved = torch.cuda.max_memory_reserved(device) / 2**30
    print(f"{label}: peak allocated={allocated:.3f} GiB, reserved={reserved:.3f} GiB")
    if dist.is_initialized():
        peaks = torch.tensor([allocated, reserved], device=device)
        dist.all_reduce(peaks, op=dist.ReduceOp.MAX)
        # Every rank must finish this CUDA collective before its temporary tensor
        # or process group can be released, not only rank 0 reading the result.
        torch.cuda.synchronize(device)
        if dist.get_rank() == 0:
            print(f"{label}: max across ranks allocated={peaks[0].item():.3f} GiB, reserved={peaks[1].item():.3f} GiB")


def finish_preflight(device, world_size):
    rank = dist.get_rank() if dist.is_initialized() else 0
    torch.cuda.synchronize(device)
    if dist.is_initialized():
        dist.barrier(device_ids=[device.index])
        torch.cuda.synchronize(device)
        dist.destroy_process_group()
    if rank == 0:
        print(f"Preflight PASS: checks and synchronized shutdown complete; world_size={world_size}.", flush=True)


def check_validation(config, model, device, val_samples):
    val_opt = dict(config["datasets"]["val"], split="val", random_crop=False)
    patch_size = val_opt.get("patch_size", 256)
    label = "Complete-image validation" if patch_size is None else "Center-crop validation"
    print(f"Validation mode: {label}; configured patch_size={patch_size}")
    val_dataset = build_dataset(val_opt)
    if len(val_dataset) == 0:
        raise ValueError("Validation dataset is empty.")
    torch.cuda.reset_peak_memory_stats(device)
    model.eval()
    with torch.no_grad():
        for index in range(min(val_samples, len(val_dataset))):
            sample = val_dataset[index]
            blur = sample["blur"].unsqueeze(0).to(device)
            event = sample["event"].unsqueeze(0).to(device)
            pred = model(blur, event)
            print(f"Validation {index}: blur={tuple(blur.shape)}, event={tuple(event.shape)}")
            del pred, blur, event
    report_peak(label, device)


def main():
    parser = argparse.ArgumentParser(description="Check single-GPU or torchrun DDP memory without saving experiment files.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--train-steps", type=int, default=3)
    parser.add_argument("--val-samples", type=int, default=3)
    args = parser.parse_args()
    if args.warmup_steps < 3 or args.train_steps < 2 or args.val_samples < 1:
        parser.error("Use at least three warmup steps, two measured train steps and one validation sample.")
    if not torch.cuda.is_available():
        parser.error("A CUDA GPU is required for actual memory measurement.")
    with open(args.config, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if config["model"].get("two_stage", False):
        parser.error("This check supports the single-stage attention experiments only.")

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = 0
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    torch.cuda.set_device(device)
    if world_size > 1:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
    set_global_seed(
        int(config["train"].get("seed", 42)) + rank,
        bool(config["train"].get("deterministic", True)),
    )
    model = build_deblur_model(**config["model"]).to(device)
    raw_model = model
    if world_size > 1:
        model = DDP(model, device_ids=[device.index], output_device=device.index, find_unused_parameters=False)
    print(f"Rank {rank}/{world_size}: parameters={sum(p.numel() for p in model.parameters()):,}, precision=float32")
    description = fusion_description(raw_model)
    if description:
        print(description)
    loss_opt = config.get("loss", {})
    criterion = build_loss(loss_opt.get("type", "l1"), loss_opt.get("edge_weight", 0.05)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["train"]["learning_rate"])
    train_opt = dict(config["datasets"]["train"], split="train", seed=int(config["train"].get("seed", 42)))
    train_dataset = build_dataset(train_opt)
    if len(train_dataset) == 0:
        parser.error("Training dataset is empty.")
    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank,
                                 seed=int(config["train"].get("seed", 42))) if world_size > 1 else None
    loader = DataLoader(train_dataset, batch_size=train_opt.get("batch_size", 1), sampler=sampler, num_workers=0)
    batch = next(iter(loader))
    batch = {key: batch[key].to(device) for key in ("blur", "event", "gt")}
    print(f"Train blur={tuple(batch['blur'].shape)}, event={tuple(batch['event'].shape)}")
    model.train()

    def train_step():
        optimizer.zero_grad(set_to_none=True)
        pred = model(batch["blur"], batch["event"])
        loss = criterion(pred, batch["gt"])
        loss.backward()
        optimizer.step()
        del pred, loss

    for _ in range(args.warmup_steps):
        train_step()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    durations = []
    for _ in range(args.train_steps):
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        train_step()
        torch.cuda.synchronize(device)
        durations.append(time.perf_counter() - started)
    report_peak("Train (forward/backward/AdamW)", device)
    print(f"Rank {rank}: mean measured train step={sum(durations) / len(durations):.4f} s ({args.train_steps} steps after {args.warmup_steps} warmup)")
    del batch
    optimizer.zero_grad(set_to_none=True)

    check_validation(config, raw_model, device, args.val_samples)
    print(f"No checkpoint/config written. DDP world_size={world_size}; unsampled image shapes are not covered.")
    finish_preflight(device, world_size)


def run_preflight():
    try:
        main()
    finally:
        if dist.is_initialized():
            # An error on one rank must not wait for a new collective barrier.
            dist.destroy_process_group()


if __name__ == "__main__":
    run_preflight()
