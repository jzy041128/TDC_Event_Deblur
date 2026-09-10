import argparse
import datetime
import os
import re
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
import yaml
from skimage.metrics import peak_signal_noise_ratio as calculate_psnr
from skimage.metrics import structural_similarity as calculate_ssim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler

from data.dataset import build_dataset
from models.losses import build_loss
from models.tdc_deblur_net import build_deblur_model
from utils.reproducibility import capture_rng_state, restore_rng_state, seed_worker, set_global_seed


class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass


class NullWriter(object):
    def write(self, message):
        pass

    def flush(self):
        pass


class RankStrideSampler(Sampler):
    """Exact distributed evaluation split without padding or duplicate samples."""

    def __init__(self, dataset, rank, world_size):
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.world_size))

    def __len__(self):
        remaining = len(self.dataset) - self.rank
        return max(0, (remaining + self.world_size - 1) // self.world_size)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to a YAML training config.")
    return parser.parse_args()


def setup_ddp():
    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError("train_ddp.py must be launched with torchrun.")
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_rank(), dist.get_world_size()


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def rank0():
    return dist.get_rank() == 0


def unwrap_model(model):
    return model.module if isinstance(model, DDP) else model


def tensor2float_img(tensor):
    img = tensor.detach().cpu().squeeze().numpy()
    if img.ndim == 4:
        img = img[0]
    img = np.transpose(img, (1, 2, 0))
    return np.clip(img, 0.0, 1.0)


def format_duration(seconds):
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def parse_best_psnr_from_log(log_path):
    best = 0.0
    if not os.path.exists(log_path):
        return best
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = re.search(r"Pred PSNR:\s*([0-9.]+)\s*dB", line)
            if match:
                best = max(best, float(match.group(1)))
    return best


def configured_path(path):
    return path not in (None, "", "~")


def make_experiment_dir(config, resume_path):
    resume_mode = resume_path and resume_path != "~" and os.path.exists(resume_path)
    if resume_mode:
        exp_dir = os.path.dirname(resume_path)
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") if rank0() else None
        obj = [timestamp]
        dist.broadcast_object_list(obj, src=0)
        exp_name = f"{config['name']}_{obj[0]}"
        exp_dir = os.path.join(config["path"]["experiments_root"], exp_name)
    if rank0():
        os.makedirs(exp_dir, exist_ok=True)
    dist.barrier()
    return exp_dir, resume_mode


def build_loaders(config, rank, world_size, seed, deterministic):
    train_opt = dict(config["datasets"]["train"])
    train_opt["split"] = "train"
    train_opt["seed"] = seed
    train_dataset = build_dataset(train_opt)
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=False,
        seed=seed,
    )
    train_num_workers = train_opt.get("num_workers", 4)
    train_loader_kwargs = {
        "batch_size": train_opt["batch_size"],
        "sampler": train_sampler,
        "num_workers": train_num_workers,
        "persistent_workers": train_num_workers > 0 and not deterministic,
        "pin_memory": True,
        "worker_init_fn": seed_worker,
        "generator": torch.Generator().manual_seed(seed + rank),
    }
    if train_num_workers > 0:
        train_loader_kwargs["prefetch_factor"] = train_opt.get("prefetch_factor", 2)
    train_loader = DataLoader(train_dataset, **train_loader_kwargs)

    val_opt = dict(config["datasets"]["val"])
    val_opt["split"] = "val"
    val_opt["random_crop"] = False
    val_dataset = build_dataset(val_opt)
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        sampler=RankStrideSampler(val_dataset, rank, world_size),
        num_workers=val_opt.get("num_workers", 0),
        pin_memory=True,
    )
    return train_loader, train_sampler, val_loader


def build_model(config, device):
    model = build_deblur_model(**dict(config.get("model", {}))).to(device)
    return DDP(
        model,
        device_ids=[device.index],
        output_device=device.index,
        find_unused_parameters=False,
    )


def validate(model, val_loader, device):
    eval_model = unwrap_model(model)
    eval_model.eval()
    total_blur_psnr, total_pred_psnr, total_ssim = 0.0, 0.0, 0.0
    with torch.no_grad():
        for val_batch in val_loader:
            val_blur = val_batch["blur"].to(device, non_blocking=True)
            val_event = val_batch["event"].to(device, non_blocking=True)
            val_gt = val_batch["gt"].to(device, non_blocking=True)
            val_pred = eval_model(val_blur, val_event)

            pred_np = tensor2float_img(val_pred)
            blur_np = tensor2float_img(val_blur)
            gt_np = tensor2float_img(val_gt)

            total_blur_psnr += calculate_psnr(gt_np, blur_np, data_range=1.0)
            total_pred_psnr += calculate_psnr(gt_np, pred_np, data_range=1.0)
            total_ssim += calculate_ssim(gt_np, pred_np, channel_axis=2, data_range=1.0)

    totals = torch.tensor(
        [total_blur_psnr, total_pred_psnr, total_ssim, len(val_loader)],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    count = max(totals[3].item(), 1.0)
    avg_blur_psnr = totals[0].item() / count
    avg_psnr = totals[1].item() / count
    avg_ssim = totals[2].item() / count
    return avg_blur_psnr, avg_psnr, avg_ssim


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    train_cfg = config.get("train", {})
    seed = int(train_cfg.get("seed", 42))
    deterministic = bool(train_cfg.get("deterministic", True))
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    local_rank, rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    set_global_seed(seed + rank, deterministic)

    resume_path = config["path"].get("resume_state")
    pretrain_path = config["path"].get("pretrain_model")
    if configured_path(resume_path) and configured_path(pretrain_path):
        raise ValueError("Set only one of path.resume_state and path.pretrain_model.")
    exp_dir, resume_mode = make_experiment_dir(config, resume_path)
    if rank0():
        sys.stdout = Logger(os.path.join(exp_dir, "train_log.txt"))
        sys.stderr = sys.stdout
    else:
        sys.stdout = NullWriter()
    print(f"Experiment dir: {exp_dir}")
    print(f"Device: {device} | DDP world_size: {world_size} | per_gpu_batch: {config['datasets']['train']['batch_size']}")
    print(f"Seed: {seed} | deterministic: {deterministic}")

    train_loader, train_sampler, val_loader = build_loaders(config, rank, world_size, seed, deterministic)
    model = build_model(config, device)
    model_cfg = config.get("model", {})
    print(
        f"Model: three_branch_progressive | base_dim: {model_cfg.get('base_dim', 32)} | "
        f"fusion_mode: {model_cfg.get('fusion_mode', 'single_ca')} | "
        f"fusion_dim: {model_cfg.get('fusion_dim', '2d')} | "
        f"cross_attn_type: {model_cfg.get('cross_attn_type', 'window')} | "
        f"single_ca_order: {model_cfg.get('single_ca_order', 'event2d_k_event3d_v')} | "
        f"cascaded_ca_order: {model_cfg.get('cascaded_ca_order', 'motion_then_struct')} | "
        f"swapped_kv_order: {model_cfg.get('swapped_kv_order', 'event3d_first')} | "
        f"key_bridge_order: {model_cfg.get('key_bridge_order', 'event3d_first')} | "
        f"encoder_sa: {model_cfg.get('encoder_self_attn', 'window')} | "
        f"deform: {model_cfg.get('deform_alignment', 'none')} | "
        f"windows(self/cross/time): {model_cfg.get('self_attn_window_size', 8)}/"
        f"{model_cfg.get('cross_attn_window_size', 8)}/{model_cfg.get('temporal_window_size', 2)} | "
        f"num_heads: {model_cfg.get('num_heads', 4)} | "
        f"decoder: {model_cfg.get('decoder_block', 'plain')} | "
        f"decoder_sa: {model_cfg.get('decoder_attention', 'none')} | "
        f"two_stage: {model_cfg.get('two_stage', False)} | "
        f"sam: {model_cfg.get('sam_mode', 'none')} | refine: {model_cfg.get('refine_type', 'shallow')}"
    )

    loss_cfg = config.get("loss", {})
    criterion = build_loss(
        loss_type=loss_cfg.get("type", "l1"),
        edge_weight=loss_cfg.get("edge_weight", 0.05),
    )
    optimizer = optim.AdamW(unwrap_model(model).parameters(), lr=config["train"]["learning_rate"])

    start_epoch = 0
    best_psnr = 0.0
    if configured_path(pretrain_path):
        if not os.path.exists(pretrain_path):
            raise FileNotFoundError(f"Pretrained model not found: {pretrain_path}")
        checkpoint = torch.load(pretrain_path, map_location=device, weights_only=False)
        if "model_state_dict" not in checkpoint:
            raise KeyError(f"Checkpoint has no model_state_dict: {pretrain_path}")
        unwrap_model(model).load_state_dict(checkpoint["model_state_dict"], strict=True)
        if rank0():
            print(f"Loaded pretrained model weights: {pretrain_path}")
            print("Optimizer is new; training starts from epoch 0 in a new experiment directory.")
    elif configured_path(resume_path):
        if os.path.exists(resume_path):
            checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
            unwrap_model(model).load_state_dict(checkpoint["model_state_dict"], strict=True)
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except ValueError:
                print("Optimizer state mismatch; optimizer is reinitialized.")
            start_epoch = checkpoint["epoch"]
            best_psnr = max(checkpoint.get("best_psnr", 0.0), parse_best_psnr_from_log(os.path.join(exp_dir, "train_log.txt")))
            rng_states = checkpoint.get("rng_states")
            if rng_states and rank < len(rng_states):
                restore_rng_state(rng_states[rank])
            print(f"Resume from epoch {start_epoch}, best_psnr={best_psnr:.2f}")
        else:
            print(f"Resume path not found: {resume_path}. Start from scratch.")

    num_epochs = config["train"]["num_epochs"]
    val_interval = int(train_cfg.get("val_interval", 1))
    if val_interval < 1:
        raise ValueError("train.val_interval must be at least 1.")
    train_start_time = time.time()
    for epoch in range(start_epoch, num_epochs):
        train_sampler.set_epoch(epoch)
        if hasattr(train_loader.dataset, "set_epoch"):
            train_loader.dataset.set_epoch(epoch)
        epoch_start_time = time.time()
        model.train()
        epoch_loss = torch.zeros((), device=device)
        steps_this_epoch = 0

        max_train_steps = config["train"].get("max_train_steps")
        for step, batch_data in enumerate(train_loader):
            blur = batch_data["blur"].to(device, non_blocking=True)
            event = batch_data["event"].to(device, non_blocking=True)
            gt = batch_data["gt"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            output = model(blur, event, return_intermediate=True)
            if isinstance(output, tuple):
                pred, stage1_pred = output
                loss = (
                    train_cfg.get("stage2_loss_weight", 1.0) * criterion(pred, gt)
                    + train_cfg.get("stage1_loss_weight", 0.5) * criterion(stage1_pred, gt)
                )
            else:
                pred = output
                loss = criterion(pred, gt)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.detach()
            steps_this_epoch += 1
            if rank0() and (step + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{step+1}/{len(train_loader)}], Loss: {loss.detach().item():.4f}")
            if max_train_steps is not None and step + 1 >= max_train_steps:
                break

        dist.all_reduce(epoch_loss, op=dist.ReduceOp.SUM)
        avg_train_loss = (epoch_loss / (max(steps_this_epoch, 1) * world_size)).item()
        skip_validation = config["train"].get("skip_validation", False)
        should_validate = (
            not skip_validation
            and ((epoch + 1) % val_interval == 0 or (epoch + 1) == num_epochs)
        )
        avg_psnr = None
        if should_validate:
            if rank0():
                print(f"\nEpoch {epoch+1} train done, avg loss: {avg_train_loss:.4f}. Validating...")
            avg_blur_psnr, avg_psnr, avg_ssim = validate(model, val_loader, device)
            if rank0():
                psnr_gain = avg_psnr - avg_blur_psnr
                print(f"Val -> Blur PSNR: {avg_blur_psnr:.2f} dB | Pred PSNR: {avg_psnr:.2f} dB | Gain: {psnr_gain:+.2f} dB | SSIM: {avg_ssim:.4f}\n")
        else:
            if rank0():
                reason = "disabled" if skip_validation else f"next interval: every {val_interval} epochs"
                print(f"\nEpoch {epoch+1} train done, avg loss: {avg_train_loss:.4f}. Validation skipped ({reason}).")

        rng_states = [None] * world_size
        dist.all_gather_object(rng_states, capture_rng_state())
        if rank0():
            elapsed_total = time.time() - train_start_time
            elapsed_epoch = time.time() - epoch_start_time
            remaining_epochs = num_epochs - epoch - 1
            avg_epoch_time = elapsed_total / max(epoch + 1 - start_epoch, 1)
            eta_seconds = avg_epoch_time * remaining_epochs
            print(
                f"Time -> Epoch: {format_duration(elapsed_epoch)} | "
                f"Elapsed: {format_duration(elapsed_total)} | ETA: {format_duration(eta_seconds)}\n"
            )

            is_best = avg_psnr is not None and avg_psnr > best_psnr
            if is_best:
                best_psnr = avg_psnr
                print(f"New best model, PSNR: {best_psnr:.2f}")

            save_dict = {
                "epoch": epoch + 1,
                "model_state_dict": unwrap_model(model).state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_psnr": best_psnr,
                "rng_states": rng_states,
                "seed": seed,
            }
            torch.save(save_dict, os.path.join(exp_dir, "latest.pth"))
            if is_best:
                torch.save(save_dict, os.path.join(exp_dir, "best.pth"))
        dist.barrier()


if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup_ddp()
