import argparse
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.dataset import build_dataset
from models.losses import build_loss
from models.tdc_deblur_net import build_deblur_model
from utils.reproducibility import set_global_seed


def report_peak(label, device):
    torch.cuda.synchronize(device)
    allocated = torch.cuda.max_memory_allocated(device) / 2**30
    reserved = torch.cuda.max_memory_reserved(device) / 2**30
    print(f"{label}: peak allocated={allocated:.3f} GiB, reserved={reserved:.3f} GiB")


def main():
    parser = argparse.ArgumentParser(description="Check one GPU without saving experiment files.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-steps", type=int, default=2)
    parser.add_argument("--val-samples", type=int, default=3)
    args = parser.parse_args()
    if args.train_steps < 2 or args.val_samples < 1:
        parser.error("Use at least two train steps and one validation sample.")
    if not torch.cuda.is_available():
        parser.error("A CUDA GPU is required for actual memory measurement.")
    with open(args.config, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if config["model"].get("two_stage", False):
        parser.error("This check supports the single-stage attention experiments only.")

    device = torch.device("cuda:0")
    set_global_seed(
        int(config["train"].get("seed", 42)),
        bool(config["train"].get("deterministic", True)),
    )
    model = build_deblur_model(**config["model"]).to(device)
    loss_opt = config.get("loss", {})
    criterion = build_loss(loss_opt.get("type", "l1"), loss_opt.get("edge_weight", 0.05)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["train"]["learning_rate"])
    train_opt = dict(config["datasets"]["train"], split="train")
    train_dataset = build_dataset(train_opt)
    if len(train_dataset) == 0:
        parser.error("Training dataset is empty.")
    loader = DataLoader(train_dataset, batch_size=train_opt.get("batch_size", 1), num_workers=0)
    batch = next(iter(loader))
    batch = {key: batch[key].to(device) for key in ("blur", "event", "gt")}
    print(f"Train blur={tuple(batch['blur'].shape)}, event={tuple(batch['event'].shape)}")
    torch.cuda.reset_peak_memory_stats(device)
    model.train()
    for _ in range(args.train_steps):
        optimizer.zero_grad(set_to_none=True)
        pred = model(batch["blur"], batch["event"])
        loss = criterion(pred, batch["gt"])
        loss.backward()
        optimizer.step()
        del pred, loss
    report_peak("Train (forward/backward/AdamW)", device)
    del batch

    val_opt = dict(config["datasets"]["val"], split="val", random_crop=False)
    if val_opt.get("patch_size") is not None:
        parser.error("Set validation patch_size: null to check complete images.")
    val_dataset = build_dataset(val_opt)
    if len(val_dataset) == 0:
        parser.error("Validation dataset is empty.")
    torch.cuda.reset_peak_memory_stats(device)
    model.eval()
    with torch.no_grad():
        for index in range(min(args.val_samples, len(val_dataset))):
            sample = val_dataset[index]
            blur = sample["blur"].unsqueeze(0).to(device)
            event = sample["event"].unsqueeze(0).to(device)
            pred = model(blur, event)
            print(f"Validation {index}: blur={tuple(blur.shape)}, event={tuple(event.shape)}")
            del pred, blur, event
    report_peak("Complete-image validation", device)
    print("No checkpoint/config written. DDP buffers and unsampled image shapes are not covered.")


if __name__ == "__main__":
    main()
