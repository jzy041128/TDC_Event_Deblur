import argparse
import sys
from pathlib import Path

import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.dataset import build_dataset


def describe_tensor(name, tensor):
    print(
        f"{name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}, "
        f"min={tensor.min().item():.6f}, max={tensor.max().item():.6f}, "
        f"mean={tensor.float().mean().item():.6f}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_tdc.yml")
    parser.add_argument("--split", default="train", choices=["train", "val"])
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    opt = dict(config["datasets"][args.split])
    opt["split"] = args.split
    if args.split == "val":
        opt["random_crop"] = False

    dataset = build_dataset(opt)
    print(f"dataset_len={len(dataset)}")
    sample = dataset[0]
    describe_tensor("sample.blur", sample["blur"])
    describe_tensor("sample.gt", sample["gt"])
    describe_tensor("sample.event", sample["event"])

    loader = DataLoader(dataset, batch_size=opt.get("batch_size", 1), shuffle=False, num_workers=0)
    batch = next(iter(loader))
    describe_tensor("batch.blur", batch["blur"])
    describe_tensor("batch.gt", batch["gt"])
    describe_tensor("batch.event", batch["event"])


if __name__ == "__main__":
    main()
