import argparse
import os

import numpy as np
import torch
import torch.distributed as dist
import yaml
from skimage.metrics import peak_signal_noise_ratio as calculate_psnr
from skimage.metrics import structural_similarity as calculate_ssim
from torch.utils.data import DataLoader, Sampler, Subset

from data.dataset import build_dataset
from models.tdc_deblur_net import build_deblur_model
from utils.reproducibility import set_global_seed


class RankStrideSampler(Sampler):
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
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint without training.")
    parser.add_argument("--config", required=True, help="Config containing the model and val dataset.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint containing model_state_dict.")
    parser.add_argument(
        "--full-resolution",
        action="store_true",
        help="Override datasets.val.patch_size and evaluate complete images.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Evaluate only the first N samples. Use 1 for a memory smoke test.",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=None,
        help="Run inference on overlapping square tiles of this size.",
    )
    parser.add_argument(
        "--tile-overlap",
        type=int,
        default=32,
        help="Overlap in pixels between adjacent inference tiles (default: 32).",
    )
    return parser.parse_args()


def setup_runtime():
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        return torch.device(f"cuda:{local_rank}"), dist.get_rank(), dist.get_world_size()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device, 0, 1


def tensor_to_float_image(tensor):
    array = tensor.detach().cpu().squeeze(0).numpy()
    return np.clip(np.transpose(array, (1, 2, 0)), 0.0, 1.0)


def reduce_totals(values, device):
    totals = torch.tensor(values, dtype=torch.float64, device=device)
    if dist.is_initialized():
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    return totals


def tile_starts(length, tile_size, overlap):
    if length <= tile_size:
        return [0]

    stride = tile_size - overlap
    distance = length - tile_size
    num_steps = (distance + stride - 1) // stride
    return [round(step * distance / num_steps) for step in range(num_steps + 1)]


def feather_weight(height, width, overlap, device, dtype):
    def axis_weight(length):
        weight = torch.ones(length, device=device, dtype=dtype)
        feather = min(overlap, length // 2)
        if feather > 0:
            ramp = torch.linspace(
                1.0 / (feather + 1), 1.0, feather, device=device, dtype=dtype
            )
            weight[:feather] = ramp
            weight[-feather:] = ramp.flip(0)
        return weight

    return axis_weight(height)[:, None] * axis_weight(width)[None, :]


def tiled_forward(model, blur, event, tile_size, overlap):
    if blur.ndim != 4 or event.ndim != 4:
        raise ValueError("Tiled inference expects BCHW blur and event tensors.")
    if blur.shape[0] != 1 or event.shape[0] != 1:
        raise ValueError("Tiled inference currently requires batch_size=1.")
    if blur.shape[-2:] != event.shape[-2:]:
        raise ValueError("Blur and event tensors must have the same spatial size.")
    if tile_size < 1:
        raise ValueError("--tile-size must be at least 1.")
    if overlap < 0 or overlap >= tile_size:
        raise ValueError("--tile-overlap must satisfy 0 <= overlap < tile_size.")

    height, width = blur.shape[-2:]
    tile_height = min(tile_size, height)
    tile_width = min(tile_size, width)
    if tile_height == height and tile_width == width:
        return model(blur, event)

    y_starts = tile_starts(height, tile_height, min(overlap, tile_height - 1))
    x_starts = tile_starts(width, tile_width, min(overlap, tile_width - 1))
    output_sum = None
    weight_sum = blur.new_zeros((1, 1, height, width))
    weight = None

    for y in y_starts:
        for x in x_starts:
            blur_tile = blur[:, :, y : y + tile_height, x : x + tile_width]
            event_tile = event[:, :, y : y + tile_height, x : x + tile_width]
            pred_tile = model(blur_tile, event_tile)
            if pred_tile.shape[-2:] != (tile_height, tile_width):
                raise ValueError(
                    "Model output size must match the input tile size, got "
                    f"{tuple(pred_tile.shape[-2:])} for {(tile_height, tile_width)}."
                )
            if output_sum is None:
                output_sum = pred_tile.new_zeros(
                    (pred_tile.shape[0], pred_tile.shape[1], height, width)
                )

            if weight is None:
                weight = feather_weight(
                    tile_height, tile_width, overlap, pred_tile.device, pred_tile.dtype
                )[None, None]
            output_sum[:, :, y : y + tile_height, x : x + tile_width] += (
                pred_tile * weight
            )
            weight_sum[:, :, y : y + tile_height, x : x + tile_width] += weight

    return output_sum / weight_sum.clamp_min(torch.finfo(output_sum.dtype).eps)


def main():
    args = parse_args()
    if args.tile_size is not None:
        if args.tile_size < 1:
            raise ValueError("--tile-size must be at least 1.")
        if args.tile_overlap < 0 or args.tile_overlap >= args.tile_size:
            raise ValueError("--tile-overlap must satisfy 0 <= overlap < tile size.")
    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    train_cfg = config.get("train", {})
    seed = int(train_cfg.get("seed", 42))
    deterministic = bool(train_cfg.get("deterministic", True))
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    device, rank, world_size = setup_runtime()
    set_global_seed(seed + rank, deterministic)

    val_opt = dict(config["datasets"]["val"])
    val_opt["split"] = "evaluation"
    val_opt["random_crop"] = False
    if args.full_resolution:
        val_opt["patch_size"] = None

    dataset = build_dataset(val_opt)
    if args.max_samples is not None:
        if args.max_samples < 1:
            raise ValueError("--max-samples must be at least 1.")
        dataset = Subset(dataset, range(min(args.max_samples, len(dataset))))
    if len(dataset) == 0:
        raise RuntimeError("No evaluation samples were found.")

    sampler = RankStrideSampler(dataset, rank, world_size)
    loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        num_workers=val_opt.get("num_workers", 0),
        pin_memory=device.type == "cuda",
    )

    model = build_deblur_model(**dict(config.get("model", {}))).to(device)
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"Checkpoint has no model_state_dict: {args.checkpoint}")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    if rank == 0:
        first = dataset[0]
        print(f"Config: {args.config}")
        print(f"Checkpoint: {args.checkpoint}")
        print(f"Evaluation samples: {len(dataset)} | world_size: {world_size}")
        print(f"Full resolution: {args.full_resolution}")
        print(
            "Tiled inference: "
            + (
                f"{args.tile_size}x{args.tile_size}, overlap={args.tile_overlap}"
                if args.tile_size is not None
                else "disabled"
            )
        )
        print(
            "First sample shapes: "
            f"blur={tuple(first['blur'].shape)} | "
            f"event={tuple(first['event'].shape)} | "
            f"gt={tuple(first['gt'].shape)}"
        )

    blur_psnr_sum = 0.0
    pred_psnr_sum = 0.0
    ssim_sum = 0.0
    sample_count = 0

    with torch.inference_mode():
        for batch in loader:
            blur = batch["blur"].to(device, non_blocking=True)
            event = batch["event"].to(device, non_blocking=True)
            gt = batch["gt"].to(device, non_blocking=True)
            if args.tile_size is None:
                pred = model(blur, event)
            else:
                pred = tiled_forward(
                    model, blur, event, args.tile_size, args.tile_overlap
                )

            blur_np = tensor_to_float_image(blur)
            pred_np = tensor_to_float_image(pred)
            gt_np = tensor_to_float_image(gt)
            blur_psnr_sum += calculate_psnr(gt_np, blur_np, data_range=1.0)
            pred_psnr_sum += calculate_psnr(gt_np, pred_np, data_range=1.0)
            ssim_sum += calculate_ssim(gt_np, pred_np, channel_axis=2, data_range=1.0)
            sample_count += 1

    totals = reduce_totals(
        [blur_psnr_sum, pred_psnr_sum, ssim_sum, sample_count], device
    )
    peak_mb = 0.0
    if device.type == "cuda":
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    peak = torch.tensor([peak_mb], dtype=torch.float64, device=device)
    if dist.is_initialized():
        dist.all_reduce(peak, op=dist.ReduceOp.MAX)

    if rank == 0:
        count = max(totals[3].item(), 1.0)
        avg_blur_psnr = totals[0].item() / count
        avg_pred_psnr = totals[1].item() / count
        avg_ssim = totals[2].item() / count
        print(
            f"Result -> Blur PSNR: {avg_blur_psnr:.4f} dB | "
            f"Pred PSNR: {avg_pred_psnr:.4f} dB | "
            f"Gain: {avg_pred_psnr - avg_blur_psnr:+.4f} dB | "
            f"SSIM: {avg_ssim:.6f}"
        )
        print(f"Peak allocated GPU memory per process (max): {peak.item():.1f} MiB")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
