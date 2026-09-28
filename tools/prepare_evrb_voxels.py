import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image


def events_to_evrb_voxel(event_path, height, width, bins):
    with np.load(event_path) as data:
        x = data["x"].astype(np.float64)
        y = data["y"].astype(np.float64)
        t = data["t"].astype(np.float64)
        p = data["p"]

    voxel = np.zeros((bins, height, width), dtype=np.float32)
    valid = (
        np.isfinite(x) & np.isfinite(y) & np.isfinite(t)
        & (x >= 0) & (x < width) & (y >= 0) & (y < height)
    )
    x, y, t, p = x[valid], y[valid], t[valid], p[valid]
    if not t.size:
        return voxel

    sign = np.where(p > 0, 1.0, -1.0).astype(np.float32)
    position = (t - t.min()) / max(float(t.max() - t.min()), 1.0) * (bins - 1)
    b0 = np.floor(position).astype(np.int64)
    b1 = np.minimum(b0 + 1, bins - 1)
    w1 = (position - b0).astype(np.float32)
    xi = np.floor(x).astype(np.int64)
    yi = np.floor(y).astype(np.int64)
    keep = (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height)
    np.add.at(voxel, (b0[keep], yi[keep], xi[keep]), sign[keep] * (1 - w1[keep]))
    np.add.at(voxel, (b1[keep], yi[keep], xi[keep]), sign[keep] * w1[keep])
    return voxel


def prepare_one(job):
    event_path, blur_path, gt_path, output_path, official_path, bins, verify = job
    with Image.open(gt_path) as image:
        width, height = image.size
    with Image.open(blur_path) as image:
        if image.size != (width, height):
            raise ValueError(f"Blur/GT size mismatch: {blur_path}")

    if output_path is not None and output_path.exists():
        with np.load(output_path) as data:
            shape = data["data"].shape
        if shape != (bins, height, width):
            raise ValueError(f"Existing cache has wrong shape {shape}: {output_path}")
        return "skipped", 0.0

    voxel = events_to_evrb_voxel(event_path, height, width, bins)
    if verify:
        with np.load(official_path) as data:
            official = data["data"]
        if official.shape != voxel.shape:
            raise ValueError(f"Official voxel has wrong shape: {official_path}")
        error = float(np.max(np.abs(official - voxel)))
        if not np.allclose(official, voxel, rtol=1e-4, atol=1e-4):
            raise ValueError(f"Official voxel mismatch: {official_path}, max error={error}")
        return "verified", error

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".npz.tmp")
    with open(temporary_path, "wb") as handle:
        np.savez_compressed(handle, data=voxel)
    os.replace(temporary_path, output_path)
    return "written", 0.0


def main():
    parser = argparse.ArgumentParser(
        description="Build EVRB voxels with spatial floor and temporal linear interpolation."
    )
    parser.add_argument("--dataroot", required=True, help="EVRB train or test directory")
    parser.add_argument("--cache-root", help="Separate output directory; never the EVRB source")
    parser.add_argument("--bins", type=int, default=6)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--verify-official", action="store_true", help="Compare against official 16-bin without writing")
    args = parser.parse_args()
    if args.bins < 2 or args.workers < 1 or (args.limit is not None and args.limit < 1):
        parser.error("--bins >= 2, --workers >= 1, and --limit >= 1 are required")
    if args.verify_official:
        if args.bins != 16 or args.cache_root:
            parser.error("--verify-official requires --bins 16 and no --cache-root")
    elif not args.cache_root:
        parser.error("--cache-root is required when generating voxels")

    dataroot = Path(args.dataroot).resolve()
    if args.cache_root:
        cache_root = Path(args.cache_root).resolve()
        source_root = dataroot.parent if dataroot.name in {"train", "test"} else dataroot
        if source_root == cache_root or source_root in cache_root.parents:
            parser.error("Cache must be outside the EVRB source tree")
    else:
        cache_root = None

    blur_paths = sorted(dataroot.glob("*/blur_processed/*.png"))
    if not blur_paths:
        raise RuntimeError(f"No EVRB images under {dataroot}")
    jobs = []
    for blur_path in blur_paths:
        sequence_dir = blur_path.parents[1]
        stem = blur_path.stem
        gt_path = sequence_dir / "gt_processed" / f"{stem}.png"
        event_path = sequence_dir / "events" / f"{stem}.npz"
        official_path = sequence_dir / "event_voxel" / f"{stem}.npz"
        if not gt_path.is_file() or not event_path.is_file():
            raise FileNotFoundError(f"Missing RGB/GT/raw events for {blur_path}")
        if args.verify_official and not official_path.is_file():
            raise FileNotFoundError(f"Missing official voxel: {official_path}")
        output_path = cache_root / sequence_dir.name / "event_voxel" / f"{stem}.npz" if cache_root else None
        jobs.append((event_path, blur_path, gt_path, output_path, official_path, args.bins, args.verify_official))

    if args.limit is not None:
        jobs = jobs[:args.limit]
    counts = {"written": 0, "skipped": 0, "verified": 0}
    max_error = 0.0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for index, (status, error) in enumerate(executor.map(prepare_one, jobs), 1):
            counts[status] += 1
            max_error = max(max_error, error)
            if index == 1 or index % 25 == 0 or index == len(jobs):
                print(f"Processed {index}/{len(jobs)} | "
                      f"written={counts['written']} skipped={counts['skipped']} "
                      f"verified={counts['verified']}", flush=True)
    print(f"Done: {len(jobs)} samples, {args.bins} bins, max verify error={max_error:.9g}")
    if cache_root:
        print(f"Cache: {cache_root}")


if __name__ == "__main__":
    main()
