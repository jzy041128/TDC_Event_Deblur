import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.dataset import events_to_voxel


def convert_one(args):
    event_path, dataroot, cache_root, bins, mode, overwrite = args
    event_path = Path(event_path)
    sequence = event_path.parents[1].name
    stem = event_path.stem
    output_path = Path(cache_root) / sequence / f"{stem}.npy"
    if output_path.exists() and not overwrite:
        return "skipped", str(output_path)

    gt_path = Path(dataroot) / sequence / "gt_down_corrected" / f"{stem}.png"
    blur_path = Path(dataroot) / sequence / "blur_down" / f"{stem}.png"
    if not gt_path.exists() or not blur_path.exists():
        raise FileNotFoundError(f"Missing RGB pair for {event_path}")
    with Image.open(gt_path) as image:
        width, height = image.size
    with Image.open(blur_path) as image:
        if image.size != (width, height):
            raise ValueError(f"Blur/GT size mismatch for {stem}: {image.size} vs {(width, height)}")

    voxel = events_to_voxel(event_path, height, width, bins, mode)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".npy.tmp")
    with open(temporary_path, "wb") as handle:
        np.save(handle, voxel)
    os.replace(temporary_path, output_path)
    return "written", str(output_path)


def main():
    parser = argparse.ArgumentParser(description="Cache REVD raw events as voxel grids.")
    parser.add_argument("--dataroot", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--bins", type=int, default=6)
    parser.add_argument("--mode", choices=("hard", "trilinear"), default="trilinear")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.bins < 1 or args.workers < 1:
        parser.error("--bins and --workers must be positive")

    dataroot = Path(args.dataroot)
    blur_paths = sorted(dataroot.glob("*/blur_down/*.png"))
    event_paths = []
    missing = []
    for blur_path in blur_paths:
        sequence = blur_path.parents[1].name
        stem = blur_path.stem
        gt_path = dataroot / sequence / "gt_down_corrected" / f"{stem}.png"
        event_path = dataroot / sequence / "warped_events" / f"{stem}.npz"
        if gt_path.exists() and event_path.exists():
            event_paths.append(event_path)
        else:
            missing.append((blur_path, gt_path, event_path))
    if missing:
        blur_path, gt_path, event_path = missing[0]
        raise RuntimeError(
            f"Found {len(missing)} incomplete RGB-led samples; first: "
            f"blur={blur_path.exists()}, gt={gt_path.exists()}, "
            f"event={event_path.exists()}, stem={blur_path.stem}"
        )

    all_events = set(dataroot.glob("*/warped_events/*.npz"))
    extra_events = all_events.difference(event_paths)
    if extra_events:
        print(
            f"Ignoring {len(extra_events)} unpaired event files without matching RGB/GT.",
            flush=True,
        )
    if args.limit is not None:
        event_paths = event_paths[: args.limit]
    if not event_paths:
        raise RuntimeError(f"No REVD event files found under {dataroot}")

    jobs = [
        (path, dataroot, args.cache_root, args.bins, args.mode, args.overwrite)
        for path in event_paths
    ]
    counts = {"written": 0, "skipped": 0}
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for index, (status, _) in enumerate(executor.map(convert_one, jobs), start=1):
            counts[status] += 1
            if index == 1 or index % 25 == 0 or index == len(jobs):
                print(
                    f"Processed {index}/{len(jobs)} | written={counts['written']} "
                    f"| skipped={counts['skipped']}",
                    flush=True,
                )

    print(
        f"Done: {len(jobs)} samples, {args.bins} bins, mode={args.mode}, "
        f"cache={args.cache_root}"
    )


if __name__ == "__main__":
    main()
