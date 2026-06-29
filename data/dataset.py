import glob
import os
import random
from collections import OrderedDict

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")


def _to_chw_float_image(path):
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr)


def _event_list_to_voxel(npz_path, height, width, num_bins):
    data = np.load(npz_path)
    x = data["x"].astype(np.int64)
    y = data["y"].astype(np.int64)
    t = data["t"].astype(np.float32)
    p = data["p"].astype(np.float32)

    voxel = np.zeros((num_bins, height, width), dtype=np.float32)
    if x.size == 0:
        return voxel

    x = np.clip(x, 0, width - 1)
    y = np.clip(y, 0, height - 1)
    polarity = np.where(p > 0, 1.0, -1.0).astype(np.float32)
    t_min = float(t.min())
    t_max = float(t.max())
    if t_max > t_min:
        bins = ((t - t_min) / (t_max - t_min + 1e-6) * (num_bins - 1)).astype(np.int64)
    else:
        bins = np.zeros_like(x, dtype=np.int64)
    bins = np.clip(bins, 0, num_bins - 1)
    np.add.at(voxel, (bins, y, x), polarity)
    return voxel


def _load_event(path, height, width, num_bins):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        event = np.load(path).astype(np.float32)
    elif ext == ".npz":
        event = _event_list_to_voxel(path, height, width, num_bins)
    else:
        raise ValueError(f"Unsupported event file extension: {path}")

    if event.ndim == 3 and event.shape[0] != num_bins and event.shape[-1] == num_bins:
        event = np.transpose(event, (2, 0, 1))
    if event.ndim != 3:
        raise ValueError(f"Event tensor must be 3D, got {event.shape} from {path}")
    return torch.from_numpy(np.ascontiguousarray(event)).float()


def _crop_triplet(blur, gt, event, patch_size, random_crop=True):
    _, h, w = gt.shape
    th, tw = patch_size, patch_size
    if h <= th or w <= tw:
        return blur, gt, event

    if random_crop:
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
    else:
        i = (h - th) // 2
        j = (w - tw) // 2
    return (
        blur[:, i : i + th, j : j + tw],
        gt[:, i : i + th, j : j + tw],
        event[:, i : i + th, j : j + tw],
    )


def _normalize_event(event):
    scale = event.abs().amax()
    return event / (scale + 1e-6)


class H5EventDeblurDataset(Dataset):
    def __init__(self, opt_dataset):
        super().__init__()
        self.dataroot = opt_dataset["dataroot"]
        self.patch_size = opt_dataset.get("patch_size", 256)
        self.random_crop = opt_dataset.get("random_crop", True)
        self.split = opt_dataset.get("split", "dataset")
        self.max_open_h5 = opt_dataset.get("max_open_h5", 2)
        self.norm_event = opt_dataset.get("norm_event", False)
        self.h5_cache = OrderedDict()

        self.h5_files = sorted(glob.glob(os.path.join(self.dataroot, "*.h5")))
        if len(self.h5_files) == 0:
            print(f"Warning: no .h5 files found in {self.dataroot}")

        self.samples = []
        for h5_path in self.h5_files:
            with h5py.File(h5_path, "r") as f:
                num_frames = len(f["images"].keys())
                for i in range(num_frames):
                    self.samples.append((h5_path, i))

        print(
            f"Loaded {self.split}: {len(self.h5_files)} h5 files, "
            f"{len(self.samples)} frames."
        )

    def __len__(self):
        return len(self.samples)

    def _get_h5_file(self, h5_path):
        if self.max_open_h5 <= 0:
            return h5py.File(h5_path, "r")
        if h5_path in self.h5_cache:
            self.h5_cache.move_to_end(h5_path)
            return self.h5_cache[h5_path]
        while len(self.h5_cache) >= self.max_open_h5:
            _, old_h5_file = self.h5_cache.popitem(last=False)
            old_h5_file.close()
        self.h5_cache[h5_path] = h5py.File(h5_path, "r")
        return self.h5_cache[h5_path]

    def __getstate__(self):
        state = self.__dict__.copy()
        state["h5_cache"] = {}
        return state

    def __del__(self):
        for h5_file in getattr(self, "h5_cache", {}).values():
            try:
                h5_file.close()
            except Exception:
                pass

    def __getitem__(self, index):
        h5_path, frame_idx = self.samples[index]
        f = self._get_h5_file(h5_path)
        try:
            img_blur = f["images"][f"image{frame_idx:09d}"][...]
            img_gt = f["sharp_images"][f"image{frame_idx:09d}"][...]
            event_voxel = f["voxels"][f"voxel{frame_idx:09d}"][...]
        finally:
            if self.max_open_h5 <= 0:
                f.close()

        img_blur = torch.from_numpy(img_blur).float() / 255.0
        img_gt = torch.from_numpy(img_gt).float() / 255.0
        event_tensor = torch.from_numpy(event_voxel).float()
        if self.norm_event:
            event_tensor = _normalize_event(event_tensor)

        img_blur, img_gt, event_tensor = _crop_triplet(
            img_blur, img_gt, event_tensor, self.patch_size, self.random_crop
        )
        return {"blur": img_blur, "gt": img_gt, "event": event_tensor}


class ImageEventDeblurDataset(Dataset):
    def __init__(self, opt_dataset):
        super().__init__()
        self.blur_root = opt_dataset["blur_root"]
        self.gt_root = opt_dataset["gt_root"]
        self.event_root = opt_dataset["event_root"]
        self.patch_size = opt_dataset.get("patch_size", 256)
        self.random_crop = opt_dataset.get("random_crop", True)
        self.split = opt_dataset.get("split", "dataset")
        self.norm_event = opt_dataset.get("norm_event", False)
        self.event_bins = opt_dataset.get("event_bins", 6)
        self.event_ext = opt_dataset.get("event_ext")

        self.samples = self._build_samples()
        print(f"Loaded {self.split}: {len(self.samples)} image/event samples.")

    def _build_samples(self):
        image_files = []
        for ext in IMAGE_EXTENSIONS:
            image_files.extend(glob.glob(os.path.join(self.blur_root, "**", f"*{ext}"), recursive=True))
        image_files = sorted(image_files)

        samples = []
        for blur_path in image_files:
            rel = os.path.relpath(blur_path, self.blur_root)
            gt_path = os.path.join(self.gt_root, rel)
            stem, _ = os.path.splitext(rel)
            event_path = self._find_event_path(stem)
            if os.path.exists(gt_path) and event_path is not None:
                samples.append((blur_path, gt_path, event_path))

        if len(samples) == 0:
            raise RuntimeError(
                "No aligned samples found. Check blur_root, gt_root, and event_root."
            )
        return samples

    def _find_event_path(self, rel_stem):
        if self.event_ext:
            path = os.path.join(self.event_root, rel_stem + self.event_ext)
            return path if os.path.exists(path) else None
        for ext in (".npy", ".npz"):
            path = os.path.join(self.event_root, rel_stem + ext)
            if os.path.exists(path):
                return path
        return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        blur_path, gt_path, event_path = self.samples[index]
        blur = _to_chw_float_image(blur_path)
        gt = _to_chw_float_image(gt_path)
        _, height, width = gt.shape
        event = _load_event(event_path, height, width, self.event_bins)
        if self.norm_event:
            event = _normalize_event(event)

        blur, gt, event = _crop_triplet(
            blur, gt, event, self.patch_size, self.random_crop
        )
        return {"blur": blur, "gt": gt, "event": event}


def build_dataset(opt_dataset):
    dataset_type = opt_dataset.get("dataset_type", "h5")
    if dataset_type == "h5":
        return H5EventDeblurDataset(opt_dataset)
    if dataset_type in {"image_event", "image_npz", "image_npy", "server_npz", "server_voxel"}:
        return ImageEventDeblurDataset(opt_dataset)
    raise ValueError(f"Unknown dataset_type: {dataset_type}")


EventDeblurDataset = H5EventDeblurDataset


if __name__ == "__main__":
    dummy_opt = {
        "dataset_type": "h5",
        "dataroot": "./datasets/train",
        "patch_size": 256,
    }
    dataset = build_dataset(dummy_opt)
    if len(dataset) > 0:
        data = dataset[0]
        print("Read sample successfully.")
        print("Blur shape:", data["blur"].shape)
        print("GT shape:", data["gt"].shape)
        print("Event shape:", data["event"].shape)
