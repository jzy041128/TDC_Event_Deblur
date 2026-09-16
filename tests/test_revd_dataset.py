import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from data.dataset import REVDEventDeblurDataset, events_to_voxel


class REVDDatasetTest(unittest.TestCase):
    def _make_sample(self, root):
        sequence = Path(root) / "sequence_a"
        for folder in ("blur_down", "gt_down_corrected", "warped_events"):
            (sequence / folder).mkdir(parents=True, exist_ok=True)
        image = np.full((6, 8, 3), 127, dtype=np.uint8)
        Image.fromarray(image).save(sequence / "blur_down" / "00001.png")
        Image.fromarray(image).save(sequence / "gt_down_corrected" / "00001.png")
        np.savez(
            sequence / "warped_events" / "00001.npz",
            x=np.array([1.25, 6.25]),
            y=np.array([2.50, 4.25]),
            t=np.array([100, 200]),
            p=np.array([1, 0], dtype=np.int16),
        )
        return sequence

    def test_raw_revd_sample(self):
        with tempfile.TemporaryDirectory() as root:
            self._make_sample(root)
            dataset = REVDEventDeblurDataset(
                {
                    "dataroot": root,
                    "patch_size": None,
                    "event_bins": 6,
                    "event_voxel_mode": "trilinear",
                }
            )
            sample = dataset[0]
            self.assertEqual(tuple(sample["blur"].shape), (3, 6, 8))
            self.assertEqual(tuple(sample["event"].shape), (6, 6, 8))
            self.assertAlmostEqual(float(sample["event"].sum()), 0.0, places=5)

    def test_cached_revd_sample(self):
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as cache:
            sequence = self._make_sample(root)
            voxel = events_to_voxel(
                sequence / "warped_events" / "00001.npz", 6, 8, 6, "trilinear"
            )
            cache_dir = Path(cache) / sequence.name
            cache_dir.mkdir(parents=True)
            np.save(cache_dir / "00001.npy", voxel)
            dataset = REVDEventDeblurDataset(
                {"dataroot": root, "event_cache_root": cache, "patch_size": None}
            )
            self.assertEqual(tuple(dataset[0]["event"].shape), (6, 6, 8))


if __name__ == "__main__":
    unittest.main()
