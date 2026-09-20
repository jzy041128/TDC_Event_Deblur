import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from data.dataset import EVRBEventDeblurDataset, build_dataset


class EVRBDatasetTest(unittest.TestCase):
    def _make_sample(self, root, event_shape=(16, 12, 18)):
        sequence = Path(root) / "00000"
        for folder in ("blur_processed", "gt_processed", "event_voxel", "events"):
            (sequence / folder).mkdir(parents=True, exist_ok=True)

        image = np.full((12, 18, 3), 127, dtype=np.uint8)
        Image.fromarray(image).save(sequence / "blur_processed" / "00001.png")
        Image.fromarray(image).save(sequence / "gt_processed" / "00001.png")
        voxel = np.linspace(-2.0, 2.0, np.prod(event_shape), dtype=np.float32).reshape(
            event_shape
        )
        np.savez(sequence / "event_voxel" / "00001.npz", data=voxel)
        return sequence

    def test_official_voxel_sample(self):
        with tempfile.TemporaryDirectory() as root:
            self._make_sample(root)
            dataset = build_dataset(
                {
                    "dataset_type": "evrb",
                    "dataroot": root,
                    "patch_size": None,
                    "event_bins": 16,
                }
            )
            sample = dataset[0]
            self.assertIsInstance(dataset, EVRBEventDeblurDataset)
            self.assertEqual(tuple(sample["blur"].shape), (3, 12, 18))
            self.assertEqual(tuple(sample["gt"].shape), (3, 12, 18))
            self.assertEqual(tuple(sample["event"].shape), (16, 12, 18))

    def test_aligned_crop_and_normalization(self):
        with tempfile.TemporaryDirectory() as root:
            self._make_sample(root)
            dataset = EVRBEventDeblurDataset(
                {
                    "dataroot": root,
                    "patch_size": 8,
                    "random_crop": False,
                    "event_bins": 16,
                    "norm_event": True,
                }
            )
            sample = dataset[0]
            self.assertEqual(tuple(sample["blur"].shape), (3, 8, 8))
            self.assertEqual(tuple(sample["event"].shape), (16, 8, 8))
            self.assertLessEqual(float(sample["event"].abs().max()), 1.0)

    def test_wrong_bin_count_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            self._make_sample(root, event_shape=(8, 12, 18))
            dataset = EVRBEventDeblurDataset(
                {"dataroot": root, "patch_size": None, "event_bins": 16}
            )
            with self.assertRaisesRegex(ValueError, "Expected a 16-bin CHW voxel"):
                dataset[0]


if __name__ == "__main__":
    unittest.main()
