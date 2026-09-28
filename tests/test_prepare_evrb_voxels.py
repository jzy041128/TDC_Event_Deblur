import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from tools.prepare_evrb_voxels import events_to_evrb_voxel, prepare_one


class EVRBVoxelTest(unittest.TestCase):
    def test_spatial_floor_and_temporal_linear(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.npz"
            np.savez(
                path,
                x=np.array([1.9, 1.1, 2.0, -1.0]),
                y=np.array([0.9, 0.1, 1.0, 0.0]),
                t=np.array([0, 5, 10, 5]),
                p=np.array([1, 0, 1, 1]),
            )
            voxel = events_to_evrb_voxel(path, 2, 3, 3)
            self.assertEqual(voxel.shape, (3, 2, 3))
            self.assertAlmostEqual(float(voxel[0, 0, 1]), 1.0)
            self.assertAlmostEqual(float(voxel[1, 0, 1]), -1.0)
            self.assertAlmostEqual(float(voxel[2, 1, 2]), 1.0)
            self.assertEqual(np.count_nonzero(voxel), 3)

    def test_temporal_interpolation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.npz"
            np.savez(
                path,
                x=np.array([0.0, 1.0, 2.0]),
                y=np.array([0.0, 0.0, 0.0]),
                t=np.array([0, 1, 4]),
                p=np.array([1, 1, 1]),
            )
            voxel = events_to_evrb_voxel(path, 1, 3, 3)
            self.assertAlmostEqual(float(voxel[0, 0, 1]), 0.5)
            self.assertAlmostEqual(float(voxel[1, 0, 1]), 0.5)

    def test_cache_is_separate_and_resumable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "EVRB" / "train" / "00000"
            cache = root / "EVRB_voxel6_matched" / "train" / "00000" / "event_voxel" / "00000.npz"
            for folder in ("blur_processed", "gt_processed", "events", "event_voxel"):
                (source / folder).mkdir(parents=True)
            image = Image.new("RGB", (3, 2))
            blur = source / "blur_processed" / "00000.png"
            gt = source / "gt_processed" / "00000.png"
            image.save(blur)
            image.save(gt)
            raw = source / "events" / "00000.npz"
            official = source / "event_voxel" / "00000.npz"
            np.savez(raw, x=[0.0], y=[0.0], t=[0], p=[1])
            np.savez(official, data=np.ones((16, 2, 3), dtype=np.float32))
            job = (raw, blur, gt, cache, official, 6, False)
            self.assertEqual(prepare_one(job)[0], "written")
            self.assertEqual(prepare_one(job)[0], "skipped")
            with np.load(cache) as data:
                self.assertEqual(data["data"].shape, (6, 2, 3))
            with np.load(official) as data:
                self.assertEqual(data["data"].shape, (16, 2, 3))


if __name__ == "__main__":
    unittest.main()
