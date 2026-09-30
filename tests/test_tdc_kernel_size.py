import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from models.tdc_deblur_net import build_deblur_model
from models.tdc_module import ShortTermTDC3D


ROOT = Path(__file__).resolve().parents[1]


class TDCKernelSizeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_default_five_tap_matches_original_formula(self):
        torch.manual_seed(42)
        layer = ShortTermTDC3D(2, 3)
        weight = layer.conv.weight
        expected_weight = torch.zeros_like(weight)
        expected_weight[:, :, 4] = weight[:, :, 4]
        expected_weight[:, :, 3] = weight[:, :, 3] - weight[:, :, 4]
        expected_weight[:, :, 2] = weight[:, :, 2] - weight[:, :, 3]
        expected_weight[:, :, 1] = weight[:, :, 1] - weight[:, :, 2]
        expected_weight[:, :, 0] = -weight[:, :, 1]

        self.assertTrue(torch.equal(layer.short_term_weight(), expected_weight))
        x = torch.randn(1, 2, 16, 8, 8)
        expected = F.conv3d(x, expected_weight, padding=(2, 1, 1))
        self.assertTrue(torch.equal(layer(x), expected))

    def test_seven_tap_preserves_time_and_backpropagates(self):
        torch.manual_seed(42)
        layer = ShortTermTDC3D(2, 3, stride=(1, 2, 2), kernel_size=7)
        x = torch.randn(1, 2, 16, 8, 8, requires_grad=True)
        output = layer(x)
        self.assertEqual(output.shape, (1, 3, 16, 4, 4))

        weight = layer.conv.weight
        expected_weight = torch.zeros_like(weight)
        for tap in range(1, 7):
            expected_weight[:, :, tap] = weight[:, :, tap] - (
                weight[:, :, tap + 1] if tap < 6 else 0
            )
        expected_weight[:, :, 0] = -weight[:, :, 1]
        self.assertTrue(torch.equal(layer.short_term_weight(), expected_weight))

        output.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(weight.grad)

    def test_invalid_temporal_kernel(self):
        for kernel_size in (0, 2, 6):
            with self.subTest(kernel_size=kernel_size):
                with self.assertRaises(ValueError):
                    ShortTermTDC3D(1, 2, kernel_size=kernel_size)

    def test_evrb_experiment_changes_only_kernel_and_epoch_limit(self):
        with (ROOT / "configs/train_tdc_evrb_scratch.yml").open(encoding="utf-8") as stream:
            baseline = yaml.safe_load(stream)
        with (ROOT / "configs/train_tdc_evrb_16bin_tdc7_scratch.yml").open(encoding="utf-8") as stream:
            trial = yaml.safe_load(stream)
        self.assertEqual(trial["model"].pop("tdc_kernel_size"), 7)
        trial["name"] = baseline["name"]
        trial["train"]["num_epochs"] = baseline["train"]["num_epochs"]
        self.assertEqual(trial, baseline)

    def test_model_uses_seven_taps_at_all_three_scales(self):
        model = build_deblur_model(base_dim=8, event_in=16, tdc_kernel_size=7)
        layers = [model.event3d_stem, *model.event3d_down]
        self.assertEqual([block.tdc.conv.kernel_size for block in layers], [(7, 3, 3)] * 3)
        model.eval()
        with torch.no_grad():
            output = model(torch.randn(1, 3, 32, 32), torch.randn(1, 16, 32, 32))
        self.assertEqual(output.shape, (1, 3, 32, 32))


if __name__ == "__main__":
    unittest.main()
