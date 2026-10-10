import copy
import io
import unittest
from pathlib import Path

import torch
import yaml

from models.tdc_deblur_net import build_deblur_model
from models.tdc_module import ShortTermTDC3D, ShortTermTDCBlock3D


ROOT = Path(__file__).resolve().parents[1]
GATED_MODE = "rgb_gated_event_then_rgb_4ca"


class Event3DConvTypeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_plain_mode_is_exactly_the_underlying_conv3d(self):
        for bins, kernel in ((6, 5), (16, 5), (16, 7)):
            with self.subTest(bins=bins, kernel=kernel):
                layer = ShortTermTDC3D(2, 4, stride=(1, 2, 2), kernel_size=kernel, conv_type="conv3d")
                x = torch.randn(1, 2, bins, 9, 11, requires_grad=True)
                output = layer(x)
                self.assertTrue(torch.equal(output, layer.conv(x)))
                self.assertEqual(output.shape, (1, 4, bins, 5, 6))
                output.square().mean().backward()
                self.assertTrue(torch.isfinite(x.grad).all())
                self.assertTrue(torch.isfinite(layer.conv.weight.grad).all())
                self.assertGreater(layer.conv.weight.grad[:, :, 0].abs().sum().item(), 0)

    def test_block_keeps_spatial_convolution_and_activation(self):
        block = ShortTermTDCBlock3D(2, 4, conv_type="conv3d")
        self.assertEqual(block.tdc.conv.kernel_size, (5, 3, 3))
        self.assertEqual(block.tdc.conv.padding, (2, 1, 1))
        self.assertIsNone(block.tdc.conv.bias)
        self.assertEqual(block.spatial.kernel_size, (1, 3, 3))
        self.assertIsNotNone(block.spatial.bias)
        x = torch.randn(1, 2, 6, 9, 11)
        expected = block.act(block.spatial(block.act(block.tdc.conv(x))))
        self.assertTrue(torch.equal(block(x), expected))

    def test_omitted_switch_matches_explicit_tdc_bitwise(self):
        torch.manual_seed(42)
        legacy = build_deblur_model(base_dim=8, fusion_mode=GATED_MODE).eval()
        torch.manual_seed(42)
        explicit = build_deblur_model(base_dim=8, fusion_mode=GATED_MODE, event3d_conv_type="tdc").eval()
        for key, value in legacy.state_dict().items():
            self.assertTrue(torch.equal(value, explicit.state_dict()[key]), key)
        inputs = (torch.randn(1, 3, 17, 19), torch.randn(1, 6, 17, 19))
        with torch.no_grad():
            self.assertTrue(torch.equal(legacy(*inputs), explicit(*inputs)))

    def test_modes_preserve_parameter_names_count_and_initialization(self):
        torch.manual_seed(42)
        tdc = build_deblur_model(base_dim=8, fusion_mode=GATED_MODE)
        torch.manual_seed(42)
        plain = build_deblur_model(base_dim=8, fusion_mode=GATED_MODE, event3d_conv_type="conv3d")
        self.assertEqual(set(tdc.state_dict()), set(plain.state_dict()))
        self.assertEqual(sum(p.numel() for p in tdc.parameters()), sum(p.numel() for p in plain.parameters()))
        for key, value in tdc.state_dict().items():
            self.assertTrue(torch.equal(value, plain.state_dict()[key]), key)

    def test_all_scales_switch_and_gated_network_backpropagates(self):
        for bins, kernel in ((6, 5), (16, 7)):
            with self.subTest(bins=bins, kernel=kernel):
                model = build_deblur_model(
                    base_dim=8, event_in=bins, fusion_mode=GATED_MODE,
                    tdc_kernel_size=kernel, event3d_conv_type="conv3d",
                )
                for block in (model.event3d_stem, *model.event3d_down):
                    self.assertEqual(block.tdc.conv_type, "conv3d")
                    self.assertEqual(block.tdc.conv.kernel_size, (kernel, 3, 3))
                    self.assertEqual(block.spatial.kernel_size, (1, 3, 3))
                output = model(torch.randn(1, 3, 17, 19), torch.randn(1, bins, 17, 19))
                self.assertEqual(output.shape, (1, 3, 17, 19))
                output.square().mean().backward()
                for name, parameter in model.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_invalid_switch_is_rejected(self):
        for factory in (ShortTermTDC3D, ShortTermTDCBlock3D):
            with self.assertRaisesRegex(ValueError, "event3d_conv_type"):
                factory(1, 2, conv_type="typo")
        with self.assertRaisesRegex(ValueError, "event3d_conv_type"):
            build_deblur_model(base_dim=8, event3d_conv_type="typo")

    def test_checkpoint_guard_preserves_legacy_tdc_and_rejects_mismatches(self):
        tdc = build_deblur_model(base_dim=8)
        plain = build_deblur_model(base_dim=8, event3d_conv_type="conv3d")
        legacy = {"model_state_dict": tdc.state_dict()}
        tdc.validate_event3d_checkpoint(legacy)
        tdc.load_state_dict(legacy["model_state_dict"], strict=True)
        with self.assertRaisesRegex(ValueError, "does not match"):
            plain.validate_event3d_checkpoint(legacy)
        with self.assertRaisesRegex(ValueError, "does not match"):
            tdc.validate_event3d_checkpoint({"event3d_conv_type": "conv3d"})

    def test_plain_checkpoint_round_trip(self):
        model = build_deblur_model(base_dim=8, fusion_mode=GATED_MODE, event3d_conv_type="conv3d").eval()
        buffer = io.BytesIO()
        torch.save({"model_state_dict": model.state_dict(), "event3d_conv_type": model.event3d_conv_type}, buffer)
        buffer.seek(0)
        checkpoint = torch.load(buffer, weights_only=True)
        clone = build_deblur_model(base_dim=8, fusion_mode=GATED_MODE, event3d_conv_type="conv3d").eval()
        clone.validate_event3d_checkpoint(checkpoint)
        clone.load_state_dict(checkpoint["model_state_dict"], strict=True)
        inputs = (torch.randn(1, 3, 17, 19), torch.randn(1, 6, 17, 19))
        with torch.no_grad():
            self.assertTrue(torch.equal(model(*inputs), clone(*inputs)))

    def test_reblur_configs_change_only_fusion_or_convolution(self):
        def load(name):
            return yaml.safe_load((ROOT / "configs" / name).read_text(encoding="utf-8"))

        gated = load("train_tdc_reblur_rgb_gated4ca_scratch.yml")
        plain = load("train_tdc_reblur_rgb_gated4ca_conv3d_scratch.yml")
        direct = load("train_tdc_reblur_direct2ca_scratch.yml")
        self.assertEqual(plain["model"]["event3d_conv_type"], "conv3d")
        self.assertEqual(direct["model"]["fusion_mode"], "direct_event_to_rgb_2ca")
        for config in (gated, plain, direct):
            self.assertEqual(config["model"]["event_in"], 6)
            self.assertEqual(config["model"]["tdc_kernel_size"], 5)
            self.assertEqual(config["train"]["num_epochs"], 200)
            self.assertEqual(config["train"]["learning_rate"], 0.0002)
            self.assertEqual(config["train"]["val_interval"], 5)
            self.assertEqual(config["train"]["seed"], 42)
            self.assertEqual(config["datasets"]["train"]["patch_size"], 256)
            self.assertIsNone(config["datasets"]["val"]["patch_size"])
            self.assertIsNone(config["path"]["pretrain_model"])
            self.assertIsNone(config["path"]["resume_state"])
        for config in (plain, direct):
            config["name"] = gated["name"]
            config["model"]["event3d_conv_type"] = gated["model"]["event3d_conv_type"]
            config["model"]["fusion_mode"] = gated["model"]["fusion_mode"]
            self.assertEqual(config, gated)

        baseline = load("train_tdc_reblur_scratch.yml")
        trial = copy.deepcopy(gated)
        trial["name"] = baseline["name"]
        trial["model"]["fusion_mode"] = baseline["model"]["fusion_mode"]
        for key in ("event_in", "tdc_kernel_size", "event3d_conv_type"):
            trial["model"].pop(key)
        trial["train"]["num_epochs"] = baseline["train"]["num_epochs"]
        self.assertEqual(trial, baseline)


if __name__ == "__main__":
    unittest.main()
