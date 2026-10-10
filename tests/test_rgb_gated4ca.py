import copy
import unittest
from pathlib import Path

import torch
import yaml

from models.tdc_deblur_net import (
    EventThenRGBFourCrossAttention2D,
    RGBGatedEventThenRGBFourCrossAttention2D,
    RGBGuidedEventExchangeGate2D,
    build_deblur_model,
)


ROOT = Path(__file__).resolve().parents[1]
GATED_MODE = "rgb_gated_event_then_rgb_4ca"


def make_fusion(gated=False):
    args = (8, "channel", 8, 2, False, 0.1)
    if gated:
        return RGBGatedEventThenRGBFourCrossAttention2D(*args)
    return EventThenRGBFourCrossAttention2D(*args, cross_event=True)


class RGBGated4CATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_neutral_gates_preserve_all_shared_initial_weights_and_prediction(self):
        torch.manual_seed(42)
        baseline = build_deblur_model(base_dim=8).eval()
        torch.manual_seed(42)
        gated = build_deblur_model(base_dim=8, fusion_mode=GATED_MODE).eval()
        for key, value in baseline.state_dict().items():
            self.assertTrue(torch.equal(value, gated.state_dict()[key]), key)
        inputs = (torch.randn(1, 3, 17, 19), torch.randn(1, 6, 17, 19))
        with torch.no_grad():
            self.assertTrue(torch.equal(baseline(*inputs), gated(*inputs)))

    def test_gate_shapes_range_and_rgb_conditioning(self):
        gate = RGBGuidedEventExchangeGate2D(8)
        inputs = [torch.randn(2, 8, 9, 11) for _ in range(3)]
        for gain in gate(*inputs):
            self.assertEqual(gain.shape, (2, 1, 9, 11))
            self.assertTrue(torch.equal(gain, torch.ones_like(gain)))
        with torch.no_grad():
            gate.to_logits.weight.normal_()
        first = gate(*inputs)
        second = gate(torch.flip(inputs[0], dims=[1]), *inputs[1:])
        for a, b in zip(first, second):
            self.assertTrue((a >= 0).all() and (a <= 2).all())
            self.assertFalse(torch.equal(a, b))

    def test_two_directional_gates_multiply_only_projected_exchange_residuals(self):
        module = make_fusion(gated=True)
        rgb, e2, e3 = [torch.randn(1, 8, 9, 11) for _ in range(3)]
        with torch.no_grad():
            module.exchange_gate.to_logits.bias.copy_(torch.tensor([-1.0, 1.0]))
        g2, g3 = module.exchange_gate(rgb, e2, e3)
        expected_e2 = e2 + module.gamma_e2 * g2 * module.inject_e2(module.e2_from_e3(e2, e3, e3))
        expected_e3 = e3 + module.gamma_e3 * g3 * module.inject_e3(module.e3_from_e2(e3, e2, e2))
        expected = module.fuse_rgb(rgb, expected_e2, expected_e3)
        self.assertTrue(torch.equal(module(rgb, e2, e3), expected))
        self.assertFalse(torch.equal(g2, g3))

    def test_zero_initialized_head_learns_then_passes_gradients_to_rgb_and_gate_body(self):
        module = make_fusion(gated=True)
        rgb, e2, e3 = [torch.randn(1, 8, 9, 11) for _ in range(3)]
        optimizer = torch.optim.AdamW(module.parameters(), lr=0.001)
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            module(rgb, e2, e3).square().mean().backward()
            for name, parameter in module.named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(module.exchange_gate.to_logits.weight.grad.abs().sum().item(), 0)
            if step == 1:
                self.assertGreater(module.exchange_gate.reduce.weight.grad.abs().sum().item(), 0)
            optimizer.step()

    def test_pretraining_requires_complete_gated_weights(self):
        baseline = make_fusion().eval()
        gated = make_fusion(gated=True).eval()
        with self.assertRaisesRegex(RuntimeError, "Missing key"):
            gated.load_state_dict(baseline.state_dict(), strict=True)
        bad = dict(gated.state_dict())
        bad.pop("inject_e2.weight")
        with self.assertRaisesRegex(RuntimeError, "Missing key"):
            gated.load_state_dict(bad, strict=True)
        partial_gate = dict(gated.state_dict())
        partial_gate.pop("exchange_gate.to_logits.weight")
        with self.assertRaisesRegex(RuntimeError, "Missing key"):
            gated.load_state_dict(partial_gate, strict=True)
        wrong_shape = dict(gated.state_dict())
        wrong_shape["inject_e2.weight"] = torch.zeros(1)
        with self.assertRaises(RuntimeError):
            gated.load_state_dict(wrong_shape, strict=True)

    def test_gated_checkpoint_round_trip_keeps_learned_gates(self):
        source = make_fusion(gated=True)
        with torch.no_grad():
            source.exchange_gate.to_logits.bias.copy_(torch.tensor([0.2, -0.3]))
        clone = make_fusion(gated=True)
        clone.load_state_dict(source.state_dict(), strict=True)
        for key, value in source.state_dict().items():
            self.assertTrue(torch.equal(value, clone.state_dict()[key]), key)
        with self.assertRaises(RuntimeError):
            make_fusion().load_state_dict(source.state_dict(), strict=True)

    def test_six_and_sixteen_bins_and_both_tdc_kernels_backpropagate(self):
        for bins, kernel in ((6, 5), (16, 5), (16, 7)):
            with self.subTest(bins=bins, kernel=kernel):
                model = build_deblur_model(
                    base_dim=8, event_in=bins, tdc_kernel_size=kernel, fusion_mode=GATED_MODE
                )
                output = model(torch.randn(1, 3, 16, 20), torch.randn(1, bins, 16, 20))
                self.assertEqual(output.shape, (1, 3, 16, 20))
                output.square().mean().backward()
                for name, parameter in model.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)

    def test_gate_configs_are_provided_and_evrb_configs_match_baseline(self):
        files = {path.name for path in (ROOT / "configs").glob("*rgb_gated4ca*.yml")}
        self.assertEqual(files, {
            "train_tdc_evrb_rgb_gated4ca_scratch.yml",
            "eval_tdc_evrb_rgb_gated4ca_full.yml",
            "train_tdc_evrb_6bin_rgb_gated4ca_scratch.yml",
            "eval_tdc_evrb_6bin_rgb_gated4ca_full.yml",
            "train_tdc_reblur_rgb_gated4ca_scratch.yml",
            "train_tdc_reblur_rgb_gated4ca_conv3d_scratch.yml",
            "train_tdc_gopro_rgb_gated4ca_scratch.yml",
            "train_tdc_gopro_rgb_gated4ca_conv3d_scratch.yml",
        })
        for suffix, bins in (("", 16), ("_6bin", 6)):
            with self.subTest(bins=bins):
                config = yaml.safe_load((ROOT / f"configs/train_tdc_evrb{suffix}_rgb_gated4ca_scratch.yml").read_text())
                baseline = yaml.safe_load((ROOT / f"configs/train_tdc_evrb{suffix}_scratch.yml").read_text(encoding="utf-8"))
                evaluation = yaml.safe_load((ROOT / f"configs/eval_tdc_evrb{suffix}_rgb_gated4ca_full.yml").read_text())
                self.assertEqual(config["model"]["fusion_mode"], GATED_MODE)
                self.assertEqual(config["model"]["tdc_kernel_size"], 5)
                self.assertEqual(config["model"]["event_in"], bins)
                self.assertEqual(config["train"]["num_epochs"], 400)
                self.assertIsNone(config["path"]["pretrain_model"])
                self.assertIsNone(config["path"]["resume_state"])
                self.assertEqual(evaluation["model"], config["model"])
                expected_val = copy.deepcopy(config["datasets"]["val"])
                expected_val["patch_size"] = None
                self.assertEqual(evaluation["datasets"]["val"], expected_val)
                trial = copy.deepcopy(config)
                trial["name"] = baseline["name"]
                trial["model"]["fusion_mode"] = baseline["model"]["fusion_mode"]
                trial["model"].pop("tdc_kernel_size")
                trial["train"]["num_epochs"] = baseline["train"]["num_epochs"]
                self.assertEqual(trial, baseline)


if __name__ == "__main__":
    unittest.main()
