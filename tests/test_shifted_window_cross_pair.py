import copy
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from models.tdc_deblur_net import (
    ShiftedWindowCrossAttentionPair2D,
    WindowCrossAttention2D,
    build_deblur_model,
    make_cross_attention_2d,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {
    "A": "train_tdc_reblur_rgb_gated4ca_channel_3gpu_scratch.yml",
    "C": "train_tdc_reblur_rgb_gated4ca_swin_ca_3gpu_scratch.yml",
    "D": "train_tdc_reblur_rgb_gated4ca_swin_sa_ca_3gpu_scratch.yml",
}


class ShiftedWindowCrossPairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_pair_updates_query_but_keeps_source_key_and_value(self):
        module = ShiftedWindowCrossAttentionPair2D(8, 8, 2)
        q, k, v = [torch.randn(1, 8, 9, 11) for _ in range(3)]
        qp, kp, vp = [F.pad(x, (0, 5, 0, 7)) for x in (q, k, v)]
        first = module.fixed(qp, kp, vp)
        expected = (first + module.shifted(qp + first, kp, vp))[:, :, :9, :11]
        self.assertTrue(torch.equal(module(q, k, v), expected))
        self.assertEqual(module.fixed.shift_size, 0)
        self.assertEqual(module.shifted.shift_size, 4)

    def test_zero_attention_returns_zero_delta_not_original_query(self):
        module = ShiftedWindowCrossAttentionPair2D(8, 8, 2)
        with torch.no_grad():
            for block in (module.fixed, module.shifted):
                block.proj.weight.zero_()
                block.proj.bias.zero_()
        inputs = [torch.randn(1, 8, 9, 11) for _ in range(3)]
        self.assertTrue(torch.equal(module(*inputs), torch.zeros_like(inputs[0])))

    def test_pair_crosses_fixed_boundary_without_wrapping_image_edges(self):
        module = ShiftedWindowCrossAttentionPair2D(4, 8, 1).eval()
        with torch.no_grad():
            for block in (module.fixed, module.shifted):
                block.q.weight.zero_()
                block.k.weight.zero_()
                block.v.weight.copy_(torch.eye(4).view(4, 4, 1, 1))
                block.proj.weight.copy_(torch.eye(4).view(4, 4, 1, 1))
                block.proj.bias.zero_()
            q, k, value = [torch.randn(1, 4, 16, 16) for _ in range(3)]
            changed = value.clone()
            changed[:, :, 6, 8] = torch.tensor([4., -2., 1., -3.])
            fixed_a = module.fixed(q, k, value)
            fixed_b = module.fixed(q, k, changed)
            self.assertTrue(torch.equal(fixed_a[:, :, 6, 7], fixed_b[:, :, 6, 7]))
            self.assertFalse(torch.allclose(module(q, k, value)[:, :, 6, 7],
                                           module(q, k, changed)[:, :, 6, 7]))
            changed = value.clone()
            changed[:, :, 6, 15] = torch.tensor([4., -2., 1., -3.])
            self.assertTrue(torch.allclose(module(q, k, value)[:, :, 6, 0],
                                          module(q, k, changed)[:, :, 6, 0]))
            q, k, value = [torch.randn(1, 4, 17, 19) for _ in range(3)]
            changed = value.clone()
            changed[:, :, 6, -1] = torch.tensor([4., -2., 1., -3.])
            self.assertTrue(torch.allclose(module(q, k, value)[:, :, 6, 0],
                                          module(q, k, changed)[:, :, 6, 0]))

    def test_old_single_shifted_factory_is_unchanged(self):
        single = make_cross_attention_2d(8, "shifted_window", 8, 2)
        self.assertIsInstance(single, WindowCrossAttention2D)
        self.assertEqual(single.shift_size, 4)
        pair = make_cross_attention_2d(8, "shifted_window_pair", 8, 2)
        self.assertIsInstance(pair, ShiftedWindowCrossAttentionPair2D)

    def test_a_keeps_original_channel_model_initialization(self):
        with (ROOT / "configs" / CONFIGS["A"]).open(encoding="utf-8") as stream:
            settings = yaml.safe_load(stream)["model"]
        settings["base_dim"] = 8
        torch.manual_seed(42)
        baseline = build_deblur_model(base_dim=8, fusion_mode="rgb_gated_event_then_rgb_4ca")
        torch.manual_seed(42)
        trial = build_deblur_model(**settings)
        self.assertEqual(baseline.state_dict().keys(), trial.state_dict().keys())
        for key, value in baseline.state_dict().items():
            self.assertTrue(torch.equal(value, trial.state_dict()[key]), key)

    def test_configs_form_controlled_three_gpu_attention_experiments(self):
        configs = {}
        for label, name in CONFIGS.items():
            with (ROOT / "configs" / name).open(encoding="utf-8") as stream:
                configs[label] = yaml.safe_load(stream)
        with (ROOT / "configs/train_tdc_reblur_rgb_gated4ca_scratch.yml").open(
            encoding="utf-8"
        ) as stream:
            baseline = yaml.safe_load(stream)
        for label, config in configs.items():
            with self.subTest(label=label):
                trial = copy.deepcopy(config)
                self.assertIn("_3GPU_", trial["name"])
                trial["name"] = baseline["name"]
                expected_sa = "restormer_shifted_window" if label == "D" else "restormer_channel"
                self.assertEqual(trial["model"].pop("encoder_self_attn_rgb"), expected_sa)
                self.assertEqual(trial["model"].pop("encoder_self_attn_event2d"), expected_sa)
                self.assertEqual(trial["model"].pop("encoder_self_attn_event3d"), "restormer_channel")
                expected_ca = "channel" if label == "A" else "shifted_window_pair"
                self.assertEqual(trial["model"]["cross_attn_type"], expected_ca)
                trial["model"]["cross_attn_type"] = "channel"
                self.assertEqual(trial, baseline)

    def test_pair_models_backpropagate_adam_step_and_checkpoint_round_trip(self):
        for label in ("C", "D"):
            for bins in (6, 16):
                with self.subTest(label=label, bins=bins):
                    with (ROOT / "configs" / CONFIGS[label]).open(encoding="utf-8") as stream:
                        settings = yaml.safe_load(stream)["model"]
                    settings.update(base_dim=8, event_in=bins)
                    model = build_deblur_model(**settings)
                    self.assertEqual(sum(isinstance(m, ShiftedWindowCrossAttentionPair2D)
                                         for m in model.modules()), 12)
                    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0002)
                    inputs = (torch.randn(1, 3, 17, 19), torch.randn(1, bins, 17, 19))
                    output = model(*inputs)
                    self.assertEqual(output.shape, inputs[0].shape)
                    output.square().mean().backward()
                    for name, parameter in model.named_parameters():
                        self.assertIsNotNone(parameter.grad, name)
                        self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                    optimizer.step()
                    clone = build_deblur_model(**settings)
                    clone.load_state_dict(model.state_dict(), strict=True)
                    with torch.no_grad():
                        self.assertTrue(torch.equal(model(*inputs), clone(*inputs)))


if __name__ == "__main__":
    unittest.main()
