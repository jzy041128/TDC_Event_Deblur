import unittest
from pathlib import Path

import torch
import yaml

from models.tdc_deblur_net import (
    EventImageChannelCrossAttention2D,
    RestormerChannelSelfAttention3D,
    RestormerShiftedWindowPair2D,
    RestormerWindowSelfAttention3D,
    build_deblur_model,
)


ROOT = Path(__file__).resolve().parents[1]
OVERRIDES = {
    "encoder_self_attn_rgb": "restormer_shifted_window",
    "encoder_self_attn_event2d": "restormer_shifted_window",
    "encoder_self_attn_event3d": "restormer_channel",
}


class EncoderSelfAttentionBranchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_legacy_modes_match_explicit_identical_overrides(self):
        for mode in (
            "window", "plain_channel", "restormer_channel",
            "restormer_window", "restormer_shifted_window",
        ):
            with self.subTest(mode=mode):
                torch.manual_seed(42)
                baseline = build_deblur_model(base_dim=8, encoder_self_attn=mode).eval()
                torch.manual_seed(42)
                explicit = build_deblur_model(
                    base_dim=8, encoder_self_attn=mode,
                    **{key: mode for key in OVERRIDES},
                ).eval()
                self.assertEqual(set(baseline.state_dict()), set(explicit.state_dict()))
                for key, value in baseline.state_dict().items():
                    self.assertTrue(torch.equal(value, explicit.state_dict()[key]), key)
                inputs = (torch.randn(1, 3, 17, 19), torch.randn(1, 6, 17, 19))
                with torch.no_grad():
                    self.assertTrue(torch.equal(baseline(*inputs), explicit(*inputs)))

    def test_mixed_branches_keep_channel_cross_attention(self):
        model = build_deblur_model(
            base_dim=8, fusion_mode="rgb_gated_event_then_rgb_4ca", **OVERRIDES
        )
        for modules in (model.rgb_self_attn, model.event2d_self_attn):
            self.assertTrue(all(isinstance(m, RestormerShiftedWindowPair2D) for m in modules))
        self.assertTrue(all(isinstance(m, RestormerChannelSelfAttention3D)
                            for m in model.event3d_self_attn))
        self.assertEqual(sum(isinstance(m, EventImageChannelCrossAttention2D)
                             for m in model.fusions.modules()), 12)

    def test_unset_and_null_branches_inherit_global_mode(self):
        model = build_deblur_model(
            base_dim=8, encoder_self_attn="restormer_window",
            encoder_self_attn_rgb=None, encoder_self_attn_event2d="RESTORMER_SHIFTED_WINDOW",
        )
        self.assertEqual(model.encoder_self_attn_types, {
            "rgb": "restormer_window", "event2d": "restormer_shifted_window",
            "event3d": "restormer_window",
        })
        self.assertIsInstance(model.event3d_self_attn[0], RestormerWindowSelfAttention3D)

    def test_invalid_branch_mode_is_reported(self):
        for key in OVERRIDES:
            for invalid in ("typo", 2, False):
                with self.subTest(key=key, invalid=invalid):
                    with self.assertRaisesRegex(ValueError, key):
                        build_deblur_model(base_dim=8, **{key: invalid})

    def test_mixed_model_backpropagates_and_round_trips(self):
        for bins in (6, 16):
            with self.subTest(bins=bins):
                settings = dict(base_dim=8, event_in=bins,
                                fusion_mode="rgb_gated_event_then_rgb_4ca", **OVERRIDES)
                model = build_deblur_model(**settings)
                inputs = (torch.randn(1, 3, 17, 19), torch.randn(1, bins, 17, 19))
                output = model(*inputs)
                self.assertEqual(output.shape, inputs[0].shape)
                output.square().mean().backward()
                for name, parameter in model.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                clone = build_deblur_model(**settings)
                clone.load_state_dict(model.state_dict(), strict=True)
                with torch.no_grad():
                    self.assertTrue(torch.equal(model(*inputs), clone(*inputs)))

    def test_reblur_config_changes_only_name_and_branch_attention(self):
        def read(name):
            with (ROOT / "configs" / name).open(encoding="utf-8") as stream:
                return yaml.safe_load(stream)

        baseline = read("train_tdc_reblur_rgb_gated4ca_scratch.yml")
        mixed = read("train_tdc_reblur_rgb_gated4ca_rgb_e2_shifted_sa_scratch.yml")
        self.assertNotEqual(mixed.pop("name"), baseline.pop("name"))
        for key, value in OVERRIDES.items():
            self.assertEqual(mixed["model"].pop(key), value)
        self.assertEqual(mixed, baseline)


if __name__ == "__main__":
    unittest.main()
