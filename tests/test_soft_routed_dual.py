import copy
import unittest
from pathlib import Path

import torch
import yaml

from models.tdc_deblur_net import (
    EventImageChannelCrossAttention2D,
    EventThenRGBFourCrossAttention2D,
    SoftRoutedDualCrossAttention2D,
    ThreeBranchStageFusion,
    WindowCrossAttention2D,
    build_deblur_model,
)


ROOT = Path(__file__).resolve().parents[1]


def make_stage(mode, attention_type="channel"):
    return ThreeBranchStageFusion(
        channels=8,
        fusion_mode=mode,
        fusion_dim="2d",
        cross_attn_type=attention_type,
        swapped_kv_order="event3d_first",
        cross_window_size=8,
        num_heads=2,
        qk_norm=False,
        gamma_init=0.1,
    )


class FusionModeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(42)

    def test_selectable_modes_build_all_attention_types(self):
        for mode in sorted(ThreeBranchStageFusion.SELECTABLE_CA_MODES):
            for attention_type in ("channel", "window", "shifted_window"):
                with self.subTest(mode=mode, attention_type=attention_type):
                    stage = make_stage(mode, attention_type)
                    if mode in ThreeBranchStageFusion.FOUR_CA_MODES:
                        module = stage.bidirectional_event_rgb
                        attentions = (
                            module.e2_from_e3,
                            module.e3_from_e2,
                            module.rgb_from_e2,
                            module.rgb_from_e3,
                        )
                    else:
                        attentions = (stage.ca1, stage.ca2)
                    expected = (
                        EventImageChannelCrossAttention2D
                        if attention_type == "channel"
                        else WindowCrossAttention2D
                    )
                    self.assertTrue(all(isinstance(item, expected) for item in attentions))
                    if attention_type != "channel":
                        expected_shift = 4 if attention_type == "shifted_window" else 0
                        self.assertTrue(
                            all(item.shift_size == expected_shift for item in attentions)
                        )

                    rgb = torch.randn(1, 8, 13, 15)
                    event2d = torch.randn_like(rgb)
                    event3d = torch.randn(1, 8, 6, 13, 15)
                    output = stage(rgb, event2d, event3d)
                    self.assertEqual(output.shape, rgb.shape)
                    output.mean().backward()
                    for name, parameter in stage.named_parameters():
                        self.assertIsNotNone(parameter.grad, name)

    def test_shifted_window_matches_window_parameter_count(self):
        fixed = make_stage("bidirectional_event_then_rgb_ca", "window")
        shifted = make_stage("bidirectional_event_then_rgb_ca", "shifted_window")
        self.assertEqual(
            sum(parameter.numel() for parameter in fixed.parameters()),
            sum(parameter.numel() for parameter in shifted.parameters()),
        )
        rgb = torch.randn(1, 8, 13, 15)
        event = torch.randn_like(rgb)
        attention = shifted.bidirectional_event_rgb.e2_from_e3
        self.assertEqual(attention(rgb, event, event).shape, rgb.shape)
        self.assertTrue(attention._mask_cache)

    def test_four_ca_routes_are_parameter_matched(self):
        cross_stage = make_stage("bidirectional_event_then_rgb_ca")
        independent_stage = make_stage("independent_event_then_rgb_4ca")
        self.assertEqual(
            sum(parameter.numel() for parameter in cross_stage.parameters()),
            sum(parameter.numel() for parameter in independent_stage.parameters()),
        )
        for stage, cross_event in ((cross_stage, True), (independent_stage, False)):
            module = stage.bidirectional_event_rgb
            self.assertIsInstance(module, EventThenRGBFourCrossAttention2D)
            self.assertEqual(module.cross_event, cross_event)
            rgb = torch.randn(1, 8, 9, 11)
            event2d = torch.randn_like(rgb)
            event3d = torch.randn_like(rgb)
            calls = {}

            def capture(name):
                return lambda _module, args: calls.__setitem__(name, args)

            handles = [
                module.e2_from_e3.register_forward_pre_hook(capture("e2")),
                module.e3_from_e2.register_forward_pre_hook(capture("e3")),
            ]
            module(rgb, event2d, event3d)
            for handle in handles:
                handle.remove()
            e2_q, e2_k, e2_v = calls["e2"]
            e3_q, e3_k, e3_v = calls["e3"]
            self.assertIs(e2_q, event2d)
            self.assertIs(e3_q, event3d)
            self.assertIs(e2_k, e2_v)
            self.assertIs(e3_k, e3_v)
            if cross_event:
                self.assertIs(e2_k, event3d)
                self.assertIs(e3_k, event2d)
            else:
                self.assertIs(e2_k, event2d)
                self.assertIs(e3_k, event3d)

    def test_soft_routed_keeps_fixed_mixed_experts(self):
        module = make_stage("soft_routed_dual_ca").soft_routed_dual
        self.assertIsInstance(module, SoftRoutedDualCrossAttention2D)
        self.assertIsInstance(module.channel_ca, EventImageChannelCrossAttention2D)
        self.assertIsInstance(module.second_ca, WindowCrossAttention2D)
        self.assertEqual(module.second_ca.shift_size, 0)
        for attention_type in ("window", "shifted_window"):
            with self.subTest(attention_type=attention_type), self.assertRaises(ValueError):
                make_stage("soft_routed_dual_ca", attention_type)

    def test_removed_modes_are_rejected(self):
        removed = (
            "cat",
            "single_ca",
            "cascaded_ca",
            "channel_ca",
            "event_conv",
            "event_window_ca",
            "plain_channel_ca",
            "event_add_channel_ca",
            "mmca_key_bridge_ca",
            "event_reorg_b32_ca",
        )
        for mode in removed:
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                make_stage(mode)

    def test_archived_config_keys_are_filtered_explicitly(self):
        model = build_deblur_model(
            base_dim=8,
            num_heads=2,
            fusion_mode="swapped_kv_cascaded_ca",
            fusion_dim="2d",
            cross_attn_type="channel",
            encoder_self_attn="restormer_channel",
            single_ca_order="event2d_k_event3d_v",
            cascaded_ca_order="motion_then_struct",
            key_bridge_order="event3d_first",
        )
        clone = copy.deepcopy(model)
        clone.load_state_dict(model.state_dict(), strict=True)

    def test_current_config_has_no_removed_selectors(self):
        with (ROOT / "configs/train_tdc_tribranch.yml").open(encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        self.assertEqual(
            config["model"]["fusion_mode"], "bidirectional_event_then_rgb_ca"
        )
        self.assertEqual(config["model"]["cross_attn_type"], "channel")
        for removed_key in (
            "single_ca_order",
            "cascaded_ca_order",
            "key_bridge_order",
        ):
            self.assertNotIn(removed_key, config["model"])


if __name__ == "__main__":
    unittest.main()
