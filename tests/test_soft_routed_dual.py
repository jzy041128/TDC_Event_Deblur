import unittest
from pathlib import Path

import torch
import yaml

from models.tdc_deblur_net import (
    BidirectionalEventThenRGBChannelFusion2D,
    EventImageChannelCrossAttention2D,
    ThreeBranchStageFusion,
    WindowCrossAttention2D,
    build_deblur_model,
)


ROOT = Path(__file__).resolve().parents[1]


def make_stage(mode):
    return ThreeBranchStageFusion(
        channels=8,
        fusion_mode=mode,
        fusion_dim="2d",
        cross_attn_type="channel",
        single_ca_order="event2d_k_event3d_v",
        cascaded_ca_order="motion_then_struct",
        swapped_kv_order="event3d_first",
        key_bridge_order="event3d_first",
        cross_window_size=8,
        temporal_window_size=2,
        num_heads=2,
        qk_norm=False,
        gamma_init=0.1,
    )


class EventFusionModeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(42)

    def test_soft_routed_mode_keeps_dynamic_channel_and_window_experts(self):
        module = make_stage("soft_routed_dual_ca").soft_routed_dual

        self.assertIsInstance(module.channel_ca, EventImageChannelCrossAttention2D)
        self.assertIsInstance(module.second_ca, WindowCrossAttention2D)
        self.assertIsNotNone(module.router)
        self.assertFalse(hasattr(module, "event_bypass"))
        self.assertFalse(hasattr(module, "branch_head"))

        event2d = torch.randn(2, 8, 11, 13)
        event3d = torch.randn_like(event2d)
        joined, gates = module.route(event2d, event3d)
        self.assertEqual(joined.shape, (2, 16, 11, 13))
        self.assertEqual(gates.shape, (2, 2, 16, 1, 1))
        torch.testing.assert_close(gates.sum(dim=1), torch.ones_like(gates[:, 0]))

    def test_removed_soft_routed_ablation_modes_are_rejected(self):
        for mode in ("fixed_half_dual_ca", "soft_routed_dual_channel_ca"):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                make_stage(mode)

    def test_soft_routed_experts_use_same_source_for_key_and_value(self):
        rgb = torch.randn(1, 8, 13, 15)
        event2d = torch.randn_like(rgb)
        event3d = torch.randn_like(rgb)
        module = make_stage("soft_routed_dual_ca").soft_routed_dual
        calls = []

        def capture(_module, args):
            calls.append(args)

        handles = [
            module.channel_ca.register_forward_pre_hook(capture),
            module.second_ca.register_forward_pre_hook(capture),
        ]
        first, second = module(rgb, event2d, event3d)
        for handle in handles:
            handle.remove()

        self.assertEqual(first.shape, rgb.shape)
        self.assertEqual(second.shape, rgb.shape)
        self.assertEqual(len(calls), 2)
        for query, key, value in calls:
            self.assertIs(query, rgb)
            self.assertIs(key, value)

    def test_four_ca_mode_preserves_event_identities_and_uses_independent_parameters(self):
        stage = make_stage("bidirectional_event_then_rgb_ca")
        module = stage.bidirectional_event_rgb
        self.assertIsInstance(module, BidirectionalEventThenRGBChannelFusion2D)

        attention_modules = (
            module.e2_from_e3,
            module.e3_from_e2,
            module.rgb_from_e2,
            module.rgb_from_e3,
        )
        for attention in attention_modules:
            self.assertIsInstance(attention, EventImageChannelCrossAttention2D)
        q_weight_ids = {id(attention.q.weight) for attention in attention_modules}
        self.assertEqual(len(q_weight_ids), 4)

        rgb = torch.randn(2, 8, 13, 15)
        event2d = torch.randn_like(rgb)
        event3d = torch.randn(2, 8, 6, 13, 15)
        event3d_2d = event3d.mean(dim=2)
        originals = [tensor.clone() for tensor in (rgb, event2d, event3d)]
        calls = {}

        def capture(name):
            return lambda _module, args: calls.__setitem__(name, args)

        handles = [
            module.e2_from_e3.register_forward_pre_hook(capture("e2_from_e3")),
            module.e3_from_e2.register_forward_pre_hook(capture("e3_from_e2")),
            module.rgb_from_e2.register_forward_pre_hook(capture("rgb_from_e2")),
            module.rgb_from_e3.register_forward_pre_hook(capture("rgb_from_e3")),
        ]
        actual = stage(rgb, event2d, event3d)
        for handle in handles:
            handle.remove()

        q, k, v = calls["e2_from_e3"]
        self.assertIs(q, event2d)
        self.assertIs(k, v)
        torch.testing.assert_close(k, event3d_2d)

        q, k, v = calls["e3_from_e2"]
        torch.testing.assert_close(q, event3d_2d)
        self.assertIs(k, event2d)
        self.assertIs(k, v)

        for name in ("rgb_from_e2", "rgb_from_e3"):
            q, k, v = calls[name]
            self.assertIs(q, rgb)
            self.assertIs(k, v)

        delta_e2 = module.e2_from_e3(event2d, event3d_2d, event3d_2d)
        delta_e3 = module.e3_from_e2(event3d_2d, event2d, event2d)
        event2d_enhanced = event2d + module.gamma_e2 * module.inject_e2(delta_e2)
        event3d_enhanced = event3d_2d + module.gamma_e3 * module.inject_e3(delta_e3)
        delta_rgb_e2 = module.rgb_from_e2(
            rgb, event2d_enhanced, event2d_enhanced
        )
        delta_rgb_e3 = module.rgb_from_e3(
            rgb, event3d_enhanced, event3d_enhanced
        )
        fused = (
            rgb
            + module.gamma_rgb_e2 * module.inject_rgb_e2(delta_rgb_e2)
            + module.gamma_rgb_e3 * module.inject_rgb_e3(delta_rgb_e3)
        )
        expected = fused + stage.gamma_ffn * stage.ffn(stage.norm_ffn(fused))
        torch.testing.assert_close(actual, expected)

        for before, after in zip(originals, (rgb, event2d, event3d)):
            torch.testing.assert_close(before, after, rtol=0, atol=0)

        actual.square().mean().backward()
        for name, parameter in stage.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_full_network_and_default_config(self):
        model = build_deblur_model(
            base_dim=8,
            num_heads=2,
            fusion_mode="bidirectional_event_then_rgb_ca",
            fusion_dim="2d",
            cross_attn_type="channel",
            encoder_self_attn="restormer_channel",
        )
        rgb = torch.randn(1, 3, 16, 20)
        event = torch.randn(1, 6, 16, 20)
        output = model(rgb, event)
        self.assertEqual(output.shape, rgb.shape)
        output.mean().backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)

        with (ROOT / "configs/train_tdc_tribranch.yml").open(encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        self.assertEqual(
            config["model"]["fusion_mode"], "bidirectional_event_then_rgb_ca"
        )
        self.assertEqual(config["model"]["encoder_self_attn"], "restormer_channel")
        self.assertIsNone(config["path"]["resume_state"])

    def test_two_dimensional_modes_reject_3d_fusion(self):
        for mode in ("soft_routed_dual_ca", "bidirectional_event_then_rgb_ca"):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                ThreeBranchStageFusion(
                    channels=8,
                    fusion_mode=mode,
                    fusion_dim="3d",
                    cross_attn_type="channel",
                    single_ca_order="event2d_k_event3d_v",
                    cascaded_ca_order="motion_then_struct",
                    swapped_kv_order="event3d_first",
                    key_bridge_order="event3d_first",
                    cross_window_size=8,
                    temporal_window_size=2,
                    num_heads=2,
                    qk_norm=False,
                    gamma_init=0.1,
                )


if __name__ == "__main__":
    unittest.main()
