import unittest
from pathlib import Path

import torch
import yaml

from models.tdc_deblur_net import (
    EventImageChannelCrossAttention2D,
    SoftRoutedDualCrossAttention2D,
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


class SoftRoutedDualTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(42)

    def test_mode_definitions_are_self_contained(self):
        dynamic = make_stage("soft_routed_dual_ca").soft_routed_dual
        fixed = make_stage("fixed_half_dual_ca").soft_routed_dual
        dual_channel = make_stage("soft_routed_dual_channel_ca").soft_routed_dual

        self.assertIsInstance(dynamic.channel_ca, EventImageChannelCrossAttention2D)
        self.assertIsInstance(dynamic.second_ca, WindowCrossAttention2D)
        self.assertIsInstance(fixed.second_ca, WindowCrossAttention2D)
        self.assertIsInstance(dual_channel.second_ca, EventImageChannelCrossAttention2D)
        self.assertIsNotNone(dynamic.router)
        self.assertIsNone(fixed.router)
        for module in (dynamic, fixed, dual_channel):
            self.assertFalse(hasattr(module, "event_bypass"))
            self.assertFalse(hasattr(module, "branch_head"))

    def test_router_is_complementary_and_fixed_mode_is_half(self):
        event2d = torch.randn(2, 8, 11, 13)
        event3d = torch.randn_like(event2d)
        for mode in ThreeBranchStageFusion.SOFT_ROUTED_DUAL_MODES:
            with self.subTest(mode=mode):
                module = make_stage(mode).soft_routed_dual
                joined, gates = module.route(event2d, event3d)
                self.assertEqual(joined.shape, (2, 16, 11, 13))
                self.assertEqual(gates.shape, (2, 2, 16, 1, 1))
                torch.testing.assert_close(gates.sum(dim=1), torch.ones_like(gates[:, 0]))
                if mode == "fixed_half_dual_ca":
                    torch.testing.assert_close(gates, torch.full_like(gates, 0.5))

    def test_parallel_experts_use_same_source_for_key_and_value(self):
        rgb = torch.randn(1, 8, 13, 15)
        event2d = torch.randn_like(rgb)
        event3d = torch.randn_like(rgb)
        for mode in ThreeBranchStageFusion.SOFT_ROUTED_DUAL_MODES:
            with self.subTest(mode=mode):
                module = make_stage(mode).soft_routed_dual
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

    def test_stage_means_time_updates_only_rgb_and_has_gradients(self):
        rgb = torch.randn(2, 8, 13, 15)
        event2d = torch.randn_like(rgb)
        event3d = torch.randn(2, 8, 6, 13, 15)
        originals = [tensor.clone() for tensor in (rgb, event2d, event3d)]
        for mode in ThreeBranchStageFusion.SOFT_ROUTED_DUAL_MODES:
            with self.subTest(mode=mode):
                stage = make_stage(mode)
                captured = []
                hook = stage.soft_routed_dual.register_forward_pre_hook(
                    lambda _module, args: captured.append(args)
                )
                actual = stage(rgb, event2d, event3d)
                hook.remove()
                torch.testing.assert_close(captured[0][2], event3d.mean(dim=2))

                channel_delta, second_delta = stage.soft_routed_dual(
                    rgb, event2d, event3d.mean(dim=2)
                )
                fused = (
                    rgb
                    + stage.gamma1 * stage.inject1(channel_delta)
                    + stage.gamma2 * stage.inject2(second_delta)
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
            fusion_mode="soft_routed_dual_ca",
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
        self.assertEqual(config["model"]["fusion_mode"], "soft_routed_dual_ca")
        self.assertEqual(config["model"]["encoder_self_attn"], "restormer_channel")
        self.assertIsNone(config["path"]["resume_state"])

    def test_invalid_dimensional_modes_fail(self):
        for mode in ThreeBranchStageFusion.SOFT_ROUTED_DUAL_MODES:
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                stage = make_stage(mode)
                stage.fusion_dim = "3d"
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
