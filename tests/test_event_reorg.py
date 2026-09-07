import copy
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from models.tdc_deblur_net import (
    EventReorganizedChannelCrossAttention2D,
    ThreeBranchStageFusion,
    build_deblur_model,
)


ROOT = Path(__file__).resolve().parents[1]


def make_stage(mode, **overrides):
    options = dict(
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
    options.update(overrides)
    return ThreeBranchStageFusion(**options)


class EventReorgTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(42)

    def test_explicit_two_map_formula_and_shared_reference(self):
        module = EventReorganizedChannelCrossAttention2D(8, 2)
        rgb, reference, content = [torch.randn(2, 8, 5, 7) for _ in range(3)]
        calls = []
        hook = module.reference.register_forward_hook(lambda *args: calls.append(1))
        actual = module(rgb, reference, content)
        hook.remove()
        self.assertEqual(len(calls), 1)
        self.assertIsNot(module.norm_k, module.norm_v)
        self.assertIsNot(module.temperature_reorg, module.temperature_read)

        def heads(x):
            return x.reshape(2, 2, 4, 35)

        q = F.normalize(heads(module.q(module.norm_q(rgb))), dim=-1)
        anchor = F.normalize(heads(module.reference(module.norm_reference(reference))), dim=-1)
        k = F.normalize(heads(module.k(module.norm_k(content))), dim=-1)
        v = heads(module.v(module.norm_v(content)))
        p = torch.softmax(torch.einsum("bhis,bhjs->bhij", anchor, k) * module.temperature_reorg, dim=-1)
        a = torch.softmax(torch.einsum("bhis,bhjs->bhij", q, anchor) * module.temperature_read, dim=-1)
        torch.testing.assert_close(p.sum(-1), torch.ones(2, 2, 4))
        torch.testing.assert_close(a.sum(-1), torch.ones(2, 2, 4))
        expected = torch.einsum("bhij,bhjk,bhks->bhis", a, p, v)
        expected = module.proj(expected.reshape(2, 8, 5, 7))
        torch.testing.assert_close(actual, expected)

    def test_both_directions_mean_time_and_keep_inputs(self):
        rgb = torch.randn(2, 8, 5, 7)
        e2 = torch.randn_like(rgb)
        e3 = torch.randn(2, 8, 6, 5, 7)
        originals = [t.clone() for t in (rgb, e2, e3)]
        for mode in sorted(ThreeBranchStageFusion.EVENT_REORG_MODES):
            with self.subTest(mode=mode):
                stage = make_stage(mode)
                captured = []
                hook = stage.event_reorg.register_forward_pre_hook(
                    lambda module, args, kwargs: captured.append(kwargs), with_kwargs=True
                )
                stage(rgb, e2, e3)
                hook.remove()
                reference, content = (e2, e3.mean(2)) if "b23" in mode else (e3.mean(2), e2)
                torch.testing.assert_close(captured[0]["reference"], reference)
                torch.testing.assert_close(captured[0]["content"], content)
                delta = stage.event_reorg(rgb, reference, content)
                fused = rgb + stage.gamma1 * stage.inject1(delta)
                expected = fused + stage.gamma_ffn * stage.ffn(stage.norm_ffn(fused))
                torch.testing.assert_close(stage(rgb, e2, e3), expected)
                for before, after in zip(originals, (rgb, e2, e3)):
                    torch.testing.assert_close(before, after, rtol=0, atol=0)

    def test_direction_swap_is_symmetric(self):
        b23 = make_stage("event_reorg_b23_ca")
        b32 = make_stage("event_reorg_b32_ca")
        b32.load_state_dict(b23.state_dict(), strict=True)
        rgb, e2, e3 = [torch.randn(1, 8, 5, 7) for _ in range(3)]
        torch.testing.assert_close(b23(rgb, e2, e3.unsqueeze(2)), b32(rgb, e3, e2.unsqueeze(2)))

    def test_zero_scales_keep_rgb_identity(self):
        for mode in sorted(ThreeBranchStageFusion.EVENT_REORG_MODES):
            stage = make_stage(mode)
            with torch.no_grad():
                stage.gamma1.zero_()
                stage.gamma_ffn.zero_()
            rgb = torch.randn(1, 8, 8, 8)
            actual = stage(rgb, torch.randn_like(rgb), torch.randn(1, 8, 6, 8, 8))
            torch.testing.assert_close(actual, rgb, rtol=0, atol=0)

    def test_invalid_modes_fail_explicitly(self):
        for mode in sorted(ThreeBranchStageFusion.EVENT_REORG_MODES):
            for override in ({"fusion_dim": "3d"}, {"cross_attn_type": "window"}):
                with self.subTest(mode=mode, override=override), self.assertRaises(ValueError):
                    make_stage(mode, **override)
        with self.assertRaises(ValueError):
            EventReorganizedChannelCrossAttention2D(8, 3)
        with self.assertRaises(ValueError):
            EventReorganizedChannelCrossAttention2D(8, 2)(
                torch.randn(1, 8, 8, 8), torch.randn(1, 8, 4, 4), torch.randn(1, 8, 8, 8)
            )

    def test_network_gradients_and_feature_routes(self):
        model = build_deblur_model(
            base_dim=8, num_heads=2, fusion_mode="event_reorg_b23_ca",
            cross_attn_type="channel", encoder_self_attn="restormer_channel",
        )
        other = build_deblur_model(
            base_dim=8, num_heads=2, fusion_mode="event_reorg_b32_ca",
            cross_attn_type="channel", encoder_self_attn="restormer_channel",
        )
        other.load_state_dict(model.state_dict(), strict=True)
        rgb = torch.randn(2, 3, 16, 20)
        event = torch.randn(2, 6, 16, 20)
        branch_outputs = []
        for net in (model, other):
            seen, handles = {}, []

            def save_output(name):
                return lambda module, args, output: seen.__setitem__(name, output.detach().clone())

            def save_input(name):
                return lambda module, args: seen.__setitem__(name, args[0].detach().clone())

            for scale in range(3):
                for branch in ("event2d", "event3d"):
                    module = getattr(net, branch + "_self_attn")[scale]
                    handles.append(module.register_forward_hook(save_output(f"{branch}_sa{scale}")))
                    if scale < 2:
                        down = getattr(net, branch + "_down")[scale]
                        handles.append(down.register_forward_pre_hook(save_input(f"{branch}_down{scale}")))
                handles.append(net.fusions[scale].register_forward_hook(save_output(f"fusion{scale}")))
                if scale < 2:
                    handles.append(net.rgb_down[scale].register_forward_pre_hook(save_input(f"rgb_down{scale}")))
            output = net(rgb, event)
            self.assertEqual(output.shape, rgb.shape)
            output.square().mean().backward()
            for name, parameter in net.named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            for scale in range(2):
                for branch in ("event2d", "event3d"):
                    torch.testing.assert_close(seen[f"{branch}_down{scale}"], seen[f"{branch}_sa{scale}"])
                torch.testing.assert_close(seen[f"rgb_down{scale}"], seen[f"fusion{scale}"])
            for handle in handles:
                handle.remove()
            branch_outputs.append(seen)
        for scale in range(3):
            for branch in ("event2d", "event3d"):
                key = f"{branch}_sa{scale}"
                torch.testing.assert_close(branch_outputs[0][key], branch_outputs[1][key], rtol=0, atol=0)
        self.assertNotEqual(model.fusions[0].event_reorg.q.weight.data_ptr(), model.fusions[1].event_reorg.q.weight.data_ptr())

    def test_default_config_and_parameter_counts(self):
        with (ROOT / "configs/train_tdc_tribranch.yml").open(encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        self.assertEqual(
            config["model"]["fusion_mode"], "bidirectional_event_then_rgb_ca"
        )
        self.assertEqual(config["model"]["encoder_self_attn"], "restormer_channel")
        self.assertIsNone(config["path"]["resume_state"])
        counts = {}
        for mode in ("event_reorg_b23_ca", "event_reorg_b32_ca", "swapped_kv_cascaded_ca"):
            options = copy.deepcopy(config["model"])
            options["fusion_mode"] = mode
            model = build_deblur_model(**options)
            counts[mode] = sum(p.numel() for p in model.parameters())
            self.assertFalse(any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in model.modules()))
        self.assertEqual(counts["event_reorg_b23_ca"], counts["event_reorg_b32_ca"])
        self.assertLess(counts["event_reorg_b23_ca"], counts["swapped_kv_cascaded_ca"])
        print("Parameter counts:", counts)


if __name__ == "__main__":
    unittest.main()
