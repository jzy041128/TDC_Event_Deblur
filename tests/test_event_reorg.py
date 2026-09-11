import unittest

import torch
import torch.nn.functional as F

from models.tdc_deblur_net import (
    EventReorganizedChannelCrossAttention2D,
    ThreeBranchStageFusion,
)


def make_stage(attention_type="channel"):
    return ThreeBranchStageFusion(
        channels=8,
        fusion_mode="event_reorg_b23_ca",
        fusion_dim="2d",
        cross_attn_type=attention_type,
        swapped_kv_order="event3d_first",
        cross_window_size=8,
        num_heads=2,
        qk_norm=False,
        gamma_init=0.1,
    )


class EventReorgTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(42)

    def test_explicit_two_map_formula(self):
        module = EventReorganizedChannelCrossAttention2D(8, 2)
        rgb, reference, content = [torch.randn(2, 8, 5, 7) for _ in range(3)]

        def heads(x):
            return x.reshape(2, 2, 4, 35)

        q = F.normalize(heads(module.q(module.norm_q(rgb))), dim=-1)
        anchor = F.normalize(
            heads(module.reference(module.norm_reference(reference))), dim=-1
        )
        k = F.normalize(heads(module.k(module.norm_k(content))), dim=-1)
        v = heads(module.v(module.norm_v(content)))
        correspondence = torch.softmax(
            (anchor @ k.transpose(-2, -1)) * module.temperature_reorg,
            dim=-1,
        )
        read = torch.softmax(
            (q @ anchor.transpose(-2, -1)) * module.temperature_read,
            dim=-1,
        )
        expected = module.proj((read @ (correspondence @ v)).reshape(2, 8, 5, 7))
        torch.testing.assert_close(module(rgb, reference, content), expected)

    def test_b23_uses_e2_reference_and_mean_e3_content(self):
        stage = make_stage()
        rgb = torch.randn(1, 8, 9, 11)
        event2d = torch.randn_like(rgb)
        event3d = torch.randn(1, 8, 6, 9, 11)
        captured = []
        handle = stage.event_reorg.register_forward_pre_hook(
            lambda _module, args, kwargs: captured.append(kwargs),
            with_kwargs=True,
        )
        output = stage(rgb, event2d, event3d)
        handle.remove()
        self.assertEqual(output.shape, rgb.shape)
        self.assertIs(captured[0]["reference"], event2d)
        torch.testing.assert_close(captured[0]["content"], event3d.mean(dim=2))

    def test_b23_remains_channel_only(self):
        for attention_type in ("window", "shifted_window"):
            with self.subTest(attention_type=attention_type), self.assertRaises(ValueError):
                make_stage(attention_type)


if __name__ == "__main__":
    unittest.main()
