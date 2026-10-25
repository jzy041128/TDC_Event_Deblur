import copy
import unittest
from pathlib import Path

import torch
import yaml

from models.tdc_deblur_net import (
    EventImageChannelCrossAttention2D,
    RestormerChannelSelfAttention2D,
    RestormerShiftedWindowPair3D,
    build_deblur_model,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/train_tdc_reblur_rgb_gated4ca_e3_shifted_sa_3gpu_scratch.yml"


def read_config(path=CONFIG):
    with path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


class E3OnlySpatialSelfAttentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_config_changes_only_name_and_e3_sa_from_a(self):
        trial = read_config()
        baseline = read_config(ROOT / "configs/train_tdc_reblur_rgb_gated4ca_channel_3gpu_scratch.yml")
        self.assertNotEqual(trial["name"], baseline["name"])
        trial["name"] = baseline["name"]
        self.assertEqual(trial["model"]["encoder_self_attn_event3d"], "restormer_shifted_window")
        trial["model"]["encoder_self_attn_event3d"] = "restormer_channel"
        self.assertEqual(trial, baseline)

    def test_only_e3_uses_spatiotemporal_pairs_at_all_scales(self):
        settings = read_config()["model"]
        settings["base_dim"] = 8
        model = build_deblur_model(**settings)
        for branch in (model.rgb_self_attn, model.event2d_self_attn):
            self.assertTrue(all(isinstance(m, RestormerChannelSelfAttention2D) for m in branch))
        self.assertEqual(len(model.event3d_self_attn), 3)
        for pair in model.event3d_self_attn:
            self.assertIsInstance(pair, RestormerShiftedWindowPair3D)
            self.assertEqual([block.shift_size for block in pair.blocks], [0, 4])
            self.assertEqual([block.temporal_window for block in pair.blocks], [2, 2])
        self.assertEqual(sum(isinstance(m, EventImageChannelCrossAttention2D)
                             for m in model.fusions.modules()), 12)

    def test_time_length_is_preserved_and_all_parameters_backpropagate(self):
        for bins in (6, 16):
            with self.subTest(bins=bins):
                settings = copy.deepcopy(read_config()["model"])
                settings.update(base_dim=8, event_in=bins)
                model = build_deblur_model(**settings)
                seen = []
                handles = [module.register_forward_hook(
                    lambda module, args, output: seen.append((args[0].shape[2], output.shape[2]))
                ) for module in model.event3d_self_attn]
                inputs = (torch.randn(1, 3, 17, 19), torch.randn(1, bins, 17, 19))
                optimizer = torch.optim.AdamW(model.parameters(), lr=0.0002)
                output = model(*inputs)
                self.assertEqual(output.shape, inputs[0].shape)
                self.assertEqual(seen, [(bins, bins)] * 3)
                for handle in handles:
                    handle.remove()
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
