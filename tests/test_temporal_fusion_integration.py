import copy
import gc
import io
import os
import unittest
from pathlib import Path

import torch
import yaml

from models.tdc_deblur_net import ThreeBranchStageFusion, build_deblur_model
from models.temporal_event_fusion import (
    TEMPORAL_FUSION_MODES, ConditionalTemporalTwoCAFusion, ConditionedBiGRUFusion,
    AnchoredLocalSTFusion, IdentityPreservingDualExpertFusion,
)
from utils.model_checkpoint import checkpoint_metadata, load_pretrained_weights, validate_pretrain_options


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_MODE = "rgb_gated_event_then_rgb_4ca"


class TemporalFusionIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_shared_weights_rng_and_public_ffn_match_existing_a(self):
        torch.manual_seed(42)
        baseline = build_deblur_model(base_dim=8, fusion_mode=REFERENCE_MODE)
        expected_rng = torch.get_rng_state()
        for mode in sorted(TEMPORAL_FUSION_MODES):
            torch.manual_seed(42)
            model = build_deblur_model(base_dim=8, fusion_mode=mode)
            self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()), mode)
            state = model.state_dict()
            for key, value in baseline.state_dict().items():
                if not key.startswith("fusions.") or any(x in key for x in (".ffn.", ".norm_ffn.", ".gamma_ffn")):
                    self.assertTrue(torch.equal(value, state[key]), f"{mode}: {key}")

    def test_stage_routing_full_e3_ffn_once_backward_and_roundtrip(self):
        for length in (6, 16):
            for mode in sorted(TEMPORAL_FUSION_MODES):
                with self.subTest(length=length, mode=mode):
                    config = dict(base_dim=8, event_in=length, fusion_mode=mode)
                    model = build_deblur_model(**config)
                    seen_e3, ffn_calls = [], []
                    hooks = []
                    for i, stage in enumerate(model.fusions):
                        hooks.append(stage.temporal_fusion.register_forward_pre_hook(lambda _, args, i=i: seen_e3.append((i, args[2].shape))))
                        hooks.append(stage.ffn.register_forward_hook(lambda _, args, result, i=i: ffn_calls.append(i)))
                    if mode in ("anchored_local_st", "conditioned_bi_gru"):
                        for stage in model.fusions[:2]:
                            self.assertIsInstance(stage.temporal_fusion, ConditionalTemporalTwoCAFusion)
                        expected = AnchoredLocalSTFusion if mode == "anchored_local_st" else ConditionedBiGRUFusion
                        self.assertIsInstance(model.fusions[2].temporal_fusion, expected)
                    if mode == "identity_preserving_dual_expert":
                        self.assertEqual([x.temporal_fusion.use_local for x in model.fusions], [False, False, True])
                    x, event = torch.randn(1, 3, 17, 19), torch.randn(1, length, 17, 19)
                    output = model(x, event, return_intermediate=True)
                    self.assertEqual(output.shape, x.shape)
                    self.assertEqual(ffn_calls, [0, 1, 2])
                    self.assertEqual([shape[2] for _, shape in seen_e3], [length] * 3)
                    output.square().mean().backward()
                    for name, parameter in model.named_parameters():
                        self.assertIsNotNone(parameter.grad, name)
                        self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                    for hook in hooks:
                        hook.remove()
                    saved = {"model_state_dict": model.state_dict(), "event3d_conv_type": "tdc", **checkpoint_metadata(model, config)}
                    buffer = io.BytesIO()
                    torch.save(saved, buffer)
                    buffer.seek(0)
                    saved = torch.load(buffer, weights_only=True)
                    clone = build_deblur_model(**config)
                    load_pretrained_weights(clone, saved)
                    with torch.no_grad():
                        torch.testing.assert_close(model(x, event), clone(x, event), rtol=0, atol=0)

    def test_three_stage_two_stage_optional_path_still_works(self):
        for sam in ("none", "standard", "event_guided"):
            model = build_deblur_model(base_dim=8, fusion_mode="cond_temporal_2ca", two_stage=True, sam_mode=sam)
            blur, event = torch.randn(1, 3, 17, 19), torch.randn(1, 6, 17, 19)
            output = model(blur, event, return_intermediate=True)
            self.assertEqual(len(output), 2)
            for result in output:
                self.assertEqual(result.shape, blur.shape)
            sum(x.square().mean() for x in output).backward()

    def test_old_modes_retain_mean_path_and_strict_loading(self):
        stage = ThreeBranchStageFusion(8, REFERENCE_MODE, "2d", "channel", "event3d_first", 8, 4, False, 0.1)
        seen = []
        handle = stage.bidirectional_event_rgb.register_forward_pre_hook(lambda _, args: seen.append(args[2]))
        rgb, e2, e3 = torch.randn(1, 8, 3, 5), torch.randn(1, 8, 3, 5), torch.randn(1, 8, 6, 3, 5)
        stage(rgb, e2, e3)
        handle.remove()
        self.assertTrue(torch.equal(seen[0], e3.mean(2)))
        old = build_deblur_model(base_dim=8, fusion_mode=REFERENCE_MODE)
        old.validate_event3d_checkpoint({"model_state_dict": old.state_dict()})
        old.load_state_dict(old.state_dict(), strict=True)

    def test_configurations_only_change_name_mode_and_options(self):
        baseline = yaml.safe_load((ROOT / "configs/train_tdc_reblur_rgb_gated4ca_channel_3gpu_scratch.yml").read_text())
        for mode in TEMPORAL_FUSION_MODES:
            path = ROOT / f"configs/train_tdc_reblur_{mode}_3gpu_scratch.yml"
            trial = yaml.safe_load(path.read_text())
            self.assertEqual(trial["model"]["fusion_mode"], mode)
            build_deblur_model(**trial["model"])
            trial["name"] = baseline["name"]
            trial["model"]["fusion_mode"] = baseline["model"]["fusion_mode"]
            trial["model"].pop("fusion_options")
            self.assertEqual(trial, baseline)

    def test_invalid_modes_options_fail_before_training(self):
        for mode in TEMPORAL_FUSION_MODES:
            with self.assertRaises(ValueError):
                build_deblur_model(base_dim=8, fusion_mode=mode, cross_attn_type="window")
        with self.assertRaises(ValueError):
            build_deblur_model(fusion_options={"temporal_reduction": 4})

    def test_metadata_guards_and_explicit_backbone_transfer(self):
        old = build_deblur_model(base_dim=8, fusion_mode=REFERENCE_MODE)
        new = build_deblur_model(base_dim=8, fusion_mode="cond_temporal_2ca")
        legacy = {"event3d_conv_type": "tdc", "model_state_dict": old.state_dict()}
        with self.assertRaisesRegex(ValueError, "metadata"):
            load_pretrained_weights(new, legacy)
        load_pretrained_weights(new, legacy, backbone_only=True)
        for key, value in old.state_dict().items():
            if not key.startswith("fusions."):
                self.assertTrue(torch.equal(value, new.state_dict()[key]), key)
        checkpoint = {"event3d_conv_type": "tdc", "model_state_dict": new.state_dict(), **checkpoint_metadata(new, {})}
        mismatched = build_deblur_model(base_dim=8, fusion_mode="cond_temporal_2ca", fusion_options={"temporal_reduction": 2})
        with self.assertRaisesRegex(ValueError, "fusion_options"):
            load_pretrained_weights(mismatched, checkpoint)
        for key in ("rgb_stem.block.0.weight", "up0.conv.block.0.weight"):
            bad = copy.deepcopy(legacy)
            if key in bad["model_state_dict"]:
                del bad["model_state_dict"][key]
                with self.assertRaisesRegex(ValueError, "Backbone initialization mismatch"):
                    load_pretrained_weights(new, bad, backbone_only=True)
        wrong_conv = build_deblur_model(base_dim=8, fusion_mode="cond_temporal_2ca", event3d_conv_type="conv3d")
        with self.assertRaisesRegex(ValueError, "does not match"):
            load_pretrained_weights(wrong_conv, legacy, backbone_only=True)
        wrong_bins = build_deblur_model(base_dim=8, event_in=16, fusion_mode="cond_temporal_2ca")
        with self.assertRaises(ValueError):
            load_pretrained_weights(wrong_bins, legacy, backbone_only=True)

    def test_backbone_only_cannot_be_used_for_resume_or_without_pretrain(self):
        for paths in ({"pretrain_backbone_only": True}, {"pretrain_backbone_only": True, "pretrain_model": "a", "resume_state": "b"}):
            with self.assertRaises(ValueError):
                validate_pretrain_options(paths)
        self.assertTrue(validate_pretrain_options({"pretrain_backbone_only": True, "pretrain_model": "a"}))

    @unittest.skipUnless(os.environ.get("TDC_FULL_SIZE_TESTS") == "1", "Opt-in full-resolution CPU synthetic check")
    def test_full_resolution_all_five_modes_two_adamw_steps_and_full_frame(self):
        self.addCleanup(torch.use_deterministic_algorithms, torch.are_deterministic_algorithms_enabled())
        torch.use_deterministic_algorithms(True)
        for mode in sorted(TEMPORAL_FUSION_MODES):
            with self.subTest(mode=mode):
                config = yaml.safe_load((ROOT / f"configs/train_tdc_reblur_{mode}_3gpu_scratch.yml").read_text())
                model = build_deblur_model(**config["model"])
                optimizer = torch.optim.AdamW(model.parameters(), lr=config["train"]["learning_rate"])
                blur, event, gt = torch.rand(1, 3, 256, 256), torch.randn(1, 6, 256, 256), torch.rand(1, 3, 256, 256)
                for _ in range(2):
                    optimizer.zero_grad(set_to_none=True)
                    output = model(blur, event)
                    self.assertEqual(output.shape, blur.shape)
                    (output - gt).abs().mean().backward()
                    for name, parameter in model.named_parameters():
                        self.assertIsNotNone(parameter.grad, name)
                        self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                    optimizer.step()
                    del output
                model.eval()
                with torch.no_grad():
                    output = model(torch.rand(1, 3, 260, 320), torch.randn(1, 6, 260, 320))
                    self.assertEqual(output.shape, (1, 3, 260, 320))
                    self.assertTrue(torch.isfinite(output).all())
                print(f"Full-size CPU synthetic PASS: {mode}, train=256x256 two AdamW steps, eval=260x320")
                del model, optimizer, blur, event, gt, output
                gc.collect()


if __name__ == "__main__":
    unittest.main()
