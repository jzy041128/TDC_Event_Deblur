import copy
import gc
import os
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from models.tdc_deblur_net import build_deblur_model
from models.temporal_event_fusion import (
    AnchoredLocalSTFusion, ConditionalTemporalRead, ConditionedBiGRUFusion,
    IdentityPreservingDualExpertFusion, LocalSTRead, UniformTemporalRead,
    normalize_fusion_options,
)
from utils.model_checkpoint import checkpoint_metadata, fusion_description, load_pretrained_weights


ROOT = Path(__file__).resolve().parents[1]
SLUGS = {"mean": "mean_v_2ca", "temporal": "temporal_2ca", "local": "local_st_2ca"}
ARCHITECTURE_MODES = ("identity_preserving_dual_expert", "conditioned_bi_gru")


def config_for(kind):
    path = ROOT / f"configs/train_tdc_reblur_{SLUGS[kind]}_allscales_3gpu_scratch.yml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def architecture_config_for(mode):
    path = ROOT / f"configs/train_tdc_reblur_{mode}_allscales_3gpu_scratch.yml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def local_reference(reader, sources, event):
    q, k, v = reader.project(sources, event)
    b, width, t, h, w = k.shape
    size, count = reader.kernel_size, reader.neighbor_count
    k = F.unfold(k.reshape(b, width * t, h, w), size, padding=size // 2)
    v = F.unfold(v.reshape(b, width * t, h, w), size, padding=size // 2)
    k, v = (x.reshape(b, width, t, count, h * w) for x in (k, v))
    scores = (q.flatten(2)[:, :, None, None] * k).sum(1) / width ** 0.5
    scores = scores + reader.time_bias(t, q).reshape(1, t, 1, 1)
    valid = F.unfold(q.new_ones(1, 1, h, w), size, padding=size // 2).bool()
    scores = scores.masked_fill(~valid[:, None], float("-inf"))
    weights = scores.reshape(b, t * count, h * w).softmax(1).reshape_as(scores)
    mixed = (weights[:, None] * v).sum((2, 3)).reshape(b, width, h, w)
    return reader.out_proj(mixed)


class MatchedReadoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_mean_is_projected_value_average_not_raw_event_mean(self):
        torch.manual_seed(42)
        reader = UniformTemporalRead(32, 64)
        sources = [torch.randn(1, 32, 3, 5, requires_grad=True) for _ in range(2)]
        event = torch.randn(1, 32, 6, 3, 5, requires_grad=True)
        result, weights = reader(sources, event, return_weights=True)
        expected = reader.out_proj(reader.v_proj(reader.norm_v(event)).mean(2))
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
        self.assertTrue(torch.equal(weights, torch.full((1, 6, 3, 5), 1 / 6)))
        changed = [torch.randn_like(x) * 10 for x in sources]
        torch.testing.assert_close(result, reader(changed, event), rtol=0, atol=0)
        grads = torch.autograd.grad(result.square().sum(), sources + [event], allow_unused=True)
        self.assertIsNone(grads[0])
        self.assertIsNone(grads[1])
        self.assertGreater(grads[2].abs().sum().item(), 0)
        state = reader.state_dict()
        self.assertTrue(all(key.startswith(("norm_v.", "v_proj.", "out_proj.")) for key in state))
        self.assertFalse(torch.allclose(result, event.mean(2)))

    def test_reader_initialization_matches_temporal_including_rng(self):
        for channels in (32, 64, 128):
            torch.manual_seed(42)
            temporal = ConditionalTemporalRead(channels, channels * 2)
            expected_rng = torch.get_rng_state()
            for kind in ("mean", "local"):
                torch.manual_seed(42)
                reader = (UniformTemporalRead(channels, channels * 2) if kind == "mean" else
                          LocalSTRead(channels, channels * 2, num_heads=1, spatial_bias=False))
                self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
                for key, value in reader.state_dict().items():
                    self.assertTrue(torch.equal(value, temporal.state_dict()[key]), (kind, channels, key))

    def test_local_one_by_one_matches_temporal_outputs_and_gradients(self):
        for length, h, w in ((1, 1, 1), (6, 3, 5), (16, 1, 3)):
            with self.subTest(length=length, h=h, w=w):
                temporal = ConditionalTemporalRead(32, 64)
                with torch.no_grad():
                    temporal.time_bias.mlp[-1].weight.normal_(std=0.1)
                local = LocalSTRead(32, 64, num_heads=1, kernel_size=1, spatial_bias=False,
                                    query_chunk=2, checkpoint_chunks=True)
                local.load_state_dict(temporal.state_dict(), strict=True)
                inputs = [torch.randn(1, 32, h, w, requires_grad=True) for _ in range(2)]
                inputs.append(torch.randn(1, 32, length, h, w, requires_grad=True))
                clones = [x.detach().clone().requires_grad_() for x in inputs]
                first, second = temporal(inputs[:2], inputs[2]), local(clones[:2], clones[2])
                torch.testing.assert_close(first, second, atol=1e-6, rtol=1e-5)
                first.square().sum().backward()
                second.square().sum().backward()
                for x, y in zip(inputs, clones):
                    torch.testing.assert_close(x.grad, y.grad, atol=2e-6, rtol=1e-4)
                for (key, p), (_, q) in zip(temporal.named_parameters(), local.named_parameters()):
                    torch.testing.assert_close(p.grad, q.grad, atol=2e-6, rtol=1e-4, msg=key)

    def test_local_three_by_three_matches_unfold_and_boundary_gradients(self):
        for h, w in ((3, 5), (1, 5), (3, 1), (1, 1)):
            reader = LocalSTRead(32, 64, num_heads=1, spatial_bias=False,
                                 query_chunk=2, checkpoint_chunks=True)
            reference = copy.deepcopy(reader)
            inputs = [torch.randn(1, 32, h, w, requires_grad=True) for _ in range(2)]
            inputs.append(torch.randn(1, 32, 6, h, w, requires_grad=True))
            clones = [x.detach().clone().requires_grad_() for x in inputs]
            out = reader(inputs[:2], inputs[2])
            expected = local_reference(reference, clones[:2], clones[2])
            torch.testing.assert_close(out, expected, atol=1e-6, rtol=1e-5)
            out.square().sum().backward()
            expected.square().sum().backward()
            for x, y in zip(inputs, clones):
                torch.testing.assert_close(x.grad, y.grad, atol=2e-6, rtol=1e-4)
            for (key, p), (_, q) in zip(reader.named_parameters(), reference.named_parameters()):
                torch.testing.assert_close(p.grad, q.grad, atol=3e-6, rtol=1e-4, msg=key)

    def test_configs_only_change_name_and_readout_and_use_all_scales(self):
        reference = config_for("temporal")
        for kind in SLUGS:
            config = config_for(kind)
            self.assertEqual(config["model"]["fusion_options"]["e3_readout"], kind)
            self.assertEqual(config["model"]["fusion_options"]["local_st_scales"], [0, 1, 2])
            self.assertEqual(config["model"]["fusion_options"]["local_st_heads"], 1)
            self.assertEqual(config["model"]["fusion_options"]["local_st_reduction"], 4)
            self.assertIsNone(config["path"]["resume_state"])
            self.assertIsNone(config["path"]["pretrain_model"])
            self.assertEqual(config["datasets"]["train"]["batch_size"], 1)
            self.assertIsNone(config["datasets"]["val"]["patch_size"])
            config["name"] = reference["name"]
            config["model"]["fusion_options"]["e3_readout"] = "temporal"
            self.assertEqual(config, reference)

    def test_all_shared_weights_rng_and_legacy_temporal_are_exact(self):
        legacy = yaml.safe_load((ROOT / "configs/train_tdc_reblur_cond_temporal_2ca_3gpu_scratch.yml").read_text())
        legacy["model"]["base_dim"] = 8
        torch.manual_seed(42)
        original = build_deblur_model(**legacy["model"])
        original_rng, original_state = torch.get_rng_state(), original.state_dict()
        for kind in SLUGS:
            config = config_for(kind)
            config["model"]["base_dim"] = 8
            torch.manual_seed(42)
            model = build_deblur_model(**config["model"])
            self.assertTrue(torch.equal(torch.get_rng_state(), original_rng), kind)
            for key, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, original_state[key]), (kind, key))
            if kind == "temporal":
                original.eval()
                model.eval()
                blur, event = torch.randn(1, 3, 17, 19), torch.randn(1, 6, 17, 19)
                first, second = original(blur, event), model(blur, event)
                torch.testing.assert_close(first, second, atol=0, rtol=0)
                first.square().sum().backward()
                second.square().sum().backward()
                for (key, p), (_, q) in zip(original.named_parameters(), model.named_parameters()):
                    torch.testing.assert_close(p.grad, q.grad, atol=0, rtol=0, msg=key)

    def test_complete_models_use_selected_reader_all_stages_and_have_no_unused_parameters(self):
        reader_types = {"mean": UniformTemporalRead, "temporal": ConditionalTemporalRead, "local": LocalSTRead}
        for length in (6, 16):
            for kind in SLUGS:
                config = config_for(kind)
                config["model"].update(base_dim=8, event_in=length)
                model = build_deblur_model(**config["model"])
                calls, ffn_calls, e3_shapes = [], [], []
                handles = []
                for i, stage in enumerate(model.fusions):
                    self.assertIsInstance(stage.temporal_fusion.temporal_read, reader_types[kind])
                    def record_reader(_, args, i=i):
                        calls.append(i)
                        e3_shapes.append(tuple(args[1].shape))
                    handles.append(stage.temporal_fusion.temporal_read.register_forward_pre_hook(record_reader))
                    handles.append(stage.ffn.register_forward_hook(lambda _, args, out, i=i: ffn_calls.append(i)))
                    if kind == "local":
                        reader = stage.temporal_fusion.temporal_read
                        self.assertEqual(reader.num_heads, 1)
                        self.assertEqual(reader.kernel_size, 3)
                        self.assertIsNone(reader.spatial_bias)
                blur, event = torch.rand(1, 3, 17, 19), torch.randn(1, length, 17, 19)
                optimizer = torch.optim.AdamW(model.parameters(), lr=0.0002)
                for _ in range(2):
                    optimizer.zero_grad(set_to_none=True)
                    result = model(blur, event)
                    self.assertEqual(result.shape, blur.shape)
                    result.square().mean().backward()
                    for key, parameter in model.named_parameters():
                        self.assertIsNotNone(parameter.grad, (kind, key))
                        self.assertTrue(torch.isfinite(parameter.grad).all(), (kind, key))
                    optimizer.step()
                self.assertEqual(calls, [0, 1, 2] * 2)
                self.assertEqual(ffn_calls, [0, 1, 2] * 2)
                self.assertEqual([s[2] for s in e3_shapes], [length] * 6)
                for handle in handles:
                    handle.remove()
                description = fusion_description(model)
                if kind == "mean":
                    self.assertIn("V-only widths", description)
                    self.assertIn("no RGB/E2-conditioned weights", description)
                if kind == "local":
                    self.assertIn("actual local stages=[0, 1, 2]", description)

    def test_checkpoint_roundtrip_and_cross_readout_rejection(self):
        models = {}
        for kind in SLUGS:
            config = config_for(kind)["model"]
            config["base_dim"] = 8
            model = build_deblur_model(**config)
            checkpoint = {"model_state_dict": model.state_dict(), "event3d_conv_type": "tdc",
                          **checkpoint_metadata(model, config)}
            clone = build_deblur_model(**config)
            load_pretrained_weights(clone, checkpoint)
            models[kind] = (model, checkpoint)
        for kind, (model, _) in models.items():
            for other, (_, checkpoint) in models.items():
                if kind != other:
                    with self.assertRaisesRegex(ValueError, "fusion_options"):
                        load_pretrained_weights(model, checkpoint)

    def test_legacy_local_and_gru_can_explicitly_use_all_three_stages(self):
        for mode in ("anchored_local_st", "conditioned_bi_gru", "identity_preserving_dual_expert"):
            model = build_deblur_model(base_dim=8, fusion_mode=mode,
                                       fusion_options={"local_st_scales": [0, 1, 2], "gru_scales": [0, 1, 2]})
            if mode == "anchored_local_st":
                self.assertTrue(all(isinstance(stage.temporal_fusion, AnchoredLocalSTFusion) for stage in model.fusions))
            elif mode == "conditioned_bi_gru":
                self.assertTrue(all(isinstance(stage.temporal_fusion, ConditionedBiGRUFusion) for stage in model.fusions))
            else:
                self.assertTrue(all(stage.temporal_fusion.use_local for stage in model.fusions))
            out = model(torch.rand(1, 3, 17, 19), torch.randn(1, 6, 17, 19))
            out.square().mean().backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_allscale_architecture_configs_match_protocol_and_preserve_legacy_scope(self):
        reference = config_for("mean")
        reference["model"]["fusion_options"].pop("e3_readout")
        for mode in ARCHITECTURE_MODES:
            config = architecture_config_for(mode)
            self.assertIn("AllScales", config["name"])
            self.assertNotIn("Coarse", config["name"])
            self.assertEqual(config["model"]["fusion_mode"], mode)
            config["name"] = reference["name"]
            config["model"]["fusion_mode"] = reference["model"]["fusion_mode"]
            self.assertEqual(config, reference)
            legacy_path = ROOT / f"configs/train_tdc_reblur_{mode}_3gpu_scratch.yml"
            legacy = yaml.safe_load(legacy_path.read_text(encoding="utf-8"))
            options = legacy["model"]["fusion_options"]
            self.assertEqual(options["local_st_scales"], [2])
            self.assertEqual(options["gru_scales"], [2])
            self.assertEqual(options["local_st_heads"], 4)

    def test_allscale_architecture_configs_run_every_stage_and_backpropagate(self):
        classes = {"identity_preserving_dual_expert": IdentityPreservingDualExpertFusion,
                   "conditioned_bi_gru": ConditionedBiGRUFusion}
        for length in (6, 16):
            for mode in ARCHITECTURE_MODES:
                with self.subTest(mode=mode, length=length):
                    config = architecture_config_for(mode)["model"]
                    config.update(base_dim=8, event_in=length)
                    model = build_deblur_model(**config)
                    calls, ffn_calls, handles = [], [], []
                    for i, stage in enumerate(model.fusions):
                        fusion = stage.temporal_fusion
                        self.assertIsInstance(fusion, classes[mode])
                        if mode == "identity_preserving_dual_expert":
                            self.assertTrue(fusion.use_local)
                            self.assertEqual(fusion.e3_read.num_heads, 1)
                            self.assertEqual(fusion.e3_read.kernel_size, 3)
                            self.assertIsNotNone(fusion.e3_read.spatial_bias)
                            self.assertFalse(hasattr(fusion, "ca_e3"))
                        def record_fusion(_, args, i=i):
                            calls.append((i, args[2].ndim, args[2].shape[2]))
                        handles.append(fusion.register_forward_pre_hook(record_fusion))
                        handles.append(stage.ffn.register_forward_hook(lambda _, args, out, i=i: ffn_calls.append(i)))
                    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0002)
                    blur, event = torch.rand(1, 3, 17, 19), torch.randn(1, length, 17, 19)
                    for _ in range(2):
                        optimizer.zero_grad(set_to_none=True)
                        output = model(blur, event)
                        self.assertEqual(output.shape, blur.shape)
                        output.square().mean().backward()
                        for key, parameter in model.named_parameters():
                            self.assertIsNotNone(parameter.grad, (mode, key))
                            self.assertTrue(torch.isfinite(parameter.grad).all(), (mode, key))
                        optimizer.step()
                    self.assertEqual(calls, [(i, 5, length) for i in range(3)] * 2)
                    self.assertEqual(ffn_calls, [0, 1, 2] * 2)
                    for handle in handles:
                        handle.remove()
                    checkpoint = {"model_state_dict": model.state_dict(), "event3d_conv_type": "tdc",
                                  **checkpoint_metadata(model, config)}
                    clone = build_deblur_model(**config)
                    load_pretrained_weights(clone, checkpoint)
                    legacy_path = ROOT / f"configs/train_tdc_reblur_{mode}_3gpu_scratch.yml"
                    legacy = yaml.safe_load(legacy_path.read_text(encoding="utf-8"))["model"]
                    legacy.update(base_dim=8, event_in=length)
                    with self.assertRaisesRegex(ValueError, "fusion_options"):
                        load_pretrained_weights(build_deblur_model(**legacy), checkpoint)

    def test_matched_options_reject_accidental_scope_or_head_changes(self):
        options = normalize_fusion_options({"e3_readout": "local"})
        self.assertEqual(options["local_st_heads"], 1)
        self.assertEqual(options["local_st_scales"], [0, 1, 2])
        for bad in ({"e3_readout": "raw_mean"}, {"e3_readout": "local", "local_st_heads": 4},
                    {"e3_readout": "local", "local_st_scales": [2]},
                    {"e3_readout": "local", "local_st_reduction": 2}):
            with self.assertRaises(ValueError):
                normalize_fusion_options(bad)
        with self.assertRaises(ValueError):
            build_deblur_model(base_dim=8, fusion_mode="conditioned_bi_gru", fusion_options={"e3_readout": "mean"})

    @unittest.skipUnless(os.environ.get("TDC_MATCHED_FULL_SIZE_TESTS") == "1", "Opt-in full-resolution CPU synthetic check")
    def test_full_resolution_two_steps_and_full_frame_all_three_readouts(self):
        self.addCleanup(torch.use_deterministic_algorithms, torch.are_deterministic_algorithms_enabled())
        torch.use_deterministic_algorithms(True)
        for kind in SLUGS:
            config = config_for(kind)
            torch.manual_seed(42)
            model = build_deblur_model(**config["model"])
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.0002)
            blur, event = torch.rand(1, 3, 256, 256), torch.randn(1, 6, 256, 256)
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                result = model(blur, event)
                (result - blur).abs().mean().backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
                optimizer.step()
                del result
            model.eval()
            with torch.no_grad():
                output = model(torch.rand(1, 3, 260, 320), torch.randn(1, 6, 260, 320))
                self.assertEqual(output.shape, (1, 3, 260, 320))
                self.assertTrue(torch.isfinite(output).all())
            print(f"Matched readout full-size CPU PASS: {kind}, two AdamW steps 256x256, eval 260x320", flush=True)
            del model, optimizer, blur, event, output
            gc.collect()


if __name__ == "__main__":
    unittest.main()
