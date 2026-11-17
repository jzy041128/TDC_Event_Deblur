import copy
import os
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from models.tdc_deblur_net import build_deblur_model
from models.temporal_event_fusion import ConditionalSpatialRead, LocalSTRead, UniformTemporalRead


def spatial_reference(reader, sources, event):
    q, k, v = reader.project(sources, event)
    b, d, t, h, w = k.shape
    size, count = reader.kernel_size, reader.neighbor_count
    k, v = (
        F.unfold(x.reshape(b, d * t, h, w), size, padding=size // 2)
        .reshape(b, d, t, count, h * w)
        for x in (k, v)
    )
    scores = (q.flatten(2)[:, :, None, None] * k).sum(1) / d ** 0.5
    valid = F.unfold(q.new_ones(1, 1, h, w), size, padding=size // 2).bool()
    weights = scores.masked_fill(~valid[:, None], float("-inf")).softmax(dim=2)
    mixed = (weights[:, None] * v).sum(3).mean(2).reshape(b, d, h, w)
    return reader.out_proj(mixed)


class SpatialReadoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_matches_per_frame_unfold_reference_and_gradients_at_boundaries(self):
        for length, h, w in ((1, 1, 1), (6, 3, 5), (16, 1, 3), (6, 3, 1)):
            with self.subTest(length=length, h=h, w=w):
                reader = ConditionalSpatialRead(8, 16, query_chunk=2)
                reference = copy.deepcopy(reader)
                inputs = [torch.randn(2, 8, h, w, requires_grad=True) for _ in range(2)]
                inputs.append(torch.randn(2, 8, length, h, w, requires_grad=True))
                clones = [x.detach().clone().requires_grad_() for x in inputs]
                actual = reader(inputs[:2], inputs[2])
                expected = spatial_reference(reference, clones[:2], clones[2])
                torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
                actual.square().sum().backward()
                expected.square().sum().backward()
                for x, y in zip(inputs, clones):
                    torch.testing.assert_close(x.grad, y.grad, atol=2e-6, rtol=1e-4)
                for (name, p), (_, q) in zip(reader.named_parameters(), reference.named_parameters()):
                    self.assertIsNotNone(p.grad, name)
                    torch.testing.assert_close(p.grad, q.grad, atol=3e-6, rtol=1e-4, msg=name)

    def test_each_frame_has_exactly_uniform_total_weight_and_masks_invalid_neighbors(self):
        reader = ConditionalSpatialRead(8, 16)
        length, h, w = 6, 3, 5
        _, valid = reader.neighbors(h, w, torch.device("cpu"))
        mask = valid[:, None, :].expand(-1, length, -1)
        logits = torch.randn(2, 1, h * w, length, 9).masked_fill(~mask[None, None], -float("inf"))
        weights = reader._attention_weights(logits.flatten(-2)).reshape_as(logits)
        torch.testing.assert_close(weights.sum(-1), torch.full((2, 1, h * w, length), 1 / length))
        self.assertTrue(torch.equal(weights.masked_select(~mask[None, None]),
                                    torch.zeros_like(weights.masked_select(~mask[None, None]))))
        # A time-specific scalar cannot affect a within-frame spatial softmax.
        offsets = torch.arange(length).reshape(1, 1, 1, length, 1).float()
        changed = reader._attention_weights((logits + offsets).flatten(-2)).reshape_as(logits)
        torch.testing.assert_close(weights, changed, atol=1e-7, rtol=1e-5)
        joint = LocalSTRead(8, 16, num_heads=1, spatial_bias=False)
        joint_weights = joint._attention_weights(logits.flatten(-2)).reshape_as(logits)
        self.assertFalse(torch.allclose(joint_weights.sum(-1), weights.sum(-1)))

    def test_single_spatial_candidate_reduces_to_projected_mean_v(self):
        reader = ConditionalSpatialRead(8, 16, kernel_size=1, query_chunk=2)
        mean = UniformTemporalRead(8, 16)
        mean.load_state_dict({key: reader.state_dict()[key] for key in mean.state_dict()}, strict=True)
        sources = [torch.randn(1, 8, 3, 5) for _ in range(2)]
        event = torch.randn(1, 8, 6, 3, 5, requires_grad=True)
        clone = event.detach().clone().requires_grad_()
        actual, expected = reader(sources, event), mean(sources, clone)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        actual.square().sum().backward()
        expected.square().sum().backward()
        torch.testing.assert_close(event.grad, clone.grad, atol=2e-6, rtol=1e-4)

    def test_single_frame_matches_local_spatiotemporal_read(self):
        reader = ConditionalSpatialRead(8, 16, query_chunk=2)
        local = LocalSTRead(8, 16, num_heads=1, spatial_bias=False, query_chunk=2)
        state = local.state_dict()
        state.update(reader.state_dict())
        local.load_state_dict(state, strict=True)
        sources, event = [torch.randn(1, 8, 3, 5) for _ in range(2)], torch.randn(1, 8, 1, 3, 5)
        torch.testing.assert_close(reader(sources, event), local(sources, event), atol=1e-6, rtol=1e-5)

    def test_no_learned_time_or_spatial_bias_and_time_permutation_invariance(self):
        reader = ConditionalSpatialRead(8, 16, query_chunk=2)
        self.assertFalse(hasattr(reader, "time_bias"))
        self.assertIsNone(reader.spatial_bias)
        sources = [torch.randn(1, 8, 3, 5) for _ in range(2)]
        event = torch.randn(1, 8, 6, 3, 5)
        permutation = torch.tensor([3, 0, 5, 1, 4, 2])
        torch.testing.assert_close(reader(sources, event), reader(sources, event[:, :, permutation]),
                                   atol=1e-6, rtol=1e-5)

    def test_chunk_and_checkpoint_variants_match_outputs_and_gradients(self):
        baseline = ConditionalSpatialRead(8, 16, query_chunk=1000, checkpoint_chunks=False)
        sources, event = [torch.randn(1, 8, 3, 5) for _ in range(2)], torch.randn(1, 8, 6, 3, 5)
        expected = baseline(sources, event)
        expected.square().sum().backward()
        for chunk, checkpoint in ((2, False), (2, True), (1000, True)):
            reader = ConditionalSpatialRead(8, 16, query_chunk=chunk, checkpoint_chunks=checkpoint)
            reader.load_state_dict(baseline.state_dict(), strict=True)
            actual = reader(sources, event)
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
            actual.square().sum().backward()
            for (name, p), (_, q) in zip(reader.named_parameters(), baseline.named_parameters()):
                self.assertIsNotNone(p.grad, name)
                torch.testing.assert_close(p.grad, q.grad, atol=3e-6, rtol=1e-4, msg=name)

    @unittest.skipUnless(os.environ.get("TDC_SPATIAL_FULL_SIZE_TESTS") == "1",
                         "Opt-in full-resolution CPU synthetic check")
    def test_full_resolution_two_adamw_steps_and_complete_frame(self):
        path = Path(__file__).resolve().parents[1] / "configs/train_tdc_reblur_spatial_2ca_allscales_3gpu_scratch.yml"
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        torch.manual_seed(42)
        model = build_deblur_model(**config["model"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["train"]["learning_rate"])
        blur, event, gt = torch.rand(1, 3, 256, 256), torch.randn(1, 6, 256, 256), torch.rand(1, 3, 256, 256)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            prediction = model(blur, event)
            self.assertEqual(prediction.shape, gt.shape)
            (prediction - gt).abs().mean().backward()
            for name, parameter in model.named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            optimizer.step()
            del prediction
        model.eval()
        with torch.no_grad():
            prediction = model(torch.rand(1, 3, 260, 320), torch.randn(1, 6, 260, 320))
        self.assertEqual(prediction.shape, (1, 3, 260, 320))
        self.assertTrue(torch.isfinite(prediction).all())


if __name__ == "__main__":
    unittest.main()
