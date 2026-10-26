import copy
import unittest

import torch
import torch.nn.functional as F

from models.temporal_event_fusion import (
    TEMPORAL_FUSION_MODES, AnchoredLocalSTFusion, ConditionalTemporalRead,
    ConditionalTemporalTwoCAFusion, ConditionedBiGRUFusion,
    EventControlledDynamicFilterFusion, IdentityPreservingDualExpertFusion,
    LocalSTRead, SpatialGate, TemporalPositionBias, build_temporal_fusion,
    grouped_dynamic_filter, normalize_fusion_options,
)


def slow_local_read(reader, sources, event3d):
    q, k, v = reader.project(sources, event3d)
    b, _, h, w = q.shape
    heads, d, t = reader.num_heads, reader.head_dim, event3d.shape[2]
    q = q.reshape(b, heads, d, h, w)
    k, v = (x.reshape(b, heads, d, t, h, w) for x in (k, v))
    bias = reader.time_bias(t, q)
    batches = []
    for bi in range(b):
        head_outputs = []
        for head in range(heads):
            pixels = []
            for y in range(h):
                for x in range(w):
                    scores, values = [], []
                    for ti in range(t):
                        for j, (dy, dx) in enumerate((dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)):
                            yy, xx = y + dy, x + dx
                            if 0 <= yy < h and 0 <= xx < w:
                                scores.append((q[bi, head, :, y, x] * k[bi, head, :, ti, yy, xx]).sum() / d ** 0.5
                                              + bias[ti, head] + reader.spatial_bias[head, j])
                                values.append(v[bi, head, :, ti, yy, xx])
                    pixels.append((torch.stack(scores).softmax(0)[:, None] * torch.stack(values)).sum(0))
            head_outputs.append(torch.stack(pixels).T.reshape(d, h, w))
        batches.append(torch.cat(head_outputs))
    return reader.out_proj(torch.stack(batches))


class TemporalFusionOperatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_options_are_copied_and_invalid_options_rejected(self):
        a = normalize_fusion_options()
        a["local_st_scales"].append(0)
        self.assertEqual(normalize_fusion_options()["local_st_scales"], [2])
        for opt in ({"typo": 1}, {"dynamic_kernel_size": 5}, {"local_st_kernel_size": 1},
                    {"temporal_position": "fixed"}, {"gru_scales": [0, 1, 2]},
                    {"temporal_min_dim": 0}, {"local_st_checkpoint_chunks": 1},
                    {"gru_hidden_ratio": 0.25}, {"dynamic_kernel_gain": float("nan")}):
            with self.subTest(opt=opt), self.assertRaises(ValueError):
                normalize_fusion_options(opt)

    def test_temporal_read_weights_and_pointwise_selection(self):
        reader = ConditionalTemporalRead(8, 16)
        sources = [torch.randn(1, 8, 3, 5) for _ in range(2)]
        event = torch.randn(1, 8, 6, 3, 5)
        out, weights = reader(sources, event, return_weights=True)
        self.assertEqual(out.shape, sources[0].shape)
        torch.testing.assert_close(weights.sum(1), torch.ones(1, 3, 5))
        changed = [x.clone() for x in sources]
        changed[0][:, :, 1, 2] = torch.randn(1, 8)
        _, second = reader(changed, event, return_weights=True)
        mask = torch.ones(3, 5, dtype=torch.bool)
        mask[1, 2] = False
        self.assertTrue(torch.equal(weights[:, :, mask], second[:, :, mask]))
        self.assertFalse(torch.equal(weights[:, :, 1, 2], second[:, :, 1, 2]))

    def test_t1_weight_is_one_and_position_is_zero(self):
        reader = ConditionalTemporalRead(8, 8)
        source, event = torch.randn(1, 8, 2, 3), torch.randn(1, 8, 1, 2, 3)
        positions = []
        hook = reader.time_bias.mlp[0].register_forward_pre_hook(lambda _, args: positions.append(args[0]))
        _, weights = reader([source], event, return_weights=True)
        hook.remove()
        self.assertTrue(torch.equal(weights, torch.ones_like(weights)))
        self.assertTrue(torch.equal(positions[0], torch.zeros(1, 2)))

    def test_gate_initialization_and_two_step_learning(self):
        gate = SpatialGate(8, 3, 2)
        sources = [torch.randn(1, 8, 5, 7) for _ in range(3)]
        self.assertTrue(torch.equal(gate(*sources), torch.full((1, 2, 5, 7), 0.5)))
        optimizer = torch.optim.AdamW(gate.parameters(), lr=0.001)
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            gate(*sources).square().mean().backward()
            self.assertGreater(gate.net[-1].weight.grad.abs().sum().item(), 0)
            if step:
                self.assertGreater(gate.net[0].weight.grad.abs().sum().item(), 0)
            optimizer.step()
        with torch.no_grad():
            gate.net[-1].weight.zero_()
            gate.net[-1].bias.fill_(10)
        self.assertTrue((gate(*sources) > 0.99).all())
        with torch.no_grad():
            gate.net[-1].bias.fill_(-10)
        self.assertTrue((gate(*sources) < 0.01).all())

    def test_local_read_matches_slow_reference_output_and_gradients(self):
        for h, w in ((3, 5), (1, 3), (3, 1), (1, 1)):
            with self.subTest(h=h, w=w):
                reader = LocalSTRead(8, 16, query_chunk=2, checkpoint_chunks=True)
                reference = copy.deepcopy(reader)
                inputs = [torch.randn(1, 8, h, w, requires_grad=True) for _ in range(2)]
                event = torch.randn(1, 8, 3, h, w, requires_grad=True)
                ref_inputs = [x.detach().clone().requires_grad_() for x in inputs]
                ref_event = event.detach().clone().requires_grad_()
                out = reader(inputs, event)
                expected = slow_local_read(reference, ref_inputs, ref_event)
                torch.testing.assert_close(out, expected, atol=1e-6, rtol=1e-5)
                out.square().sum().backward()
                expected.square().sum().backward()
                for x, y in zip(inputs + [event], ref_inputs + [ref_event]):
                    torch.testing.assert_close(x.grad, y.grad, atol=1e-6, rtol=1e-4)
                for (name, p), (_, q) in zip(reader.named_parameters(), reference.named_parameters()):
                    self.assertIsNotNone(p.grad, name)
                    torch.testing.assert_close(p.grad, q.grad, atol=2e-6, rtol=1e-4)

    def test_local_chunk_and_checkpoint_variants_agree(self):
        baseline = LocalSTRead(8, 8, query_chunk=10000, checkpoint_chunks=False)
        sources, event = [torch.randn(1, 8, 5, 7)], torch.randn(1, 8, 6, 5, 7)
        expected = baseline(sources, event)
        expected.square().sum().backward()
        for chunk, enabled in ((3, False), (3, True), (10000, True)):
            reader = LocalSTRead(8, 8, query_chunk=chunk, checkpoint_chunks=enabled)
            reader.load_state_dict(baseline.state_dict())
            out = reader(sources, event)
            torch.testing.assert_close(out, expected)
            out.square().sum().backward()
            for (name, p), (_, q) in zip(reader.named_parameters(), baseline.named_parameters()):
                torch.testing.assert_close(p.grad, q.grad, atol=1e-5, rtol=1e-4)

    def test_dynamic_filter_matches_unfold_without_group_mixing(self):
        x = torch.randn(2, 12, 5, 7, requires_grad=True)
        k = torch.randn(2, 4, 9, 5, 7, requires_grad=True)
        patches = F.unfold(F.pad(x, (1, 1, 1, 1), mode="replicate"), 3).reshape(2, 4, 3, 9, 5, 7)
        expected = (patches * k.unsqueeze(2)).sum(3).reshape_as(x)
        out = grouped_dynamic_filter(x, k)
        torch.testing.assert_close(out, expected)
        g = torch.autograd.grad(out.square().sum(), (x, k), retain_graph=True)
        ref_g = torch.autograd.grad(expected.square().sum(), (x, k))
        for first, second in zip(g, ref_g):
            torch.testing.assert_close(first, second, atol=1e-4, rtol=1e-5)

    def test_dynamic_kernels_zero_sum_and_constant_response(self):
        module = EventControlledDynamicFilterFusion(32)
        k = module.kernels(torch.randn(1, 32, 7, 9))
        torch.testing.assert_close(k.sum(2), torch.zeros(1, 4, 7, 9), atol=1e-7, rtol=0)
        constant = torch.randn(1, 32, 1, 1).expand(1, 32, 7, 9)
        torch.testing.assert_close(grouped_dynamic_filter(constant, k), torch.zeros_like(constant), atol=1e-7, rtol=0)

    def test_all_modes_shape_gradients_inputs_unchanged_and_event_used(self):
        for channels in (32, 64, 128):
            for length in (1, 6, 16):
                for mode in sorted(TEMPORAL_FUSION_MODES):
                    with self.subTest(channels=channels, length=length, mode=mode):
                        module = build_temporal_fusion(mode, channels, 4, 0.1, 2)
                        rgb, e2 = [torch.randn(1, channels, 3, 5, requires_grad=True) for _ in range(2)]
                        e3 = torch.randn(1, channels, length, 3, 5, requires_grad=True)
                        snapshots = [x.detach().clone() for x in (rgb, e2, e3)]
                        out = module(rgb, e2, e3)
                        self.assertEqual(out.shape, rgb.shape)
                        self.assertTrue(torch.isfinite(out).all())
                        out.square().mean().backward()
                        for x, saved in zip((rgb, e2, e3), snapshots):
                            self.assertTrue(torch.equal(x.detach(), saved))
                            self.assertIsNotNone(x.grad)
                            self.assertTrue(torch.isfinite(x.grad).all())
                            self.assertGreater(x.grad.abs().sum().item(), 0)
                        for name, parameter in module.named_parameters():
                            self.assertIsNotNone(parameter.grad, name)
                            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                        with torch.no_grad():
                            self.assertFalse(torch.equal(out.detach(), module(rgb, e2, e3.flip(1))))

    def test_invalid_shapes_and_channels_are_rejected(self):
        for mode in TEMPORAL_FUSION_MODES:
            module = build_temporal_fusion(mode, 32, 4, 0.1, 2)
            rgb, e2, e3 = torch.randn(1, 32, 3, 5), torch.randn(1, 32, 3, 5), torch.randn(1, 32, 6, 3, 5)
            for bad in (e3.mean(2), e3[:, :, :0], e3[:, :, :, :2], e3.double()):
                with self.subTest(mode=mode), self.assertRaises(ValueError):
                    module(rgb, e2, bad)
        with self.assertRaises(ValueError):
            ConditionalTemporalRead(8, 24)
        with self.assertRaises(ValueError):
            LocalSTRead(32, 32, num_heads=3)
        with self.assertRaises(ValueError):
            EventControlledDynamicFilterFusion(10, num_heads=2)

    def test_gru_directions_are_independent_and_receive_gradients(self):
        module = ConditionedBiGRUFusion(32)
        self.assertIsNot(module.fwd_cell.zr_conv.weight, module.bwd_cell.zr_conv.weight)
        inputs = [torch.randn(1, 32, 3, 5), torch.randn(1, 32, 3, 5), torch.randn(1, 32, 6, 3, 5)]
        module(*inputs).square().mean().backward()
        for cell in (module.fwd_cell, module.bwd_cell):
            self.assertGreater(cell.zr_conv.weight.grad.abs().sum().item(), 0)
            self.assertGreater(cell.candidate_conv.weight.grad.abs().sum().item(), 0)

    def test_dual_expert_queries_original_rgb_only(self):
        for stage in (0, 2):
            module = IdentityPreservingDualExpertFusion(32, stage_index=stage)
            rgb, e2, e3 = torch.randn(1, 32, 3, 5), torch.randn(1, 32, 3, 5), torch.randn(1, 32, 6, 3, 5)
            seen = []
            hook = module.e3_read.register_forward_pre_hook(lambda _, args: seen.append(args[0]))
            module(rgb, e2, e3)
            hook.remove()
            self.assertEqual(len(seen[0]), 1)
            self.assertIs(seen[0][0], rgb)


if __name__ == "__main__":
    unittest.main()
