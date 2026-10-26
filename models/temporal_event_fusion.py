"""Optional fusion readers that retain E3's temporal axis until content selection."""

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from models.fusion_common import (
    ChannelLayerNorm2D,
    ChannelLayerNorm3D,
    EventImageChannelCrossAttention2D,
)


TEMPORAL_FUSION_MODES = frozenset({
    "cond_temporal_2ca", "event_controlled_dynamic_filter", "anchored_local_st",
    "conditioned_bi_gru", "identity_preserving_dual_expert",
})
DEFAULT_FUSION_OPTIONS = {
    "temporal_reduction": 4, "temporal_min_dim": 8,
    "temporal_position": "quadratic_mlp", "gate_reduction": 4, "gate_min_dim": 8,
    "dynamic_groups": 4, "dynamic_kernel_size": 3, "dynamic_kernel_gain": 1.0,
    "local_st_scales": [2], "local_st_heads": 4, "local_st_reduction": 4,
    "local_st_min_dim": 8, "local_st_kernel_size": 3, "local_st_query_chunk": 512,
    "local_st_checkpoint_chunks": True, "gru_scales": [2], "gru_hidden_ratio": 0.5,
}


def normalize_fusion_options(options=None):
    if options is not None and not isinstance(options, dict):
        raise ValueError("fusion_options must be a dictionary or null")
    result = copy.deepcopy(DEFAULT_FUSION_OPTIONS)
    unknown = set(options or {}) - set(result)
    if unknown:
        raise ValueError(f"Unknown fusion_options: {sorted(unknown)}")
    result.update(copy.deepcopy(options or {}))
    for key in (
        "temporal_reduction", "temporal_min_dim", "gate_reduction", "gate_min_dim",
        "dynamic_groups", "local_st_heads", "local_st_reduction", "local_st_min_dim",
        "local_st_query_chunk",
    ):
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"fusion_options.{key} must be a positive integer")
    if result["temporal_position"] != "quadratic_mlp":
        raise ValueError("Only temporal_position=quadratic_mlp is supported")
    for key in ("dynamic_kernel_size", "local_st_kernel_size"):
        if type(result[key]) is not int or result[key] != 3:
            raise ValueError(f"MVP requires {key}=3")
    for key in ("local_st_scales", "gru_scales"):
        if result[key] != [2] or any(type(x) is not int for x in result[key]):
            raise ValueError(f"MVP requires {key}=[2]; heavy readers run only at stage 2")
    if type(result["local_st_checkpoint_chunks"]) is not bool:
        raise ValueError("local_st_checkpoint_chunks must be boolean")
    gain = result["dynamic_kernel_gain"]
    if isinstance(gain, bool) or not isinstance(gain, (int, float)) or not math.isfinite(gain) or gain <= 0:
        raise ValueError("dynamic_kernel_gain must be finite and positive")
    if isinstance(result["gru_hidden_ratio"], bool) or result["gru_hidden_ratio"] != 0.5:
        raise ValueError("MVP requires gru_hidden_ratio=0.5")
    return result


def _check_inputs(rgb, event2d, event3d, channels):
    if rgb.ndim != 4 or event2d.shape != rgb.shape or rgb.shape[1] != channels:
        raise ValueError(f"R and E2 must have identical [B,{channels},H,W] shapes")
    b, c, h, w = rgb.shape
    if min(b, h, w) < 1 or event3d.ndim != 5 or event3d.shape[:2] != (b, c) or event3d.shape[3:] != (h, w):
        raise ValueError("E3 must be [B,C,T,H,W] matching R/E2 in B,C,H,W")
    if event3d.shape[2] < 1:
        raise ValueError("E3 must contain at least one temporal position")
    if any(x.device != rgb.device or x.dtype != rgb.dtype for x in (event2d, event3d)):
        raise ValueError("R, E2 and E3 must have the same device and dtype")
    if not rgb.is_floating_point():
        raise ValueError("Fusion inputs must be floating point tensors")


class TemporalPositionBias(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(2, 16), nn.GELU(), nn.Linear(16, output_dim))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, length, reference):
        tau = reference.new_zeros(1) if length == 1 else torch.linspace(
            -1, 1, length, device=reference.device, dtype=reference.dtype
        )
        return self.mlp(torch.stack((tau, tau.square()), dim=-1))


class _ProjectedReader(nn.Module):
    def __init__(self, channels, condition_channels, width):
        super().__init__()
        if type(channels) is not int or type(width) is not int or channels < 1 or width < 1:
            raise ValueError("Reader channels and projection width must be positive integers")
        if condition_channels not in (channels, channels * 2):
            raise ValueError("condition_channels must be C or 2C")
        self.channels = channels
        self.width = width
        self.source_norms = nn.ModuleList([
            ChannelLayerNorm2D(channels) for _ in range(condition_channels // channels)
        ])
        self.norm_k = ChannelLayerNorm3D(channels)
        self.norm_v = ChannelLayerNorm3D(channels)
        self.q_proj = nn.Conv2d(condition_channels, width, 1, bias=False)
        self.k_proj = nn.Conv3d(channels, width, 1, bias=False)
        self.v_proj = nn.Conv3d(channels, width, 1, bias=False)
        self.out_proj = nn.Conv2d(width, channels, 1, bias=False)

    def project(self, query_sources, event3d):
        if not isinstance(query_sources, (tuple, list)) or len(query_sources) != len(self.source_norms):
            raise ValueError("query_sources count does not match condition_channels")
        first = query_sources[0]
        for source in query_sources:
            _check_inputs(first, source, event3d, self.channels)
        q = self.q_proj(torch.cat([
            norm(source) for norm, source in zip(self.source_norms, query_sources)
        ], dim=1))
        return q, self.k_proj(self.norm_k(event3d)), self.v_proj(self.norm_v(event3d))


class ConditionalTemporalRead(_ProjectedReader):
    def __init__(self, channels, condition_channels, reduction=4, min_dim=8):
        super().__init__(channels, condition_channels, max(channels // reduction, min_dim))
        self.time_bias = TemporalPositionBias(1)

    def forward(self, query_sources, event3d, return_weights=False):
        q, k, v = self.project(query_sources, event3d)
        # Explicitly disable autocast for scores and softmax, including future AMP callers.
        with torch.autocast(device_type=q.device.type, enabled=False):
            logits = (q.float().unsqueeze(2) * k.float()).sum(dim=1) / math.sqrt(self.width)
            logits = logits + self.time_bias(event3d.shape[2], q).float().view(1, -1, 1, 1)
            weights = torch.softmax(logits, dim=1)
        mixed = (weights.to(v.dtype).unsqueeze(1) * v).sum(dim=2)
        result = self.out_proj(mixed).to(query_sources[0].dtype)
        return (result, weights) if return_weights else result


class SpatialGate(nn.Module):
    def __init__(self, channels, n_inputs, n_gates=1, reduction=4, min_dim=8):
        super().__init__()
        self.norms = nn.ModuleList([ChannelLayerNorm2D(channels) for _ in range(n_inputs)])
        hidden = max(channels // reduction, min_dim)
        self.net = nn.Sequential(
            nn.Conv2d(n_inputs * channels, hidden, 1),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden), nn.GELU(),
            nn.Conv2d(hidden, n_gates, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, *sources):
        if len(sources) != len(self.norms) or any(x.shape != sources[0].shape for x in sources):
            raise ValueError("SpatialGate sources must match its input count and shape")
        logits = self.net(torch.cat([norm(x) for norm, x in zip(self.norms, sources)], dim=1))
        return torch.sigmoid(logits.float()).to(sources[0].dtype)


class LocalSTRead(_ProjectedReader):
    def __init__(self, channels, condition_channels, num_heads=4, reduction=4, min_dim=8,
                 query_chunk=512, checkpoint_chunks=True):
        width = max(channels // reduction, min_dim)
        if num_heads < 1 or width % num_heads:
            raise ValueError("LocalSTRead total projection width must be divisible by heads")
        if query_chunk < 1:
            raise ValueError("LocalSTRead query_chunk must be positive")
        super().__init__(channels, condition_channels, width)
        self.num_heads = num_heads
        self.head_dim = width // num_heads
        self.query_chunk = query_chunk
        self.checkpoint_chunks = checkpoint_chunks
        self.time_bias = TemporalPositionBias(num_heads)
        self.spatial_bias = nn.Parameter(torch.zeros(num_heads, 9))
        self._neighbor_cache = None

    def neighbors(self, height, width, device):
        key = (height, width, device)
        if self._neighbor_cache is None or self._neighbor_cache[0] != key:
            y, x = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij")
            dy, dx = torch.meshgrid(torch.arange(-1, 2, device=device), torch.arange(-1, 2, device=device), indexing="ij")
            yy, xx = y.flatten()[:, None] + dy.flatten(), x.flatten()[:, None] + dx.flatten()
            valid = (yy >= 0) & (yy < height) & (xx >= 0) & (xx < width)
            index = yy.clamp(0, height - 1) * width + xx.clamp(0, width - 1)
            # Bounded, graph-free cache: never retains feature tensors from earlier steps.
            self._neighbor_cache = (key, index, valid)
        return self._neighbor_cache[1:]

    def forward(self, query_sources, event3d):
        q, k, v = self.project(query_sources, event3d)
        b, _, height, width = q.shape
        length, n = event3d.shape[2], height * width
        q = q.reshape(b, self.num_heads, self.head_dim, n).permute(0, 1, 3, 2)
        k = k.reshape(b, self.num_heads, self.head_dim, length, n).permute(0, 1, 3, 4, 2)
        v = v.reshape(b, self.num_heads, self.head_dim, length, n).permute(0, 1, 3, 4, 2)
        index, valid = self.neighbors(height, width, q.device)
        time_bias = self.time_bias(length, q).transpose(0, 1)
        outputs = []
        for start in range(0, n, self.query_chunk):
            end = min(start + self.query_chunk, n)

            # Bind indices now: backward checkpoint recomputation happens after this loop.
            def read_chunk(q_all, k_all, v_all, tb, sb, start=start, end=end):
                idx, mask = index[start:end], valid[start:end]
                kc = k_all.index_select(3, idx.reshape(-1)).reshape(
                    b, self.num_heads, length, end - start, 9, self.head_dim
                ).permute(0, 1, 3, 2, 4, 5).reshape(
                    b, self.num_heads, end - start, length * 9, self.head_dim
                )
                vc = v_all.index_select(3, idx.reshape(-1)).reshape(
                    b, self.num_heads, length, end - start, 9, self.head_dim
                ).permute(0, 1, 3, 2, 4, 5).reshape_as(kc)
                with torch.autocast(device_type=q_all.device.type, enabled=False):
                    logits = torch.einsum("bhnd,bhnld->bhnl", q_all[:, :, start:end].float(), kc.float()) / math.sqrt(self.head_dim)
                    bias = (tb.float()[:, :, None] + sb.float()[:, None, :]).reshape(self.num_heads, 1, length * 9)
                    mask = mask[:, None, :].expand(-1, length, -1).reshape(end - start, length * 9)
                    logits = (logits + bias).masked_fill(~mask[None, None], float("-inf"))
                    weights = torch.softmax(logits, dim=-1)
                return torch.einsum("bhnl,bhnld->bhnd", weights.to(vc.dtype), vc)

            if self.checkpoint_chunks and self.training and torch.is_grad_enabled():
                output = checkpoint(read_chunk, q, k, v, time_bias, self.spatial_bias, use_reentrant=False)
            else:
                output = read_chunk(q, k, v, time_bias, self.spatial_bias)
            outputs.append(output)
        mixed = torch.cat(outputs, dim=2).permute(0, 1, 3, 2).reshape(b, self.width, height, width)
        return self.out_proj(mixed).to(query_sources[0].dtype)


def grouped_dynamic_filter(x, kernels):
    if x.ndim != 4 or kernels.ndim != 5:
        raise ValueError("Dynamic filter expects X[B,C,H,W], K[B,G,9,H,W]")
    b, c, h, w = x.shape
    if kernels.shape[0] != b or kernels.shape[1] < 1 or kernels.shape[2:] != (9, h, w) or c % kernels.shape[1]:
        raise ValueError("Dynamic kernel groups/shapes do not match X")
    if kernels.device != x.device or kernels.dtype != x.dtype:
        raise ValueError("Dynamic kernels must match X device and dtype")
    groups = kernels.shape[1]
    padded = F.pad(x, (1, 1, 1, 1), mode="replicate")
    result = torch.zeros_like(x).reshape(b, groups, c // groups, h, w)
    for j, (dy, dx) in enumerate((dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)):
        shifted = padded[:, :, 1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
        result = result + shifted.reshape(b, groups, c // groups, h, w) * kernels[:, :, j].unsqueeze(2)
    return result.reshape(b, c, h, w)


class _FusionBase(nn.Module):
    def __init__(self, channels, num_heads=4, gamma_init=0.1, options=None, stage_index=0):
        super().__init__()
        if type(channels) is not int or type(num_heads) is not int or channels < 1 or num_heads < 1 or channels % num_heads:
            raise ValueError("Fusion channels must be positive and divisible by num_heads")
        if not math.isfinite(gamma_init) or gamma_init <= 0:
            raise ValueError("New fusion gamma_init must be finite and positive")
        self.channels, self.num_heads = channels, num_heads
        self.gamma_init, self.stage_index = gamma_init, stage_index
        self.options = normalize_fusion_options(options)

    def ca(self):
        return EventImageChannelCrossAttention2D(self.channels, self.num_heads)

    def inject(self):
        return nn.Conv2d(self.channels, self.channels, 3, padding=1)

    def gamma(self):
        return nn.Parameter(torch.full((1,), float(self.gamma_init)))

    def temporal_reader(self, sources=2):
        return ConditionalTemporalRead(
            self.channels, self.channels * sources,
            self.options["temporal_reduction"], self.options["temporal_min_dim"],
        )

    def local_reader(self, sources=2):
        opt = self.options
        return LocalSTRead(self.channels, self.channels * sources, opt["local_st_heads"],
                           opt["local_st_reduction"], opt["local_st_min_dim"],
                           opt["local_st_query_chunk"], opt["local_st_checkpoint_chunks"])

    def gate(self, sources, n_gates=1):
        return SpatialGate(self.channels, sources, n_gates,
                           self.options["gate_reduction"], self.options["gate_min_dim"])


class ConditionalTemporalTwoCAFusion(_FusionBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ca_e2, self.ca_e3 = self.ca(), self.ca()
        self.inject_e2, self.inject_e3 = self.inject(), self.inject()
        self.gamma_e2, self.gamma_e3 = self.gamma(), self.gamma()
        self.temporal_read = self.temporal_reader()
        self.spatial_gate = self.gate(3)

    def forward(self, rgb, event2d, event3d):
        _check_inputs(rgb, event2d, event3d, self.channels)
        x = rgb + self.gamma_e2 * self.inject_e2(self.ca_e2(rgb, event2d, event2d))
        memory = self.temporal_read([x, event2d], event3d)
        delta = self.inject_e3(self.ca_e3(x, memory, memory))
        return x + self.gamma_e3 * self.spatial_gate(x, event2d, memory) * delta


class EventControlledDynamicFilterFusion(_FusionBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.groups = self.options["dynamic_groups"]
        if self.channels % self.groups:
            raise ValueError("Fusion channels must be divisible by dynamic_groups")
        self.ca_e2 = self.ca()
        self.inject_e2, self.inject_dyn = self.inject(), self.inject()
        self.gamma_e2, self.gamma_dyn = self.gamma(), self.gamma()
        self.temporal_read = self.temporal_reader()
        self.spatial_gate = self.gate(2)
        width = max(self.channels // self.options["temporal_reduction"], self.options["temporal_min_dim"])
        self.kernel_head = nn.Sequential(nn.Conv2d(self.channels, width, 1), nn.GELU(),
                                         nn.Conv2d(width, self.groups * 9, 1))
        nn.init.normal_(self.kernel_head[-1].weight, 0, 0.02)
        nn.init.zeros_(self.kernel_head[-1].bias)

    def kernels(self, memory):
        b, _, h, w = memory.shape
        values = torch.tanh(self.kernel_head(memory)).reshape(b, self.groups, 9, h, w)
        return (self.options["dynamic_kernel_gain"] / 9.0) * (values - values.mean(dim=2, keepdim=True))

    def forward(self, rgb, event2d, event3d):
        _check_inputs(rgb, event2d, event3d, self.channels)
        x = rgb + self.gamma_e2 * self.inject_e2(self.ca_e2(rgb, event2d, event2d))
        memory = self.temporal_read([x, event2d], event3d)
        delta = grouped_dynamic_filter(x, self.kernels(memory))
        return x + self.gamma_dyn * self.spatial_gate(x, memory) * self.inject_dyn(delta)


class AnchoredLocalSTFusion(_FusionBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ca_e2 = self.ca()
        self.inject_e2, self.inject_st = self.inject(), self.inject()
        self.gamma_e2, self.gamma_st = self.gamma(), self.gamma()
        self.local_st_read = self.local_reader()
        self.spatial_gate = self.gate(3)

    def forward(self, rgb, event2d, event3d):
        _check_inputs(rgb, event2d, event3d, self.channels)
        x = rgb + self.gamma_e2 * self.inject_e2(self.ca_e2(rgb, event2d, event2d))
        memory = self.local_st_read([x, event2d], event3d)
        return x + self.gamma_st * self.spatial_gate(x, event2d, memory) * self.inject_st(memory)


class ConditionedConvGRUCell(nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.zr_conv = nn.Conv2d(hidden_channels * 3, hidden_channels * 2, 3, padding=1)
        self.candidate_conv = nn.Conv2d(hidden_channels * 3, hidden_channels, 3, padding=1)
        nn.init.zeros_(self.zr_conv.bias)
        nn.init.zeros_(self.candidate_conv.bias)

    def forward(self, z, previous, context):
        gates = torch.sigmoid(self.zr_conv(torch.cat((z, previous, context), dim=1)).float()).to(z.dtype)
        update, reset = gates.chunk(2, dim=1)
        candidate = torch.tanh(self.candidate_conv(torch.cat((z, reset * previous, context), dim=1)).float()).to(z.dtype)
        return (1 - update) * previous + update * candidate


class ConditionedBiGRUFusion(_FusionBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.channels % 2:
            raise ValueError("ConditionedBiGRUFusion requires even channels")
        hidden = self.channels // 2
        self.norm_rgb, self.norm_e2 = ChannelLayerNorm2D(self.channels), ChannelLayerNorm2D(self.channels)
        self.norm_e3 = ChannelLayerNorm3D(self.channels)
        self.conv_context = nn.Conv2d(self.channels * 2, hidden, 1)
        self.conv_z = nn.Conv3d(self.channels, hidden, 1)
        self.init_fwd, self.init_bwd = nn.Conv2d(hidden, hidden, 1), nn.Conv2d(hidden, hidden, 1)
        self.time_bias = TemporalPositionBias(hidden)
        self.fwd_cell, self.bwd_cell = ConditionedConvGRUCell(hidden), ConditionedConvGRUCell(hidden)
        self.conv_merge = nn.Conv2d(hidden * 2, self.channels, 1)
        self.ca_e2, self.ca_memory = self.ca(), self.ca()
        self.inject_e2, self.inject_memory = self.inject(), self.inject()
        self.gamma_e2, self.gamma_memory = self.gamma(), self.gamma()

    def forward(self, rgb, event2d, event3d):
        _check_inputs(rgb, event2d, event3d, self.channels)
        context = self.conv_context(torch.cat((self.norm_rgb(rgb), self.norm_e2(event2d)), dim=1))
        z = self.conv_z(self.norm_e3(event3d))
        length = z.shape[2]
        z = z + self.time_bias(length, z).transpose(0, 1)[None, :, :, None, None]
        hf, hb = torch.tanh(self.init_fwd(context)), torch.tanh(self.init_bwd(context))
        for t in range(length):
            hf = self.fwd_cell(z[:, :, t], hf, context)
        for t in range(length - 1, -1, -1):
            hb = self.bwd_cell(z[:, :, t], hb, context)
        memory = self.conv_merge(torch.cat((hf, hb), dim=1))
        d2 = self.inject_e2(self.ca_e2(rgb, event2d, event2d))
        dt = self.inject_memory(self.ca_memory(rgb, memory, memory))
        return rgb + self.gamma_e2 * d2 + self.gamma_memory * dt


class IdentityPreservingDualExpertFusion(_FusionBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ca_e2 = self.ca()
        self.inject_e2, self.inject_e3 = self.inject(), self.inject()
        self.gamma_e2, self.gamma_e3 = self.gamma(), self.gamma()
        self.spatial_gate = self.gate(3, 2)
        self.use_local = self.stage_index in self.options["local_st_scales"]
        if self.use_local:
            self.e3_read = self.local_reader(sources=1)
        else:
            self.e3_read = self.temporal_reader(sources=1)
            self.ca_e3 = self.ca()

    def forward(self, rgb, event2d, event3d):
        _check_inputs(rgb, event2d, event3d, self.channels)
        d2 = self.inject_e2(self.ca_e2(rgb, event2d, event2d))
        memory = self.e3_read([rgb], event3d)
        d3 = self.inject_e3(memory if self.use_local else self.ca_e3(rgb, memory, memory))
        g2, g3 = self.spatial_gate(rgb, d2, d3).chunk(2, dim=1)
        return rgb + self.gamma_e2 * g2 * d2 + self.gamma_e3 * g3 * d3


def build_temporal_fusion(mode, channels, num_heads, gamma_init, stage_index, fusion_options=None):
    options = normalize_fusion_options(fusion_options)
    if stage_index not in (0, 1, 2):
        raise ValueError("stage_index must be 0, 1 or 2")
    if mode == "anchored_local_st" and stage_index not in options["local_st_scales"]:
        mode = "cond_temporal_2ca"
    if mode == "conditioned_bi_gru" and stage_index not in options["gru_scales"]:
        mode = "cond_temporal_2ca"
    classes = {
        "cond_temporal_2ca": ConditionalTemporalTwoCAFusion,
        "event_controlled_dynamic_filter": EventControlledDynamicFilterFusion,
        "anchored_local_st": AnchoredLocalSTFusion,
        "conditioned_bi_gru": ConditionedBiGRUFusion,
        "identity_preserving_dual_expert": IdentityPreservingDualExpertFusion,
    }
    if mode not in classes:
        raise ValueError(f"Unknown temporal fusion mode: {mode}")
    return classes[mode](channels, num_heads, gamma_init, options, stage_index)
