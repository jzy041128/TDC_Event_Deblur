import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d

from models.tdc_module import ShortTermTDCBlock3D


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)


class ConvBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            conv3x3(in_channels, out_channels, stride),
            nn.LeakyReLU(0.2, inplace=True),
            conv3x3(out_channels, out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1)
        self.refine = ConvBlock2D(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.refine(torch.cat([x, skip], dim=1))


class ResidualUpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        merged_channels = out_channels + skip_channels
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1)
        self.main = ConvBlock2D(merged_channels, out_channels)
        self.shortcut = nn.Conv2d(merged_channels, out_channels, 1)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        merged = torch.cat([x, skip], dim=1)
        return self.shortcut(merged) + self.main(merged)


class SupervisedAttentionModule(nn.Module):
    def __init__(self, channels, image_channels=3, event_channels=0):
        super().__init__()
        self.feature = nn.Conv2d(channels, channels, 3, padding=1)
        self.attention = nn.Sequential(
            nn.Conv2d(image_channels + event_channels, channels, 3, padding=1),
            nn.Sigmoid(),
        )
        self.project = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, feature, stage1_image, event_guide=None):
        attention_input = stage1_image
        if event_guide is not None:
            attention_input = torch.cat([stage1_image, event_guide], dim=1)
        attended = self.feature(feature) * self.attention(attention_input)
        return feature + self.project(attended)


class ShallowRefineNet(nn.Module):
    def __init__(self, channels, image_channels=3):
        super().__init__()
        self.body = nn.Sequential(
            ConvBlock2D(channels + image_channels * 2, channels),
            ConvBlock2D(channels, channels),
            nn.Conv2d(channels, image_channels, 3, padding=1),
        )

    def forward(self, feature, stage1_image, blur):
        return stage1_image + self.body(torch.cat([feature, stage1_image, blur], dim=1))


class LightUNetRefineNet(nn.Module):
    def __init__(self, channels, image_channels=3):
        super().__init__()
        self.stem = ConvBlock2D(channels + image_channels * 2, channels)
        self.down = ConvBlock2D(channels, channels * 2, stride=2)
        self.bottleneck = ConvBlock2D(channels * 2, channels * 2)
        self.up = ResidualUpBlock(channels * 2, channels, channels)
        self.to_image = nn.Conv2d(channels, image_channels, 3, padding=1)

    def forward(self, feature, stage1_image, blur):
        x0 = self.stem(torch.cat([feature, stage1_image, blur], dim=1))
        x1 = self.bottleneck(self.down(x0))
        return stage1_image + self.to_image(self.up(x1, x0))


class ChannelLayerNorm2D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()


class ChannelLayerNorm3D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3).contiguous()


class PlainChannelSelfAttention2D(nn.Module):
    def __init__(self, channels, num_heads, qk_norm=False):
        super().__init__()
        if channels % num_heads:
            raise ValueError("channels must be divisible by num_heads")
        self.num_heads = num_heads
        self.qk_norm = qk_norm
        self.norm = ChannelLayerNorm2D(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        residual = x
        b, c, h, w = x.shape
        head_dim = c // self.num_heads
        spatial_dim = h * w
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=1)
        q = q.reshape(b, self.num_heads, head_dim, spatial_dim)
        k = k.reshape(b, self.num_heads, head_dim, spatial_dim)
        v = v.reshape(b, self.num_heads, head_dim, spatial_dim)
        if self.qk_norm:
            q = F.normalize(q, dim=-1)
            k = F.normalize(k, dim=-1)
            scale = 1.0
        else:
            scale = spatial_dim ** -0.5
        weights = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * scale, dim=-1)
        attended = torch.matmul(weights, v).reshape(b, c, h, w)
        return residual + self.gamma * self.proj(attended)


class PlainChannelSelfAttention3D(nn.Module):
    def __init__(self, channels, num_heads, qk_norm=False):
        super().__init__()
        if channels % num_heads:
            raise ValueError("channels must be divisible by num_heads")
        self.num_heads = num_heads
        self.qk_norm = qk_norm
        self.norm = ChannelLayerNorm3D(channels)
        self.qkv = nn.Conv3d(channels, channels * 3, 1, bias=False)
        self.proj = nn.Conv3d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        residual = x
        b, c, t, h, w = x.shape
        head_dim = c // self.num_heads
        spatiotemporal_dim = t * h * w
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=1)
        q = q.reshape(b, self.num_heads, head_dim, spatiotemporal_dim)
        k = k.reshape(b, self.num_heads, head_dim, spatiotemporal_dim)
        v = v.reshape(b, self.num_heads, head_dim, spatiotemporal_dim)
        if self.qk_norm:
            q = F.normalize(q, dim=-1)
            k = F.normalize(k, dim=-1)
            scale = 1.0
        else:
            scale = spatiotemporal_dim ** -0.5
        weights = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * scale, dim=-1)
        attended = torch.matmul(weights, v).reshape(b, c, t, h, w)
        return residual + self.gamma * self.proj(attended)


class GatedDconvFFN2D(nn.Module):
    def __init__(self, channels, expansion_factor=2.0):
        super().__init__()
        hidden_channels = int(channels * expansion_factor)
        self.project_in = nn.Conv2d(channels, hidden_channels * 2, 1)
        self.dwconv = nn.Conv2d(
            hidden_channels * 2,
            hidden_channels * 2,
            3,
            padding=1,
            groups=hidden_channels * 2,
        )
        self.project_out = nn.Conv2d(hidden_channels, channels, 1)

    def forward(self, x):
        x1, x2 = self.dwconv(self.project_in(x)).chunk(2, dim=1)
        return self.project_out(F.gelu(x1) * x2)


class GatedDconvFFN3D(nn.Module):
    def __init__(self, channels, expansion_factor=2.0):
        super().__init__()
        hidden_channels = int(channels * expansion_factor)
        self.project_in = nn.Conv3d(channels, hidden_channels * 2, 1)
        self.dwconv = nn.Conv3d(
            hidden_channels * 2,
            hidden_channels * 2,
            3,
            padding=1,
            groups=hidden_channels * 2,
        )
        self.project_out = nn.Conv3d(hidden_channels, channels, 1)

    def forward(self, x):
        x1, x2 = self.dwconv(self.project_in(x)).chunk(2, dim=1)
        return self.project_out(F.gelu(x1) * x2)


class RestormerChannelSelfAttention2D(nn.Module):
    def __init__(self, channels, num_heads):
        super().__init__()
        if channels % num_heads:
            raise ValueError("channels must be divisible by num_heads")
        self.num_heads = num_heads
        self.norm1 = ChannelLayerNorm2D(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.qkv_dwconv = nn.Conv2d(
            channels * 3,
            channels * 3,
            3,
            padding=1,
            groups=channels * 3,
            bias=False,
        )
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.project_out = nn.Conv2d(channels, channels, 1, bias=False)
        self.norm2 = ChannelLayerNorm2D(channels)
        self.ffn = GatedDconvFFN2D(channels)

    def forward(self, x):
        residual = x
        b, c, h, w = x.shape
        head_dim = c // self.num_heads
        q, k, v = self.qkv_dwconv(self.qkv(self.norm1(x))).chunk(3, dim=1)
        q = q.reshape(b, self.num_heads, head_dim, h * w)
        k = k.reshape(b, self.num_heads, head_dim, h * w)
        v = v.reshape(b, self.num_heads, head_dim, h * w)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        weights = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * self.temperature, dim=-1)
        attended = torch.matmul(weights, v).view(b, c, h, w)
        x = residual + self.project_out(attended)
        return x + self.ffn(self.norm2(x))


class RestormerChannelSelfAttention3D(nn.Module):
    def __init__(self, channels, num_heads):
        super().__init__()
        if channels % num_heads:
            raise ValueError("channels must be divisible by num_heads")
        self.num_heads = num_heads
        self.norm1 = ChannelLayerNorm3D(channels)
        self.qkv = nn.Conv3d(channels, channels * 3, 1, bias=False)
        self.qkv_dwconv = nn.Conv3d(
            channels * 3,
            channels * 3,
            3,
            padding=1,
            groups=channels * 3,
            bias=False,
        )
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.project_out = nn.Conv3d(channels, channels, 1, bias=False)
        self.norm2 = ChannelLayerNorm3D(channels)
        self.ffn = GatedDconvFFN3D(channels)

    def forward(self, x):
        residual = x
        b, c, t, h, w = x.shape
        head_dim = c // self.num_heads
        q, k, v = self.qkv_dwconv(self.qkv(self.norm1(x))).chunk(3, dim=1)
        q = q.reshape(b, self.num_heads, head_dim, t * h * w)
        k = k.reshape(b, self.num_heads, head_dim, t * h * w)
        v = v.reshape(b, self.num_heads, head_dim, t * h * w)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        weights = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * self.temperature, dim=-1)
        attended = torch.matmul(weights, v).view(b, c, t, h, w)
        x = residual + self.project_out(attended)
        return x + self.ffn(self.norm2(x))


class ImageGuidedDeformAlignment2D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = ChannelLayerNorm2D(channels)
        self.offset = nn.Conv2d(channels, 18, 3, padding=1)
        self.deform = DeformConv2d(channels, channels, 3, padding=1)
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)

    def forward(self, x):
        normalized = self.norm(x)
        return x + self.deform(normalized, self.offset(normalized))


def partition_2d(x, window_size):
    b, c, h, w = x.shape
    hp = math.ceil(h / window_size) * window_size
    wp = math.ceil(w / window_size) * window_size
    x = F.pad(x, (0, wp - w, 0, hp - h))
    x = x.view(b, c, hp // window_size, window_size, wp // window_size, window_size)
    x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
    return x.view(-1, window_size * window_size, c), (b, h, w, hp, wp)


def reverse_2d(windows, shape, window_size):
    b, h, w, hp, wp = shape
    c = windows.shape[-1]
    x = windows.view(b, hp // window_size, wp // window_size, window_size, window_size, c)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous().view(b, c, hp, wp)
    return x[:, :, :h, :w]


def partition_3d(x, temporal_window, spatial_window):
    b, c, t, h, w = x.shape
    tp = math.ceil(t / temporal_window) * temporal_window
    hp = math.ceil(h / spatial_window) * spatial_window
    wp = math.ceil(w / spatial_window) * spatial_window
    x = F.pad(x, (0, wp - w, 0, hp - h, 0, tp - t))
    x = x.view(
        b,
        c,
        tp // temporal_window,
        temporal_window,
        hp // spatial_window,
        spatial_window,
        wp // spatial_window,
        spatial_window,
    )
    x = x.permute(0, 2, 4, 6, 3, 5, 7, 1).contiguous()
    tokens = temporal_window * spatial_window * spatial_window
    return x.view(-1, tokens, c), (b, t, h, w, tp, hp, wp)


def reverse_3d(windows, shape, temporal_window, spatial_window):
    b, t, h, w, tp, hp, wp = shape
    c = windows.shape[-1]
    x = windows.view(
        b,
        tp // temporal_window,
        hp // spatial_window,
        wp // spatial_window,
        temporal_window,
        spatial_window,
        spatial_window,
        c,
    )
    x = x.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous().view(b, c, tp, hp, wp)
    return x[:, :, :t, :h, :w]


def attention(q, k, v, num_heads, qk_norm=False):
    batch_windows, tokens, channels = q.shape
    head_dim = channels // num_heads
    q = q.view(batch_windows, tokens, num_heads, head_dim).transpose(1, 2)
    k = k.view(batch_windows, tokens, num_heads, head_dim).transpose(1, 2)
    v = v.view(batch_windows, tokens, num_heads, head_dim).transpose(1, 2)
    if qk_norm:
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        scale = 1.0
    else:
        scale = head_dim ** -0.5
    weights = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * scale, dim=-1)
    out = torch.matmul(weights, v).transpose(1, 2).contiguous()
    return out.view(batch_windows, tokens, channels)


def cosine_window_attention(q, k, v, num_heads, temperature, attn_mask=None):
    batch_windows, tokens, channels = q.shape
    head_dim = channels // num_heads
    q = q.view(batch_windows, tokens, num_heads, head_dim).transpose(1, 2)
    k = k.view(batch_windows, tokens, num_heads, head_dim).transpose(1, 2)
    v = v.view(batch_windows, tokens, num_heads, head_dim).transpose(1, 2)
    q = F.normalize(q, dim=-1)
    k = F.normalize(k, dim=-1)
    weights = torch.matmul(q, k.transpose(-2, -1)) * temperature.unsqueeze(0)
    if attn_mask is not None:
        num_windows = attn_mask.shape[0]
        batch_size = batch_windows // num_windows
        weights = weights.view(batch_size, num_windows, num_heads, tokens, tokens)
        weights = weights + attn_mask.to(dtype=weights.dtype)[None, :, None, :, :]
        weights = weights.view(batch_windows, num_heads, tokens, tokens)
    weights = torch.softmax(weights, dim=-1)
    out = torch.matmul(weights, v).transpose(1, 2).contiguous()
    return out.view(batch_windows, tokens, channels)


def shifted_window_mask_2d(hp, wp, window_size, shift_size, device):
    mask = torch.zeros((1, 1, hp, wp), device=device)
    slices = (
        slice(0, -window_size),
        slice(-window_size, -shift_size),
        slice(-shift_size, None),
    )
    region = 0
    for h_slice in slices:
        for w_slice in slices:
            mask[:, :, h_slice, w_slice] = region
            region += 1
    windows, _ = partition_2d(mask, window_size)
    windows = windows.squeeze(-1)
    differences = windows.unsqueeze(1) - windows.unsqueeze(2)
    return differences.ne(0).to(mask.dtype) * -100.0


def shifted_window_mask_3d(hp, wp, temporal_window, spatial_window, shift_size, device):
    # The same spatial boundary mask is shared by every temporal-window group.
    mask = torch.zeros((1, 1, temporal_window, hp, wp), device=device)
    slices = (
        slice(0, -spatial_window),
        slice(-spatial_window, -shift_size),
        slice(-shift_size, None),
    )
    region = 0
    for h_slice in slices:
        for w_slice in slices:
            mask[:, :, :, h_slice, w_slice] = region
            region += 1
    windows, _ = partition_3d(mask, temporal_window, spatial_window)
    windows = windows.squeeze(-1)
    differences = windows.unsqueeze(1) - windows.unsqueeze(2)
    return differences.ne(0).to(mask.dtype) * -100.0


class WindowSelfAttention2D(nn.Module):
    def __init__(self, channels, window_size, num_heads, qk_norm=False):
        super().__init__()
        if channels % num_heads:
            raise ValueError("channels must be divisible by num_heads")
        self.window_size = window_size
        self.num_heads = num_heads
        self.qk_norm = qk_norm
        self.norm = ChannelLayerNorm2D(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        residual = x
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=1)
        q, shape = partition_2d(q, self.window_size)
        k, _ = partition_2d(k, self.window_size)
        v, _ = partition_2d(v, self.window_size)
        out = attention(q, k, v, self.num_heads, self.qk_norm)
        out = reverse_2d(out, shape, self.window_size)
        return residual + self.gamma * self.proj(out)


class WindowSelfAttention3D(nn.Module):
    def __init__(self, channels, spatial_window, temporal_window, num_heads, qk_norm=False):
        super().__init__()
        if channels % num_heads:
            raise ValueError("channels must be divisible by num_heads")
        self.spatial_window = spatial_window
        self.temporal_window = temporal_window
        self.num_heads = num_heads
        self.qk_norm = qk_norm
        self.norm = ChannelLayerNorm3D(channels)
        self.qkv = nn.Conv3d(channels, channels * 3, 1, bias=False)
        self.proj = nn.Conv3d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        residual = x
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=1)
        q, shape = partition_3d(q, self.temporal_window, self.spatial_window)
        k, _ = partition_3d(k, self.temporal_window, self.spatial_window)
        v, _ = partition_3d(v, self.temporal_window, self.spatial_window)
        out = attention(q, k, v, self.num_heads, self.qk_norm)
        out = reverse_3d(out, shape, self.temporal_window, self.spatial_window)
        return residual + self.gamma * self.proj(out)


class RestormerWindowSelfAttention2D(nn.Module):
    def __init__(self, channels, window_size, num_heads, shift_size=0):
        super().__init__()
        if channels % num_heads:
            raise ValueError("channels must be divisible by num_heads")
        if not 0 <= shift_size < window_size:
            raise ValueError("shift_size must be in [0, window_size)")
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_heads = num_heads
        self.norm1 = ChannelLayerNorm2D(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.qkv_dwconv = nn.Conv2d(
            channels * 3,
            channels * 3,
            3,
            padding=1,
            groups=channels * 3,
            bias=False,
        )
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.project_out = nn.Conv2d(channels, channels, 1, bias=False)
        self.norm2 = ChannelLayerNorm2D(channels)
        self.ffn = GatedDconvFFN2D(channels)
        self._mask_cache = {}

    def forward(self, x):
        residual = x
        q, k, v = self.qkv_dwconv(self.qkv(self.norm1(x))).chunk(3, dim=1)
        if self.shift_size:
            shifts = (-self.shift_size, -self.shift_size)
            q = torch.roll(q, shifts=shifts, dims=(-2, -1))
            k = torch.roll(k, shifts=shifts, dims=(-2, -1))
            v = torch.roll(v, shifts=shifts, dims=(-2, -1))
        q, shape = partition_2d(q, self.window_size)
        k, _ = partition_2d(k, self.window_size)
        v, _ = partition_2d(v, self.window_size)
        attn_mask = None
        if self.shift_size:
            _, _, _, hp, wp = shape
            mask_key = (hp, wp, q.device.type, q.device.index)
            attn_mask = self._mask_cache.get(mask_key)
            if attn_mask is None:
                attn_mask = shifted_window_mask_2d(
                    hp, wp, self.window_size, self.shift_size, q.device
                )
                self._mask_cache[mask_key] = attn_mask
        out = cosine_window_attention(
            q, k, v, self.num_heads, self.temperature, attn_mask
        )
        out = reverse_2d(out, shape, self.window_size)
        if self.shift_size:
            out = torch.roll(
                out,
                shifts=(self.shift_size, self.shift_size),
                dims=(-2, -1),
            )
        x = residual + self.project_out(out)
        return x + self.ffn(self.norm2(x))


class RestormerWindowSelfAttention3D(nn.Module):
    def __init__(
        self,
        channels,
        spatial_window,
        temporal_window,
        num_heads,
        shift_size=0,
    ):
        super().__init__()
        if channels % num_heads:
            raise ValueError("channels must be divisible by num_heads")
        if not 0 <= shift_size < spatial_window:
            raise ValueError("shift_size must be in [0, spatial_window)")
        self.spatial_window = spatial_window
        self.temporal_window = temporal_window
        self.shift_size = shift_size
        self.num_heads = num_heads
        self.norm1 = ChannelLayerNorm3D(channels)
        self.qkv = nn.Conv3d(channels, channels * 3, 1, bias=False)
        self.qkv_dwconv = nn.Conv3d(
            channels * 3,
            channels * 3,
            3,
            padding=1,
            groups=channels * 3,
            bias=False,
        )
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.project_out = nn.Conv3d(channels, channels, 1, bias=False)
        self.norm2 = ChannelLayerNorm3D(channels)
        self.ffn = GatedDconvFFN3D(channels)
        self._mask_cache = {}

    def forward(self, x):
        residual = x
        q, k, v = self.qkv_dwconv(self.qkv(self.norm1(x))).chunk(3, dim=1)
        if self.shift_size:
            shifts = (-self.shift_size, -self.shift_size)
            q = torch.roll(q, shifts=shifts, dims=(-2, -1))
            k = torch.roll(k, shifts=shifts, dims=(-2, -1))
            v = torch.roll(v, shifts=shifts, dims=(-2, -1))
        q, shape = partition_3d(q, self.temporal_window, self.spatial_window)
        k, _ = partition_3d(k, self.temporal_window, self.spatial_window)
        v, _ = partition_3d(v, self.temporal_window, self.spatial_window)
        attn_mask = None
        if self.shift_size:
            _, _, _, _, _, hp, wp = shape
            mask_key = (hp, wp, q.device.type, q.device.index)
            attn_mask = self._mask_cache.get(mask_key)
            if attn_mask is None:
                attn_mask = shifted_window_mask_3d(
                    hp,
                    wp,
                    self.temporal_window,
                    self.spatial_window,
                    self.shift_size,
                    q.device,
                )
                self._mask_cache[mask_key] = attn_mask
        out = cosine_window_attention(
            q, k, v, self.num_heads, self.temperature, attn_mask
        )
        out = reverse_3d(
            out, shape, self.temporal_window, self.spatial_window
        )
        if self.shift_size:
            out = torch.roll(
                out,
                shifts=(self.shift_size, self.shift_size),
                dims=(-2, -1),
            )
        x = residual + self.project_out(out)
        return x + self.ffn(self.norm2(x))


class RestormerShiftedWindowPair2D(nn.Module):
    def __init__(self, channels, window_size, num_heads):
        super().__init__()
        self.blocks = nn.Sequential(
            RestormerWindowSelfAttention2D(channels, window_size, num_heads),
            RestormerWindowSelfAttention2D(
                channels,
                window_size,
                num_heads,
                shift_size=window_size // 2,
            ),
        )

    def forward(self, x):
        return self.blocks(x)


class RestormerShiftedWindowPair3D(nn.Module):
    def __init__(
        self,
        channels,
        spatial_window,
        temporal_window,
        num_heads,
    ):
        super().__init__()
        self.blocks = nn.Sequential(
            RestormerWindowSelfAttention3D(
                channels, spatial_window, temporal_window, num_heads
            ),
            RestormerWindowSelfAttention3D(
                channels,
                spatial_window,
                temporal_window,
                num_heads,
                shift_size=spatial_window // 2,
            ),
        )

    def forward(self, x):
        return self.blocks(x)


class WindowCrossAttention2D(nn.Module):
    def __init__(self, channels, window_size, num_heads, qk_norm=False):
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        self.qk_norm = qk_norm
        self.norm_q = ChannelLayerNorm2D(channels)
        self.norm_k = ChannelLayerNorm2D(channels)
        self.norm_v = ChannelLayerNorm2D(channels)
        self.q = nn.Conv2d(channels, channels, 1, bias=False)
        self.k = nn.Conv2d(channels, channels, 1, bias=False)
        self.v = nn.Conv2d(channels, channels, 1, bias=False)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, query, key, value):
        q, shape = partition_2d(self.q(self.norm_q(query)), self.window_size)
        k, _ = partition_2d(self.k(self.norm_k(key)), self.window_size)
        v, _ = partition_2d(self.v(self.norm_v(value)), self.window_size)
        out = attention(q, k, v, self.num_heads, self.qk_norm)
        return self.proj(reverse_2d(out, shape, self.window_size))


class WindowCrossAttention3D(nn.Module):
    def __init__(self, channels, spatial_window, temporal_window, num_heads, qk_norm=False):
        super().__init__()
        self.spatial_window = spatial_window
        self.temporal_window = temporal_window
        self.num_heads = num_heads
        self.qk_norm = qk_norm
        self.norm_q = ChannelLayerNorm3D(channels)
        self.norm_k = ChannelLayerNorm3D(channels)
        self.norm_v = ChannelLayerNorm3D(channels)
        self.q = nn.Conv3d(channels, channels, 1, bias=False)
        self.k = nn.Conv3d(channels, channels, 1, bias=False)
        self.v = nn.Conv3d(channels, channels, 1, bias=False)
        self.proj = nn.Conv3d(channels, channels, 1)

    def forward(self, query, key, value):
        q, shape = partition_3d(self.q(self.norm_q(query)), self.temporal_window, self.spatial_window)
        k, _ = partition_3d(self.k(self.norm_k(key)), self.temporal_window, self.spatial_window)
        v, _ = partition_3d(self.v(self.norm_v(value)), self.temporal_window, self.spatial_window)
        out = attention(q, k, v, self.num_heads, self.qk_norm)
        out = reverse_3d(out, shape, self.temporal_window, self.spatial_window)
        return self.proj(out)


class EventImageChannelCrossAttention2D(nn.Module):
    def __init__(self, channels, num_heads):
        super().__init__()
        if channels % num_heads:
            raise ValueError("channels must be divisible by num_heads")
        self.num_heads = num_heads
        self.norm_q = ChannelLayerNorm2D(channels)
        self.norm_k = ChannelLayerNorm2D(channels)
        self.norm_v = ChannelLayerNorm2D(channels)
        self.q = nn.Conv2d(channels, channels, 1, bias=False)
        self.k = nn.Conv2d(channels, channels, 1, bias=False)
        self.v = nn.Conv2d(channels, channels, 1, bias=False)
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.proj = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, query, key, value=None):
        value = key if value is None else value
        if query.shape != key.shape or query.shape != value.shape:
            raise ValueError("query, key, and value must have the same shape")
        b, c, h, w = query.shape
        head_dim = c // self.num_heads

        q = self.q(self.norm_q(query)).view(b, self.num_heads, head_dim, h * w)
        k = self.k(self.norm_k(key)).view(b, self.num_heads, head_dim, h * w)
        v = self.v(self.norm_v(value)).view(b, self.num_heads, head_dim, h * w)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        weights = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * self.temperature, dim=-1)
        out = torch.matmul(weights, v).view(b, c, h, w)
        return self.proj(out)


class EventReorganizedChannelCrossAttention2D(nn.Module):
    """Read event content through a shared reference-channel coordinate system."""

    def __init__(self, channels, num_heads):
        super().__init__()
        if channels % num_heads:
            raise ValueError("channels must be divisible by num_heads")
        self.num_heads = num_heads
        self.norm_q = ChannelLayerNorm2D(channels)
        self.norm_reference = ChannelLayerNorm2D(channels)
        self.norm_k = ChannelLayerNorm2D(channels)
        self.norm_v = ChannelLayerNorm2D(channels)
        self.q = nn.Conv2d(channels, channels, 1, bias=False)
        self.reference = nn.Conv2d(channels, channels, 1, bias=False)
        self.k = nn.Conv2d(channels, channels, 1, bias=False)
        self.v = nn.Conv2d(channels, channels, 1, bias=False)
        self.temperature_reorg = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperature_read = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.proj = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, rgb, reference, content):
        if rgb.shape != reference.shape or rgb.shape != content.shape:
            raise ValueError("RGB, reference and content features must have the same shape")
        b, c, h, w = rgb.shape
        head_shape = (b, self.num_heads, c // self.num_heads, h * w)
        q = F.normalize(self.q(self.norm_q(rgb)).reshape(head_shape), dim=-1)
        reference = F.normalize(
            self.reference(self.norm_reference(reference)).reshape(head_shape), dim=-1
        )
        k = F.normalize(self.k(self.norm_k(content)).reshape(head_shape), dim=-1)
        v = self.v(self.norm_v(content)).reshape(head_shape)

        # The same reference is Q in P and K in A. Keep its channel order between them:
        # no intermediate output projection, V reprojection or event residual.
        correspondence = torch.softmax(
            (reference @ k.transpose(-2, -1)) * self.temperature_reorg, dim=-1
        )
        reorganized = correspondence @ v
        read_weights = torch.softmax(
            (q @ reference.transpose(-2, -1)) * self.temperature_read, dim=-1
        )
        out = (read_weights @ reorganized).reshape(b, c, h, w)
        return self.proj(out)


class SoftRoutedDualCrossAttention2D(nn.Module):
    """Route E2/E3 channels to two parallel, same-source K/V experts."""

    def __init__(self, channels, window_size, num_heads, qk_norm=False):
        super().__init__()
        self.channels = channels
        self.proj_e2 = nn.Sequential(
            ChannelLayerNorm2D(channels),
            nn.Conv2d(channels, channels, 1, bias=False),
        )
        self.proj_e3 = nn.Sequential(
            ChannelLayerNorm2D(channels),
            nn.Conv2d(channels, channels, 1, bias=False),
        )
        router_hidden = max(channels // 4, 8)
        self.router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, router_hidden, 1),
            nn.GELU(),
            nn.Conv2d(router_hidden, channels * 4, 1),
        )
        nn.init.zeros_(self.router[-1].weight)
        nn.init.zeros_(self.router[-1].bias)

        self.proj_channel = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.GELU(),
        )
        self.proj_second = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.GELU(),
        )
        self.channel_ca = EventImageChannelCrossAttention2D(channels, num_heads)
        self.second_ca = WindowCrossAttention2D(
            channels, window_size, num_heads, qk_norm
        )

    def route(self, event2d, event3d):
        x2 = self.proj_e2(event2d)
        x3 = self.proj_e3(event3d)
        joined = torch.cat([x2, x3], dim=1)
        logits = self.router(joined).view(
            joined.shape[0], 2, joined.shape[1], 1, 1
        )
        gates = torch.softmax(logits, dim=1)
        return joined, gates

    def forward(self, rgb, event2d, event3d):
        if rgb.shape != event2d.shape or rgb.shape != event3d.shape:
            raise ValueError(
                "RGB, Event2D, and mean Event3D features must have the same shape"
            )
        joined, gates = self.route(event2d, event3d)
        channel_event = self.proj_channel(gates[:, 0] * joined)
        second_event = self.proj_second(gates[:, 1] * joined)
        channel_out = self.channel_ca(rgb, channel_event, channel_event)
        second_out = self.second_ca(rgb, second_event, second_event)
        return channel_out, second_out


class BidirectionalEventThenRGBChannelFusion2D(nn.Module):
    """Cross-enhance E2/E3 first, then let RGB read both identities."""

    def __init__(self, channels, num_heads, gamma_init):
        super().__init__()
        self.e2_from_e3 = EventImageChannelCrossAttention2D(channels, num_heads)
        self.e3_from_e2 = EventImageChannelCrossAttention2D(channels, num_heads)
        self.rgb_from_e2 = EventImageChannelCrossAttention2D(channels, num_heads)
        self.rgb_from_e3 = EventImageChannelCrossAttention2D(channels, num_heads)

        self.inject_e2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.inject_e3 = nn.Conv2d(channels, channels, 3, padding=1)
        self.inject_rgb_e2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.inject_rgb_e3 = nn.Conv2d(channels, channels, 3, padding=1)

        self.gamma_e2 = nn.Parameter(torch.ones(1) * gamma_init)
        self.gamma_e3 = nn.Parameter(torch.ones(1) * gamma_init)
        self.gamma_rgb_e2 = nn.Parameter(torch.ones(1) * gamma_init)
        self.gamma_rgb_e3 = nn.Parameter(torch.ones(1) * gamma_init)

    def forward(self, rgb, event2d, event3d):
        if rgb.shape != event2d.shape or rgb.shape != event3d.shape:
            raise ValueError(
                "RGB, Event2D, and mean Event3D features must have the same shape"
            )

        delta_e2 = self.e2_from_e3(event2d, event3d, event3d)
        delta_e3 = self.e3_from_e2(event3d, event2d, event2d)
        event2d_enhanced = event2d + self.gamma_e2 * self.inject_e2(delta_e2)
        event3d_enhanced = event3d + self.gamma_e3 * self.inject_e3(delta_e3)

        delta_rgb_e2 = self.rgb_from_e2(
            rgb, event2d_enhanced, event2d_enhanced
        )
        delta_rgb_e3 = self.rgb_from_e3(
            rgb, event3d_enhanced, event3d_enhanced
        )
        return (
            rgb
            + self.gamma_rgb_e2 * self.inject_rgb_e2(delta_rgb_e2)
            + self.gamma_rgb_e3 * self.inject_rgb_e3(delta_rgb_e3)
        )


class PlainChannelCrossAttention2D(nn.Module):
    def __init__(self, channels, num_heads):
        super().__init__()
        if channels % num_heads:
            raise ValueError("channels must be divisible by num_heads")
        self.num_heads = num_heads
        self.norm_q = ChannelLayerNorm2D(channels)
        self.norm_kv = ChannelLayerNorm2D(channels)
        self.q = nn.Conv2d(channels, channels, 1, bias=False)
        self.k = nn.Conv2d(channels, channels, 1, bias=False)
        self.v = nn.Conv2d(channels, channels, 1, bias=False)
        self.proj = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, image, event):
        if image.shape != event.shape:
            raise ValueError("image and fused event features must have the same shape")
        b, c, h, w = image.shape
        head_dim = c // self.num_heads
        spatial_dim = h * w

        q = self.q(self.norm_q(image)).view(b, self.num_heads, head_dim, spatial_dim)
        event = self.norm_kv(event)
        k = self.k(event).view(b, self.num_heads, head_dim, spatial_dim)
        v = self.v(event).view(b, self.num_heads, head_dim, spatial_dim)

        weights = torch.softmax(
            torch.matmul(q, k.transpose(-2, -1)) * (spatial_dim ** -0.5), dim=-1
        )
        out = torch.matmul(weights, v).view(b, c, h, w)
        return self.proj(out)


class CosineWindowCrossAttention2D(nn.Module):
    def __init__(self, channels, window_size, num_heads):
        super().__init__()
        if channels % num_heads:
            raise ValueError("channels must be divisible by num_heads")
        self.window_size = window_size
        self.num_heads = num_heads
        self.norm_q = ChannelLayerNorm2D(channels)
        self.norm_kv = ChannelLayerNorm2D(channels)
        self.q = nn.Conv2d(channels, channels, 1, bias=False)
        self.k = nn.Conv2d(channels, channels, 1, bias=False)
        self.v = nn.Conv2d(channels, channels, 1, bias=False)
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.proj = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, image, event):
        if image.shape != event.shape:
            raise ValueError("image and fused event features must have the same shape")
        q, shape = partition_2d(self.q(self.norm_q(image)), self.window_size)
        event = self.norm_kv(event)
        k, _ = partition_2d(self.k(event), self.window_size)
        v, _ = partition_2d(self.v(event), self.window_size)

        batch_windows, tokens, channels = q.shape
        head_dim = channels // self.num_heads
        q = q.view(batch_windows, tokens, self.num_heads, head_dim).transpose(1, 2)
        k = k.view(batch_windows, tokens, self.num_heads, head_dim).transpose(1, 2)
        v = v.view(batch_windows, tokens, self.num_heads, head_dim).transpose(1, 2)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        weights = torch.softmax(
            torch.matmul(q, k.transpose(-2, -1)) * self.temperature, dim=-1
        )
        out = torch.matmul(weights, v).transpose(1, 2).contiguous()
        out = out.view(batch_windows, tokens, channels)
        return self.proj(reverse_2d(out, shape, self.window_size))


class ConvFFN2D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 1),
            nn.GELU(),
            nn.Conv2d(channels * 2, channels * 2, 3, padding=1, groups=channels * 2),
            nn.GELU(),
            nn.Conv2d(channels * 2, channels, 1),
        )

    def forward(self, x):
        return self.net(x)


class ThreeBranchStageFusion(nn.Module):
    SINGLE_ORDERS = {"event2d_k_event3d_v", "event3d_k_event2d_v"}
    CASCADE_ORDERS = {"motion_then_struct", "struct_then_motion", "event_then_rgb"}
    SWAPPED_KV_ORDERS = {"event3d_first", "event2d_first"}
    KEY_BRIDGE_ORDERS = {"event3d_first", "event2d_first"}
    EVENT_REORG_MODES = {"event_reorg_b23_ca", "event_reorg_b32_ca"}
    SOFT_ROUTED_DUAL_MODES = {"soft_routed_dual_ca"}
    BIDIRECTIONAL_EVENT_RGB_MODES = {"bidirectional_event_then_rgb_ca"}
    TWO_DIM_ONLY_MODES = {
        "channel_ca",
        "event_conv",
        "event_window_ca",
        "plain_channel_ca",
        "event_add_channel_ca",
    }

    def __init__(
        self,
        channels,
        fusion_mode,
        fusion_dim,
        cross_attn_type,
        single_ca_order,
        cascaded_ca_order,
        swapped_kv_order,
        key_bridge_order,
        cross_window_size,
        temporal_window_size,
        num_heads,
        qk_norm,
        gamma_init,
    ):
        super().__init__()
        self.fusion_mode = fusion_mode.lower()
        self.fusion_dim = fusion_dim.lower()
        self.cross_attn_type = cross_attn_type.lower()
        self.single_ca_order = single_ca_order.lower()
        self.cascaded_ca_order = cascaded_ca_order.lower()
        self.swapped_kv_order = swapped_kv_order.lower()
        self.key_bridge_order = key_bridge_order.lower()
        valid_modes = {
            "cat",
            "single_ca",
            "cascaded_ca",
            "swapped_kv_cascaded_ca",
            "parallel_swapped_kv_cat_ca",
            "mmca_key_bridge_ca",
        } | (
            self.TWO_DIM_ONLY_MODES
            | self.EVENT_REORG_MODES
            | self.SOFT_ROUTED_DUAL_MODES
            | self.BIDIRECTIONAL_EVENT_RGB_MODES
        )
        if self.fusion_mode not in valid_modes:
            raise ValueError(f"Unknown fusion_mode: {fusion_mode}")
        if self.fusion_dim not in {"2d", "3d"}:
            raise ValueError(f"Unknown fusion_dim: {fusion_dim}")
        if self.cross_attn_type not in {"window", "channel"}:
            raise ValueError(f"Unknown cross_attn_type: {cross_attn_type}")
        if (
            self.fusion_mode in {"single_ca", "cascaded_ca"}
            and self.cross_attn_type == "channel"
            and self.fusion_dim != "2d"
        ):
            raise ValueError("cross_attn_type: channel requires fusion_dim: 2d")
        if self.fusion_mode in self.TWO_DIM_ONLY_MODES and self.fusion_dim != "2d":
            raise ValueError(f"fusion_mode: {self.fusion_mode} requires fusion_dim: 2d")
        if self.single_ca_order not in self.SINGLE_ORDERS:
            raise ValueError(f"Unknown single_ca_order: {single_ca_order}")
        if self.cascaded_ca_order not in self.CASCADE_ORDERS:
            raise ValueError(f"Unknown cascaded_ca_order: {cascaded_ca_order}")
        if self.swapped_kv_order not in self.SWAPPED_KV_ORDERS:
            raise ValueError(f"Unknown swapped_kv_order: {swapped_kv_order}")
        if self.key_bridge_order not in self.KEY_BRIDGE_ORDERS:
            raise ValueError(f"Unknown key_bridge_order: {key_bridge_order}")
        channel_2d_modes = (
            {
                "swapped_kv_cascaded_ca",
                "parallel_swapped_kv_cat_ca",
                "mmca_key_bridge_ca",
            }
            | self.EVENT_REORG_MODES
            | self.SOFT_ROUTED_DUAL_MODES
            | self.BIDIRECTIONAL_EVENT_RGB_MODES
        )
        if self.fusion_mode in channel_2d_modes and (
            self.fusion_dim != "2d" or self.cross_attn_type != "channel"
        ):
            raise ValueError(
                f"{self.fusion_mode} requires fusion_dim: 2d and cross_attn_type: channel"
            )

        if self.fusion_mode == "cat":
            if self.fusion_dim == "2d":
                self.cat = nn.Sequential(ConvBlock2D(channels * 3, channels), nn.Conv2d(channels, channels, 3, padding=1))
            else:
                self.cat = nn.Sequential(
                    nn.Conv3d(channels * 3, channels, 1),
                    nn.GELU(),
                    nn.Conv3d(channels, channels, 3, padding=1, groups=channels),
                    nn.GELU(),
                    nn.Conv3d(channels, channels, 1),
                )
        elif self.fusion_mode in {"channel_ca", "event_conv", "event_window_ca", "plain_channel_ca"}:
            self.event_aggregate = ConvBlock2D(channels * 2, channels)
            if self.fusion_mode == "channel_ca":
                self.channel_ca = EventImageChannelCrossAttention2D(channels, num_heads)
            elif self.fusion_mode == "event_window_ca":
                self.event_window_ca = CosineWindowCrossAttention2D(
                    channels, cross_window_size, num_heads
                )
            elif self.fusion_mode == "plain_channel_ca":
                self.plain_channel_ca = PlainChannelCrossAttention2D(channels, num_heads)
        elif self.fusion_mode == "event_add_channel_ca":
            self.channel_ca = EventImageChannelCrossAttention2D(channels, num_heads)
        elif self.fusion_mode in self.EVENT_REORG_MODES:
            self.event_reorg = EventReorganizedChannelCrossAttention2D(channels, num_heads)
            self.inject1 = nn.Conv2d(channels, channels, 3, padding=1)
        elif self.fusion_mode in self.SOFT_ROUTED_DUAL_MODES:
            self.soft_routed_dual = SoftRoutedDualCrossAttention2D(
                channels,
                cross_window_size,
                num_heads,
                qk_norm=qk_norm,
            )
            self.inject1 = nn.Conv2d(channels, channels, 3, padding=1)
            self.inject2 = nn.Conv2d(channels, channels, 3, padding=1)
        elif self.fusion_mode in self.BIDIRECTIONAL_EVENT_RGB_MODES:
            self.bidirectional_event_rgb = BidirectionalEventThenRGBChannelFusion2D(
                channels, num_heads, gamma_init
            )
        elif self.fusion_dim == "2d":
            self.inject1 = nn.Conv2d(channels, channels, 3, padding=1)
            if self.cross_attn_type == "channel":
                make_ca = lambda: EventImageChannelCrossAttention2D(channels, num_heads)
            else:
                make_ca = lambda: WindowCrossAttention2D(
                    channels, cross_window_size, num_heads, qk_norm
                )
            if self.fusion_mode in {
                "cascaded_ca",
                "swapped_kv_cascaded_ca",
                "parallel_swapped_kv_cat_ca",
                "mmca_key_bridge_ca",
            }:
                self.ca1 = make_ca()
                self.ca2 = make_ca()
                self.inject2 = nn.Conv2d(channels, channels, 3, padding=1)
                if self.fusion_mode == "parallel_swapped_kv_cat_ca":
                    self.parallel_cat = nn.Conv2d(channels * 2, channels, 1)
            else:
                self.ca = make_ca()
        else:
            self.inject1 = nn.Conv3d(channels, channels, 3, padding=1)
            if self.fusion_mode == "cascaded_ca":
                self.ca1 = WindowCrossAttention3D(
                    channels, cross_window_size, temporal_window_size, num_heads, qk_norm
                )
                self.ca2 = WindowCrossAttention3D(
                    channels, cross_window_size, temporal_window_size, num_heads, qk_norm
                )
                self.inject2 = nn.Conv3d(channels, channels, 3, padding=1)
            else:
                self.ca = WindowCrossAttention3D(
                    channels, cross_window_size, temporal_window_size, num_heads, qk_norm
                )

        if self.fusion_mode not in self.BIDIRECTIONAL_EVENT_RGB_MODES:
            self.gamma1 = nn.Parameter(torch.ones(1) * gamma_init)
            if self.fusion_mode in {
                "cascaded_ca",
                "swapped_kv_cascaded_ca",
                "parallel_swapped_kv_cat_ca",
                "mmca_key_bridge_ca",
            } | self.SOFT_ROUTED_DUAL_MODES:
                self.gamma2 = nn.Parameter(torch.ones(1) * gamma_init)
        self.norm_ffn = ChannelLayerNorm2D(channels)
        self.ffn = ConvFFN2D(channels)
        self.gamma_ffn = nn.Parameter(torch.ones(1) * gamma_init)

    @staticmethod
    def expand_time(x, time_steps):
        return x.unsqueeze(2).expand(-1, -1, time_steps, -1, -1)

    def single_2d(self, rgb, event2d, event3d):
        event3d = event3d.mean(dim=2)
        if self.single_ca_order == "event2d_k_event3d_v":
            delta = self.ca(rgb, event2d, event3d)
        else:
            delta = self.ca(rgb, event3d, event2d)
        return rgb + self.gamma1 * self.inject1(delta)

    def cascade_2d(self, rgb, event2d, event3d):
        event3d = event3d.mean(dim=2)
        if self.cascaded_ca_order == "motion_then_struct":
            x = rgb + self.gamma1 * self.inject1(self.ca1(rgb, event3d, event3d))
            return x + self.gamma2 * self.inject2(self.ca2(x, event2d, event2d))
        if self.cascaded_ca_order == "struct_then_motion":
            x = rgb + self.gamma1 * self.inject1(self.ca1(rgb, event2d, event2d))
            return x + self.gamma2 * self.inject2(self.ca2(x, event3d, event3d))
        event = event3d + self.gamma1 * self.inject1(self.ca1(event3d, event2d, event2d))
        return rgb + self.gamma2 * self.inject2(self.ca2(rgb, event, event))

    def swapped_kv_cascade_2d(self, rgb, event2d, event3d):
        event3d = event3d.mean(dim=2)
        if self.swapped_kv_order == "event3d_first":
            x = rgb + self.gamma1 * self.inject1(self.ca1(rgb, event3d, event2d))
            return x + self.gamma2 * self.inject2(self.ca2(x, event2d, event3d))
        x = rgb + self.gamma1 * self.inject1(self.ca1(rgb, event2d, event3d))
        return x + self.gamma2 * self.inject2(self.ca2(x, event3d, event2d))

    def parallel_swapped_kv_cat_2d(self, rgb, event2d, event3d):
        event3d = event3d.mean(dim=2)
        delta_e3k_e2v = self.gamma1 * self.inject1(self.ca1(rgb, event3d, event2d))
        delta_e2k_e3v = self.gamma2 * self.inject2(self.ca2(rgb, event2d, event3d))
        delta = self.parallel_cat(torch.cat([delta_e3k_e2v, delta_e2k_e3v], dim=1))
        return rgb + delta

    def mmca_key_bridge_2d(self, rgb, event2d, event3d):
        event3d = event3d.mean(dim=2)
        if self.key_bridge_order == "event3d_first":
            bridge = self.ca1(rgb, event3d, event3d)
            x = rgb + self.gamma1 * self.inject1(bridge)
            return x + self.gamma2 * self.inject2(self.ca2(rgb, bridge, event2d))
        bridge = self.ca1(rgb, event2d, event2d)
        x = rgb + self.gamma1 * self.inject1(bridge)
        return x + self.gamma2 * self.inject2(self.ca2(rgb, bridge, event3d))

    def event_reorg_2d(self, rgb, event2d, event3d):
        event3d = event3d.mean(dim=2)
        if self.fusion_mode == "event_reorg_b23_ca":
            delta = self.event_reorg(rgb, reference=event2d, content=event3d)
        else:
            delta = self.event_reorg(rgb, reference=event3d, content=event2d)
        return rgb + self.gamma1 * self.inject1(delta)

    def soft_routed_dual_2d(self, rgb, event2d, event3d):
        event3d = event3d.mean(dim=2)
        channel_delta, second_delta = self.soft_routed_dual(rgb, event2d, event3d)
        return (
            rgb
            + self.gamma1 * self.inject1(channel_delta)
            + self.gamma2 * self.inject2(second_delta)
        )

    def bidirectional_event_then_rgb_2d(self, rgb, event2d, event3d):
        return self.bidirectional_event_rgb(rgb, event2d, event3d.mean(dim=2))

    def single_3d(self, rgb, event2d, event3d):
        time_steps = event3d.shape[2]
        rgb = self.expand_time(rgb, time_steps)
        event2d = self.expand_time(event2d, time_steps)
        if self.single_ca_order == "event2d_k_event3d_v":
            delta = self.ca(rgb, event2d, event3d)
        else:
            delta = self.ca(rgb, event3d, event2d)
        return (rgb + self.gamma1 * self.inject1(delta)).mean(dim=2)

    def cascade_3d(self, rgb, event2d, event3d):
        time_steps = event3d.shape[2]
        rgb = self.expand_time(rgb, time_steps)
        event2d = self.expand_time(event2d, time_steps)
        if self.cascaded_ca_order == "motion_then_struct":
            x = rgb + self.gamma1 * self.inject1(self.ca1(rgb, event3d, event3d))
            x = x + self.gamma2 * self.inject2(self.ca2(x, event2d, event2d))
        elif self.cascaded_ca_order == "struct_then_motion":
            x = rgb + self.gamma1 * self.inject1(self.ca1(rgb, event2d, event2d))
            x = x + self.gamma2 * self.inject2(self.ca2(x, event3d, event3d))
        else:
            event = event3d + self.gamma1 * self.inject1(self.ca1(event3d, event2d, event2d))
            x = rgb + self.gamma2 * self.inject2(self.ca2(rgb, event, event))
        return x.mean(dim=2)

    def forward(self, rgb, event2d, event3d):
        if self.fusion_mode == "cat":
            if self.fusion_dim == "2d":
                delta = self.cat(torch.cat([rgb, event2d, event3d.mean(dim=2)], dim=1))
            else:
                time_steps = event3d.shape[2]
                rgb3d = self.expand_time(rgb, time_steps)
                event2d3d = self.expand_time(event2d, time_steps)
                delta = self.cat(torch.cat([rgb3d, event2d3d, event3d], dim=1)).mean(dim=2)
            fused = rgb + self.gamma1 * delta
        elif self.fusion_mode == "channel_ca":
            event = self.event_aggregate(torch.cat([event2d, event3d.mean(dim=2)], dim=1))
            fused = rgb + self.gamma1 * self.channel_ca(rgb, event)
        elif self.fusion_mode == "event_conv":
            event = self.event_aggregate(torch.cat([event2d, event3d.mean(dim=2)], dim=1))
            fused = rgb + self.gamma1 * event
        elif self.fusion_mode == "event_window_ca":
            event = self.event_aggregate(torch.cat([event2d, event3d.mean(dim=2)], dim=1))
            fused = rgb + self.gamma1 * self.event_window_ca(rgb, event)
        elif self.fusion_mode == "plain_channel_ca":
            event = self.event_aggregate(torch.cat([event2d, event3d.mean(dim=2)], dim=1))
            fused = rgb + self.gamma1 * self.plain_channel_ca(rgb, event)
        elif self.fusion_mode == "event_add_channel_ca":
            event = event2d + event3d.mean(dim=2)
            fused = rgb + self.gamma1 * self.channel_ca(rgb, event)
        elif self.fusion_dim == "2d":
            if self.fusion_mode == "single_ca":
                fused = self.single_2d(rgb, event2d, event3d)
            elif self.fusion_mode == "swapped_kv_cascaded_ca":
                fused = self.swapped_kv_cascade_2d(rgb, event2d, event3d)
            elif self.fusion_mode == "parallel_swapped_kv_cat_ca":
                fused = self.parallel_swapped_kv_cat_2d(rgb, event2d, event3d)
            elif self.fusion_mode == "mmca_key_bridge_ca":
                fused = self.mmca_key_bridge_2d(rgb, event2d, event3d)
            elif self.fusion_mode in self.EVENT_REORG_MODES:
                fused = self.event_reorg_2d(rgb, event2d, event3d)
            elif self.fusion_mode in self.SOFT_ROUTED_DUAL_MODES:
                fused = self.soft_routed_dual_2d(rgb, event2d, event3d)
            elif self.fusion_mode in self.BIDIRECTIONAL_EVENT_RGB_MODES:
                fused = self.bidirectional_event_then_rgb_2d(rgb, event2d, event3d)
            else:
                fused = self.cascade_2d(rgb, event2d, event3d)
        else:
            fused = self.single_3d(rgb, event2d, event3d) if self.fusion_mode == "single_ca" else self.cascade_3d(rgb, event2d, event3d)
        return fused + self.gamma_ffn * self.ffn(self.norm_ffn(fused))


class ThreeBranchProgressiveDeblurNet(nn.Module):
    def __init__(
        self,
        rgb_in=3,
        event_in=6,
        base_dim=32,
        fusion_mode="single_ca",
        fusion_dim="2d",
        cross_attn_type="window",
        single_ca_order="event2d_k_event3d_v",
        cascaded_ca_order="motion_then_struct",
        swapped_kv_order="event3d_first",
        key_bridge_order="event3d_first",
        encoder_self_attn="window",
        self_attn_window_size=8,
        cross_attn_window_size=8,
        temporal_window_size=2,
        num_heads=4,
        qk_norm=False,
        gamma_init=0.1,
        deform_alignment="none",
        decoder_block="plain",
        decoder_attention="none",
        two_stage=False,
        sam_mode="none",
        refine_type="shallow",
    ):
        super().__init__()
        encoder_self_attn = encoder_self_attn.lower()
        deform_alignment = deform_alignment.lower()
        decoder_block = decoder_block.lower()
        decoder_attention = decoder_attention.lower()
        sam_mode = sam_mode.lower()
        refine_type = refine_type.lower()
        if encoder_self_attn not in {
            "window",
            "plain_channel",
            "restormer_channel",
            "restormer_window",
            "restormer_shifted_window",
        }:
            raise ValueError(f"Unknown encoder_self_attn: {encoder_self_attn}")
        if deform_alignment not in {"none", "image_guided"}:
            raise ValueError(f"Unknown deform_alignment: {deform_alignment}")
        if decoder_block not in {"plain", "residual"}:
            raise ValueError(f"Unknown decoder_block: {decoder_block}")
        if decoder_attention not in {"none", "restormer_channel"}:
            raise ValueError(f"Unknown decoder_attention: {decoder_attention}")
        if sam_mode not in {"none", "standard", "event_guided"}:
            raise ValueError(f"Unknown sam_mode: {sam_mode}")
        if refine_type not in {"shallow", "light_unet"}:
            raise ValueError(f"Unknown refine_type: {refine_type}")
        if not two_stage and sam_mode != "none":
            raise ValueError("sam_mode requires two_stage: true")

        dims = [base_dim, base_dim * 2, base_dim * 4]
        self.two_stage = two_stage
        self.sam_mode = sam_mode
        self.rgb_stem = ConvBlock2D(rgb_in, dims[0])
        self.event2d_stem = ConvBlock2D(event_in, dims[0])
        self.event3d_stem = ShortTermTDCBlock3D(1, dims[0])
        self.rgb_down = nn.ModuleList([ConvBlock2D(dims[i], dims[i + 1], stride=2) for i in range(2)])
        self.event2d_down = nn.ModuleList([ConvBlock2D(dims[i], dims[i + 1], stride=2) for i in range(2)])
        self.event3d_down = nn.ModuleList([
            ShortTermTDCBlock3D(dims[i], dims[i + 1], stride=(1, 2, 2)) for i in range(2)
        ])
        self.rgb_alignment = nn.ModuleList([
            ImageGuidedDeformAlignment2D(dim) if deform_alignment == "image_guided" else nn.Identity()
            for dim in dims
        ])
        if encoder_self_attn == "window":
            self.rgb_self_attn = nn.ModuleList([
                WindowSelfAttention2D(dim, self_attn_window_size, num_heads, qk_norm) for dim in dims
            ])
            self.event2d_self_attn = nn.ModuleList([
                WindowSelfAttention2D(dim, self_attn_window_size, num_heads, qk_norm) for dim in dims
            ])
            self.event3d_self_attn = nn.ModuleList([
                WindowSelfAttention3D(
                    dim,
                    self_attn_window_size,
                    temporal_window_size,
                    num_heads,
                    qk_norm,
                )
                for dim in dims
            ])
        elif encoder_self_attn == "plain_channel":
            self.rgb_self_attn = nn.ModuleList([
                PlainChannelSelfAttention2D(dim, num_heads, qk_norm) for dim in dims
            ])
            self.event2d_self_attn = nn.ModuleList([
                PlainChannelSelfAttention2D(dim, num_heads, qk_norm) for dim in dims
            ])
            self.event3d_self_attn = nn.ModuleList([
                PlainChannelSelfAttention3D(dim, num_heads, qk_norm) for dim in dims
            ])
        elif encoder_self_attn == "restormer_channel":
            self.rgb_self_attn = nn.ModuleList([
                RestormerChannelSelfAttention2D(dim, num_heads) for dim in dims
            ])
            self.event2d_self_attn = nn.ModuleList([
                RestormerChannelSelfAttention2D(dim, num_heads) for dim in dims
            ])
            self.event3d_self_attn = nn.ModuleList([
                RestormerChannelSelfAttention3D(dim, num_heads) for dim in dims
            ])
        elif encoder_self_attn == "restormer_window":
            self.rgb_self_attn = nn.ModuleList([
                RestormerWindowSelfAttention2D(
                    dim, self_attn_window_size, num_heads
                )
                for dim in dims
            ])
            self.event2d_self_attn = nn.ModuleList([
                RestormerWindowSelfAttention2D(
                    dim, self_attn_window_size, num_heads
                )
                for dim in dims
            ])
            self.event3d_self_attn = nn.ModuleList([
                RestormerWindowSelfAttention3D(
                    dim,
                    self_attn_window_size,
                    temporal_window_size,
                    num_heads,
                )
                for dim in dims
            ])
        else:
            self.rgb_self_attn = nn.ModuleList([
                RestormerShiftedWindowPair2D(
                    dim, self_attn_window_size, num_heads
                )
                for dim in dims
            ])
            self.event2d_self_attn = nn.ModuleList([
                RestormerShiftedWindowPair2D(
                    dim, self_attn_window_size, num_heads
                )
                for dim in dims
            ])
            self.event3d_self_attn = nn.ModuleList([
                RestormerShiftedWindowPair3D(
                    dim,
                    self_attn_window_size,
                    temporal_window_size,
                    num_heads,
                )
                for dim in dims
            ])
        self.fusions = nn.ModuleList([
            ThreeBranchStageFusion(
                dim,
                fusion_mode,
                fusion_dim,
                cross_attn_type,
                single_ca_order,
                cascaded_ca_order,
                swapped_kv_order,
                key_bridge_order,
                cross_attn_window_size,
                temporal_window_size,
                num_heads,
                qk_norm,
                gamma_init,
            )
            for dim in dims
        ])
        up_block = UpBlock if decoder_block == "plain" else ResidualUpBlock
        self.up1 = up_block(dims[2], dims[1], dims[1])
        self.up0 = up_block(dims[1], dims[0], dims[0])
        if decoder_attention == "restormer_channel":
            self.up1_attention = RestormerChannelSelfAttention2D(dims[1], num_heads)
            self.up0_attention = RestormerChannelSelfAttention2D(dims[0], num_heads)
        else:
            self.up1_attention = nn.Identity()
            self.up0_attention = nn.Identity()
        self.reconstruct = nn.Sequential(ConvBlock2D(dims[0], dims[0]), nn.Conv2d(dims[0], rgb_in, 3, padding=1))
        if self.two_stage:
            if self.sam_mode == "standard":
                self.sam = SupervisedAttentionModule(dims[0], rgb_in)
            elif self.sam_mode == "event_guided":
                self.event_guide = ConvBlock2D(dims[0] * 2, dims[0])
                self.sam = SupervisedAttentionModule(dims[0], rgb_in, dims[0])
            refine_net = ShallowRefineNet if refine_type == "shallow" else LightUNetRefineNet
            self.refine = refine_net(dims[0], rgb_in)

    def attend(self, scale, rgb, event2d, event3d):
        rgb = self.rgb_alignment[scale](rgb)
        return (
            self.rgb_self_attn[scale](rgb),
            self.event2d_self_attn[scale](event2d),
            self.event3d_self_attn[scale](event3d),
        )

    def forward(self, blur, event, return_intermediate=False):
        rgb = self.rgb_stem(blur)
        event2d = self.event2d_stem(event)
        event3d = self.event3d_stem(event.unsqueeze(1))
        rgb, event2d, event3d = self.attend(0, rgb, event2d, event3d)
        event2d_h, event3d_h = event2d, event3d
        fused0 = self.fusions[0](rgb, event2d, event3d)

        rgb = self.rgb_down[0](fused0)
        event2d = self.event2d_down[0](event2d)
        event3d = self.event3d_down[0](event3d)
        rgb, event2d, event3d = self.attend(1, rgb, event2d, event3d)
        fused1 = self.fusions[1](rgb, event2d, event3d)

        rgb = self.rgb_down[1](fused1)
        event2d = self.event2d_down[1](event2d)
        event3d = self.event3d_down[1](event3d)
        rgb, event2d, event3d = self.attend(2, rgb, event2d, event3d)
        fused2 = self.fusions[2](rgb, event2d, event3d)

        x = self.up1_attention(self.up1(fused2, fused1))
        x = self.up0_attention(self.up0(x, fused0))
        stage1_image = blur + self.reconstruct(x)
        if not self.two_stage:
            return stage1_image

        refine_feature = x
        if self.sam_mode == "standard":
            refine_feature = self.sam(x, stage1_image)
        elif self.sam_mode == "event_guided":
            event_guide = self.event_guide(torch.cat([event2d_h, event3d_h.mean(dim=2)], dim=1))
            refine_feature = self.sam(x, stage1_image, event_guide)
        final_image = self.refine(refine_feature, stage1_image, blur)
        if return_intermediate:
            return final_image, stage1_image
        return final_image


def build_deblur_model(**model_cfg):
    model_type = model_cfg.pop("model_type", model_cfg.pop("type", "progressive_fusion"))
    if model_type != "progressive_fusion":
        raise ValueError("Only model.type: progressive_fusion is supported by the three-branch network.")
    return ThreeBranchProgressiveDeblurNet(**model_cfg)


if __name__ == "__main__":
    model = build_deblur_model(base_dim=16)
    blur = torch.randn(1, 3, 64, 64)
    event = torch.randn(1, 6, 64, 64)
    print(model(blur, event).shape)
