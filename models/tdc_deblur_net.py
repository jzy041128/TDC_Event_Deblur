import math

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.norm_kv = ChannelLayerNorm2D(channels)
        self.q = nn.Conv2d(channels, channels, 1, bias=False)
        self.k = nn.Conv2d(channels, channels, 1, bias=False)
        self.v = nn.Conv2d(channels, channels, 1, bias=False)
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.proj = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, image, event):
        if image.shape != event.shape:
            raise ValueError("image and fused event features must have the same shape")
        b, c, h, w = image.shape
        head_dim = c // self.num_heads

        q = self.q(self.norm_q(image)).view(b, self.num_heads, head_dim, h * w)
        event = self.norm_kv(event)
        k = self.k(event).view(b, self.num_heads, head_dim, h * w)
        v = self.v(event).view(b, self.num_heads, head_dim, h * w)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        weights = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * self.temperature, dim=-1)
        out = torch.matmul(weights, v).view(b, c, h, w)
        return self.proj(out)


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

    def __init__(
        self,
        channels,
        fusion_mode,
        fusion_dim,
        single_ca_order,
        cascaded_ca_order,
        cross_window_size,
        temporal_window_size,
        num_heads,
        qk_norm,
        gamma_init,
    ):
        super().__init__()
        self.fusion_mode = fusion_mode.lower()
        self.fusion_dim = fusion_dim.lower()
        self.single_ca_order = single_ca_order.lower()
        self.cascaded_ca_order = cascaded_ca_order.lower()
        if self.fusion_mode not in {"cat", "single_ca", "cascaded_ca", "channel_ca"}:
            raise ValueError(f"Unknown fusion_mode: {fusion_mode}")
        if self.fusion_dim not in {"2d", "3d"}:
            raise ValueError(f"Unknown fusion_dim: {fusion_dim}")
        if self.fusion_mode == "channel_ca" and self.fusion_dim != "2d":
            raise ValueError("fusion_mode: channel_ca requires fusion_dim: 2d")
        if self.single_ca_order not in self.SINGLE_ORDERS:
            raise ValueError(f"Unknown single_ca_order: {single_ca_order}")
        if self.cascaded_ca_order not in self.CASCADE_ORDERS:
            raise ValueError(f"Unknown cascaded_ca_order: {cascaded_ca_order}")

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
        elif self.fusion_mode == "channel_ca":
            self.event_aggregate = ConvBlock2D(channels * 2, channels)
            self.channel_ca = EventImageChannelCrossAttention2D(channels, num_heads)
        elif self.fusion_dim == "2d":
            self.inject1 = nn.Conv2d(channels, channels, 3, padding=1)
            if self.fusion_mode == "cascaded_ca":
                self.ca1 = WindowCrossAttention2D(channels, cross_window_size, num_heads, qk_norm)
                self.ca2 = WindowCrossAttention2D(channels, cross_window_size, num_heads, qk_norm)
                self.inject2 = nn.Conv2d(channels, channels, 3, padding=1)
            else:
                self.ca = WindowCrossAttention2D(channels, cross_window_size, num_heads, qk_norm)
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

        self.gamma1 = nn.Parameter(torch.ones(1) * gamma_init)
        if self.fusion_mode == "cascaded_ca":
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
        elif self.fusion_dim == "2d":
            fused = self.single_2d(rgb, event2d, event3d) if self.fusion_mode == "single_ca" else self.cascade_2d(rgb, event2d, event3d)
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
        single_ca_order="event2d_k_event3d_v",
        cascaded_ca_order="motion_then_struct",
        self_attn_window_size=8,
        cross_attn_window_size=8,
        temporal_window_size=2,
        num_heads=4,
        qk_norm=False,
        gamma_init=0.1,
    ):
        super().__init__()
        dims = [base_dim, base_dim * 2, base_dim * 4]
        self.rgb_stem = ConvBlock2D(rgb_in, dims[0])
        self.event2d_stem = ConvBlock2D(event_in, dims[0])
        self.event3d_stem = ShortTermTDCBlock3D(1, dims[0])
        self.rgb_down = nn.ModuleList([ConvBlock2D(dims[i], dims[i + 1], stride=2) for i in range(2)])
        self.event2d_down = nn.ModuleList([ConvBlock2D(dims[i], dims[i + 1], stride=2) for i in range(2)])
        self.event3d_down = nn.ModuleList([
            ShortTermTDCBlock3D(dims[i], dims[i + 1], stride=(1, 2, 2)) for i in range(2)
        ])
        self.rgb_self_attn = nn.ModuleList([
            WindowSelfAttention2D(dim, self_attn_window_size, num_heads, qk_norm) for dim in dims
        ])
        self.event2d_self_attn = nn.ModuleList([
            WindowSelfAttention2D(dim, self_attn_window_size, num_heads, qk_norm) for dim in dims
        ])
        self.event3d_self_attn = nn.ModuleList([
            WindowSelfAttention3D(dim, self_attn_window_size, temporal_window_size, num_heads, qk_norm)
            for dim in dims
        ])
        self.fusions = nn.ModuleList([
            ThreeBranchStageFusion(
                dim,
                fusion_mode,
                fusion_dim,
                single_ca_order,
                cascaded_ca_order,
                cross_attn_window_size,
                temporal_window_size,
                num_heads,
                qk_norm,
                gamma_init,
            )
            for dim in dims
        ])
        self.up1 = UpBlock(dims[2], dims[1], dims[1])
        self.up0 = UpBlock(dims[1], dims[0], dims[0])
        self.reconstruct = nn.Sequential(ConvBlock2D(dims[0], dims[0]), nn.Conv2d(dims[0], rgb_in, 3, padding=1))

    def attend(self, scale, rgb, event2d, event3d):
        return (
            self.rgb_self_attn[scale](rgb),
            self.event2d_self_attn[scale](event2d),
            self.event3d_self_attn[scale](event3d),
        )

    def forward(self, blur, event):
        rgb = self.rgb_stem(blur)
        event2d = self.event2d_stem(event)
        event3d = self.event3d_stem(event.unsqueeze(1))
        rgb, event2d, event3d = self.attend(0, rgb, event2d, event3d)
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

        x = self.up1(fused2, fused1)
        x = self.up0(x, fused0)
        return blur + self.reconstruct(x)


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
