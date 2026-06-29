import torch
import torch.nn as nn
import torch.nn.functional as F

from models.sta_module import CrossModalSTA
from models.tdc_module import TDCBlock3D


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)


class ConvBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            conv3x3(in_channels, out_channels, stride=stride),
            nn.LeakyReLU(0.2, True),
            conv3x3(out_channels, out_channels),
            nn.LeakyReLU(0.2, True),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.refine = ConvBlock2D(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        return self.refine(torch.cat([x, skip], dim=1))


class ThreeScaleImageEncoder(nn.Module):
    def __init__(self, rgb_in=3, base_dim=32):
        super().__init__()
        self.enc0 = ConvBlock2D(rgb_in, base_dim)
        self.enc1 = ConvBlock2D(base_dim, base_dim * 2, stride=2)
        self.enc2 = ConvBlock2D(base_dim * 2, base_dim * 4, stride=2)

    def forward(self, x):
        feat0 = self.enc0(x)
        feat1 = self.enc1(feat0)
        feat2 = self.enc2(feat1)
        return [feat0, feat1, feat2]


class EventVoxel2DEncoder(nn.Module):
    def __init__(self, event_in=6, base_dim=32):
        super().__init__()
        self.enc0 = ConvBlock2D(event_in, base_dim)
        self.enc1 = ConvBlock2D(base_dim, base_dim * 2, stride=2)
        self.enc2 = ConvBlock2D(base_dim * 2, base_dim * 4, stride=2)

    def forward(self, event):
        feat0 = self.enc0(event)
        feat1 = self.enc1(feat0)
        feat2 = self.enc2(feat1)
        return [feat0, feat1, feat2]


class EventVoxel3DEncoder(nn.Module):
    def __init__(self, base_dim=32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(1, base_dim, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(base_dim),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(base_dim, base_dim, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(base_dim),
            nn.LeakyReLU(0.2, True),
        )
        self.down1 = nn.Sequential(
            nn.Conv3d(base_dim, base_dim * 2, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(base_dim * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(base_dim * 2, base_dim * 2, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.BatchNorm3d(base_dim * 2),
            nn.LeakyReLU(0.2, True),
        )
        self.down2 = nn.Sequential(
            nn.Conv3d(base_dim * 2, base_dim * 4, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(base_dim * 4),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(base_dim * 4, base_dim * 4, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.BatchNorm3d(base_dim * 4),
            nn.LeakyReLU(0.2, True),
        )

    @staticmethod
    def pool_time(x):
        return x.mean(dim=2)

    def forward(self, event, return_3d=False):
        x = event.unsqueeze(1)
        feat0_3d = self.stem(x)
        feat1_3d = self.down1(feat0_3d)
        feat2_3d = self.down2(feat1_3d)
        feats_3d = [feat0_3d, feat1_3d, feat2_3d]
        feats_2d = [self.pool_time(feat) for feat in feats_3d]
        if return_3d:
            return feats_2d, feats_3d
        return feats_2d, None


class EventTDC3DEncoder(nn.Module):
    def __init__(self, base_dim=32):
        super().__init__()
        self.stem = nn.Sequential(
            TDCBlock3D(1, base_dim),
            nn.Conv3d(base_dim, base_dim, kernel_size=1),
            nn.LeakyReLU(0.2, True),
        )
        self.down1 = nn.Sequential(
            TDCBlock3D(base_dim, base_dim * 2),
            nn.Conv3d(base_dim * 2, base_dim * 2, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.LeakyReLU(0.2, True),
        )
        self.down2 = nn.Sequential(
            TDCBlock3D(base_dim * 2, base_dim * 4),
            nn.Conv3d(base_dim * 4, base_dim * 4, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.LeakyReLU(0.2, True),
        )

    @staticmethod
    def pool_time(x):
        return x.mean(dim=2)

    def forward(self, event, return_3d=False):
        x = event.unsqueeze(1)
        feat0_3d = self.stem(x)
        feat1_3d = self.down1(feat0_3d)
        feat2_3d = self.down2(feat1_3d)
        feats_3d = [feat0_3d, feat1_3d, feat2_3d]
        feats_2d = [self.pool_time(feat) for feat in feats_3d]
        if return_3d:
            return feats_2d, feats_3d
        return feats_2d, None


class EventEncoderWith3DOption(nn.Module):
    def __init__(self, event_in=6, base_dim=32, encoder_type='2d'):
        super().__init__()
        self.encoder_type = encoder_type.lower()
        if self.encoder_type == '2d':
            self.encoder = EventVoxel2DEncoder(event_in=event_in, base_dim=base_dim)
        elif self.encoder_type == '3d':
            self.encoder = EventVoxel3DEncoder(base_dim=base_dim)
        else:
            raise ValueError(f'Unknown event_encoder_type: {encoder_type}')

    def forward(self, event, return_3d=False):
        if self.encoder_type == '3d':
            return self.encoder(event, return_3d=return_3d)
        feats_2d = self.encoder(event)
        if return_3d:
            return feats_2d, None
        return feats_2d, None


class DualEventEncoder(nn.Module):
    def __init__(self, event_in=6, base_dim=32):
        super().__init__()
        dims = [base_dim, base_dim * 2, base_dim * 4]
        self.voxel_2d = EventVoxel2DEncoder(event_in=event_in, base_dim=base_dim)
        self.tdc_3d = EventTDC3DEncoder(base_dim=base_dim)
        self.merge = nn.ModuleList([
            ConvBlock2D(dim * 2, dim) for dim in dims
        ])

    def forward(self, event):
        voxel_feats = self.voxel_2d(event)
        temporal_feats, _ = self.tdc_3d(event)
        return [
            merge(torch.cat([voxel, temporal], dim=1))
            for merge, voxel, temporal in zip(self.merge, voxel_feats, temporal_feats)
        ]


class EventGate(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, event_feat):
        return self.gate(event_feat)


class ConvFFN2D(nn.Module):
    def __init__(self, channels, expansion=2):
        super().__init__()
        hidden_channels = channels * expansion
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                groups=hidden_channels,
            ),
            nn.GELU(),
            nn.Conv2d(hidden_channels, channels, kernel_size=1),
        )

    def forward(self, x):
        return self.net(x)


class ChannelCrossAttention2D(nn.Module):
    def __init__(self, channels, qkv_mode='rgb_qk_event_v'):
        super().__init__()
        self.qkv_mode = qkv_mode
        self.temperature = nn.Parameter(torch.ones(1, 1, 1))
        self.q_img = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.k_img = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.v_img = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.q_event = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.k_event = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.v_event = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, img_feat, event_feat):
        if self.qkv_mode in ('rgb_qk_event_v', 'rgb_qk_event_v_gate'):
            q = self.q_img(img_feat)
            k = self.k_img(img_feat)
            v = self.v_event(event_feat)
        elif self.qkv_mode == 'rgb_q_event_kv':
            q = self.q_img(img_feat)
            k = self.k_event(event_feat)
            v = self.v_event(event_feat)
        else:
            raise ValueError(f'Unknown qkv_mode: {self.qkv_mode}')

        b, c, h, w = q.shape
        q = q.view(b, c, -1)
        k = k.view(b, c, -1)
        v = v.view(b, c, -1)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.temperature
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v).view(b, c, h, w)
        return self.proj(out)


class WindowCrossAttention2D(nn.Module):
    def __init__(self, channels, window_size=8, num_heads=4, qkv_mode='rgb_qk_event_v'):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(f'channels ({channels}) must be divisible by num_heads ({num_heads})')
        self.channels = channels
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv_mode = qkv_mode

        self.q_img = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.k_img = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.v_img = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.q_event = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.k_event = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.v_event = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def make_qkv(self, img_feat, event_feat):
        if self.qkv_mode in ('rgb_qk_event_v', 'rgb_qk_event_v_gate'):
            return self.q_img(img_feat), self.k_img(img_feat), self.v_event(event_feat)
        if self.qkv_mode == 'rgb_q_event_kv':
            return self.q_img(img_feat), self.k_event(event_feat), self.v_event(event_feat)
        raise ValueError(f'Unknown qkv_mode: {self.qkv_mode}')

    def partition_windows(self, x):
        b, c, h, w = x.shape
        ws = self.window_size
        pad_h = (ws - h % ws) % ws
        pad_w = (ws - w % ws) % ws
        x = F.pad(x, (0, pad_w, 0, pad_h))
        hp, wp = x.shape[-2:]
        x = x.view(b, c, hp // ws, ws, wp // ws, ws)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
        x = x.view(-1, ws * ws, c)
        return x, (h, w, hp, wp)

    def reverse_windows(self, x, shape_info, batch_size):
        h, w, hp, wp = shape_info
        ws = self.window_size
        x = x.view(batch_size, hp // ws, wp // ws, ws, ws, self.channels)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.view(batch_size, self.channels, hp, wp)
        return x[:, :, :h, :w]

    def forward(self, img_feat, event_feat):
        q, k, v = self.make_qkv(img_feat, event_feat)
        b = q.shape[0]
        q_windows, shape_info = self.partition_windows(q)
        k_windows, _ = self.partition_windows(k)
        v_windows, _ = self.partition_windows(v)

        def split_heads(x):
            x = x.view(x.shape[0], x.shape[1], self.num_heads, self.head_dim)
            return x.permute(0, 2, 1, 3)

        q_windows = split_heads(q_windows)
        k_windows = split_heads(k_windows)
        v_windows = split_heads(v_windows)
        attn = torch.matmul(q_windows, k_windows.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v_windows)
        out = out.permute(0, 2, 1, 3).contiguous().view(out.shape[0], -1, self.channels)
        out = self.reverse_windows(out, shape_info, b)
        return self.proj(out)


class RGBQEventKVWindowAttention(nn.Module):
    def __init__(self, channels, window_size=8, num_heads=4, qk_norm=False):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(f'channels ({channels}) must be divisible by num_heads ({num_heads})')
        self.channels = channels
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        self.qk_norm = qk_norm

        self.q_img = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.k_event = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.v_event = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def partition_windows(self, x):
        b, c, h, w = x.shape
        ws = self.window_size
        pad_h = (ws - h % ws) % ws
        pad_w = (ws - w % ws) % ws
        x = F.pad(x, (0, pad_w, 0, pad_h))
        hp, wp = x.shape[-2:]
        x = x.view(b, c, hp // ws, ws, wp // ws, ws)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
        x = x.view(-1, ws * ws, c)
        return x, (h, w, hp, wp)

    def reverse_windows(self, x, shape_info, batch_size):
        h, w, hp, wp = shape_info
        ws = self.window_size
        x = x.view(batch_size, hp // ws, wp // ws, ws, ws, self.channels)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.view(batch_size, self.channels, hp, wp)
        return x[:, :, :h, :w]

    def forward(self, img_feat, key_feat, value_feat):
        q = self.q_img(img_feat)
        k = self.k_event(key_feat)
        v = self.v_event(value_feat)
        b = q.shape[0]
        q_windows, shape_info = self.partition_windows(q)
        k_windows, _ = self.partition_windows(k)
        v_windows, _ = self.partition_windows(v)

        def split_heads(x):
            x = x.view(x.shape[0], x.shape[1], self.num_heads, self.head_dim)
            return x.permute(0, 2, 1, 3)

        q_windows = split_heads(q_windows)
        k_windows = split_heads(k_windows)
        v_windows = split_heads(v_windows)
        if self.qk_norm:
            q_windows = F.normalize(q_windows, dim=-1)
            k_windows = F.normalize(k_windows, dim=-1)
        attn = torch.matmul(q_windows, k_windows.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v_windows)
        out = out.permute(0, 2, 1, 3).contiguous().view(out.shape[0], -1, self.channels)
        out = self.reverse_windows(out, shape_info, b)
        return self.proj(out)


class RGBQEventKVWindowAttention3D(nn.Module):
    def __init__(self, channels, window_size=8, temporal_window=2, num_heads=4, qk_norm=False):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(f'channels ({channels}) must be divisible by num_heads ({num_heads})')
        self.channels = channels
        self.window_size = window_size
        self.temporal_window = temporal_window
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        self.qk_norm = qk_norm

        self.q_img = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.k_event = nn.Conv3d(channels, channels, kernel_size=1, bias=False)
        self.v_event = nn.Conv3d(channels, channels, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def partition_windows(self, x):
        b, c, t, h, w = x.shape
        wt = min(self.temporal_window, t)
        ws = self.window_size
        pad_t = (wt - t % wt) % wt
        pad_h = (ws - h % ws) % ws
        pad_w = (ws - w % ws) % ws
        x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_t))
        tp, hp, wp = x.shape[-3:]
        x = x.view(b, c, tp // wt, wt, hp // ws, ws, wp // ws, ws)
        x = x.permute(0, 2, 4, 6, 3, 5, 7, 1).contiguous()
        x = x.view(-1, wt * ws * ws, c)
        return x, (t, h, w, tp, hp, wp, wt, ws)

    def reverse_windows(self, x, shape_info, batch_size):
        t, h, w, tp, hp, wp, wt, ws = shape_info
        x = x.view(batch_size, tp // wt, hp // ws, wp // ws, wt, ws, ws, self.channels)
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()
        x = x.view(batch_size, self.channels, tp, hp, wp)
        return x[:, :, :t, :h, :w]

    def forward(self, img_feat, key_feat_3d, value_feat_3d):
        q_2d = self.q_img(img_feat)
        k = self.k_event(key_feat_3d)
        v = self.v_event(value_feat_3d)
        b, _, t, _, _ = k.shape
        q = q_2d.unsqueeze(2).expand(-1, -1, t, -1, -1)

        q_windows, shape_info = self.partition_windows(q)
        k_windows, _ = self.partition_windows(k)
        v_windows, _ = self.partition_windows(v)

        def split_heads(x):
            x = x.view(x.shape[0], x.shape[1], self.num_heads, self.head_dim)
            return x.permute(0, 2, 1, 3)

        q_windows = split_heads(q_windows)
        k_windows = split_heads(k_windows)
        v_windows = split_heads(v_windows)
        if self.qk_norm:
            q_windows = F.normalize(q_windows, dim=-1)
            k_windows = F.normalize(k_windows, dim=-1)
        attn = torch.matmul(q_windows, k_windows.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v_windows)
        out = out.permute(0, 2, 1, 3).contiguous().view(out.shape[0], -1, self.channels)
        out = self.reverse_windows(out, shape_info, b)
        out = out.mean(dim=2)
        return self.proj(out)


class FusionBlock(nn.Module):
    def __init__(self, channels, fusion_op='cat_gate', qkv_mode='rgb_qk_event_v', window_size=8, num_heads=4):
        super().__init__()
        self.fusion_op = fusion_op
        self.qkv_mode = qkv_mode
        self.gate = EventGate(channels)
        if fusion_op == 'window':
            self.attn = WindowCrossAttention2D(channels, window_size=window_size, num_heads=num_heads, qkv_mode=qkv_mode)
        elif fusion_op == 'channel':
            self.attn = ChannelCrossAttention2D(channels, qkv_mode=qkv_mode)
        elif fusion_op == 'cat_gate':
            self.attn = None
        else:
            raise ValueError(f'Unknown fusion_op: {fusion_op}')
        self.out = ConvBlock2D(channels * 2, channels)

    def forward(self, img_feat, event_feat):
        gate = self.gate(event_feat)
        if self.attn is None:
            event_out = gate * event_feat
        else:
            event_out = self.attn(img_feat, event_feat)
            if self.qkv_mode == 'rgb_qk_event_v_gate':
                event_out = gate * event_out
        return self.out(torch.cat([img_feat, event_out], dim=1))


class MultiScaleTDCEventDeblurNet(nn.Module):
    def __init__(
        self,
        rgb_in=3,
        event_in=6,
        base_dim=32,
        fusion_type='hybrid',
        qkv_mode='rgb_qk_event_v',
        window_size=8,
        num_heads=4,
    ):
        super().__init__()
        dims = [base_dim, base_dim * 2, base_dim * 4]
        fusion_ops = self.resolve_fusion_ops(fusion_type)
        self.image_encoder = ThreeScaleImageEncoder(rgb_in=rgb_in, base_dim=base_dim)
        self.event_encoder = DualEventEncoder(event_in=event_in, base_dim=base_dim)
        self.fusions = nn.ModuleList([
            FusionBlock(dim, op, qkv_mode=qkv_mode, window_size=window_size, num_heads=num_heads)
            for dim, op in zip(dims, fusion_ops)
        ])
        self.up1 = UpBlock(dims[2], dims[1], dims[1])
        self.up0 = UpBlock(dims[1], dims[0], dims[0])
        self.reconstruct = nn.Sequential(
            ConvBlock2D(dims[0], dims[0]),
            nn.Conv2d(dims[0], rgb_in, kernel_size=3, padding=1),
        )

    @staticmethod
    def resolve_fusion_ops(fusion_type):
        fusion_type = fusion_type.lower()
        if fusion_type == 'hybrid':
            return ['cat_gate', 'cat_gate', 'window']
        if fusion_type == 'channel_window':
            return ['cat_gate', 'channel', 'window']
        if fusion_type == 'window_all':
            return ['window', 'window', 'window']
        if fusion_type == 'cat_all':
            return ['cat_gate', 'cat_gate', 'cat_gate']
        raise ValueError(f'Unknown fusion_type: {fusion_type}')

    def forward(self, blur, event):
        img_feats = self.image_encoder(blur)
        event_feats = self.event_encoder(event)
        fused = [
            fusion(img_feat, event_feat)
            for fusion, img_feat, event_feat in zip(self.fusions, img_feats, event_feats)
        ]
        x = self.up1(fused[2], fused[1])
        x = self.up0(x, fused[0])
        residual = self.reconstruct(x)
        return residual + blur


class LateFusionTDCEventDeblurNet(nn.Module):
    def __init__(
        self,
        rgb_in=3,
        event_in=6,
        base_dim=32,
        late_fusion_mode='kv_2d_k_tdc_v',
        window_size=8,
        num_heads=4,
        qk_norm=False,
        event_encoder_type='2d',
        attention_dim='2d',
        temporal_window=2,
    ):
        super().__init__()
        self.late_fusion_mode = late_fusion_mode.lower()
        self.event_encoder_type = event_encoder_type.lower()
        self.attention_dim = attention_dim.lower()
        dims = [base_dim, base_dim * 2, base_dim * 4]
        h4_dim = dims[2]

        self.image_encoder = ThreeScaleImageEncoder(rgb_in=rgb_in, base_dim=base_dim)
        self.event_2d_encoder = EventEncoderWith3DOption(
            event_in=event_in,
            base_dim=base_dim,
            encoder_type=event_encoder_type,
        )
        self.event_tdc_encoder = EventTDC3DEncoder(base_dim=base_dim)
        self.event_merge_h4 = ConvBlock2D(h4_dim * 2, h4_dim)
        self.attn = RGBQEventKVWindowAttention(
            h4_dim,
            window_size=window_size,
            num_heads=num_heads,
            qk_norm=qk_norm,
        )
        self.attn3d = RGBQEventKVWindowAttention3D(
            h4_dim,
            window_size=window_size,
            temporal_window=temporal_window,
            num_heads=num_heads,
            qk_norm=qk_norm,
        )
        self.cat_fusion = nn.Sequential(
            ConvBlock2D(h4_dim * 3, h4_dim),
            nn.Conv2d(h4_dim, h4_dim, kernel_size=3, padding=1),
        )
        self.inject = nn.Sequential(
            ConvBlock2D(h4_dim, h4_dim),
            nn.Conv2d(h4_dim, h4_dim, kernel_size=3, padding=1),
        )
        self.norm_img_h4 = nn.GroupNorm(1, h4_dim)
        self.norm_event2d_h4 = nn.GroupNorm(1, h4_dim)
        self.norm_tdc_h4 = nn.GroupNorm(1, h4_dim)
        self.norm_merged_h4 = nn.GroupNorm(1, h4_dim)
        self.norm_fused_h4 = nn.GroupNorm(1, h4_dim)
        self.ffn_h4 = ConvFFN2D(h4_dim)
        self.gamma_attn = nn.Parameter(torch.ones(1) * 0.1)
        self.gamma_ffn = nn.Parameter(torch.ones(1) * 0.1)
        self.up1 = UpBlock(dims[2], dims[1], dims[1])
        self.up0 = UpBlock(dims[1], dims[0], dims[0])
        self.reconstruct = nn.Sequential(
            ConvBlock2D(dims[0], dims[0]),
            nn.Conv2d(dims[0], rgb_in, kernel_size=3, padding=1),
        )

    @staticmethod
    def _strip_plus_mode(late_fusion_mode):
        if late_fusion_mode.endswith('_plus'):
            return late_fusion_mode[:-5], True
        if late_fusion_mode in ('a_plus', 'b_plus', 'c_plus'):
            return late_fusion_mode[0], True
        return late_fusion_mode, False

    def fuse_h4(self, img_h4, event2d_h4, tdc_h4, event3d_h4=None, tdc3d_h4=None):
        base_mode, use_plus = self._strip_plus_mode(self.late_fusion_mode)
        if base_mode in ('cat', 'concat'):
            fusion_delta = self.cat_fusion(torch.cat([img_h4, event2d_h4, tdc_h4], dim=1))
            return img_h4 + fusion_delta
        img_for_attn = self.norm_img_h4(img_h4) if use_plus else img_h4
        event2d_for_attn = self.norm_event2d_h4(event2d_h4) if use_plus else event2d_h4
        tdc_for_attn = self.norm_tdc_h4(tdc_h4) if use_plus else tdc_h4
        event3d_for_attn = self.norm_event2d_h4(event3d_h4) if use_plus and event3d_h4 is not None else event3d_h4
        tdc3d_for_attn = self.norm_tdc_h4(tdc3d_h4) if use_plus and tdc3d_h4 is not None else tdc3d_h4

        if self.attention_dim == '3d':
            if event3d_for_attn is None or tdc3d_for_attn is None:
                raise ValueError('attention_dim=3d requires event_encoder_type=3d and 3D TDC features.')
            if base_mode in ('kv_2d_k_tdc_v', 'a'):
                attn_event = self.attn3d(img_for_attn, key_feat_3d=event3d_for_attn, value_feat_3d=tdc3d_for_attn)
            elif base_mode in ('kv_2d_only', 'event2d_only', 'e2d'):
                attn_event = self.attn3d(img_for_attn, key_feat_3d=event3d_for_attn, value_feat_3d=event3d_for_attn)
            elif base_mode in ('kv_tdc_only', 'tdc_only'):
                attn_event = self.attn3d(img_for_attn, key_feat_3d=tdc3d_for_attn, value_feat_3d=tdc3d_for_attn)
            else:
                raise ValueError(f'attention_dim=3d currently supports A/event2d_only/tdc_only, got: {self.late_fusion_mode}')
        elif base_mode in ('kv_2d_only', 'event2d_only', 'e2d'):
            attn_event = self.attn(img_for_attn, key_feat=event2d_for_attn, value_feat=event2d_for_attn)
        elif base_mode in ('kv_tdc_only', 'tdc_only'):
            attn_event = self.attn(img_for_attn, key_feat=tdc_for_attn, value_feat=tdc_for_attn)
        elif base_mode in ('kv_2d_k_tdc_v', 'a'):
            attn_event = self.attn(img_for_attn, key_feat=event2d_for_attn, value_feat=tdc_for_attn)
        elif base_mode in ('kv_2d_v_tdc_k', 'b'):
            attn_event = self.attn(img_for_attn, key_feat=tdc_for_attn, value_feat=event2d_for_attn)
        elif base_mode in ('kv_merged', 'c'):
            merged_event = self.event_merge_h4(torch.cat([event2d_h4, tdc_h4], dim=1))
            if use_plus:
                merged_event = self.norm_merged_h4(merged_event)
            attn_event = self.attn(img_for_attn, key_feat=merged_event, value_feat=merged_event)
        else:
            raise ValueError(f'Unknown late_fusion_mode: {self.late_fusion_mode}')
        if use_plus:
            fused = img_h4 + self.gamma_attn * self.inject(attn_event)
            return fused + self.gamma_ffn * self.ffn_h4(self.norm_fused_h4(fused))
        return img_h4 + self.inject(attn_event)

    def forward(self, blur, event):
        img_feats = self.image_encoder(blur)
        need_3d = self.attention_dim == '3d'
        event2d_feats, event3d_feats = self.event_2d_encoder(event, return_3d=need_3d)
        tdc_feats, tdc3d_feats = self.event_tdc_encoder(event, return_3d=need_3d)

        event3d_h4 = event3d_feats[2] if event3d_feats is not None else None
        tdc3d_h4 = tdc3d_feats[2] if tdc3d_feats is not None else None
        fused_h4 = self.fuse_h4(img_feats[2], event2d_feats[2], tdc_feats[2], event3d_h4, tdc3d_h4)
        x = self.up1(fused_h4, img_feats[1])
        x = self.up0(x, img_feats[0])
        residual = self.reconstruct(x)
        return residual + blur


class NoEventDeblurNet(nn.Module):
    def __init__(self, rgb_in=3, base_dim=32):
        super().__init__()
        self.encoder_rgb = nn.Sequential(
            nn.Conv2d(rgb_in, base_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(base_dim, base_dim, kernel_size=3, padding=1, stride=2),
            nn.LeakyReLU(0.2, True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base_dim, base_dim, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(base_dim, base_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(base_dim, rgb_in, kernel_size=3, padding=1),
        )

    def forward(self, blur, event=None):
        residual = self.decoder(self.encoder_rgb(blur))
        return residual + blur


class ConcatEventDeblurNet(nn.Module):
    def __init__(self, rgb_in=3, event_in=6, base_dim=32):
        super().__init__()
        self.encoder_rgb = nn.Sequential(
            nn.Conv2d(rgb_in, base_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(base_dim, base_dim, kernel_size=3, padding=1, stride=2),
            nn.LeakyReLU(0.2, True),
        )
        self.encoder_event = nn.Sequential(
            TDCBlock3D(in_channels=1, out_channels=base_dim),
            nn.Conv3d(base_dim, base_dim, kernel_size=(1, 3, 3), padding=(0, 1, 1), stride=(1, 2, 2)),
            nn.LeakyReLU(0.2, True),
        )
        self.time_pool = nn.AdaptiveAvgPool3d((1, None, None))
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(base_dim * 2, base_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base_dim, base_dim, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(base_dim, base_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(base_dim, rgb_in, kernel_size=3, padding=1),
        )

    def forward(self, blur, event):
        feat_rgb = self.encoder_rgb(blur)
        feat_event = self.encoder_event(event.unsqueeze(1))
        feat_event = self.time_pool(feat_event).squeeze(2)
        fused = self.fusion_conv(torch.cat([feat_rgb, feat_event], dim=1))
        residual = self.decoder(fused)
        return residual + blur


class TDC_Deblur_Net(nn.Module):
    def __init__(self, rgb_in=3, event_in=6, base_dim=32):
        super().__init__()
        self.encoder_rgb = nn.Sequential(
            nn.Conv2d(rgb_in, base_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(base_dim, base_dim, kernel_size=3, padding=1, stride=2),
        )
        self.encoder_event = nn.Sequential(
            TDCBlock3D(in_channels=1, out_channels=base_dim),
            nn.Conv3d(base_dim, base_dim, kernel_size=(1, 3, 3), padding=(0, 1, 1), stride=(1, 2, 2)),
            nn.LeakyReLU(0.2, True),
        )
        self.sta_fusion = CrossModalSTA(in_channels=base_dim)
        self.time_pool = nn.AdaptiveAvgPool3d((1, None, None))
        self.fusion_conv = nn.Conv2d(base_dim * 2, base_dim, kernel_size=3, padding=1)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base_dim, base_dim, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(base_dim, base_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(base_dim, rgb_in, kernel_size=3, padding=1),
        )

    def forward(self, blur, event):
        feat_rgb = self.encoder_rgb(blur)
        feat_event = self.encoder_event(event.unsqueeze(1))
        fused_event_3d = self.sta_fusion(feat_rgb, feat_event)
        fused_event_2d = self.time_pool(fused_event_3d).squeeze(2)
        bottleneck_feat = self.fusion_conv(torch.cat([feat_rgb, fused_event_2d], dim=1))
        out = self.decoder(bottleneck_feat)
        return out + blur


def build_deblur_model(
    model_type='attention',
    rgb_in=3,
    event_in=6,
    base_dim=32,
    fusion_type='hybrid',
    qkv_mode='rgb_qk_event_v',
    late_fusion_mode='kv_2d_k_tdc_v',
    window_size=8,
    num_heads=4,
    qk_norm=False,
    event_encoder_type='2d',
    attention_dim='2d',
    temporal_window=2,
):
    model_type = model_type.lower()
    if model_type in ('late_fusion', 'late_fusion_tdc', 'single_fusion'):
        return LateFusionTDCEventDeblurNet(
            rgb_in=rgb_in,
            event_in=event_in,
            base_dim=base_dim,
            late_fusion_mode=late_fusion_mode,
            window_size=window_size,
            num_heads=num_heads,
            qk_norm=qk_norm,
            event_encoder_type=event_encoder_type,
            attention_dim=attention_dim,
            temporal_window=temporal_window,
        )
    if model_type in ('multiscale', 'multiscale_tdc', 'tdc_multiscale'):
        return MultiScaleTDCEventDeblurNet(
            rgb_in=rgb_in,
            event_in=event_in,
            base_dim=base_dim,
            fusion_type=fusion_type,
            qkv_mode=qkv_mode,
            window_size=window_size,
            num_heads=num_heads,
        )
    if model_type in ('attention', 'tdc_attention', 'sta'):
        return TDC_Deblur_Net(rgb_in=rgb_in, event_in=event_in, base_dim=base_dim)
    if model_type in ('concat', 'event_concat'):
        return ConcatEventDeblurNet(rgb_in=rgb_in, event_in=event_in, base_dim=base_dim)
    if model_type in ('no_event', 'image_only', 'blur_only'):
        return NoEventDeblurNet(rgb_in=rgb_in, base_dim=base_dim)
    raise ValueError(f'Unknown model type: {model_type}')


if __name__ == '__main__':
    dummy_blur = torch.randn(2, 3, 128, 128)
    dummy_event = torch.randn(2, 6, 128, 128)
    model = build_deblur_model(model_type='multiscale', base_dim=24)
    out = model(dummy_blur, dummy_event)
    print('Input blur:', dummy_blur.shape)
    print('Input event:', dummy_event.shape)
    print('Output:', out.shape)
    print('Params:', sum(p.numel() for p in model.parameters() if p.requires_grad))
