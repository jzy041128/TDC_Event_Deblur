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

    def forward(self, event):
        x = event.unsqueeze(1)
        feat0_3d = self.stem(x)
        feat1_3d = self.down1(feat0_3d)
        feat2_3d = self.down2(feat1_3d)
        return [self.pool_time(feat0_3d), self.pool_time(feat1_3d), self.pool_time(feat2_3d)]


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
        temporal_feats = self.tdc_3d(event)
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
    window_size=8,
    num_heads=4,
):
    model_type = model_type.lower()
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
