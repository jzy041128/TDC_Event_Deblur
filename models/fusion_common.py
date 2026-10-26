import torch
import torch.nn as nn
import torch.nn.functional as F


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
