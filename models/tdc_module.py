import torch
import torch.nn as nn
import torch.nn.functional as F


class ShortTermTDC3D(nn.Module):
    """BN-free short-term temporal difference convolution in kernel space."""

    def __init__(self, in_channels, out_channels, stride=(1, 1, 1), groups=1):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=(5, 3, 3),
            stride=stride,
            padding=(2, 1, 1),
            groups=groups,
            bias=False,
        )

    def short_term_weight(self):
        weight = self.conv.weight
        diff_weight = torch.zeros_like(weight)
        diff_weight[:, :, 4] = weight[:, :, 4]
        diff_weight[:, :, 3] = weight[:, :, 3] - weight[:, :, 4]
        diff_weight[:, :, 2] = weight[:, :, 2] - weight[:, :, 3]
        diff_weight[:, :, 1] = weight[:, :, 1] - weight[:, :, 2]
        diff_weight[:, :, 0] = -weight[:, :, 1]
        return diff_weight

    def forward(self, x):
        return F.conv3d(
            x,
            self.short_term_weight(),
            bias=None,
            stride=self.conv.stride,
            padding=self.conv.padding,
            groups=self.conv.groups,
        )


class ShortTermTDCBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=(1, 1, 1)):
        super().__init__()
        self.tdc = ShortTermTDC3D(in_channels, out_channels, stride=stride)
        self.spatial = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=(1, 3, 3),
            padding=(0, 1, 1),
            bias=True,
        )
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.act(self.tdc(x))
        return self.act(self.spatial(x))
