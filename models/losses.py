import torch
import torch.nn as nn
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))


class EdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        kernel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        kernel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        self.register_buffer('kernel_x', kernel_x)
        self.register_buffer('kernel_y', kernel_y)

    def gradient(self, x):
        channels = x.shape[1]
        kernel_x = self.kernel_x.repeat(channels, 1, 1, 1)
        kernel_y = self.kernel_y.repeat(channels, 1, 1, 1)
        grad_x = F.conv2d(x, kernel_x, padding=1, groups=channels)
        grad_y = F.conv2d(x, kernel_y, padding=1, groups=channels)
        return grad_x, grad_y

    def forward(self, pred, target):
        pred_x, pred_y = self.gradient(pred)
        target_x, target_y = self.gradient(target)
        return F.l1_loss(pred_x, target_x) + F.l1_loss(pred_y, target_y)


class RestorationLoss(nn.Module):
    def __init__(self, loss_type='l1', edge_weight=0.05):
        super().__init__()
        self.loss_type = loss_type.lower()
        self.edge_weight = edge_weight
        if self.loss_type == 'l1':
            self.base_loss = nn.L1Loss()
        elif self.loss_type in ('charbonnier', 'charb'):
            self.base_loss = CharbonnierLoss()
        elif self.loss_type in ('charbonnier_edge', 'charb_edge'):
            self.base_loss = CharbonnierLoss()
            self.edge_loss = EdgeLoss()
        else:
            raise ValueError(f'Unknown loss type: {loss_type}')

    def forward(self, pred, target):
        loss = self.base_loss(pred, target)
        if self.loss_type in ('charbonnier_edge', 'charb_edge'):
            loss = loss + self.edge_weight * self.edge_loss(pred, target)
        return loss


def build_loss(loss_type='l1', edge_weight=0.05):
    return RestorationLoss(loss_type=loss_type, edge_weight=edge_weight)
