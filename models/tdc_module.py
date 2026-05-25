import torch
import torch.nn as nn

class TDCBlock3D(nn.Module):
    """
    基于 3D 卷积的时序差分模块 (Temporal Difference Convolution)
    专为处理事件体素设计
    输入维度要求: (Batch, Channels, Time, Height, Width) -> (B, 1, 6, H, W)
    """
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)):
        super().__init__()
        # 1. 基础的 3D 卷积，用于提取时空联合特征
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        
        # 2. 空间注意力机制的雏形 (可选，为了让网络更关注运动区域)
        # 这里用一个简单的 1x1x1 卷积融合通道
        self.diff_conv = nn.Conv3d(out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # x 形状: (B, C, T, H, W)
        
        # 1. 先进行标准的 3D 卷积提取特征
        out = self.conv3d(x)
        
        # 2. 核心操作：计算时序差分 (TDC)
        # 提取时间维度上的动态变化：后一个时间步 减去 前一个时间步
        # out[:, :, 1:, :, :] 取 T=1 到末尾
        # out[:, :, :-1, :, :] 取 T=0 到倒数第二
        diff = out[:, :, 1:, :, :] - out[:, :, :-1, :, :]
        
        # 为了保持时间维度 T (例如 6) 不变，我们在时间维度的最前面补零 (Zero Padding)
        # 因为第一帧没有更前面的一帧可以减了
        pad = torch.zeros_like(out[:, :, 0:1, :, :])
        diff = torch.cat([pad, diff], dim=2)
        
        # 3. 差分特征通过卷积处理后，作为高频补偿加回到原特征上
        out = out + self.diff_conv(diff)
        
        out = self.bn(out)
        out = self.relu(out)
        return out

# ==========================================
# 简单的维度测试
# ==========================================
if __name__ == '__main__':
    # 模拟我们 DataLoader 读出来的事件 Voxel 维度 (B, C_event, H, W) -> (4, 6, 256, 256)
    dummy_event_2d = torch.randn(4, 6, 256, 256)
    
    print("原始事件体素维度 (2D形式):", dummy_event_2d.shape)
    
    # 导师的核心思想：把 2D 的 (B, 6, H, W) 变形为 3D 的 (B, 1, 6, H, W)
    # 相当于变成了单通道(C=1)，时间步长(T=6) 的 3D 视频数据
    dummy_event_3d = dummy_event_2d.unsqueeze(1) 
    print("转换后的 3D 维度 (B, C, T, H, W):", dummy_event_3d.shape)
    
    # 初始化 TDC 模块 (输入通道数为 1，输出通道数设为 32)
    tdc_block = TDCBlock3D(in_channels=1, out_channels=32)
    
    # 送入网络
    out = tdc_block(dummy_event_3d)
    print("TDC 模块输出维度:", out.shape)