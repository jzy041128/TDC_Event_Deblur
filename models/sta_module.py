import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalSTA(nn.Module):
    """
    修正版：严格遵循 "RGB(Q,K) 指导 Event(V)" 的通道交叉注意力
    创新点：利用 RGB 的空间结构生成注意力权重，去滤除 3D Event 的噪声并增强边缘。
    """
    def __init__(self, in_channels):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1, 1, 1)) # 温度系数

        # 1. RGB (单帧) 专属：生成 Q 和 K
        # 用于计算空间结构的通道相关性矩阵
        self.q_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.k_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)

        # 2. Event (3D体素) 专属：生成 V
        # 用于提供包含高频运动特征的实际内容
        self.v_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1, bias=False)

        # 3. 融合后的输出映射 (3D)
        self.project_out = nn.Conv3d(in_channels, in_channels, kernel_size=1, bias=False)

    def forward(self, rgb_feat, event_feat_3d):
        # rgb_feat 形状: (B, C, H, W)
        # event_feat_3d 形状: (B, C, T, H, W)
        B, C, T, H, W = event_feat_3d.shape

        # --- 第一步：RGB 生成 Q 和 K (提取结构特征) ---
        q = self.q_conv(rgb_feat) # (B, C, H, W)
        k = self.k_conv(rgb_feat) # (B, C, H, W)

        # 展平 H 和 W，准备计算通道注意力
        q_flat = q.view(B, C, -1) # (B, C, H*W)
        k_flat = k.view(B, C, -1) # (B, C, H*W)

        # --- 第二步：Event 生成 V (提取运动特征) ---
        v = self.v_conv(event_feat_3d) # (B, C, T, H, W)
        # 展平 时间 T 和 空间 H,W
        v_flat = v.view(B, C, -1) # (B, C, T*H*W)

        # --- 第三步：核心指导！RGB 自己算注意力滤网 ---
        # q 和 k 转置相乘，得到 RGB 通道间的结构亲和力矩阵 (B, C, C)
        attn = torch.matmul(q_flat, k_flat.transpose(-2, -1)) * self.temperature
        attn = F.softmax(attn, dim=-1) # 这是纯粹由 RGB 决定的指导矩阵！

        # --- 第四步：用 RGB 的滤网去过滤 Event 的 V ---
        # (B, C, C) 乘以 (B, C, T*H*W) -> 结果依然是 (B, C, T*H*W)
        fused_feat_flat = torch.matmul(attn, v_flat)
        
        # 恢复成完整的 3D 形状
        fused_feat_3d = fused_feat_flat.view(B, C, T, H, W)

        # --- 第五步：映射并加上 Event 的残差 ---
        out = self.project_out(fused_feat_3d) + event_feat_3d

        return out

# =================测试代码=================
if __name__ == '__main__':
    dummy_rgb = torch.randn(2, 32, 128, 128)
    dummy_event = torch.randn(2, 32, 6, 128, 128)
    
    sta = CrossModalSTA(in_channels=32)
    out = sta(dummy_rgb, dummy_event)
    
    print("【严格版】RGB(Q,K) 指导 Event(V) 输出维度:", out.shape)