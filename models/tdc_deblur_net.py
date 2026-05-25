import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. 简化的时空上下文注意力模块 (魔改自 STA)
# 导师要求：用单帧 RGB 指导 事件体素
# ==========================================
class CrossModalSTA(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # 这是一个简化的交叉注意力示意
        # rgb_feat 生成注意力权重，去增强 event_feat
        self.conv_rgb = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv_event = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv_out = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, rgb_feat, event_feat):
        # 1. RGB 特征生成 Spatial Attention Map
        attention_map = self.sigmoid(self.conv_rgb(rgb_feat))
        
        # 2. 用 RGB 的注意力去调制 (Modulate) 事件特征
        event_projected = self.conv_event(event_feat)
        fused_feat = event_projected * attention_map
        
        # 3. 融合后输出
        return self.conv_out(fused_feat + rgb_feat)

# ==========================================
# 2. 编解码器主网络 (Encoder-Decoder)
# 导师要求：改成去模糊架构
# ==========================================
class TDC_Deblur_Net(nn.Module):
    def __init__(self, rgb_in=3, event_in=6, base_dim=64):
        super().__init__()
        
        # --- 双分支 Encoder (分别提取 RGB 和 Event 的浅层特征) ---
        self.encoder_rgb = nn.Sequential(
            nn.Conv2d(rgb_in, base_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(base_dim, base_dim, kernel_size=3, padding=1, stride=2) # 下采样
        )
        
        # 这里预留给你接原论文的 TDC (时序差分卷积) 模块
        # 因为事件 Voxel 天生带有时间维度，用 TDC 处理再合适不过了
        self.encoder_event = nn.Sequential(
            nn.Conv2d(event_in, base_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(base_dim, base_dim, kernel_size=3, padding=1, stride=2) # 下采样
        )
        
        # --- 核心融合层 (导师要求的 RGB 指导 Event) ---
        self.sta_fusion = CrossModalSTA(in_channels=base_dim)
        
        # --- Decoder (解码恢复清晰图像) ---
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base_dim, base_dim, kernel_size=4, stride=2, padding=1), # 上采样
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(base_dim, base_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(base_dim, rgb_in, kernel_size=3, padding=1) # 输出 3 通道 RGB
        )

    def forward(self, blur, event):
        # 1. 编码提取特征
        feat_rgb = self.encoder_rgb(blur)
        feat_event = self.encoder_event(event)
        
        # 2. 特征融合 (STA模块)
        feat_fused = self.sta_fusion(feat_rgb, feat_event)
        
        # 3. 解码输出清晰图像 (加上 blur 做全局残差连接，去模糊常用技巧)
        out = self.decoder(feat_fused)
        return out + blur

# ==========================================
# 简单的维度测试
# ==========================================
if __name__ == '__main__':
    # 模拟我们之前 DataLoader 读出来的维度
    dummy_blur = torch.randn(4, 3, 256, 256)
    dummy_event = torch.randn(4, 6, 256, 256)
    
    model = TDC_Deblur_Net()
    out = model(dummy_blur, dummy_event)
    
    print("输入 Blur 维度:", dummy_blur.shape)
    print("输入 Event 维度:", dummy_event.shape)
    print("输出 清晰图像 维度:", out.shape)