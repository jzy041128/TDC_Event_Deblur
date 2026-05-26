import torch
import torch.nn as nn
from models.tdc_module import TDCBlock3D
from models.sta_module import CrossModalSTA

class TDC_Deblur_Net(nn.Module):
    def __init__(self, rgb_in=3, event_in=6, base_dim=32):
        super().__init__()
        
        # ==========================================
        # 1. RGB 编码器 (提取单帧 2D 空间特征)
        # ==========================================
        self.encoder_rgb = nn.Sequential(
            nn.Conv2d(rgb_in, base_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(base_dim, base_dim, kernel_size=3, padding=1, stride=2) # 空间下采样
        )

        # ==========================================
        # 2. Event 编码器 (提取 3D 时空动态特征)
        # ==========================================
        self.encoder_event = nn.Sequential(
            # 接入导师的创新点：3D TDC 模块
            TDCBlock3D(in_channels=1, out_channels=base_dim),
            # 空间下采样 (stride=(1,2,2) 保证时间维度 T=6 不变，只对 H 和 W 缩小一半)
            nn.Conv3d(base_dim, base_dim, kernel_size=(1, 3, 3), padding=(0, 1, 1), stride=(1, 2, 2)),
            nn.LeakyReLU(0.2, True)
        )

        # ==========================================
        # 3. 跨模态融合 (导师的创新点：RGB 指导 Event)
        # ==========================================
        self.sta_fusion = CrossModalSTA(in_channels=base_dim)

        # ==========================================
        # 4. 3D 转 2D 桥接层
        # ==========================================
        # 融合后的 Event 依然是 3D 的，我们需要把它压扁回 2D，才能和 RGB 拼接去解码
        self.time_pool = nn.AdaptiveAvgPool3d((1, None, None)) # 把时间维度平均池化掉
        self.fusion_conv = nn.Conv2d(base_dim * 2, base_dim, kernel_size=3, padding=1)

        # ==========================================
        # 5. 解码器 (恢复 2D 清晰图像)
        # ==========================================
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base_dim, base_dim, kernel_size=4, stride=2, padding=1), # 上采样放大
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(base_dim, base_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(base_dim, rgb_in, kernel_size=3, padding=1) # 输出 3 通道 RGB
        )

    def forward(self, blur, event):
        # blur: (B, 3, H, W)
        # event: (B, 6, H, W)
        
        # --- 1. 特征编码 ---
        feat_rgb = self.encoder_rgb(blur) # 输出: (B, base_dim, H/2, W/2)

        # 核心变形：升维打击，把通道 6 变成时间维度 6
        event_3d = event.unsqueeze(1)     # 变成: (B, 1, 6, H, W)
        feat_event = self.encoder_event(event_3d) # 输出: (B, base_dim, 6, H/2, W/2)

        # --- 2. 跨模态融合 ---
        fused_event_3d = self.sta_fusion(feat_rgb, feat_event) # 输出依然保留 3D

        # --- 3. 3D 转 2D 桥接与拼接 ---
        # 把时间 T=6 压缩掉，去掉多余的维度 1
        fused_event_2d = self.time_pool(fused_event_3d).squeeze(2) # 变成: (B, base_dim, H/2, W/2)
        
        # 拼接 RGB 特征和融合后的 Event 特征
        concat_feat = torch.cat([feat_rgb, fused_event_2d], dim=1) # 通道翻倍
        bottleneck_feat = self.fusion_conv(concat_feat) # 通道恢复到 base_dim

        # --- 4. 解码输出 ---
        out = self.decoder(bottleneck_feat)
        
        return out + blur # 全局残差，去模糊必备

# ==========================================
# 测试代码
# ==========================================
if __name__ == '__main__':
    dummy_blur = torch.randn(4, 3, 256, 256)
    dummy_event = torch.randn(4, 6, 256, 256)
    
    model = TDC_Deblur_Net()
    out = model(dummy_blur, dummy_event)
    
    print("\n--- 终极网络大合体测试 ---")
    print("输入 Blur 维度:", dummy_blur.shape)
    print("输入 Event 维度:", dummy_event.shape)
    print("输出 清晰图像 维度:", out.shape)
    print("网络参数量:", sum(p.numel() for p in model.parameters() if p.requires_grad))