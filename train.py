from models.tdc_deblur_net import TDC_Deblur_Net
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from data.dataset import EventDeblurDataset


# ========================================================
# 训练主循环
# ========================================================
def main():
    # 1. 读取配置
    with open('configs/train_tdc.yml', 'r') as f:
        opt = yaml.safe_load(f)

    device = torch.device('cpu')#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"正在使用设备: {device}")

    # 2. 准备数据
    train_dataset = EventDeblurDataset(opt['datasets']['train'])
    
    # 注意：在 WSL 里刚开始测试时，num_workers 最好先设为 0，防止多进程读取 h5 报错
    train_loader = DataLoader(
        train_dataset, 
        batch_size=opt['datasets']['train']['batch_size'], 
        shuffle=True, 
        num_workers=0 
    )

    # 3. 初始化模型、损失函数和优化器
    model = TDC_Deblur_Net().to(device)
    criterion = nn.L1Loss()  # 去模糊常用的 L1 Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=opt['train']['lr'])

    epochs = opt['train']['epochs']
    print("\n🚀 开始训练大循环...")

    # 4. 真正开始训练
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for i, batch in enumerate(train_loader):
            # 将数据搬运到显卡上
            blur = batch['blur'].to(device)
            gt = batch['gt'].to(device)
            event = batch['event'].to(device)

            # --- 核心三步曲 ---
            optimizer.zero_grad()            # 清空上一轮梯度
            output = model(blur, event)      # 前向传播 (推断)
            loss = criterion(output, gt)     # 计算误差
            loss.backward()                  # 反向传播 (求导)
            optimizer.step()                 # 更新权重

            epoch_loss += loss.item()
            
            # 打印进度 (这里设置为每个 batch 都打印，方便你看到 loss 变化)
            print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

if __name__ == '__main__':
    main()