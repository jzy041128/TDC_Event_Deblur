import os
import sys
import yaml
import time
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

# 导入极其权威的 skimage 指标计算
from skimage.metrics import peak_signal_noise_ratio as calculate_psnr
from skimage.metrics import structural_similarity as calculate_ssim

from data.dataset import EventDeblurDataset
from models.losses import build_loss
from models.tdc_deblur_net import build_deblur_model

class Logger(object):
    """
    终端输出双向拦截器：既在屏幕上打印，又同时写入 txt 文件
    """
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush() # 实时强制写入硬盘

    def flush(self):
        pass

def tensor2img(tensor):
    """
    将 PyTorch 的 Tensor (B, C, H, W) 转换为 NumPy 的图像格式 (H, W, C)
    以便送入 skimage 计算指标
    """
    img = tensor.detach().cpu().squeeze().numpy() # 假设 batch_size=1 时验证
    if img.ndim == 4: # 如果 batch > 1，只取第一张算
        img = img[0]
    img = np.transpose(img, (1, 2, 0)) # CHW -> HWC
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img


def tensor2float_img(tensor):
    img = tensor.detach().cpu().squeeze().numpy()
    if img.ndim == 4:
        img = img[0]
    img = np.transpose(img, (1, 2, 0))
    return np.clip(img, 0.0, 1.0)


def format_duration(seconds):
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def main():
    # 1. 加载配置
    with open('configs/train_tdc.yml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 2. 建立实验文件夹；恢复训练时继续使用 checkpoint 所在目录
    resume_path = config['path'].get('resume_state')
    resume_mode = resume_path and resume_path != '~' and os.path.exists(resume_path)
    if resume_mode:
        exp_dir = os.path.dirname(resume_path)
    else:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_name = f"{config['name']}_{timestamp}"
        exp_dir = os.path.join(config['path']['experiments_root'], exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    sys.stdout = Logger(os.path.join(exp_dir, "train_log.txt"))
    if resume_mode:
        print(f"🔁 Resume 模式：继续使用原实验目录 {exp_dir}")
    else:
        print(f"🚀 本次实验的所有存档将保存在: {exp_dir}")

    # 3. 设置设备 (2060 GPU 全开)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"💻 正在使用设备: {device}")

    # 4. 加载数据 (Train 和 Val)
    train_dataset = EventDeblurDataset(opt_dataset=config['datasets']['train'])
    train_loader = DataLoader(train_dataset, 
                              batch_size=config['datasets']['train']['batch_size'], 
                              shuffle=True, 
                              num_workers=config['datasets']['train'].get('num_workers', 4),
                              pin_memory=True)
    
    val_opt = dict(config['datasets']['val'])
    val_opt['random_crop'] = False
    val_dataset = EventDeblurDataset(opt_dataset=val_opt)
    val_loader = DataLoader(val_dataset, 
                            batch_size=1, 
                            shuffle=False, 
                            num_workers=0, 
                            pin_memory=True)

    # 5. 初始化模型、损失函数、优化器
    model_cfg = config.get('model', {})
    model_type = model_cfg.get('type', 'attention')
    base_dim = model_cfg.get('base_dim', 32)
    fusion_type = model_cfg.get('fusion_type', 'hybrid')
    qkv_mode = model_cfg.get('qkv_mode', 'rgb_qk_event_v')
    late_fusion_mode = model_cfg.get('late_fusion_mode', 'kv_2d_k_tdc_v')
    window_size = model_cfg.get('window_size', 8)
    num_heads = model_cfg.get('num_heads', 4)
    model = build_deblur_model(
        model_type=model_type,
        base_dim=base_dim,
        fusion_type=fusion_type,
        qkv_mode=qkv_mode,
        late_fusion_mode=late_fusion_mode,
        window_size=window_size,
        num_heads=num_heads,
    ).to(device)
    print(
        f"Model type: {model_type} | base_dim: {base_dim} | "
        f"fusion_type: {fusion_type} | qkv_mode: {qkv_mode} | "
        f"late_fusion_mode: {late_fusion_mode} | "
        f"window_size: {window_size} | num_heads: {num_heads}"
    )
    loss_cfg = config.get('loss', {})
    criterion = build_loss(
        loss_type=loss_cfg.get('type', 'l1'),
        edge_weight=loss_cfg.get('edge_weight', 0.05),
    )
    print(f"Loss type: {loss_cfg.get('type', 'l1')} | edge_weight: {loss_cfg.get('edge_weight', 0.05)}")
    optimizer = optim.AdamW(model.parameters(), lr=config['train']['learning_rate'])

    # 6. 👇 核心功能：断点续训 (Resume) 👇
    start_epoch = 0
    best_psnr = 0.0
    if resume_path and resume_path != '~':
        if os.path.exists(resume_path):
            print(f"🔄 发现存档文件！正在从 {resume_path} 恢复训练...")
            checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            best_psnr = checkpoint.get('best_psnr', 0.0)
            print(f"✅ 成功恢复至第 {start_epoch} 个 Epoch！历史最佳 PSNR: {best_psnr:.2f} dB")
        else:
            print(f"⚠️ 找不到存档文件 {resume_path}，将从头开始训练！")

    # 7. 开始训练大循环
    num_epochs = config['train']['num_epochs']
    train_start_time = time.time()
    
    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()
        model.train()
        epoch_loss = 0.0
        
        # --- 训练阶段 ---
        for step, batch_data in enumerate(train_loader):
            blur = batch_data['blur'].to(device)
            event = batch_data['event'].to(device)
            gt = batch_data['gt'].to(device)

            optimizer.zero_grad()
            pred = model(blur, event)
            loss = criterion(pred, gt)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            if (step + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{step+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        avg_train_loss = epoch_loss / len(train_loader)
        print(f"\n⏳ Epoch {epoch+1} 训练结束，平均 Loss: {avg_train_loss:.4f}，开始在 Test 集上验证...")
        model.eval()
        total_blur_psnr, total_pred_psnr, total_ssim = 0.0, 0.0, 0.0
        
        with torch.no_grad():
            for val_batch in val_loader:
                val_blur = val_batch['blur'].to(device)
                val_event = val_batch['event'].to(device)
                val_gt = val_batch['gt'].to(device)
                
                val_pred = model(val_blur, val_event)
                
                # 转换格式计算指标
                pred_np = tensor2float_img(val_pred)
                blur_np = tensor2float_img(val_blur)
                gt_np = tensor2float_img(val_gt)
                
                # 计算 PSNR 和 SSIM (channel_axis=2 代表颜色通道在最后)
                total_blur_psnr += calculate_psnr(gt_np, blur_np, data_range=1.0)
                total_pred_psnr += calculate_psnr(gt_np, pred_np, data_range=1.0)
                total_ssim += calculate_ssim(gt_np, pred_np, channel_axis=2, data_range=1.0)
                
        avg_blur_psnr = total_blur_psnr / len(val_loader)
        avg_psnr = total_pred_psnr / len(val_loader)
        avg_ssim = total_ssim / len(val_loader)
        psnr_gain = avg_psnr - avg_blur_psnr
        print(f"📈 验证结果 -> Blur PSNR: {avg_blur_psnr:.2f} dB | Pred PSNR: {avg_psnr:.2f} dB | Gain: {psnr_gain:+.2f} dB | SSIM: {avg_ssim:.4f}\n")

        elapsed_total = time.time() - train_start_time
        elapsed_epoch = time.time() - epoch_start_time
        finished_epochs = epoch + 1 - start_epoch
        remaining_epochs = num_epochs - epoch - 1
        avg_epoch_time = elapsed_total / max(finished_epochs, 1)
        eta_seconds = avg_epoch_time * remaining_epochs
        print(
            f"⏱️ 时间统计 -> Epoch耗时: {format_duration(elapsed_epoch)} | "
            f"已用: {format_duration(elapsed_total)} | "
            f"预计剩余: {format_duration(eta_seconds)}\n"
        )

        # --- 存档阶段 (Save Checkpoints) ---
        save_dict = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_psnr': best_psnr
        }
        
        # 1. 每跑完一个 Epoch，覆盖保存一次 latest.pth (防停电)
        torch.save(save_dict, os.path.join(exp_dir, 'latest.pth'))
        
        # 2. 如果分刷出了新高，保存为 best.pth
        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            print(f"✨ 突破历史记录！正在保存最佳模型 (PSNR: {best_psnr:.2f})")
            torch.save(save_dict, os.path.join(exp_dir, 'best.pth'))

if __name__ == '__main__':
    main()
