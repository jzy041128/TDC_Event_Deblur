import os
import torch
import h5py
import numpy as np
import random
import glob
from torch.utils.data import Dataset

class EventDeblurDataset(Dataset):
    def __init__(self, opt_dataset):
        super().__init__()
        self.dataroot = opt_dataset['dataroot']
        self.patch_size = opt_dataset.get('patch_size', 256)
        self.random_crop = opt_dataset.get('random_crop', True)
        
        # 1. 找到目录下所有的 .h5 文件
        self.h5_files = glob.glob(os.path.join(self.dataroot, '*.h5'))
        if len(self.h5_files) == 0:
            print(f"警告: 在 {self.dataroot} 下没有找到任何 .h5 文件！")
            
        # 2. 建立全局索引映射
        # 因为一个 .h5 文件（通常是一个视频序列）里面包含多帧图像
        self.samples = []
        for h5_path in self.h5_files:
            # 只读模式打开获取帧数
            with h5py.File(h5_path, 'r') as f:
                num_frames = len(f['images'].keys())
                for i in range(num_frames):
                    self.samples.append((h5_path, i))
                    
        print(f"成功加载数据集，共找到 {len(self.h5_files)} 个序列，总计 {len(self.samples)} 帧有效数据。")
                            
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, index):
        h5_path, frame_idx = self.samples[index]
        
        # 3. 按照 EFNet 的键值格式读取数据
        with h5py.File(h5_path, 'r') as f:
            # 格式化字符串为 9 位数字，如 'image000000001'
            img_blur = f['images'][f'image{frame_idx:09d}'][...]
            img_gt = f['sharp_images'][f'image{frame_idx:09d}'][...]
            event_voxel = f['voxels'][f'voxel{frame_idx:09d}'][...]
        
        # 4. 转换为 Tensor 并归一化图像到 0~1 (原数据已经是 C, H, W 了)
        img_blur = torch.from_numpy(img_blur).float() / 255.0
        img_gt = torch.from_numpy(img_gt).float() / 255.0
        event_tensor = torch.from_numpy(event_voxel).float()
        
        # 5. 随机裁剪 (保证图像和事件流裁剪同一块区域)
        c, h, w = img_gt.shape
        th, tw = self.patch_size, self.patch_size
        if h > th and w > tw:
            if self.random_crop:
                i = random.randint(0, h - th)
                j = random.randint(0, w - tw)
            else:
                i = (h - th) // 2
                j = (w - tw) // 2
            img_blur = img_blur[:, i:i+th, j:j+tw]
            img_gt = img_gt[:, i:i+th, j:j+tw]
            event_tensor = event_tensor[:, i:i+th, j:j+tw]
            
        return {
            'blur': img_blur,
            'gt': img_gt,
            'event': event_tensor
        }

# ================= 测试代码 =================
if __name__ == '__main__':
    # 模拟配置文件
    dummy_opt = {
        'dataroot': './datasets/train',
        'patch_size': 256
    }
    dataset = EventDeblurDataset(dummy_opt)
    
    if len(dataset) > 0:
        data = dataset[0]
        print("\n读取成功！张量维度如下：")
        print("Blur 图像 shape:", data['blur'].shape)
        print("GT   图像 shape:", data['gt'].shape)
        print("事件 Voxel shape:", data['event'].shape)
