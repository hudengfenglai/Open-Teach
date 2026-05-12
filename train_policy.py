"""
行为克隆策略训练
从采集的演示数据中学习 image -> joint action 的映射。

运行方式:
    cd Open-Teach
    export LD_LIBRARY_PATH=/home/hu/miniconda3/envs/openteach_v2/lib
    export PATH=/home/hu/miniconda3/envs/openteach_v2/bin:$PATH
    /home/hu/miniconda3/envs/openteach_v2/bin/python train_policy.py

训练完成后模型保存在: trained_models/grasp_policy.pt
"""

import os
import h5py
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import argparse
from tqdm import tqdm


# ============================================================
# 数据集
# ============================================================
class GraspDemoDataset(Dataset):
    """从演示数据加载 (image, joint_state) 对"""
    
    def __init__(self, data_path, num_demos=10, image_size=84):
        self.images = []
        self.actions = []
        self.image_size = image_size
        
        print(f"加载演示数据...")
        for demo_id in range(1, num_demos + 1):
            demo_dir = os.path.join(data_path, f'demonstration_{demo_id}')
            if not os.path.exists(demo_dir):
                continue
            
            # 加载关节角度
            joint_path = os.path.join(demo_dir, 'allegro_commanded_joint_states.h5')
            with h5py.File(joint_path, 'r') as f:
                joints = f['positions'][:]
            
            # 加载视频帧
            video_path = os.path.join(demo_dir, 'cam_0_rgb_video.avi')
            cap = cv2.VideoCapture(video_path)
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx < len(joints):
                    # 缩放图像
                    frame = cv2.resize(frame, (image_size, image_size))
                    frame = frame.astype(np.float32) / 255.0
                    frame = np.transpose(frame, (2, 0, 1))  # HWC -> CHW
                    self.images.append(frame)
                    self.actions.append(joints[frame_idx])
                frame_idx += 1
            cap.release()
        
        self.images = np.array(self.images, dtype=np.float32)
        self.actions = np.array(self.actions, dtype=np.float32)
        print(f"  加载完成: {len(self.images)} 帧, 来自 {num_demos} 个演示")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return torch.tensor(self.images[idx]), torch.tensor(self.actions[idx])


# ============================================================
# 策略网络
# ============================================================
class GraspPolicy(nn.Module):
    """
    视觉行为克隆策略
    输入: RGB图像 (3, 84, 84)
    输出: 16个关节角度
    """
    
    def __init__(self, action_dim=16):
        super().__init__()
        
        # CNN 视觉编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),   # 84 -> 42
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 42 -> 21
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), # 21 -> 11
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), # 11 -> 6
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),                      # 6 -> 1
            nn.Flatten(),
        )
        
        # MLP 动作头
        self.action_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )
    
    def forward(self, x):
        features = self.encoder(x)
        action = self.action_head(features)
        return action


# ============================================================
# 训练
# ============================================================
def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    
    # 数据集
    dataset = GraspDemoDataset(args.data_path, args.num_demos, args.image_size)
    
    # 划分训练/验证
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    print(f"  训练集: {train_size} 帧")
    print(f"  验证集: {val_size} 帧")
    
    # 模型
    model = GraspPolicy(action_dim=16).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.MSELoss()
    
    # 计算参数量
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量: {num_params:,}")
    
    # 训练循环
    os.makedirs(args.save_path, exist_ok=True)
    best_val_loss = float('inf')
    
    print(f"\n开始训练 ({args.epochs} epochs)...")
    print("-" * 50)
    
    for epoch in range(args.epochs):
        # 训练
        model.train()
        train_loss = 0
        for images, actions in train_loader:
            images, actions = images.to(device), actions.to(device)
            
            pred_actions = model(images)
            loss = criterion(pred_actions, actions)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, actions in val_loader:
                images, actions = images.to(device), actions.to(device)
                pred_actions = model(images)
                loss = criterion(pred_actions, actions)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        scheduler.step()
        
        # 打印进度
        if (epoch + 1) % 10 == 0 or epoch == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch+1:3d}/{args.epochs} | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f} | "
                  f"LR: {lr:.6f}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(args.save_path, 'grasp_policy_best.pt'))
    
    # 保存最终模型
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'val_loss': val_loss,
    }, os.path.join(args.save_path, 'grasp_policy.pt'))
    
    print("-" * 50)
    print(f"训练完成!")
    print(f"  最佳验证损失: {best_val_loss:.6f}")
    print(f"  模型保存在: {args.save_path}/grasp_policy.pt")
    print(f"  最佳模型: {args.save_path}/grasp_policy_best.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='extracted_data', help='演示数据路径')
    parser.add_argument('--num_demos', type=int, default=10, help='使用的演示数量')
    parser.add_argument('--save_path', type=str, default='trained_models', help='模型保存路径')
    parser.add_argument('--epochs', type=int, default=200, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=64, help='批大小')
    parser.add_argument('--lr', type=float, default=3e-4, help='学习率')
    parser.add_argument('--image_size', type=int, default=84, help='输入图像大小')
    args = parser.parse_args()
    
    train(args)
