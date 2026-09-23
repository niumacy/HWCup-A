"""
在Q1生成的特征上验证Baseline模型

用于:
1. 验证 Q1 特征的可用性
2. 检查与附件2 联合使用的接口一致性
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
import sys
import os

sys.path.insert(0, '/mnt/data2/home/chenyu/HuaWeiCup/huaweibei-E')
from src.models.baseline_model import build_model


class MultimodalDataset(Dataset):
    def __init__(self, data_dict):
        self.text = torch.from_numpy(np.array(data_dict['text']).astype(np.float32))
        self.audio = torch.from_numpy(np.array(data_dict['audio']).astype(np.float32))
        self.vision = torch.from_numpy(np.array(data_dict['vision']).astype(np.float32))
        self.cls_labels = torch.from_numpy(np.array(data_dict['classification_labels']).astype(np.int64))
        self.reg_labels = torch.from_numpy(np.array(data_dict['regression_labels']).astype(np.float32))

    def __len__(self):
        return len(self.cls_labels)

    def __getitem__(self, idx):
        return {
            'text': self.text[idx],
            'audio': self.audio[idx],
            'vision': self.vision[idx],
            'cls_label': self.cls_labels[idx],
            'reg_label': self.reg_labels[idx],
        }


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in dataloader:
        text = batch['text'].to(device)
        audio = batch['audio'].to(device)
        vision = batch['vision'].to(device)
        cls_label = batch['cls_label'].to(device)
        reg_label = batch['reg_label'].to(device)

        optimizer.zero_grad()
        cls_logits, reg_pred = model(text, audio, vision)

        cls_loss = nn.CrossEntropyLoss()(cls_logits, cls_label)
        reg_loss = nn.SmoothL1Loss()(reg_pred, reg_label)
        loss = cls_loss + reg_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (cls_logits.argmax(dim=1) == cls_label).sum().item()
        total += cls_label.size(0)

    return total_loss / len(dataloader), correct / total


def main():
    print("=" * 70)
    print("验证: Q1 特征在 Baseline 模型上的可用性")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载 Q1 生成的特征
    with open('/mnt/data2/home/chenyu/HuaWeiCup/huaweibei-E/data/processed/features_q1/q1_features_combined.pkl', 'rb') as f:
        q1_data = pickle.load(f)

    data = q1_data['all']
    print(f"\n📊 Q1 数据:")
    print(f"   样本数: {len(data['id'])}")
    print(f"   text shape: {np.array(data['text']).shape}")
    print(f"   audio shape: {np.array(data['audio']).shape}")
    print(f"   vision shape: {np.array(data['vision']).shape}")

    # 简单划分: 80% train, 20% test
    N = len(data['id'])
    indices = np.random.RandomState(42).permutation(N)
    train_size = int(N * 0.8)

    train_data = {k: (np.array(data[k])[indices[:train_size]] if hasattr(data[k], '__len__') and not isinstance(data[k], str) else [data[k][i] for i in indices[:train_size]]) for k in data.keys()}
    test_data = {k: (np.array(data[k])[indices[train_size:]] if hasattr(data[k], '__len__') and not isinstance(data[k], str) else [data[k][i] for i in indices[train_size:]]) for k in data.keys()}

    # 转 list (因为 text 等是 ndarray)
    for k in train_data:
        if isinstance(train_data[k], np.ndarray) and train_data[k].ndim == 0:
            train_data[k] = [train_data[k].item()]
        elif isinstance(train_data[k], np.ndarray):
            train_data[k] = train_data[k].tolist() if train_data[k].ndim == 1 else train_data[k]
        if isinstance(test_data[k], np.ndarray) and test_data[k].ndim == 0:
            test_data[k] = [test_data[k].item()]
        elif isinstance(test_data[k], np.ndarray):
            test_data[k] = test_data[k].tolist() if test_data[k].ndim == 1 else test_data[k]

    train_loader = DataLoader(MultimodalDataset(train_data), batch_size=16, shuffle=True)
    test_loader = DataLoader(MultimodalDataset(test_data), batch_size=16, shuffle=False)

    # 模型
    model = build_model('baseline', hidden_dim=128).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    print(f"\n🚀 训练 Baseline 模型 (Q1 特征, 50 epochs)")
    print("-" * 70)
    for epoch in range(50):
        loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}: loss={loss:.4f}, train_acc={train_acc:.4f}")

    # 评估
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            text = batch['text'].to(device)
            audio = batch['audio'].to(device)
            vision = batch['vision'].to(device)
            cls_label = batch['cls_label']

            cls_logits, _ = model(text, audio, vision)
            correct += (cls_logits.argmax(dim=1).cpu() == cls_label).sum().item()
            total += cls_label.size(0)

    test_acc = correct / total
    print(f"\n📈 测试集准确率: {test_acc:.4f}")
    print(f"\n✅ Q1 特征在 Baseline 上可训练!")
    print(f"   训练集: {train_size} 样本")
    print(f"   测试集: {N - train_size} 样本")


if __name__ == '__main__':
    main()
