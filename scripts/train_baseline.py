"""
Baseline 模型训练脚本
直接在附件2上训练，验证 pipeline 可用
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
import argparse
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.baseline_model import build_model
from src.data_loader import load_attachment2


# ── 数据集 ─────────────────────────────────────────────────────────────────────

class MultimodalDataset(Dataset):
    def __init__(self, data_dict):
        self.text = torch.from_numpy(np.array(data_dict['text']).astype(np.float32))
        self.audio = torch.from_numpy(np.array(data_dict['audio']).astype(np.float32))
        self.vision = torch.from_numpy(np.array(data_dict['vision']).astype(np.float32))
        self.cls_labels = torch.from_numpy(np.array(data_dict['classification_labels']).astype(np.int64))
        self.reg_labels = torch.from_numpy(np.array(data_dict['regression_labels']).astype(np.float32))
        self.raw_text = data_dict['raw_text']
        self.ids = data_dict['id']

    def __len__(self):
        return len(self.cls_labels)

    def __getitem__(self, idx):
        return {
            'text': self.text[idx],
            'audio': self.audio[idx],
            'vision': self.vision[idx],
            'cls_label': self.cls_labels[idx],
            'reg_label': self.reg_labels[idx],
            'id': self.ids[idx],
        }


# ── 训练函数 ──────────────────────────────────────────────────────────────────

def train_epoch(model, dataloader, optimizer, device, cls_weight=1.0, reg_weight=1.0):
    model.train()
    total_cls_loss = 0
    total_reg_loss = 0
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

        # 多任务损失
        cls_loss = nn.CrossEntropyLoss()(cls_logits, cls_label)
        reg_loss = nn.SmoothL1Loss()(reg_pred, reg_label)
        loss = cls_weight * cls_loss + reg_weight * reg_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_cls_loss += cls_loss.item()
        total_reg_loss += reg_loss.item()
        correct += (cls_logits.argmax(dim=1) == cls_label).sum().item()
        total += cls_label.size(0)

    acc = correct / total
    avg_cls_loss = total_cls_loss / len(dataloader)
    avg_reg_loss = total_reg_loss / len(dataloader)
    return avg_cls_loss, avg_reg_loss, acc


def evaluate(model, dataloader, device):
    model.eval()
    all_preds_cls = []
    all_preds_reg = []
    all_labels_cls = []
    all_labels_reg = []

    with torch.no_grad():
        for batch in dataloader:
            text = batch['text'].to(device)
            audio = batch['audio'].to(device)
            vision = batch['vision'].to(device)
            cls_label = batch['cls_label']
            reg_label = batch['reg_label']

            cls_logits, reg_pred = model(text, audio, vision)

            all_preds_cls.append(cls_logits.argmax(dim=1).cpu().numpy())
            all_preds_reg.append(reg_pred.cpu().numpy())
            all_labels_cls.append(cls_label.numpy())
            all_labels_reg.append(reg_label.numpy())

    all_preds_cls = np.concatenate(all_preds_cls)
    all_preds_reg = np.concatenate(all_preds_reg)
    all_labels_cls = np.concatenate(all_labels_cls)
    all_labels_reg = np.concatenate(all_labels_reg)

    # 分类指标
    from sklearn.metrics import accuracy_score, f1_score
    acc = accuracy_score(all_labels_cls, all_preds_cls)
    f1 = f1_score(all_labels_cls, all_preds_cls, average='weighted')

    # 回归指标
    mae = np.abs(all_preds_reg - all_labels_reg).mean()
    from scipy.stats import pearsonr
    corr, _ = pearsonr(all_preds_reg, all_labels_reg)

    return {
        'Accuracy': acc,
        'F1': f1,
        'MAE': mae,
        'Pearson': corr,
    }


def main():
    parser = argparse.ArgumentParser(description='Baseline 模型训练')
    parser.add_argument('--data', type=str, default='huaweibei-E/data/raw_attachment2/aligned_50.pkl')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--model', type=str, default='baseline',
                        choices=['baseline', 'attention'])
    parser.add_argument('--output_dir', type=str, default='huaweibei-E/outputs')
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    # 设备
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  设备: {device}")

    # 加载数据
    print(f"📂 加载数据: {args.data}")
    data = load_attachment2(args.data)
    train_data = data['train']
    valid_data = data['valid']
    test_data = data['test']

    train_loader = DataLoader(MultimodalDataset(train_data), batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(MultimodalDataset(valid_data), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(MultimodalDataset(test_data), batch_size=args.batch_size, shuffle=False)

    print(f"   train: {len(train_data['id'])} | valid: {len(valid_data['id'])} | test: {len(test_data['id'])}")

    # 模型
    model = build_model(
        args.model,
        text_dim=768,
        audio_dim=74,
        vision_dim=35,
        hidden_dim=args.hidden_dim,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   模型参数: {total_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_f1 = 0
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("开始训练")
    print("=" * 70)

    for epoch in range(args.epochs):
        cls_loss, reg_loss, train_acc = train_epoch(
            model, train_loader, optimizer, device
        )
        scheduler.step()

        val_metrics = evaluate(model, valid_loader, device)

        marker = " ★" if val_metrics['F1'] > best_val_f1 else ""
        best_val_f1 = max(best_val_f1, val_metrics['F1'])

        print(f"Epoch {epoch+1:2d}/{args.epochs} | "
              f"cls_loss={cls_loss:.4f} reg_loss={reg_loss:.4f} | "
              f"train_acc={train_acc:.4f} | "
              f"val: Acc={val_metrics['Accuracy']:.4f} F1={val_metrics['F1']:.4f} "
              f"MAE={val_metrics['MAE']:.4f} Pearson={val_metrics['Pearson']:.4f}"
              + marker)

        if marker:  # 保存最佳模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
            }, f'{args.output_dir}/baseline_best.pt')
            print(f"   💾 已保存最佳模型")

    # 最终测试
    print("\n" + "=" * 70)
    print("测试集评估")
    print("=" * 70)
    checkpoint = torch.load(f'{args.output_dir}/baseline_best.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    test_metrics = evaluate(model, test_loader, device)

    print(f"   Accuracy: {test_metrics['Accuracy']:.4f}")
    print(f"   F1 (weighted): {test_metrics['F1']:.4f}")
    print(f"   MAE: {test_metrics['MAE']:.4f}")
    print(f"   Pearson: {test_metrics['Pearson']:.4f}")

    print("\n✅ Baseline 训练完成!")


if __name__ == '__main__':
    main()
