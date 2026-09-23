"""
快速优化方案: F1错误分析 + Focal Loss + 特征归一化
目标: Acc 从 0.6864 提升到 0.70+
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
import argparse
import os
import sys
import json
from datetime import datetime

# ── Focal Loss ─────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """Focal Loss: 聚焦难分样本，降低易分样本的权重"""
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha  # 类别权重
        self.gamma = gamma
    
    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # p_t
        focal_weight = (1 - pt) ** self.gamma
        focal_loss = focal_weight * ce_loss
        
        if self.alpha is not None:
            if self.alpha.device != logits.device:
                self.alpha = self.alpha.to(logits.device)
            focal_loss = self.alpha[targets] * focal_loss
        
        return focal_loss.mean()


class BoundaryAwareFocalLoss(nn.Module):
    """边界感知 Focal Loss: 边界样本额外加权"""
    def __init__(self, alpha=None, gamma=2.0, boundary_threshold=0.3, boundary_weight=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.boundary_threshold = boundary_threshold
        self.boundary_weight = boundary_weight
    
    def forward(self, cls_logits, targets, reg_labels=None):
        # Focal Loss
        ce_loss = F.cross_entropy(cls_logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_weight = (1 - pt) ** self.gamma
        focal_loss = focal_weight * ce_loss
        
        if self.alpha is not None:
            if self.alpha.device != cls_logits.device:
                self.alpha = self.alpha.to(cls_logits.device)
            focal_loss = self.alpha[targets] * focal_loss
        
        # 边界样本额外加权
        if reg_labels is not None:
            is_boundary = (reg_labels.abs() < self.boundary_threshold).float()
            boundary_mask = 1.0 + is_boundary * (self.boundary_weight - 1.0)
            focal_loss = focal_loss * boundary_mask
        
        return focal_loss.mean()


# ── 特征归一化 ─────────────────────────────────────────────────────────────────
def normalize_features(batch):
    """对每个模态做 L2 归一化"""
    text, audio, vision = batch['text'], batch['audio'], batch['vision']
    
    # 文本: (B, T, 768) -> 沿最后一维归一化
    text_norm = torch.norm(text, p=2, dim=-1, keepdim=True) + 1e-8
    text = text / text_norm
    
    # 音频: (B, T, 74)
    audio_norm = torch.norm(audio, p=2, dim=-1, keepdim=True) + 1e-8
    audio = audio / audio_norm
    
    # 视觉: (B, T, 35)
    vision_norm = torch.norm(vision, p=2, dim=-1, keepdim=True) + 1e-8
    vision = vision / vision_norm
    
    return text, audio, vision


def clip_audio_features(audio, clip_value=100.0):
    """语音特征裁剪，去除异常值"""
    return torch.clamp(audio, min=-clip_value, max=clip_value)


# ── 数据集 ─────────────────────────────────────────────────────────────────────
class MultimodalDataset(Dataset):
    def __init__(self, data_dict, normalize=False, clip_audio=True):
        self.text = np.array(data_dict['text']).astype(np.float32)
        self.audio = np.array(data_dict['audio']).astype(np.float32)
        self.vision = np.array(data_dict['vision']).astype(np.float32)
        
        # 语音特征裁剪
        if clip_audio:
            self.audio = np.clip(self.audio, -100, 100)
        
        self.cls_labels = np.array(data_dict['classification_labels']).astype(np.int64)
        self.reg_labels = np.array(data_dict['regression_labels']).astype(np.float32)
        self.raw_text = data_dict['raw_text']
        self.ids = data_dict['id']
        self.normalize = normalize

    def __len__(self):
        return len(self.cls_labels)

    def __getitem__(self, idx):
        text = torch.from_numpy(self.text[idx])
        audio = torch.from_numpy(self.audio[idx])
        vision = torch.from_numpy(self.vision[idx])
        
        # 训练时归一化
        if self.normalize:
            text, audio, vision = normalize_features({
                'text': text, 'audio': audio, 'vision': vision
            })
        
        return {
            'text': text,
            'audio': audio,
            'vision': vision,
            'cls_label': torch.tensor(self.cls_labels[idx]),
            'reg_label': torch.tensor(self.reg_labels[idx]),
            'id': self.ids[idx],
        }


# ── 模型 (复用现有架构) ─────────────────────────────────────────────────────────
sys.path.insert(0, '/mnt/data2/home/chenyu/HuaWeiCup/huaweibei-E')
from src.models.baseline_model import LateFusionBaseline, GatedLateFusion
from src.models.msa_gmoe import MSAGMoE, build_msa_gmoe
from src.data_loader import load_attachment2


# ── 训练函数 ──────────────────────────────────────────────────────────────────
def train_epoch(model, dataloader, optimizer, device, 
                criterion_cls, criterion_reg,
                cls_weight=1.0, reg_weight=0.5,
                use_normalize=False):
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

        # 归一化 (测试时用)
        if use_normalize:
            text, audio, vision = normalize_features(batch)
            text = text.to(device)
            audio = audio.to(device)
            vision = vision.to(device)

        optimizer.zero_grad()
        cls_logits, reg_pred = model(text, audio, vision)

        cls_loss = criterion_cls(cls_logits, cls_label)
        reg_loss = criterion_reg(reg_pred, reg_label)
        loss = cls_weight * cls_loss + reg_weight * reg_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_cls_loss += cls_loss.item()
        total_reg_loss += reg_loss.item()
        correct += (cls_logits.argmax(dim=1) == cls_label).sum().item()
        total += cls_label.size(0)

    return total_cls_loss / len(dataloader), total_reg_loss / len(dataloader), correct / total


@torch.no_grad()
def evaluate(model, dataloader, device, return_confusion=False):
    model.eval()
    all_preds_cls = []
    all_preds_reg = []
    all_labels_cls = []
    all_labels_reg = []

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

    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
    
    acc = accuracy_score(all_labels_cls, all_preds_cls)
    f1 = f1_score(all_labels_cls, all_preds_cls, average='weighted')
    f1_macro = f1_score(all_labels_cls, all_preds_cls, average='macro')
    mae = np.abs(all_preds_reg - all_labels_reg).mean()
    from scipy.stats import pearsonr
    corr, _ = pearsonr(all_preds_reg, all_labels_reg)

    result = {
        'Accuracy': acc,
        'F1': f1,
        'F1_macro': f1_macro,
        'MAE': mae,
        'Pearson': corr,
    }
    
    if return_confusion:
        cm = confusion_matrix(all_labels_cls, all_preds_cls)
        result['confusion_matrix'] = cm.tolist()
        # 详细错误分析
        result['per_class_acc'] = (cm.diagonal() / cm.sum(axis=1)).tolist()
        result['total_samples'] = len(all_labels_cls)
        result['correct'] = int((all_preds_cls == all_labels_cls).sum())
        result['errors'] = int((all_preds_cls != all_labels_cls).sum())
    
    return result


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


import random


# ── 主程序 ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Focal Loss + 特征归一化优化')
    parser.add_argument('--data', type=str, default='E题数据/附件2-数据集特征文件/aligned_50.pkl')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=8e-4)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model', type=str, default='latefusion', 
                       choices=['latefusion', 'glf', 'msa_gmoe'])
    parser.add_argument('--normalize', action='store_true', default=True, help='启用特征归一化')
    parser.add_argument('--focal_gamma', type=float, default=2.0, help='Focal Loss gamma')
    parser.add_argument('--boundary_weight', type=float, default=2.0, help='边界样本加权倍数')
    parser.add_argument('--cls_weight', type=float, default=1.0, help='分类损失权重')
    parser.add_argument('--reg_weight', type=float, default=0.5, help='回归损失权重')
    parser.add_argument('--label_smoothing', type=float, default=0.05, help='标签平滑')
    parser.add_argument('--output_dir', type=str, default='huaweibei-E/outputs')
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    # 设备
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  设备: {device}")
    set_seed(args.seed)
    print(f"🎲 随机种子: {args.seed}")

    # 加载数据
    print(f"📂 加载数据: {args.data}")
    data = load_attachment2(args.data)
    train_loader = DataLoader(MultimodalDataset(data['train'], normalize=False), 
                            batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(MultimodalDataset(data['valid']), 
                             batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(MultimodalDataset(data['test']), 
                            batch_size=args.batch_size, shuffle=False)
    
    print(f"   train: {len(data['train']['id'])} | valid: {len(data['valid']['id'])} | test: {len(data['test']['id'])}")

    # 模型
    if args.model == 'latefusion':
        model = LateFusionBaseline(text_dim=768, audio_dim=74, vision_dim=35,
                                   hidden_dim=args.hidden_dim, dropout=0.35).to(device)
    elif args.model == 'glf':
        model = GatedLateFusion(text_dim=768, audio_dim=74, vision_dim=35,
                               hidden_dim=args.hidden_dim, dropout=0.35).to(device)
    elif args.model == 'msa_gmoe':
        model = build_msa_gmoe(text_dim=768, audio_dim=74, vision_dim=35,
                               hidden_dim=192, dropout=0.3).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   模型: {args.model} | 参数: {total_params:,}")
    print(f"   优化: normalize={args.normalize}, focal_gamma={args.focal_gamma}, boundary_weight={args.boundary_weight}")

    # 类别权重
    cls_counts = np.bincount(np.array(data['train']['classification_labels']).astype(np.int64))
    cls_weights = (1.0 / cls_counts) * cls_counts.sum() / len(cls_counts)
    cls_weight_tensor = torch.tensor(cls_weights, dtype=torch.float32, device=device)
    print(f"   类别权重: {cls_weights.round(2)}")

    # 损失函数: Focal Loss + 边界加权
    criterion_cls = BoundaryAwareFocalLoss(
        alpha=cls_weight_tensor, 
        gamma=args.focal_gamma,
        boundary_threshold=0.3,
        boundary_weight=args.boundary_weight
    )
    criterion_reg = nn.SmoothL1Loss()

    # 优化器 + 学习率调度
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=15, T_mult=2, eta_min=1e-6
    )

    best_val_acc = 0
    best_test = None
    os.makedirs(args.output_dir, exist_ok=True)

    tag = f"focal_{args.model}_norm{int(args.normalize)}_seed{args.seed}"
    print(f"\n{'='*70}\n开始训练\n{'='*70}")

    for epoch in range(args.epochs):
        cls_loss, reg_loss, train_acc = train_epoch(
            model, train_loader, optimizer, device,
            criterion_cls, criterion_reg,
            cls_weight=args.cls_weight, reg_weight=args.reg_weight
        )
        scheduler.step()

        val_metrics = evaluate(model, valid_loader, device)
        test_metrics = evaluate(model, test_loader, device)

        marker = " ★" if val_metrics['Accuracy'] > best_val_acc else ""
        best_val_acc = max(best_val_acc, val_metrics['Accuracy'])
        
        print(f"Epoch {epoch+1:2d}/{args.epochs} | "
              f"cls={cls_loss:.4f} reg={reg_loss:.4f} | "
              f"train={train_acc:.4f} | "
              f"val: Acc={val_metrics['Accuracy']:.4f} F1={val_metrics['F1']:.4f} "
              f"MAE={val_metrics['MAE']:.4f} | "
              f"test: Acc={test_metrics['Accuracy']:.4f} F1={test_metrics['F1']:.4f}"
              + marker)

        if marker:
            best_test = test_metrics
            best_test['epoch'] = epoch + 1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'test_metrics': test_metrics,
                'args': vars(args),
            }, f'{args.output_dir}/{tag}_best.pt')
            print(f"   💾 已保存 (val_acc={val_metrics['Accuracy']:.4f})")

    # 最终结果
    print("\n" + "=" * 70)
    print(f"最佳模型测试结果: {tag}")
    print("=" * 70)
    print(f"   Accuracy: {best_test['Accuracy']:.4f}")
    print(f"   F1 (weighted): {best_test['F1']:.4f}")
    print(f"   F1 (macro): {best_test['F1_macro']:.4f}")
    print(f"   MAE: {best_test['MAE']:.4f}")
    print(f"   Pearson: {best_test['Pearson']:.4f}")

    # 保存报告
    report = {
        'tag': tag,
        'best_val_acc': float(best_val_acc),
        'test_metrics': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                        for k, v in best_test.items()},
        'config': vars(args),
        'improvement_vs_baseline': {
            'baseline_acc': 0.6864,
            'current_acc': float(best_test['Accuracy']),
            'delta': float(best_test['Accuracy']) - 0.6864
        }
    }
    report_name = f'{args.output_dir}/reports/{tag}_report.json'
    os.makedirs(os.path.dirname(report_name), exist_ok=True)
    with open(report_name, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n📊 报告保存到: {report_name}")
    print(f"\n✅ 完成! Acc 提升: {best_test['Accuracy'] - 0.6864:+.4f}")


if __name__ == '__main__':
    main()
