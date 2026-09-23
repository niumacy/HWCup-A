"""
增强版 Baseline 训练 - V2

改进:
1. 多模型对比 (LateFusion / LSTM-Attention / TCN)
2. Cosine LR schedule with warmup
3. Label Smoothing
4. 早停 (Early Stopping)
5. 完整指标 (Acc, F1, Precision, Recall, MAE, Pearson, Corr)
6. 训练曲线 + 混淆矩阵

最终目标: Acc > 0.75, F1 > 0.70, MAE < 0.65, Pearson > 0.70
"""
import os
import sys
import json
import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, mean_absolute_error
)
from scipy.stats import pearsonr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.baseline_model import build_model
from src.data_loader import load_attachment2


# ──────────────────────────────────────────────────────────────────────────────
# 数据集
# ──────────────────────────────────────────────────────────────────────────────

class MultimodalDataset(Dataset):
    def __init__(self, data_dict):
        self.text = torch.from_numpy(np.array(data_dict['text']).astype(np.float32))
        self.audio = torch.from_numpy(np.array(data_dict['audio']).astype(np.float32))
        self.vision = torch.from_numpy(np.array(data_dict['vision']).astype(np.float32))
        self.cls_labels = torch.from_numpy(np.array(data_dict['classification_labels']).astype(np.int64))
        self.reg_labels = torch.from_numpy(np.array(data_dict['regression_labels']).astype(np.float32))
        self.ids = data_dict.get('id', [''] * len(self.cls_labels))

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


# ──────────────────────────────────────────────────────────────────────────────
# 训练工具
# ──────────────────────────────────────────────────────────────────────────────

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, num_classes=3, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=-1)
        # 平滑后的 one-hot
        smooth = torch.full_like(log_probs, self.smoothing / self.num_classes)
        smooth.scatter_(1, targets.unsqueeze(1), 1 - self.smoothing + self.smoothing / self.num_classes)
        loss = -(smooth * log_probs).sum(dim=-1).mean()
        return loss


def get_lr_scheduler(optimizer, total_steps, warmup_steps=0):
    """Warmup + Cosine Decay"""
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(model, dataloader, optimizer, criterion_cls, criterion_reg, device, cls_weight=0.7, grad_clip=1.0):
    model.train()
    total_loss = 0
    total_cls_loss = 0
    total_reg_loss = 0
    correct = 0
    total = 0
    batch_count = 0

    for batch in dataloader:
        text = batch['text'].to(device)
        audio = batch['audio'].to(device)
        vision = batch['vision'].to(device)
        cls_label = batch['cls_label'].to(device)
        reg_label = batch['reg_label'].to(device)

        optimizer.zero_grad()
        cls_logits, reg_pred = model(text, audio, vision)

        loss_cls = criterion_cls(cls_logits, cls_label)
        loss_reg = criterion_reg(reg_pred, reg_label)
        loss = cls_weight * loss_cls + (1 - cls_weight) * loss_reg

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        total_loss += loss.item()
        total_cls_loss += loss_cls.item()
        total_reg_loss += loss_reg.item()
        correct += (cls_logits.argmax(dim=1) == cls_label).sum().item()
        total += cls_label.size(0)
        batch_count += 1

    return {
        'loss': total_loss / batch_count,
        'cls_loss': total_cls_loss / batch_count,
        'reg_loss': total_reg_loss / batch_count,
        'acc': correct / total,
    }


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    all_cls_preds = []
    all_cls_logits = []
    all_reg_preds = []
    all_cls_labels = []
    all_reg_labels = []

    for batch in dataloader:
        text = batch['text'].to(device)
        audio = batch['audio'].to(device)
        vision = batch['vision'].to(device)
        cls_logits, reg_pred = model(text, audio, vision)

        all_cls_logits.append(cls_logits.cpu().numpy())
        all_cls_preds.append(cls_logits.argmax(dim=1).cpu().numpy())
        all_reg_preds.append(reg_pred.cpu().numpy())
        all_cls_labels.append(batch['cls_label'].numpy())
        all_reg_labels.append(batch['reg_label'].numpy())

    all_cls_preds = np.concatenate(all_cls_preds)
    all_cls_logits = np.concatenate(all_cls_logits)
    all_reg_preds = np.concatenate(all_reg_preds)
    all_cls_labels = np.concatenate(all_cls_labels)
    all_reg_labels = np.concatenate(all_reg_labels)

    # 分类指标
    acc = accuracy_score(all_cls_labels, all_cls_preds)
    f1 = f1_score(all_cls_labels, all_cls_preds, average='weighted')
    f1_macro = f1_score(all_cls_labels, all_cls_preds, average='macro')
    prec = precision_score(all_cls_labels, all_cls_preds, average='weighted', zero_division=0)
    rec = recall_score(all_cls_labels, all_cls_preds, average='weighted', zero_division=0)

    # 回归指标
    mae = mean_absolute_error(all_reg_labels, all_reg_preds)
    corr, _ = pearsonr(all_reg_preds, all_reg_labels)

    return {
        'Accuracy': acc,
        'F1_weighted': f1,
        'F1_macro': f1_macro,
        'Precision': prec,
        'Recall': rec,
        'MAE': mae,
        'Pearson': corr,
        'cm': confusion_matrix(all_cls_labels, all_cls_preds),
        'cls_logits': all_cls_logits,
        'reg_preds': all_reg_preds,
        'cls_labels': all_cls_labels,
        'reg_labels': all_reg_labels,
    }


def plot_curves(history, save_path):
    """绘制训练曲线"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Training Curves (best val F1: {history['best_val_f1']:.4f})", fontsize=14)

    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss', color='blue')
    axes[0, 0].plot(history['val_loss'], label='Val Loss', color='red')
    axes[0, 0].set_title('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # Accuracy
    axes[0, 1].plot(history['train_acc'], label='Train Acc', color='blue')
    axes[0, 1].plot(history['val_acc'], label='Val Acc', color='red')
    axes[0, 1].set_title('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # F1
    axes[0, 2].plot(history['val_f1'], label='Val F1', color='green')
    axes[0, 2].set_title('Val F1 (weighted)')
    axes[0, 2].legend()
    axes[0, 2].grid(alpha=0.3)

    # MAE
    axes[1, 0].plot(history['val_mae'], label='Val MAE', color='orange')
    axes[1, 0].set_title('Val MAE')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    # Pearson
    axes[1, 1].plot(history['val_pearson'], label='Val Pearson', color='purple')
    axes[1, 1].set_title('Val Pearson')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    # Learning rate
    axes[1, 2].plot(history['lr'], label='LR', color='brown')
    axes[1, 2].set_title('Learning Rate')
    axes[1, 2].legend()
    axes[1, 2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   📊 训练曲线已保存: {save_path}")


def plot_confusion_matrix(cm, save_path):
    """绘制混淆矩阵"""
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=['Neg', 'Neu', 'Pos'], yticklabels=['Neg', 'Neu', 'Pos'],
           title='Confusion Matrix',
           ylabel='True', xlabel='Predicted')
    # 标注
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   📊 混淆矩阵已保存: {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# 主训练
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='增强版 Baseline 训练 (V2)')
    parser.add_argument('--data', type=str,
                        default='/mnt/data2/home/chenyu/HuaWeiCup/E题数据/附件2-数据集特征文件/aligned_50.pkl')
    parser.add_argument('--model', type=str, default='latefusion',
                        choices=['latefusion', 'lstmattn', 'tcn', 'gmt', 'glf'],
                        help='模型类型: latefusion / lstmattn / tcn / gmt / glf')
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.4)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--cls_weight', type=float, default=0.7)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--early_stop_patience', type=int, default=15)
    parser.add_argument('--use_q1', action='store_true', help='拼接 Q1 100 条样本到训练集')
    parser.add_argument('--q1_path', type=str,
                        default='/mnt/data2/home/chenyu/HuaWeiCup/huaweibei-E/data/processed/features_q1/q1_features_combined.pkl')
    parser.add_argument('--output_dir', type=str,
                        default='/mnt/data2/home/chenyu/HuaWeiCup/huaweibei-E/outputs')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # 随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  设备: {device}")

    # 加载数据
    data = load_attachment2(args.data)
    train_data = data['train']
    valid_data = data['valid']
    test_data = data['test']

    # 可选: 拼接 Q1 数据到训练集
    if args.use_q1 and os.path.exists(args.q1_path):
        with open(args.q1_path, 'rb') as f:
            q1_data = pickle.load(f)['all']
        print(f"\n📦 拼接 Q1 数据: {len(q1_data['id'])} 条")
        # 合并 train + q1
        train_data = {
            k: (
                np.concatenate([np.array(train_data[k]), np.array(q1_data[k])]) if k not in ['id', 'raw_text', 'annotation']
                else list(train_data[k]) + list(q1_data[k])
            )
            for k in train_data.keys()
        }

    train_loader = DataLoader(MultimodalDataset(train_data), batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(MultimodalDataset(valid_data), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(MultimodalDataset(test_data), batch_size=args.batch_size, shuffle=False)

    print(f"\n📂 数据加载:")
    print(f"   train: {len(train_data['id'])}")
    print(f"   valid: {len(valid_data['id'])}")
    print(f"   test:  {len(test_data['id'])}")

    # 模型
    model_kwargs = dict(
        text_dim=768, audio_dim=74, vision_dim=35,
        hidden_dim=args.hidden_dim, num_classes=3, dropout=args.dropout
    )
    model = build_model(args.model, **model_kwargs).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n🏗️  模型: {args.model}")
    print(f"   总参数: {total_params:,}")
    print(f"   可训练: {trainable:,}")

    # 损失函数
    criterion_cls = LabelSmoothingCrossEntropy(num_classes=3, smoothing=args.label_smoothing)
    criterion_reg = nn.SmoothL1Loss()

    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Scheduler
    total_steps = len(train_loader) * args.epochs
    warmup_steps = len(train_loader) * args.warmup_epochs
    scheduler = get_lr_scheduler(optimizer, total_steps, warmup_steps)

    # 训练日志
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'val_f1': [], 'val_f1_macro': [],
        'val_mae': [], 'val_pearson': [],
        'val_precision': [], 'val_recall': [],
        'lr': [],
        'best_val_f1': 0,
        'best_epoch': 0,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    best_val_f1 = 0
    patience_counter = 0
    best_model_state = None

    print("\n" + "=" * 70)
    print("🚀 开始训练")
    print("=" * 70)

    for epoch in range(args.epochs):
        # 训练
        train_metrics = train_one_epoch(
            model, train_loader, optimizer,
            criterion_cls, criterion_reg, device,
            cls_weight=args.cls_weight,
            grad_clip=args.grad_clip,
        )
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # 验证 (用 CrossEntropy 评估, 不用 smoothing)
        val_metrics = evaluate(model, valid_loader, device)

        # 更新历史
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['acc'])
        history['val_loss'].append(val_metrics.get('loss', 0))
        history['val_acc'].append(val_metrics['Accuracy'])
        history['val_f1'].append(val_metrics['F1_weighted'])
        history['val_f1_macro'].append(val_metrics['F1_macro'])
        history['val_mae'].append(val_metrics['MAE'])
        history['val_pearson'].append(val_metrics['Pearson'])
        history['val_precision'].append(val_metrics['Precision'])
        history['val_recall'].append(val_metrics['Recall'])
        history['lr'].append(current_lr)

        is_best = val_metrics['F1_weighted'] > best_val_f1
        if is_best:
            best_val_f1 = val_metrics['F1_weighted']
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            history['best_val_f1'] = best_val_f1
            history['best_epoch'] = epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1

        # 打印
        marker = " ★ Best" if is_best else ""
        if (epoch + 1) % 5 == 0 or is_best or epoch == 0:
            print(f"Epoch {epoch+1:2d}/{args.epochs} | "
                  f"loss={train_metrics['loss']:.4f} | "
                  f"train_acc={train_metrics['acc']:.4f} | "
                  f"val: Acc={val_metrics['Accuracy']:.4f} F1={val_metrics['F1_weighted']:.4f} "
                  f"Prec={val_metrics['Precision']:.4f} Rec={val_metrics['Recall']:.4f} | "
                  f"MAE={val_metrics['MAE']:.4f} Pearson={val_metrics['Pearson']:.4f} "
                  f"| lr={current_lr:.2e}{marker}")

        # 早停
        if patience_counter >= args.early_stop_patience:
            print(f"\n⏹️  早停: {args.early_stop_patience} epoch 内 F1 未提升")
            break

    # 加载最佳模型, 在测试集上评估
    print("\n" + "=" * 70)
    print("📈 测试集评估 (使用最佳验证模型)")
    print("=" * 70)
    model.load_state_dict(best_model_state)

    test_metrics = evaluate(model, test_loader, device)

    # 打印
    print(f"\n   Accuracy:           {test_metrics['Accuracy']:.4f}")
    print(f"   F1 (weighted):      {test_metrics['F1_weighted']:.4f}")
    print(f"   F1 (macro):         {test_metrics['F1_macro']:.4f}")
    print(f"   Precision:          {test_metrics['Precision']:.4f}")
    print(f"   Recall:             {test_metrics['Recall']:.4f}")
    print(f"   MAE:                {test_metrics['MAE']:.4f}")
    print(f"   Pearson:            {test_metrics['Pearson']:.4f}")
    print(f"\n   混淆矩阵:")
    print(test_metrics['cm'])

    # 保存最佳模型
    model_path = f'{args.output_dir}/{args.model}_best.pt'
    torch.save({
        'model_state_dict': best_model_state,
        'model_kwargs': model_kwargs,
        'model_type': args.model,
        'test_metrics': {k: v for k, v in test_metrics.items() if k not in ['cm', 'cls_logits', 'reg_preds', 'cls_labels', 'reg_labels']},
        'history': history,
        'args': vars(args),
    }, model_path)
    print(f"\n💾 模型已保存: {model_path}")

    # 绘制曲线
    curves_path = f'{args.output_dir}/figures/{args.model}_training_curves.png'
    os.makedirs(os.path.dirname(curves_path), exist_ok=True)
    plot_curves(history, curves_path)

    # 绘制混淆矩阵
    cm_path = f'{args.output_dir}/figures/{args.model}_confusion_matrix.png'
    plot_confusion_matrix(test_metrics['cm'], cm_path)

    # 保存测试结果 JSON
    def to_python(obj):
        """numpy 转 python 原生类型"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: to_python(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [to_python(v) for v in obj]
        return obj

    report = {
        'model_type': args.model,
        'best_epoch': history['best_epoch'],
        'best_val_f1': history['best_val_f1'],
        'test_metrics': to_python({k: v for k, v in test_metrics.items()
                                    if k not in ['cls_logits', 'reg_preds', 'cls_labels', 'reg_labels']}),
        'config': vars(args),
        'timestamp': datetime.now().isoformat(),
    }
    json_path = f'{args.output_dir}/reports/{args.model}_report.json'
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"📋 报告已保存: {json_path}")

    print("\n✅ 训练完成!")
    return test_metrics


if __name__ == '__main__':
    main()
