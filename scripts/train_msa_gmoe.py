"""
训练 MSA-GMoE 模型

支持:
- 多 seed 训练
- Mixup 数据增强
- 对比学习正则化
- 多任务损失 (分类 + 回归)
- MoE 负载均衡损失
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
import random
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.msa_gmoe import MSAGMoE, build_msa_gmoe, contrastive_loss
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


# ── Mixup 增强 ────────────────────────────────────────────────────────────────

def mixup_batch(text, audio, vision, cls_label, reg_label, alpha=0.3):
    """Mixup 数据增强"""
    if alpha <= 0:
        return text, audio, vision, cls_label, reg_label, 1.0
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1.0 - lam)
    B = text.size(0)
    idx = torch.randperm(B, device=text.device)
    text_m = lam * text + (1 - lam) * text[idx]
    audio_m = lam * audio + (1 - lam) * audio[idx]
    vision_m = lam * vision + (1 - lam) * vision[idx]
    return text_m, audio_m, vision_m, cls_label, cls_label[idx], reg_label, reg_label[idx], lam


# ── 训练 / 评估函数 ───────────────────────────────────────────────────────────

def train_epoch(model, dataloader, optimizer, device,
                cls_weight=1.0, reg_weight=1.0,
                contrastive_weight=0.1, moe_balance_weight=0.1,
                mixup_alpha=0.3, use_contrastive=False, mixup_prob=0.5):
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

        # Mixup 增强 (随机概率)
        use_mixup = (mixup_alpha > 0) and (random.random() < mixup_prob)
        if use_mixup:
            text_in, audio_in, vision_in, cls_a, cls_b, reg_a, reg_b, lam = mixup_batch(
                text, audio, vision, cls_label, reg_label, mixup_alpha
            )
        else:
            text_in, audio_in, vision_in = text, audio, vision

        cls_logits, reg_pred, extras = model(text_in, audio_in, vision_in, return_gate_weights=True)

        # 分类损失 (mixup 时用混合标签)
        if use_mixup:
            cls_loss = lam * F.cross_entropy(cls_logits, cls_a) + (1 - lam) * F.cross_entropy(cls_logits, cls_b)
            reg_loss = lam * F.smooth_l1_loss(reg_pred, reg_a) + (1 - lam) * F.smooth_l1_loss(reg_pred, reg_b)
        else:
            cls_loss = F.cross_entropy(cls_logits, cls_label)
            reg_loss = F.smooth_l1_loss(reg_pred, reg_label)

        loss = cls_weight * cls_loss + reg_weight * reg_loss

        # MoE 负载均衡损失
        loss = loss + moe_balance_weight * extras['moe_balance_loss']

        # 对比学习损失 (仅 mixup 时不计算)
        if use_contrastive and not use_mixup and extras['contrastive_feats'] is not None:
            c_loss = contrastive_loss(extras['contrastive_feats'])
            loss = loss + contrastive_weight * c_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_cls_loss += cls_loss.item()
        total_reg_loss += reg_loss.item()
        correct += (cls_logits.argmax(dim=1) == cls_label).sum().item()
        total += cls_label.size(0)

    acc = correct / total
    return total_cls_loss / len(dataloader), total_reg_loss / len(dataloader), acc


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    all_preds_cls = []
    all_preds_reg = []
    all_labels_cls = []
    all_labels_reg = []
    all_gate_weights = []

    for batch in dataloader:
        text = batch['text'].to(device)
        audio = batch['audio'].to(device)
        vision = batch['vision'].to(device)
        cls_label = batch['cls_label']
        reg_label = batch['reg_label']

        cls_logits, reg_pred, extras = model(text, audio, vision, return_gate_weights=True)

        all_preds_cls.append(cls_logits.argmax(dim=1).cpu().numpy())
        all_preds_reg.append(reg_pred.cpu().numpy())
        all_labels_cls.append(cls_label.numpy())
        all_labels_reg.append(reg_label.numpy())
        if extras['gate_weights'] is not None:
            all_gate_weights.append(extras['gate_weights'].cpu().numpy())

    all_preds_cls = np.concatenate(all_preds_cls)
    all_preds_reg = np.concatenate(all_preds_reg)
    all_labels_cls = np.concatenate(all_labels_cls)
    all_labels_reg = np.concatenate(all_labels_reg)

    from sklearn.metrics import accuracy_score, f1_score
    acc = accuracy_score(all_labels_cls, all_preds_cls)
    f1 = f1_score(all_labels_cls, all_preds_cls, average='weighted')
    f1_macro = f1_score(all_labels_cls, all_preds_cls, average='macro')
    mae = np.abs(all_preds_reg - all_labels_reg).mean()
    from scipy.stats import pearsonr
    corr, _ = pearsonr(all_preds_reg, all_labels_reg)

    gate_info = None
    if all_gate_weights:
        gw = np.concatenate(all_gate_weights, axis=0)
        gate_info = gw.mean(axis=0).tolist()

    return {
        'Accuracy': acc,
        'F1': f1,
        'F1_macro': f1_macro,
        'MAE': mae,
        'Pearson': corr,
        'gate_weights_avg': gate_info,
    }


# ── 主程序 ─────────────────────────────────────────────────────────────────────

def set_seed(seed):
    """固定所有随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description='训练 MSA-GMoE 模型')
    parser.add_argument('--data', type=str, default='huaweibei-E/data/raw_attachment2/aligned_50.pkl')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=8e-4)
    parser.add_argument('--hidden_dim', type=int, default=192)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use_contrastive', action='store_true', default=False)
    parser.add_argument('--contrastive_weight', type=float, default=0.05)
    parser.add_argument('--moe_balance_weight', type=float, default=0.01)
    parser.add_argument('--reg_weight', type=float, default=0.5)
    parser.add_argument('--mixup_alpha', type=float, default=0.2)
    parser.add_argument('--mixup_prob', type=float, default=0.5)
    parser.add_argument('--output_dir', type=str,
                       default='/mnt/data2/home/chenyu/HuaWeiCup/huaweibei-E/outputs')
    parser.add_argument('--tag', type=str, default='msa_gmoe')
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    # 设备
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  设备: {device}")

    # 固定种子
    set_seed(args.seed)
    print(f"🎲 随机种子: {args.seed}")

    # 加载数据
    print(f"📂 加载数据: {args.data}")
    data = load_attachment2(args.data)
    train_loader = DataLoader(MultimodalDataset(data['train']),
                              batch_size=args.batch_size, shuffle=True, drop_last=False)
    valid_loader = DataLoader(MultimodalDataset(data['valid']),
                              batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(MultimodalDataset(data['test']),
                             batch_size=args.batch_size, shuffle=False)

    print(f"   train: {len(data['train']['id'])} | valid: {len(data['valid']['id'])} | test: {len(data['test']['id'])}")

    # 模型
    model = build_msa_gmoe(
        text_dim=768, audio_dim=74, vision_dim=35,
        hidden_dim=args.hidden_dim,
        num_heads=4,
        dropout=0.3,
        use_contrastive=args.use_contrastive,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   模型参数: {total_params:,}")

    # 类别权重 (处理类别不均衡)
    cls_counts = np.bincount(np.array(data['train']['classification_labels']).astype(np.int64))
    cls_weights = (1.0 / cls_counts) * cls_counts.sum() / len(cls_counts)
    cls_weight_tensor = torch.tensor(cls_weights, dtype=torch.float32, device=device)
    print(f"   类别权重: {cls_weights}")

    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    best_val_f1 = 0
    best_test = None
    os.makedirs(args.output_dir, exist_ok=True)

    save_name = f'{args.tag}_seed{args.seed}.pt'
    print(f"\n{'='*70}\n开始训练 (tag={args.tag}, seed={args.seed})\n{'='*70}")

    for epoch in range(args.epochs):
        cls_loss, reg_loss, train_acc = train_epoch(
            model, train_loader, optimizer, device,
            cls_weight=1.0, reg_weight=args.reg_weight,
            contrastive_weight=args.contrastive_weight,
            moe_balance_weight=args.moe_balance_weight,
            mixup_alpha=args.mixup_alpha,
            use_contrastive=args.use_contrastive,
            mixup_prob=args.mixup_prob,
        )
        scheduler.step()

        val_metrics = evaluate(model, valid_loader, device)
        # 当前 epoch 的测试评估 (仅记录)
        test_metrics = evaluate(model, test_loader, device)

        marker = " ★" if val_metrics['F1'] > best_val_f1 else ""
        best_val_f1 = max(best_val_f1, val_metrics['F1'])
        msg = (f"Epoch {epoch+1:2d}/{args.epochs} | "
              f"cls={cls_loss:.4f} reg={reg_loss:.4f} | "
              f"train_acc={train_acc:.4f} | "
              f"val: Acc={val_metrics['Accuracy']:.4f} F1={val_metrics['F1']:.4f} "
              f"MAE={val_metrics['MAE']:.4f} Pearson={val_metrics['Pearson']:.4f} | "
              f"test: Acc={test_metrics['Accuracy']:.4f} F1={test_metrics['F1']:.4f}"
              + marker)
        print(msg, flush=True)

        if marker:
            best_test = test_metrics
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'test_metrics': test_metrics,
                'args': vars(args),
            }, f'{args.output_dir}/{save_name}')
            print(f"   💾 已保存最佳模型 (val F1={val_metrics['F1']:.4f})")

    # 最终结果
    print("\n" + "=" * 70)
    print(f"最佳模型在测试集上的表现")
    print("=" * 70)
    if best_test:
        print(f"   Accuracy: {best_test['Accuracy']:.4f}")
        print(f"   F1 (weighted): {best_test['F1']:.4f}")
        print(f"   F1 (macro): {best_test['F1_macro']:.4f}")
        print(f"   MAE: {best_test['MAE']:.4f}")
        print(f"   Pearson: {best_test['Pearson']:.4f}")
        if best_test.get('gate_weights_avg'):
            print(f"   Gate weights (avg): {[f'{w:.3f}' for w in best_test['gate_weights_avg']]}")

    # 转换 float32 -> float 让 JSON 可序列化
    if best_test:
        for k, v in best_test.items():
            if isinstance(v, np.ndarray):
                best_test[k] = v.tolist()
            elif isinstance(v, (np.floating, np.integer)):
                best_test[k] = float(v)
    if best_test.get('gate_weights_avg') is not None:
        best_test['gate_weights_avg'] = [float(w) for w in best_test['gate_weights_avg']]

    # 保存报告
    report = {
        'tag': args.tag,
        'seed': args.seed,
        'best_val_f1': float(best_val_f1),
        'test_metrics': best_test,
        'config': vars(args),
        'total_params': int(total_params),
    }
    report_name = f'{args.output_dir}/reports/{args.tag}_seed{args.seed}_report.json'
    os.makedirs(os.path.dirname(report_name), exist_ok=True)
    with open(report_name, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n📊 报告保存到: {report_name}")

    print(f"\n✅ 训练完成!")


if __name__ == '__main__':
    main()
