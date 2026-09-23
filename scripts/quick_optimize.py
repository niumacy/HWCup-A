"""
快速优化: 标准 CE + 增强 Neutral 权重 + 更大模型
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

sys.path.insert(0, '/mnt/data2/home/chenyu/HuaWeiCup/huaweibei-E')
from src.models.baseline_model import LateFusionBaseline, GatedLateFusion
from src.models.msa_gmoe import build_msa_gmoe
from src.data_loader import load_attachment2


class MultimodalDataset(Dataset):
    def __init__(self, data_dict):
        self.text = torch.from_numpy(np.array(data_dict['text']).astype(np.float32))
        self.audio = torch.from_numpy(np.array(data_dict['audio']).astype(np.float32))
        self.vision = torch.from_numpy(np.array(data_dict['vision']).astype(np.float32))
        # clip audio异常值
        self.audio = torch.clamp(self.audio, min=-100, max=100)
        self.cls_labels = torch.from_numpy(np.array(data_dict['classification_labels']).astype(np.int64))
        self.reg_labels = torch.from_numpy(np.array(data_dict['regression_labels']).astype(np.float32))
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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(model, dataloader, optimizer, device, cls_weight, reg_weight):
    model.train()
    total_cls, total_reg, correct, total = 0, 0, 0, 0
    for batch in dataloader:
        text, audio, vision = batch['text'].to(device), batch['audio'].to(device), batch['vision'].to(device)
        cls_label, reg_label = batch['cls_label'].to(device), batch['reg_label'].to(device)
        
        optimizer.zero_grad()
        cls_logits, reg_pred = model(text, audio, vision)
        
        cls_loss = F.cross_entropy(cls_logits, cls_label, weight=cls_weight)
        reg_loss = F.smooth_l1_loss(reg_pred, reg_label)
        loss = cls_weight.sum() * 0.01 * cls_loss + reg_weight * reg_loss  # 调整权重
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_cls += cls_loss.item()
        total_reg += reg_loss.item()
        correct += (cls_logits.argmax(1) == cls_label).sum().item()
        total += cls_label.size(0)
    return total_cls/len(dataloader), total_reg/len(dataloader), correct/total


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    preds_cls, preds_reg, labels_cls, labels_reg = [], [], [], []
    for batch in dataloader:
        text, audio, vision = batch['text'].to(device), batch['audio'].to(device), batch['vision'].to(device)
        cls_logits, reg_pred = model(text, audio, vision)
        preds_cls.append(cls_logits.argmax(1).cpu().numpy())
        preds_reg.append(reg_pred.cpu().numpy())
        labels_cls.append(batch['cls_label'].numpy())
        labels_reg.append(batch['reg_label'].numpy())
    
    preds_cls = np.concatenate(preds_cls)
    preds_reg = np.concatenate(preds_reg)
    labels_cls = np.concatenate(labels_cls)
    labels_reg = np.concatenate(labels_reg)
    
    from sklearn.metrics import accuracy_score, f1_score
    acc = accuracy_score(labels_cls, preds_cls)
    f1 = f1_score(labels_cls, preds_cls, average='weighted')
    f1_macro = f1_score(labels_cls, preds_cls, average='macro')
    mae = np.abs(preds_reg - labels_reg).mean()
    from scipy.stats import pearsonr
    corr, _ = pearsonr(preds_reg, labels_reg)
    
    return {'Accuracy': acc, 'F1': f1, 'F1_macro': f1_macro, 'MAE': mae, 'Pearson': corr}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='E题数据/附件2-数据集特征文件/aligned_50.pkl')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--hidden_dim', type=int, default=384)
    parser.add_argument('--dropout', type=float, default=0.35)
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--model', type=str, default='latefusion', choices=['latefusion', 'glf'])
    parser.add_argument('--cls_weight_mult', type=float, default=1.5, help='Neutral类权重倍数')
    parser.add_argument('--output_dir', type=str, default='huaweibei-E/outputs')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ {device}")
    set_seed(args.seed)
    
    data = load_attachment2(args.data)
    train_loader = DataLoader(MultimodalDataset(data['train']), batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(MultimodalDataset(data['valid']), batch_size=args.batch_size)
    test_loader = DataLoader(MultimodalDataset(data['test']), batch_size=args.batch_size)
    
    # 模型
    if args.model == 'latefusion':
        model = LateFusionBaseline(text_dim=768, audio_dim=74, vision_dim=35,
                                  hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    else:
        model = GatedLateFusion(text_dim=768, audio_dim=74, vision_dim=35,
                                hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    
    print(f"模型: {args.model}, hidden={args.hidden_dim}, dropout={args.dropout}")
    
    # 增强的类别权重
    cls_counts = np.bincount(np.array(data['train']['classification_labels'], dtype=np.int64), minlength=3).astype(np.float32)
    weights = 1.0 / cls_counts
    weights = weights / weights.sum() * 3  # 归一化
    weights[1] *= args.cls_weight_mult  # 增强 Neutral
    weights = torch.tensor(weights, dtype=torch.float32, device=device)
    print(f"类别权重: Neg={weights[0]:.2f}, Neu={weights[1]:.2f}, Pos={weights[2]:.2f}")
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2, eta_min=1e-6)
    
    best_acc = 0
    best_test = None
    tag = f"{args.model}_h{args.hidden_dim}_n{args.cls_weight_mult}_s{args.seed}"
    
    for epoch in range(args.epochs):
        cls_loss, reg_loss, train_acc = train_epoch(model, train_loader, optimizer, device, weights, 0.5)
        scheduler.step()
        
        val_m = evaluate(model, valid_loader, device)
        test_m = evaluate(model, test_loader, device)
        
        marker = " ★" if val_m['Accuracy'] > best_acc else ""
        best_acc = max(best_acc, val_m['Accuracy'])
        
        if marker:
            best_test = test_m
            torch.save({'epoch': epoch, 'model': model.state_dict(), 'val': val_m, 'test': test_m},
                      f'{args.output_dir}/{tag}_best.pt')
        
        print(f"Epoch {epoch+1:2d}/{args.epochs} | "
              f"train={train_acc:.4f} | val:Acc={val_m['Accuracy']:.4f} F1={val_m['F1']:.4f} "
              f"MAE={val_m['MAE']:.4f} | test:Acc={test_m['Accuracy']:.4f} F1={test_m['F1']:.4f}" + marker)
    
    print(f"\n{'='*60}\n最佳结果: {tag}\n{'='*60}")
    print(f"Accuracy: {best_test['Accuracy']:.4f}")
    print(f"F1 (weighted): {best_test['F1']:.4f}")
    print(f"F1 (macro): {best_test['F1_macro']:.4f}")
    print(f"MAE: {best_test['MAE']:.4f}")
    print(f"Pearson: {best_test['Pearson']:.4f}")
    print(f"提升 vs 基线(0.6864): {best_test['Accuracy'] - 0.6864:+.4f}")
    
    # 保存报告
    os.makedirs(f'{args.output_dir}/reports', exist_ok=True)
    with open(f'{args.output_dir}/reports/{tag}_report.json', 'w') as f:
        json.dump({'tag': tag, 'test': {k: float(v) for k, v in best_test.items()}, 'args': vars(args)}, f, indent=2)
    print(f"✅ 完成!")


if __name__ == '__main__':
    main()
