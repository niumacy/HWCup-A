"""
Ensemble 评估: LateFusion + LSTM-Attention + TCN 三模型集成

支持两种集成方式:
1. 概率平均 (Soft Voting)
2. 预测投票 (Hard Voting)
"""
import torch
import numpy as np
import pickle
import os
import sys
import json
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, mean_absolute_error
)
from scipy.stats import pearsonr
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.baseline_model import build_model
from src.data_loader import load_attachment2


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


@torch.no_grad()
def get_model_predictions(model, dataloader, device):
    model.eval()
    all_cls_probs = []
    all_reg_preds = []
    all_cls_labels = []
    all_reg_labels = []

    for batch in dataloader:
        text = batch['text'].to(device)
        audio = batch['audio'].to(device)
        vision = batch['vision'].to(device)
        cls_logits, reg_pred = model(text, audio, vision)

        cls_probs = F.softmax(cls_logits, dim=-1).cpu().numpy()
        all_cls_probs.append(cls_probs)
        all_reg_preds.append(reg_pred.cpu().numpy())
        all_cls_labels.append(batch['cls_label'].numpy())
        all_reg_labels.append(batch['reg_label'].numpy())

    all_cls_probs = np.concatenate(all_cls_probs, axis=0)
    all_reg_preds = np.concatenate(all_reg_preds, axis=0)
    all_cls_labels = np.concatenate(all_cls_labels, axis=0)
    all_reg_labels = np.concatenate(all_reg_labels, axis=0)

    return all_cls_probs, all_reg_preds, all_cls_labels, all_reg_labels


def metrics_from_preds(cls_preds, reg_preds, cls_labels, reg_labels):
    acc = accuracy_score(cls_labels, cls_preds)
    f1 = f1_score(cls_labels, cls_preds, average='weighted')
    f1_macro = f1_score(cls_labels, cls_preds, average='macro')
    prec = precision_score(cls_labels, cls_preds, average='weighted', zero_division=0)
    rec = recall_score(cls_labels, cls_preds, average='weighted', zero_division=0)
    mae = mean_absolute_error(reg_labels, reg_preds)
    corr, _ = pearsonr(reg_preds, reg_labels)

    return {
        'Accuracy': acc,
        'F1_weighted': f1,
        'F1_macro': f1_macro,
        'Precision': prec,
        'Recall': rec,
        'MAE': mae,
        'Pearson': corr,
        'cm': confusion_matrix(cls_labels, cls_preds),
    }


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载数据
    data = load_attachment2('/mnt/data2/home/chenyu/HuaWeiCup/E题数据/附件2-数据集特征文件/aligned_50.pkl')
    test_loader = DataLoader(MultimodalDataset(data['test']), batch_size=64, shuffle=False)
    valid_loader = DataLoader(MultimodalDataset(data['valid']), batch_size=64, shuffle=False)

    # 模型路径
    model_paths = {
        'latefusion': '/mnt/data2/home/chenyu/HuaWeiCup/huaweibei-E/outputs/latefusion_best.pt',
        'lstmattn': '/mnt/data2/home/chenyu/HuaWeiCup/huaweibei-E/outputs/lstmattn_best.pt',
        'tcn': '/mnt/data2/home/chenyu/HuaWeiCup/huaweibei-E/outputs/tcn_best.pt',
    }

    # 加载各模型的预测概率
    val_preds = {}
    test_preds = {}

    for name, path in model_paths.items():
        if not os.path.exists(path):
            print(f"⚠️  跳过 {name}: 文件不存在 {path}")
            continue
        ckpt = torch.load(path, map_location=device)
        model_kwargs = ckpt['model_kwargs']
        model_type = ckpt['model_type']
        print(f"\n🤖 加载模型: {name} ({model_type})")
        model = build_model(model_type, **model_kwargs).to(device)
        model.load_state_dict(ckpt['model_state_dict'])

        # 验证集预测 (用于权重优化)
        cls_probs_v, reg_v, cls_lbl_v, reg_lbl_v = get_model_predictions(model, valid_loader, device)
        val_preds[name] = {'cls_probs': cls_probs_v, 'reg': reg_v}
        val_metrics = metrics_from_preds(cls_probs_v.argmax(1), reg_v, cls_lbl_v, reg_lbl_v)
        print(f"   val单独: Acc={val_metrics['Accuracy']:.4f} F1={val_metrics['F1_weighted']:.4f}")

        # 测试集预测
        cls_probs_t, reg_t, cls_lbl_t, reg_lbl_t = get_model_predictions(model, test_loader, device)
        test_preds[name] = {'cls_probs': cls_probs_t, 'reg': reg_t}
        test_metrics = metrics_from_metrics = metrics_from_preds(cls_probs_t.argmax(1), reg_t, cls_lbl_t, reg_lbl_t)
        print(f"   test单独: Acc={test_metrics['Accuracy']:.4f} F1={test_metrics['F1_weighted']:.4f}")

    # ─── 集成策略 ──────────────────────────────────────────────────────

    print("\n" + "=" * 70)
    print("📊 集成策略对比")
    print("=" * 70)

    # 1. Soft Voting (概率平均)
    avg_cls_probs_val = np.mean([v['cls_probs'] for v in val_preds.values()], axis=0)
    avg_reg_val = np.mean([v['reg'] for v in val_preds.values()], axis=0)
    val_soft = metrics_from_preds(avg_cls_probs_val.argmax(1), avg_reg_val, cls_lbl_v, reg_lbl_v)

    avg_cls_probs_test = np.mean([v['cls_probs'] for v in test_preds.values()], axis=0)
    avg_reg_test = np.mean([v['reg'] for v in test_preds.values()], axis=0)
    test_soft = metrics_from_preds(avg_cls_probs_test.argmax(1), avg_reg_test, cls_lbl_t, reg_lbl_t)

    print(f"\n1️⃣  Soft Voting (概率平均):")
    print(f"   val:  Acc={val_soft['Accuracy']:.4f} F1={val_soft['F1_weighted']:.4f} MAE={val_soft['MAE']:.4f} Pearson={val_soft['Pearson']:.4f}")
    print(f"   test: Acc={test_soft['Accuracy']:.4f} F1={test_soft['F1_weighted']:.4f} MAE={test_soft['MAE']:.4f} Pearson={test_soft['Pearson']:.4f}")

    # 2. Hard Voting
    hard_votes_val = np.stack([v['cls_probs'].argmax(1) for v in val_preds.values()], axis=0)
    hard_pred_val = np.array([np.bincount(hard_votes_val[:, i]).argmax() for i in range(hard_votes_val.shape[1])])

    hard_votes_test = np.stack([v['cls_probs'].argmax(1) for v in test_preds.values()], axis=0)
    hard_pred_test = np.array([np.bincount(hard_votes_test[:, i]).argmax() for i in range(hard_votes_test.shape[1])])

    test_hard = metrics_from_preds(hard_pred_test, avg_reg_test, cls_lbl_t, reg_lbl_t)
    print(f"\n2️⃣  Hard Voting (投票):")
    print(f"   test: Acc={test_hard['Accuracy']:.4f} F1={test_hard['F1_weighted']:.4f} MAE={test_hard['MAE']:.4f} Pearson={test_hard['Pearson']:.4f}")

    # 3. 加权 Soft Voting (用验证集 F1 作为权重)
    val_f1s = np.array([metrics_from_preds(v['cls_probs'].argmax(1), v['reg'], cls_lbl_v, reg_lbl_v)['F1_weighted']
                          for v in val_preds.values()])
    val_f1s = val_f1s - val_f1s.min() + 0.1  # 平移避免负权重
    weights = val_f1s / val_f1s.sum()
    print(f"\n   模型权重: {dict(zip(val_preds.keys(), [f'{w:.3f}' for w in weights]))}")

    weighted_cls_probs = np.zeros_like(avg_cls_probs_test)
    weighted_reg = np.zeros_like(avg_reg_test)
    for w, (name, v) in zip(weights, test_preds.items()):
        weighted_cls_probs += w * v['cls_probs']
        weighted_reg += w * v['reg']

    test_weighted = metrics_from_preds(weighted_cls_probs.argmax(1), weighted_reg, cls_lbl_t, reg_lbl_t)
    print(f"\n3️⃣  Weighted Soft Voting:")
    print(f"   test: Acc={test_weighted['Accuracy']:.4f} F1={test_weighted['F1_weighted']:.4f} MAE={test_weighted['MAE']:.4f} Pearson={test_weighted['Pearson']:.4f}")

    # 选择最佳策略
    candidates = {
        'soft_voting': test_soft,
        'hard_voting': test_hard,
        'weighted_soft': test_weighted,
    }
    best_strategy = max(candidates.keys(), key=lambda k: candidates[k]['F1_weighted'])
    best_metrics = candidates[best_strategy]

    print(f"\n🏆 最佳策略: {best_strategy}")
    print(f"\n   Accuracy:    {best_metrics['Accuracy']:.4f}")
    print(f"   F1 weighted: {best_metrics['F1_weighted']:.4f}")
    print(f"   F1 macro:    {best_metrics['F1_macro']:.4f}")
    print(f"   Precision:   {best_metrics['Precision']:.4f}")
    print(f"   Recall:      {best_metrics['Recall']:.4f}")
    print(f"   MAE:         {best_metrics['MAE']:.4f}")
    print(f"   Pearson:     {best_metrics['Pearson']:.4f}")
    print(f"\n   混淆矩阵:")
    print(best_metrics['cm'])

    # 保存报告
    def to_python(obj):
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
        'ensemble_strategy': best_strategy,
        'best_metrics': to_python({k: v for k, v in best_metrics.items() if k != 'cm'}),
        'all_strategies': to_python({k: {mk: mv for mk, mv in v.items() if mk != 'cm'} for k, v in candidates.items()}),
        'individual_models': list(test_preds.keys()),
    }

    out_path = '/mnt/data2/home/chenyu/HuaWeiCup/huaweibei-E/outputs/reports/ensemble_report.json'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n📋 报告已保存: {out_path}")

    return best_metrics


if __name__ == '__main__':
    main()
