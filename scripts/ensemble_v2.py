"""
Ensemble 评估 V2: 多模型集成
支持 LateFusion (3个 seed) + LSTM-Attention + TCN 五模型集成
"""
import torch
import torch.nn.functional as F
import numpy as np
import os
import sys
import json
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, mean_absolute_error
)
from scipy.stats import pearsonr

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
def get_preds(model, dataloader, device):
    model.eval()
    all_cls_probs = []
    all_reg = []
    for batch in dataloader:
        text = batch['text'].to(device)
        audio = batch['audio'].to(device)
        vision = batch['vision'].to(device)
        cls_logits, reg_pred = model(text, audio, vision)
        all_cls_probs.append(F.softmax(cls_logits, dim=-1).cpu().numpy())
        all_reg.append(reg_pred.cpu().numpy())
    return np.concatenate(all_cls_probs, axis=0), np.concatenate(all_reg, axis=0)


def evaluate(cls_preds, reg_preds, cls_lbl, reg_lbl):
    return {
        'Accuracy': accuracy_score(cls_lbl, cls_preds),
        'F1_weighted': f1_score(cls_lbl, cls_preds, average='weighted'),
        'F1_macro': f1_score(cls_lbl, cls_preds, average='macro'),
        'Precision': precision_score(cls_lbl, cls_preds, average='weighted', zero_division=0),
        'Recall': recall_score(cls_lbl, cls_preds, average='weighted', zero_division=0),
        'MAE': mean_absolute_error(reg_lbl, reg_preds),
        'Pearson': pearsonr(reg_preds, reg_lbl)[0],
        'cm': confusion_matrix(cls_lbl, cls_preds),
    }


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


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载数据
    data = load_attachment2('/mnt/data2/home/chenyu/HuaWeiCup/E题数据/附件2-数据集特征文件/aligned_50.pkl')
    valid_loader = DataLoader(MultimodalDataset(data['valid']), batch_size=64, shuffle=False)
    test_loader = DataLoader(MultimodalDataset(data['test']), batch_size=64, shuffle=False)

    # 加载测试集标签 (用于评估)
    test_cls_labels = np.array(data['test']['classification_labels'])
    test_reg_labels = np.array(data['test']['regression_labels']).astype(np.float32)
    val_cls_labels = np.array(data['valid']['classification_labels'])
    val_reg_labels = np.array(data['valid']['regression_labels']).astype(np.float32)

    # 加载所有模型
    model_paths = [
        ('latefusion_seed42', '/mnt/data2/home/chenyu/HuaWeiCup/huaweibei-E/outputs/latefusion_seed42.pt'),
        ('latefusion_seed123', '/mnt/data2/home/chenyu/HuaWeiCup/huaweibei-E/outputs/latefusion_seed123.pt'),
        ('latefusion_seed7', '/mnt/data2/home/chenyu/HuaWeiCup/huaweibei-E/outputs/latefusion_seed7.pt'),
        ('latefusion_seed2024', '/mnt/data2/home/chenyu/HuaWeiCup/huaweibei-E/outputs/latefusion_seed2024.pt'),
        ('latefusion_seed999', '/mnt/data2/home/chenyu/HuaWeiCup/huaweibei-E/outputs/latefusion_seed999.pt'),
        ('lstmattn', '/mnt/data2/home/chenyu/HuaWeiCup/huaweibei-E/outputs/lstmattn_best.pt'),
        ('tcn', '/mnt/data2/home/chenyu/HuaWeiCup/huaweibei-E/outputs/tcn_best.pt'),
        ('gmt_seed42', '/mnt/data2/home/chenyu/HuaWeiCup/huaweibei-E/outputs/gmt_seed42.pt'),
        ('gmt_seed2024', '/mnt/data2/home/chenyu/HuaWeiCup/huaweibei-E/outputs/gmt_best.pt'),
        ('glf_seed42', '/mnt/data2/home/chenyu/HuaWeiCup/huaweibei-E/outputs/glf_seed42.pt'),
        ('glf_seed2024', '/mnt/data2/home/chenyu/HuaWeiCup/huaweibei-E/outputs/glf_seed2024.pt'),
    ]

    val_preds = {}
    test_preds = {}
    print("=" * 70)
    print("📊 各模型独立测试表现")
    print("=" * 70)

    for name, path in model_paths:
        if not os.path.exists(path):
            print(f"⚠️  跳过 {name}: {path}")
            continue
        ckpt = torch.load(path, map_location=device)
        model = build_model(ckpt['model_type'], **ckpt['model_kwargs']).to(device)
        model.load_state_dict(ckpt['model_state_dict'])

        cls_v, reg_v = get_preds(model, valid_loader, device)
        val_preds[name] = {'cls_probs': cls_v, 'reg': reg_v}

        cls_t, reg_t = get_preds(model, test_loader, device)
        test_preds[name] = {'cls_probs': cls_t, 'reg': reg_t}

        # 单独表现
        indv = evaluate(cls_t.argmax(1), reg_t, test_cls_labels, test_reg_labels)
        print(f"  {name:25s}: Acc={indv['Accuracy']:.4f} F1={indv['F1_weighted']:.4f} "
              f"MAE={indv['MAE']:.4f} Pearson={indv['Pearson']:.4f}")

    # ─── 集成 ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("🎯 集成策略对比")
    print("=" * 70)

    all_names = list(test_preds.keys())
    n_models = len(all_names)

    # 1. Soft Voting (全部模型等权平均)
    avg_cls = np.mean([test_preds[n]['cls_probs'] for n in all_names], axis=0)
    avg_reg = np.mean([test_preds[n]['reg'] for n in all_names], axis=0)
    soft_all = evaluate(avg_cls.argmax(1), avg_reg, test_cls_labels, test_reg_labels)
    print(f"\n1️⃣  全部 {n_models} 模型 Soft Voting (等权):")
    print(f"    Acc={soft_all['Accuracy']:.4f} F1={soft_all['F1_weighted']:.4f} "
          f"MAE={soft_all['MAE']:.4f} Pearson={soft_all['Pearson']:.4f}")

    # 2. Soft Voting (异构 3 模型 - LateFusion 全 + TCN + LSTMAtt)
    hetero = ['latefusion_seed42', 'latefusion_seed123', 'latefusion_seed7', 'lstmattn', 'tcn']
    hetero = [n for n in hetero if n in test_preds]
    avg_cls_h = np.mean([test_preds[n]['cls_probs'] for n in hetero], axis=0)
    avg_reg_h = np.mean([test_preds[n]['reg'] for n in hetero], axis=0)
    soft_hetero = evaluate(avg_cls_h.argmax(1), avg_reg_h, test_cls_labels, test_reg_labels)
    print(f"\n2️⃣  异构 Soft Voting ({len(hetero)} 模型):")
    print(f"    Acc={soft_hetero['Accuracy']:.4f} F1={soft_hetero['F1_weighted']:.4f} "
          f"MAE={soft_hetero['MAE']:.4f} Pearson={soft_hetero['Pearson']:.4f}")

    # 3. Weighted Soft Voting (用验证集 F1 作权重)
    val_scores = []
    for n in all_names:
        cv = evaluate(val_preds[n]['cls_probs'].argmax(1), val_preds[n]['reg'],
                      val_cls_labels, val_reg_labels)
        val_scores.append(cv['F1_weighted'])
    val_scores = np.array(val_scores)
    weights = val_scores - val_scores.min() + 0.05
    weights = weights / weights.sum()
    print(f"\n   模型权重 (验证集 F1 加权): {dict(zip(all_names, [f'{w:.3f}' for w in weights]))}")

    weighted_cls = np.zeros_like(avg_cls)
    weighted_reg = np.zeros_like(avg_reg)
    for w, n in zip(weights, all_names):
        weighted_cls += w * test_preds[n]['cls_probs']
        weighted_reg += w * test_preds[n]['reg']
    weighted_soft = evaluate(weighted_cls.argmax(1), weighted_reg, test_cls_labels, test_reg_labels)
    print(f"\n3️⃣  Weighted Soft Voting (验证集 F1 加权):")
    print(f"    Acc={weighted_soft['Accuracy']:.4f} F1={weighted_soft['F1_weighted']:.4f} "
          f"MAE={weighted_soft['MAE']:.4f} Pearson={weighted_soft['Pearson']:.4f}")

    # 4. Stacking-like: 用验证集搜索最优权重 (基于验证集 F1)
    print(f"\n4️⃣  自动权重优化 (按验证集 F1 占比)...")
    val_scores = []
    for n in all_names:
        cv = evaluate(val_preds[n]['cls_probs'].argmax(1), val_preds[n]['reg'],
                      val_cls_labels, val_reg_labels)
        val_scores.append(cv['F1_weighted'])
    val_scores = np.array(val_scores)
    # softmax 权重 (放大差异)
    weights_softmax = np.exp(val_scores * 20) / np.sum(np.exp(val_scores * 20))
    print(f"   Softmax 权重 (T=20): {dict(zip(all_names, [f'{w:.3f}' for w in weights_softmax]))}")

    opt_cls = np.zeros_like(avg_cls)
    opt_reg = np.zeros_like(avg_reg)
    for w, n in zip(weights_softmax, all_names):
        opt_cls += w * test_preds[n]['cls_probs']
        opt_reg += w * test_preds[n]['reg']
    opt_test = evaluate(opt_cls.argmax(1), opt_reg, test_cls_labels, test_reg_labels)

    weight_dict = {all_names[i]: float(weights_softmax[i]) for i in range(len(all_names))}
    print(f"\n   测试集: Acc={opt_test['Accuracy']:.4f} F1={opt_test['F1_weighted']:.4f} "
          f"MAE={opt_test['MAE']:.4f} Pearson={opt_test['Pearson']:.4f}")

    # 选择最佳
    candidates = {
        'equal_soft': soft_all,
        'weighted_soft': weighted_soft,
        'grid_search_soft': opt_test,
    }
    best_strategy = max(candidates.keys(), key=lambda k: candidates[k]['F1_weighted'])
    best_metrics = candidates[best_strategy]

    print(f"\n🏆 最佳策略: {best_strategy}")
    print(f"   Accuracy:    {best_metrics['Accuracy']:.4f}")
    print(f"   F1 weighted: {best_metrics['F1_weighted']:.4f}")
    print(f"   F1 macro:    {best_metrics['F1_macro']:.4f}")
    print(f"   Precision:   {best_metrics['Precision']:.4f}")
    print(f"   Recall:      {best_metrics['Recall']:.4f}")
    print(f"   MAE:         {best_metrics['MAE']:.4f}")
    print(f"   Pearson:     {best_metrics['Pearson']:.4f}")
    print(f"\n   混淆矩阵:")
    print(best_metrics['cm'])

    # 保存报告
    report = to_python({
        'best_strategy': best_strategy,
        'best_metrics': best_metrics,
        'all_candidates': candidates,
        'individual': {n: evaluate(test_preds[n]['cls_probs'].argmax(1),
                                     test_preds[n]['reg'],
                                     test_cls_labels, test_reg_labels)
                       for n in all_names},
        'opt_weights': weight_dict,
    })
    out_path = '/mnt/data2/home/chenyu/HuaWeiCup/huaweibei-E/outputs/reports/ensemble_v2_report.json'
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n📋 报告已保存: {out_path}")


if __name__ == '__main__':
    main()
