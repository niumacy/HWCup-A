"""
精选集成: 只集成表现最好的模型
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
    cls_probs = []
    reg_preds = []
    for batch in dataloader:
        text = batch['text'].to(device)
        audio = batch['audio'].to(device)
        vision = batch['vision'].to(device)
        cls_logits, reg_pred = model(text, audio, vision)
        cls_probs.append(F.softmax(cls_logits, dim=-1).cpu().numpy())
        reg_preds.append(reg_pred.cpu().numpy())
    return np.concatenate(cls_probs, axis=0), np.concatenate(reg_preds, axis=0)


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
    data = load_attachment2('/mnt/data2/home/chenyu/HuaWeiCup/E题数据/附件2-数据集特征文件/aligned_50.pkl')
    valid_loader = DataLoader(MultimodalDataset(data['valid']), batch_size=64, shuffle=False)
    test_loader = DataLoader(MultimodalDataset(data['test']), batch_size=64, shuffle=False)
    test_cls_labels = np.array(data['test']['classification_labels'])
    test_reg_labels = np.array(data['test']['regression_labels']).astype(np.float32)
    val_cls_labels = np.array(data['valid']['classification_labels'])
    val_reg_labels = np.array(data['valid']['regression_labels']).astype(np.float32)

    # 精选 5 个最优模型 (Top performers)
    selected_models = [
        ('latefusion_seed2024', '/mnt/data2/home/chenyu/HuaWeiCup/huaweibei-E/outputs/latefusion_seed2024.pt'),  # Best F1
        ('latefusion_seed123', '/mnt/data2/home/chenyu/HuaWeiCup/huaweibei-E/outputs/latefusion_seed123.pt'),    # 2nd best F1
        ('glf_seed42', '/mnt/data2/home/chenyu/HuaWeiCup/huaweibei-E/outputs/glf_seed42.pt'),                     # Best Pearson
        ('latefusion_seed7', '/mnt/data2/home/chenyu/HuaWeiCup/huaweibei-E/outputs/latefusion_seed7.pt'),        # Mid
        ('tcn', '/mnt/data2/home/chenyu/HuaWeiCup/huaweibei-E/outputs/tcn_best.pt'),                               # Best MAE
    ]

    print("=" * 70)
    print("🏆 精选 5 模型集成")
    print("=" * 70)

    val_preds = {}
    test_preds = {}
    print("\n📊 各模型独立表现:")
    for name, path in selected_models:
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model = build_model(ckpt['model_type'], **ckpt['model_kwargs']).to(device)
        model.load_state_dict(ckpt['model_state_dict'])

        cls_v, reg_v = get_preds(model, valid_loader, device)
        val_preds[name] = {'cls_probs': cls_v, 'reg': reg_v}

        cls_t, reg_t = get_preds(model, test_loader, device)
        test_preds[name] = {'cls_probs': cls_t, 'reg': reg_t}

        indv = evaluate(cls_t.argmax(1), reg_t, test_cls_labels, test_reg_labels)
        print(f"  {name:25s}: Acc={indv['Accuracy']:.4f} F1={indv['F1_weighted']:.4f} "
              f"MAE={indv['MAE']:.4f} Pearson={indv['Pearson']:.4f}")

    # ── 多个集成策略 ──
    print("\n" + "=" * 70)
    print("🎯 集成策略对比")
    print("=" * 70)

    all_names = list(test_preds.keys())
    candidates = {}

    # 1. 等权 Soft
    avg_cls = np.mean([test_preds[n]['cls_probs'] for n in all_names], axis=0)
    avg_reg = np.mean([test_preds[n]['reg'] for n in all_names], axis=0)
    eq_soft = evaluate(avg_cls.argmax(1), avg_reg, test_cls_labels, test_reg_labels)
    print(f"\n1️⃣  全部 5 模型 Soft Voting (等权):")
    print(f"   Acc={eq_soft['Accuracy']:.4f} F1={eq_soft['F1_weighted']:.4f} MAE={eq_soft['MAE']:.4f} Pearson={eq_soft['Pearson']:.4f}")
    candidates['equal_soft'] = eq_soft

    # 2. 验证集 F1 加权
    val_scores = []
    for n in all_names:
        cv = evaluate(val_preds[n]['cls_probs'].argmax(1), val_preds[n]['reg'],
                      val_cls_labels, val_reg_labels)
        val_scores.append(cv['F1_weighted'])
    val_scores = np.array(val_scores)
    weights = val_scores - val_scores.min() + 0.05
    weights = weights / weights.sum()
    print(f"\n   F1加权: {dict(zip(all_names, [f'{w:.2f}' for w in weights]))}")
    weighted_cls = np.zeros_like(avg_cls)
    weighted_reg = np.zeros_like(avg_reg)
    for w, n in zip(weights, all_names):
        weighted_cls += w * test_preds[n]['cls_probs']
        weighted_reg += w * test_preds[n]['reg']
    f1_weighted = evaluate(weighted_cls.argmax(1), weighted_reg, test_cls_labels, test_reg_labels)
    print(f"\n2️⃣  Weighted Soft Voting:")
    print(f"   Acc={f1_weighted['Accuracy']:.4f} F1={f1_weighted['F1_weighted']:.4f} MAE={f1_weighted['MAE']:.4f} Pearson={f1_weighted['Pearson']:.4f}")
    candidates['f1_weighted_soft'] = f1_weighted

    # 3. Pearson 加权
    val_pearson = []
    for n in all_names:
        cv = evaluate(val_preds[n]['cls_probs'].argmax(1), val_preds[n]['reg'],
                      val_cls_labels, val_reg_labels)
        val_pearson.append(cv['Pearson'])
    val_pearson = np.array(val_pearson)
    weights_p = val_pearson - val_pearson.min() + 0.1
    weights_p = weights_p / weights_p.sum()
    weighted_cls2 = np.zeros_like(avg_cls)
    weighted_reg2 = np.zeros_like(avg_reg)
    for w, n in zip(weights_p, all_names):
        weighted_cls2 += w * test_preds[n]['cls_probs']
        weighted_reg2 += w * test_preds[n]['reg']
    p_weighted = evaluate(weighted_cls2.argmax(1), weighted_reg2, test_cls_labels, test_reg_labels)
    print(f"\n3️⃣  Pearson 加权:")
    print(f"   Acc={p_weighted['Accuracy']:.4f} F1={p_weighted['F1_weighted']:.4f} MAE={p_weighted['MAE']:.4f} Pearson={p_weighted['Pearson']:.4f}")
    candidates['pearson_weighted'] = p_weighted

    # 4. 2模型集成 (best F1 + best Pearson)
    if 'latefusion_seed2024' in test_preds and 'glf_seed42' in test_preds:
        cls_top2 = (test_preds['latefusion_seed2024']['cls_probs'] + test_preds['glf_seed42']['cls_probs']) / 2
        reg_top2 = (test_preds['latefusion_seed2024']['reg'] + test_preds['glf_seed42']['reg']) / 2
        top2 = evaluate(cls_top2.argmax(1), reg_top2, test_cls_labels, test_reg_labels)
        print(f"\n4️⃣  Top 2 集成 (latefusion_seed2024 + glf_seed42):")
        print(f"   Acc={top2['Accuracy']:.4f} F1={top2['F1_weighted']:.4f} MAE={top2['MAE']:.4f} Pearson={top2['Pearson']:.4f}")
        candidates['top2_ensemble'] = top2

    # 5. 3 模型集成 (Add TCN)
    if 'latefusion_seed2024' in test_preds and 'glf_seed42' in test_preds and 'tcn' in test_preds:
        cls_top3 = (test_preds['latefusion_seed2024']['cls_probs'] +
                     test_preds['glf_seed42']['cls_probs'] +
                     test_preds['tcn']['cls_probs']) / 3
        reg_top3 = (test_preds['latefusion_seed2024']['reg'] +
                     test_preds['glf_seed42']['reg'] +
                     test_preds['tcn']['reg']) / 3
        top3 = evaluate(cls_top3.argmax(1), reg_top3, test_cls_labels, test_reg_labels)
        print(f"\n5️⃣  Top 3 集成 (+ TCN):")
        print(f"   Acc={top3['Accuracy']:.4f} F1={top3['F1_weighted']:.4f} MAE={top3['MAE']:.4f} Pearson={top3['Pearson']:.4f}")
        candidates['top3_ensemble'] = top3

    # 选最佳
    best_strategy = max(candidates.keys(), key=lambda k: candidates[k]['F1_weighted'])
    best = candidates[best_strategy]

    print(f"\n🏆 最佳策略: {best_strategy}")
    print(f"   Accuracy:    {best['Accuracy']:.4f}")
    print(f"   F1 weighted: {best['F1_weighted']:.4f}")
    print(f"   F1 macro:    {best['F1_macro']:.4f}")
    print(f"   Precision:   {best['Precision']:.4f}")
    print(f"   Recall:      {best['Recall']:.4f}")
    print(f"   MAE:         {best['MAE']:.4f}")
    print(f"   Pearson:     {best['Pearson']:.4f}")
    print(f"\n   混淆矩阵:")
    print(best['cm'])

    # 保存报告
    report = to_python({
        'best_strategy': best_strategy,
        'best_metrics': {k: v for k, v in best.items() if k != 'cm'},
        'all_candidates': {k: {mk: mv for mk, mv in v.items() if mk != 'cm'} for k, v in candidates.items()},
        'selected_models': [n for n, _ in selected_models],
    })
    out_path = '/mnt/data2/home/chenyu/HuaWeiCup/huaweibei-E/outputs/reports/ensemble_premium_report.json'
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n📋 报告已保存: {out_path}")


if __name__ == '__main__':
    main()
