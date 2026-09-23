"""
MSA-GMoE 加入集成的评估脚本
对比 baseline ensemble 是否有显著提升
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
import json
import pickle
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from scipy.stats import pearsonr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.baseline_model import (
    LateFusionBaseline, GatedLateFusion, TCNFusion,
    LSTMCrossAttention, GatedMultimodalTransformer
)
from src.models.msa_gmoe import MSAGMoE


class MultimodalDataset(Dataset):
    def __init__(self, data_dict):
        self.text = torch.from_numpy(np.array(data_dict['text']).astype(np.float32))
        self.audio = torch.from_numpy(np.array(data_dict['audio']).astype(np.float32))
        self.vision = torch.from_numpy(np.array(data_dict['vision']).astype(np.float32))
        self.cls_labels = torch.from_numpy(np.array(data_dict['classification_labels']).astype(np.int64))
        self.reg_labels = np.array(data_dict['regression_labels']).astype(np.float32)
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


def load_model(name, path, device, extra_kwargs=None):
    """根据 name 加载不同模型"""
    if extra_kwargs is None:
        extra_kwargs = {}
    ckpt = torch.load(path, map_location=device, weights_only=False)
    args_dict = ckpt.get('args', {})

    if name == 'latefusion':
        hd = extra_kwargs.get('hidden_dim', args_dict.get('hidden_dim', 320))
        dp = extra_kwargs.get('dropout', args_dict.get('dropout', 0.4))
        model = LateFusionBaseline(text_dim=768, audio_dim=74, vision_dim=35,
                                    hidden_dim=hd, dropout=dp)
    elif name == 'glf':
        model = GatedLateFusion(text_dim=768, audio_dim=74, vision_dim=35,
                                 hidden_dim=256, dropout=0.4)
    elif name == 'tcn':
        hd = extra_kwargs.get('hidden_dim', args_dict.get('hidden_dim', 192))
        dp = extra_kwargs.get('dropout', args_dict.get('dropout', 0.3))
        model = TCNFusion(text_dim=768, audio_dim=74, vision_dim=35,
                          hidden_dim=hd, dropout=dp)
    elif name == 'lstmattn':
        model = LSTMCrossAttention(text_dim=768, audio_dim=74, vision_dim=35,
                                    hidden_dim=256, dropout=0.3)
    elif name == 'gmt':
        hd = extra_kwargs.get('hidden_dim', args_dict.get('hidden_dim', 192))
        dp = extra_kwargs.get('dropout', args_dict.get('dropout', 0.4))
        model = GatedMultimodalTransformer(text_dim=768, audio_dim=74, vision_dim=35,
                                            hidden_dim=hd, dropout=dp)
    elif name == 'msa_gmoe':
        hd = args_dict.get('hidden_dim', 192)
        uc = args_dict.get('use_contrastive', False)
        model = MSAGMoE(text_dim=768, audio_dim=74, vision_dim=35,
                         hidden_dim=hd, use_contrastive=uc)
    else:
        raise ValueError(f"Unknown model name: {name}")

    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    model.to(device).eval()
    return model


@torch.no_grad()
def get_preds(model, dataloader, device, model_type):
    """获取预测概率和回归预测"""
    cls_probs = []
    reg_preds = []
    for batch in dataloader:
        text = batch['text'].to(device)
        audio = batch['audio'].to(device)
        vision = batch['vision'].to(device)
        # 根据模型类型调用
        if model_type == 'msa_gmoe':
            cls_logits, reg_pred, _ = model(text, audio, vision)
        else:
            cls_logits, reg_pred = model(text, audio, vision)
        cls_probs.append(F.softmax(cls_logits, dim=-1).cpu().numpy())
        reg_preds.append(reg_pred.cpu().numpy())
    return np.concatenate(cls_probs, axis=0), np.concatenate(reg_preds, axis=0)


def evaluate(cls_preds, reg_preds, cls_lbl, reg_lbl):
    return {
        'Accuracy': float(accuracy_score(cls_lbl, cls_preds)),
        'F1_weighted': float(f1_score(cls_lbl, cls_preds, average='weighted')),
        'F1_macro': float(f1_score(cls_lbl, cls_preds, average='macro')),
        'MAE': float(mean_absolute_error(reg_lbl, reg_preds)),
        'Pearson': float(pearsonr(reg_preds, reg_lbl)[0]),
    }


def main():
    data_path = '/mnt/data2/home/chenyu/HuaWeiCup/huaweibei-E/data/raw_attachment2/aligned_50.pkl'
    output_dir = '/mnt/data2/home/chenyu/HuaWeiCup/huaweibei-E/outputs'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 加载数据
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    test_loader = DataLoader(MultimodalDataset(data['test']), batch_size=64, shuffle=False)
    valid_loader = DataLoader(MultimodalDataset(data['valid']), batch_size=64, shuffle=False)

    test_cls = np.array(data['test']['classification_labels'])
    test_reg = np.array(data['test']['regression_labels']).astype(np.float32)
    val_cls = np.array(data['valid']['classification_labels'])
    val_reg = np.array(data['valid']['regression_labels']).astype(np.float32)

    # ── 要评估/集成的模型 ──
    # (name, model_type, path, extra_kwargs for model construction)
    candidates = [
        ('latefusion2024', 'latefusion', f'{output_dir}/latefusion_seed2024.pt',
         {'hidden_dim': 320, 'dropout': 0.45}),
        ('glf42', 'glf', f'{output_dir}/glf_seed42.pt',
         {'hidden_dim': 256, 'dropout': 0.4}),
        ('tcn', 'tcn', f'{output_dir}/tcn_best.pt',
         {'hidden_dim': 192, 'dropout': 0.3}),
        ('lstmattn', 'lstmattn', f'{output_dir}/lstmattn_best.pt',
         {'hidden_dim': 256, 'dropout': 0.3}),
        ('gmt42', 'gmt', f'{output_dir}/gmt_seed42.pt',
         {'hidden_dim': 192, 'dropout': 0.4}),
        ('msa_gmoe_s42', 'msa_gmoe', f'{output_dir}/msa_gmoe_seed42.pt', {}),
        ('msa_gmoe_s7', 'msa_gmoe', f'{output_dir}/msa_gmoe_seed7.pt', {}),
        ('msa_gmoe_s2024', 'msa_gmoe', f'{output_dir}/msa_gmoe_seed2024.pt', {}),
    ]

    # 检查哪些模型存在
    available = []
    for entry in candidates:
        name = entry[0]
        path = entry[2]
        kwargs = entry[3] if len(entry) > 3 else {}
        if os.path.exists(path):
            available.append((name, path, kwargs))
        else:
            print(f"⚠️  {name} not found, skipping")

    print(f"\n可用模型数: {len(available)}")

    # ── 单模型表现 ──
    print("\n" + "=" * 70)
    print("📊 各模型独立表现 (测试集)")
    print("=" * 70)
    print(f"{'Model':<22} {'Acc':>7} {'F1':>7} {'MAE':>7} {'Pearson':>8}")
    print("-" * 70)

    test_preds = {}
    val_preds = {}
    single_results = {}

    for name, path, extra_kwargs in available:
        # 从 name 推断 model_type
        if 'latefusion' in name:
            mt = 'latefusion'
        elif 'glf' in name:
            mt = 'glf'
        elif 'tcn' in name:
            mt = 'tcn'
        elif 'lstmattn' in name:
            mt = 'lstmattn'
        elif 'gmt' in name:
            mt = 'gmt'
        elif 'msa' in name:
            mt = 'msa_gmoe'
        else:
            mt = 'latefusion'

        try:
            model = load_model(mt, path, device, extra_kwargs)
            cv, rv = get_preds(model, valid_loader, device, mt)
            ct, rt = get_preds(model, test_loader, device, mt)
            val_preds[name] = {'cls_probs': cv, 'reg': rv}
            test_preds[name] = {'cls_probs': ct, 'reg': rt}
            m = evaluate(ct.argmax(1), rt, test_cls, test_reg)
            single_results[name] = m
            print(f"{name:<22} {m['Accuracy']:>7.4f} {m['F1_weighted']:>7.4f} "
                  f"{m['MAE']:>7.4f} {m['Pearson']:>8.4f}")
        except Exception as e:
            print(f"❌ {name}: {e}")
            import traceback; traceback.print_exc()

    # ── 集成策略 ──
    print("\n" + "=" * 70)
    print("🎯 集成策略对比")
    print("=" * 70)

    all_names = list(test_preds.keys())

    ensembles = {}

    if len(all_names) >= 2:
        # 全部等权
        avg_cls = np.mean([test_preds[n]['cls_probs'] for n in all_names], axis=0)
        avg_reg = np.mean([test_preds[n]['reg'] for n in all_names], axis=0)
        m = evaluate(avg_cls.argmax(1), avg_reg, test_cls, test_reg)
        ensembles['all_equal'] = m
        print(f"\n[All Equal Soft]    Acc={m['Accuracy']:.4f} F1={m['F1_weighted']:.4f} "
              f"MAE={m['MAE']:.4f} Pearson={m['Pearson']:.4f}")

        # F1 加权 (用验证集 F1 做权重)
        val_f1 = np.array([evaluate(val_preds[n]['cls_probs'].argmax(1),
                                     val_preds[n]['reg'],
                                     val_cls, val_reg)['F1_weighted']
                           for n in all_names])
        w_f1 = np.maximum(val_f1 - val_f1.min() + 0.05, 0)
        w_f1 = w_f1 / w_f1.sum() if w_f1.sum() > 0 else np.ones_like(w_f1) / len(w_f1)
        weighted_cls = sum(w_f1[i] * test_preds[n]['cls_probs'] for i, n in enumerate(all_names))
        weighted_reg = sum(w_f1[i] * test_preds[n]['reg'] for i, n in enumerate(all_names))
        m = evaluate(weighted_cls.argmax(1), weighted_reg, test_cls, test_reg)
        ensembles['f1_weighted'] = m
        print(f"[F1 Weighted]       Acc={m['Accuracy']:.4f} F1={m['F1_weighted']:.4f} "
              f"MAE={m['MAE']:.4f} Pearson={m['Pearson']:.4f}")

    # Top 2 / Top 3
    if 'latefusion2024' in test_preds and 'glf42' in test_preds:
        cls2 = (test_preds['latefusion2024']['cls_probs'] + test_preds['glf42']['cls_probs']) / 2
        reg2 = (test_preds['latefusion2024']['reg'] + test_preds['glf42']['reg']) / 2
        m = evaluate(cls2.argmax(1), reg2, test_cls, test_reg)
        ensembles['top2_baseline'] = m
        print(f"\n[Top2 Original]     Acc={m['Accuracy']:.4f} F1={m['F1_weighted']:.4f} "
              f"MAE={m['MAE']:.4f} Pearson={m['Pearson']:.4f}")

    # Top 3 (LF + GLF + TCN)
    if all(n in test_preds for n in ['latefusion2024', 'glf42', 'tcn']):
        cls3 = sum(test_preds[n]['cls_probs'] for n in ['latefusion2024', 'glf42', 'tcn']) / 3
        reg3 = sum(test_preds[n]['reg'] for n in ['latefusion2024', 'glf42', 'tcn']) / 3
        m = evaluate(cls3.argmax(1), reg3, test_cls, test_reg)
        ensembles['top3_baseline'] = m
        print(f"[Top3 Baseline]     Acc={m['Accuracy']:.4f} F1={m['F1_weighted']:.4f} "
              f"MAE={m['MAE']:.4f} Pearson={m['Pearson']:.4f}")

    # MSA-GMoE 单独 / 加入集成
    msa_names = [n for n in all_names if 'msa_gmoe' in n]
    if msa_names:
        # MSA-GMoE 平均
        avg_cls = np.mean([test_preds[n]['cls_probs'] for n in msa_names], axis=0)
        avg_reg = np.mean([test_preds[n]['reg'] for n in msa_names], axis=0)
        m = evaluate(avg_cls.argmax(1), avg_reg, test_cls, test_reg)
        ensembles['msa_avg'] = m
        print(f"\n[MSA-GMoE Avg]      Acc={m['Accuracy']:.4f} F1={m['F1_weighted']:.4f} "
              f"MAE={m['MAE']:.4f} Pearson={m['Pearson']:.4f}")

        # LF(2024) + GLF(42) + MSA-GMoE(avg)
        if 'latefusion2024' in test_preds and 'glf42' in test_preds:
            cls = (test_preds['latefusion2024']['cls_probs'] +
                   test_preds['glf42']['cls_probs'] + avg_cls) / 3
            reg = (test_preds['latefusion2024']['reg'] +
                   test_preds['glf42']['reg'] + avg_reg) / 3
            m = evaluate(cls.argmax(1), reg, test_cls, test_reg)
            ensembles['lf_glf_msa'] = m
            print(f"[LF+GLF+MSA]        Acc={m['Accuracy']:.4f} F1={m['F1_weighted']:.4f} "
                  f"MAE={m['MAE']:.4f} Pearson={m['Pearson']:.4f}")

        # LF(2024) + MSA-GMoE 2-model ensemble
        if 'latefusion2024' in test_preds:
            cls = (test_preds['latefusion2024']['cls_probs'] + avg_cls) / 2
            reg = (test_preds['latefusion2024']['reg'] + avg_reg) / 2
            m = evaluate(cls.argmax(1), reg, test_cls, test_reg)
            ensembles['lf_msa'] = m
            print(f"[LF+MSA]            Acc={m['Accuracy']:.4f} F1={m['F1_weighted']:.4f} "
                  f"MAE={m['MAE']:.4f} Pearson={m['Pearson']:.4f}")

        # MSA + GLF(42)
        if 'glf42' in test_preds:
            cls = (test_preds['glf42']['cls_probs'] + avg_cls) / 2
            reg = (test_preds['glf42']['reg'] + avg_reg) / 2
            m = evaluate(cls.argmax(1), reg, test_cls, test_reg)
            ensembles['msa_glf'] = m
            print(f"[MSA+GLF]           Acc={m['Accuracy']:.4f} F1={m['F1_weighted']:.4f} "
                  f"MAE={m['MAE']:.4f} Pearson={m['Pearson']:.4f}")

    # ── 汇总最佳 ──
    if ensembles:
        best_name = max(ensembles.keys(), key=lambda k: ensembles[k]['F1_weighted'])
        best = ensembles[best_name]
        print(f"\n🏆 最佳集成策略: {best_name}")
        print(f"   Acc={best['Accuracy']:.4f} F1={best['F1_weighted']:.4f} "
              f"MAE={best['MAE']:.4f} Pearson={best['Pearson']:.4f}")

        # ── 保存报告 ──
        report = {
            'single_results': single_results,
            'ensembles': ensembles,
            'best_ensemble': best_name,
            'best_metrics': best,
            'available_models': [n for n, _, _ in available],
        }
        out_path = f'{output_dir}/reports/msa_gmoe_ensemble_report.json'
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n📋 报告已保存: {out_path}")


if __name__ == '__main__':
    main()
