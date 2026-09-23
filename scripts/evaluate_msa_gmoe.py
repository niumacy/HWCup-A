"""
MSA-GMoE 模型综合评估脚本
对所有模型（GMT, GLF, LateFusion, TCN, MSA-GMoE）做完整对比
"""
import torch
import torch.nn as nn
import numpy as np
import pickle
import json
import os
import sys
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from scipy.stats import pearsonr
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.baseline_model import LateFusionBaseline, GatedLateFusion, TCNFusion, LSTMCrossAttention, GatedMultimodalTransformer
from src.models.msa_gmoe import MSAGMoE


class MultimodalDataset(Dataset):
    def __init__(self, data_dict):
        self.text = torch.from_numpy(np.array(data_dict['text']).astype(np.float32))
        self.audio = torch.from_numpy(np.array(data_dict['audio']).astype(np.float32))
        self.vision = torch.from_numpy(np.array(data_dict['vision']).astype(np.float32))
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


def load_model(model_name, model_path, device):
    """加载模型"""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # 优先从 checkpoint['args'] 读超参数
    args_dict = {}
    if isinstance(checkpoint, dict) and 'args' in checkpoint:
        args_dict = checkpoint['args']

    if model_name == 'latefusion':
        hidden_dim = args_dict.get('hidden_dim', 512)
        dropout = args_dict.get('dropout', 0.4)
        model = LateFusionBaseline(text_dim=768, audio_dim=74, vision_dim=35,
                                   hidden_dim=hidden_dim, dropout=dropout)
    elif model_name == 'glf':
        model = GatedLateFusion(text_dim=768, audio_dim=74, vision_dim=35, hidden_dim=256)
    elif model_name == 'tcn':
        model = TCNFusion(text_dim=768, audio_dim=74, vision_dim=35, hidden_dim=128)
    elif model_name == 'lstmattn':
        model = LSTMCrossAttention(text_dim=768, audio_dim=74, vision_dim=35, hidden_dim=256)
    elif model_name == 'gmt':
        hidden_dim = args_dict.get('hidden_dim', 256)
        model = GatedMultimodalTransformer(text_dim=768, audio_dim=74, vision_dim=35,
                                            hidden_dim=hidden_dim)
    elif model_name == 'msa_gmoe':
        hidden_dim = args_dict.get('hidden_dim', 192)
        use_contrastive = args_dict.get('use_contrastive', False)
        model = MSAGMoE(text_dim=768, audio_dim=74, vision_dim=35,
                        hidden_dim=hidden_dim,
                        use_contrastive=use_contrastive)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def evaluate_model(model, dataloader, device, model_name):
    """评估单个模型"""
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

        if model_name == 'msa_gmoe':
            cls_logits, reg_pred, _ = model(text, audio, vision)
        else:
            cls_logits, reg_pred = model(text, audio, vision)

        all_preds_cls.append(cls_logits.argmax(dim=1).cpu().numpy())
        all_preds_reg.append(reg_pred.cpu().numpy())
        all_labels_cls.append(cls_label.numpy())
        all_labels_reg.append(reg_label.numpy())

    all_preds_cls = np.concatenate(all_preds_cls)
    all_preds_reg = np.concatenate(all_preds_reg)
    all_labels_cls = np.concatenate(all_labels_cls)
    all_labels_reg = np.concatenate(all_labels_reg)

    acc = accuracy_score(all_labels_cls, all_preds_cls)
    f1_w = f1_score(all_labels_cls, all_preds_cls, average='weighted')
    f1_m = f1_score(all_labels_cls, all_preds_cls, average='macro')
    mae = np.abs(all_preds_reg - all_labels_reg).mean()
    corr, _ = pearsonr(all_preds_reg, all_labels_reg)

    return {
        'Accuracy': float(acc),
        'F1_weighted': float(f1_w),
        'F1_macro': float(f1_m),
        'MAE': float(mae),
        'Pearson': float(corr),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str,
                       default='/mnt/data2/home/chenyu/HuaWeiCup/huaweibei-E/data/raw_attachment2/aligned_50.pkl')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--output_dir', type=str,
                       default='/mnt/data2/home/chenyu/HuaWeiCup/huaweibei-E/outputs')
    args = parser.parse_args()

    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Device: {device}")

    # 加载数据
    with open(args.data, 'rb') as f:
        data = pickle.load(f)
    test_loader = DataLoader(MultimodalDataset(data['test']), batch_size=64, shuffle=False)
    valid_loader = DataLoader(MultimodalDataset(data['valid']), batch_size=64, shuffle=False)

    # 模型列表
    models = [
        ('LateFusion', 'latefusion', 'latefusion_seed2024.pt'),
        ('GLF', 'glf', 'glf_seed42.pt'),
        ('TCN', 'tcn', 'tcn_best.pt'),
        ('LSTM-Attn', 'lstmattn', 'lstmattn_best.pt'),
        ('GMT', 'gmt', 'gmt_seed42.pt'),
        ('MSA-GMoE', 'msa_gmoe', 'msa_gmoe_seed42.pt'),
    ]

    results = []
    for name, model_type, ckpt_file in models:
        ckpt_path = f'{args.output_dir}/{ckpt_file}'
        if not os.path.exists(ckpt_path):
            print(f"⚠️  Skip {name} (not found: {ckpt_file})")
            continue

        print(f"\n{'='*60}")
        print(f"评估 {name} ...")
        try:
            model = load_model(model_type, ckpt_path, device)
            # 先用 valid 确认能跑
            valid_metrics = evaluate_model(model, valid_loader, device, model_type)
            print(f"  Valid: Acc={valid_metrics['Accuracy']:.4f} F1={valid_metrics['F1_weighted']:.4f} "
                  f"MAE={valid_metrics['MAE']:.4f} Pearson={valid_metrics['Pearson']:.4f}")
            test_metrics = evaluate_model(model, test_loader, device, model_type)
            print(f"  Test:  Acc={test_metrics['Accuracy']:.4f} F1={test_metrics['F1_weighted']:.4f} "
                  f"MAE={test_metrics['MAE']:.4f} Pearson={test_metrics['Pearson']:.4f}")
            results.append({
                'name': name,
                'model_type': model_type,
                'ckpt': ckpt_file,
                'valid': valid_metrics,
                'test': test_metrics,
            })
        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback; traceback.print_exc()

    # 保存结果
    print(f"\n{'='*60}")
    print("所有模型对比:")
    print(f"{'Model':<15} {'Acc':>7} {'F1':>7} {'MAE':>7} {'Pearson':>8}")
    print("-" * 60)
    for r in sorted(results, key=lambda x: -x['test']['F1_weighted']):
        print(f"{r['name']:<15} {r['test']['Accuracy']:>7.4f} {r['test']['F1_weighted']:>7.4f} "
              f"{r['test']['MAE']:>7.4f} {r['test']['Pearson']:>8.4f}")

    out = f'{args.output_dir}/reports/msa_gmoe_comparison.json'
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n📊 对比报告保存到: {out}")


if __name__ == '__main__':
    main()
