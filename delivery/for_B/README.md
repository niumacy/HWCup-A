# 给成员 B 的交付包（Q2 鲁棒预测）

> **接收人**: 成员 B
> **目标**: Q2 - 鲁棒情感预测
> **数据基础**: 附件 2 的 16k 样本（训练用）+ 成员 A 提供的 100 条样本（评估用）

---

## 🎯 你应该优先看的 5 个文件

| 顺序 | 文件 | 说明 |
|------|------|------|
| 1 | `../00_common/docs/feature_spec.md` | 接口规范、字段含义 |
| 2 | `../00_common/code/src/data_loader.py` | 数据加载器（直接用） |
| 3 | `../00_common/code/src/feature_extractor/aligner.py` | 时序对齐（直接用） |
| 4 | `reports/COMPREHENSIVE_REPORT.md` | 当前最优结果与所有尝试过的模型 |
| 5 | `code/baseline_model.py` | Baseline + LateFusion + GLF + TCN 等实现 |

---

## 📂 目录结构说明

```
for_B/
├── code/                    ← 你能直接复用的代码
│   └── baseline_model.py    ← 含: Baseline, LateFusion, GLF, TCN, GMT, LSTM-Attention
├── models/                  ← 5 个已训练好的最佳权重（可作为起点）
│   ├── baseline_best.pt
│   ├── latefusion_seed2024.pt
│   ├── glf_seed42.pt
│   ├── msa_gmoe_seed7.pt
│   └── msa_gmoe_seed42.pt
└── reports/                 ← 13 份训练报告
    ├── COMPREHENSIVE_REPORT.md       ← 综合报告
    ├── final_optimized_result.json   ← 当前最优
    ├── final_baseline_report.md      ← baseline 训练总结
    └── ... 单模型报告
```

---

## 🔧 5 分钟上手

### 加载 Q1 100 条样本

```python
import sys
sys.path.insert(0, '../00_common/code')

from src.data_loader import load_q1_features, get_dataloader

data = load_q1_features('../00_common/data/q1_features_combined.pkl')
all_data = data['all']

print('text:', all_data['text'].shape)     # (100, 50, 768)
print('audio:', all_data['audio'].shape)   # (100, 50, 74)
print('vision:', all_data['vision'].shape) # (100, 50, 35)

loader = get_dataloader(all_data, batch_size=64, shuffle=False)
```

### 同时加载附件 2 + Q1

```python
from src.data_loader import load_all
both = load_all(
    version='aligned',
    q1_extra_path='../00_common/data/q1_features_combined.pkl',
    data_root='/mnt/data2/home/chenyu/HuaWeiCup/E题数据/附件2-数据集特征文件',
)
# both = {'train': ..., 'valid': ..., 'test': ..., 'q1': {'all': ...}}
```

### 复用 Baseline 模型

```python
import sys
sys.path.insert(0, 'code')
from baseline_model import MultimodalBaseline, LateFusionModel

model = LateFusionModel(text_dim=768, audio_dim=74, vision_dim=35, hidden_dim=320)
model.load_state_dict(torch.load('models/latefusion_seed2024.pt', map_location='cpu'))
model.eval()
```

### 复用对齐模块

```python
from src.feature_extractor.aligner import TemporalAligner

aligner = TemporalAligner(target_len=50, mode='uniform')
text_a, audio_a, vision_a = aligner.align_50(text, audio, vision)
```

---

## 📊 当前最优结果（请以此为基线）

| 策略 | Acc | F1 | MAE | Pearson |
|------|-----|----|----|---------|
| **加权集成 (45% LF + 45% GLF + 10% MSA-GMoE)** | **0.6905** | **0.6796** | 0.6443 | 0.6713 |
| LateFusion 单模型 (seed=2024) | 0.6850 | 0.6788 | 0.6839 | 0.6516 |
| GLF 单模型 (seed=42) | 0.6713 | 0.6626 | **0.6509** | **0.6657** |
| MSA-GMoE 单模型 (seed=7) | 0.6754 | 0.6730 | 0.6573 | 0.6634 |

**你的鲁棒性提升目标**: Acc ≥ 0.70, F1 ≥ 0.69

---

## ⚠️ 重要约定

1. **不要直接拼接 Q1 100 条 + 附件 2 训练**
   - text_bert 维度顺序相反
   - audio/vision 数值分布不同
2. **Q1 100 条可用于**:
   - 推理 / 案例分析
   - 鲁棒性测试（用 Q1 验证附件 2 训练的模型是否过拟合）
   - 域迁移评估
3. **baseline_model.py 中已包含 5 个模型类**, 直接复用:
   - `BaselineModel` (1-epoch 快速验证)
   - `LateFusionModel` (Acc/F1 强)
   - `GLFModel` (MAE/Pearson 强)
   - `TCNModel`, `GMTModel`, `LSTMAttentionModel`

---

## 📋 已尝试的负结果（不必重做）

| 实验 | 结论 |
|------|------|
| 5-Fold CV Ensemble | -1.7% Acc, 不推荐 |
| Label Smoothing 0.1 | -2.3% Acc, 不推荐 |
| LSTM Temporal | -2.8% Acc, 不推荐 |
| Sample Weighting | -6.7% Acc, 不推荐 |
| Focal LateFusion | -2.6% Acc, 不推荐 |
| Multi-config Ensemble (8 模型) | 比精选 2-3 模型差, 弱模型拖累 |

详见 `reports/COMPREHENSIVE_REPORT.md` 第五节。

---

## 🚀 建议的下一步

1. **鲁棒性方向**: 在 `LateFusion` 基础上加对抗训练 (FGSM/PGD)
2. **架构方向**: 复用 `MSA-GMoE` 思路（已实现）, 探索更多模态路由
3. **集成方向**: 在 weighted ensemble 基础上加 stacking
4. **数据方向**: 用 Q1 100 条做域外测试集

---

## 📞 联系方式

如需新增导出物或接口变更，请联系成员 A。
