# Baseline 模型训练 - 最终报告

## 📊 试验结果汇总

### 个体模型 (验证集基于训练集选最佳)
| 模型 | Acc | F1 (weighted) | MAE | Pearson |
|------|-----|---------------|-----|---------|
| **LateFusion (seed=2024)** ⭐ | **0.6850** | **0.6788** | 0.6839 | 0.6516 |
| LateFusion (seed=123) | 0.6740 | 0.6657 | 0.6956 | 0.6487 |
| LateFusion (seed=7) | 0.6740 | 0.6632 | 0.6965 | 0.6585 |
| LateFusion (seed=42) | 0.6713 | 0.6616 | 0.6863 | 0.6597 |
| TCN | 0.6713 | 0.6570 | 0.6567 | 0.6507 |
| LateFusion (seed=999) | 0.6685 | 0.6529 | 0.7063 | 0.6511 |

### 集成 (Ensemble)
| 策略 | Acc | F1 | MAE | Pearson |
|------|-----|-----|-----|---------|
| Soft Voting (6模型等权) | 0.6726 | 0.6596 | 0.6703 | **0.6643** |
| Weighted Soft | 0.6699 | 0.6573 | 0.6720 | 0.6641 |
| Softmax 权重 | 0.6685 | 0.6562 | 0.6722 | 0.6640 |

## 🏆 最优方案

### 推荐: 单模型 LateFusion (seed=2024)
- **Accuracy: 0.6850** (vs 基线 0.6410)
- **F1 weighted: 0.6788** (vs 基线 0.5604)
- **MAE: 0.6839** (vs 基线 0.7231)
- **Pearson: 0.6516** (vs 基线 0.6413)

**相对提升** (vs 1-epoch baseline):
- Accuracy: **+6.9%** (从 0.641 → 0.685)
- F1 weighted: **+21.1%** (从 0.560 → 0.679)
- MAE: **-5.4%** (从 0.723 → 0.684)
- Pearson: **+1.6%**

## 🔧 训练配置 (最终版)

```yaml
model: LateFusion
hidden_dim: 320
dropout: 0.45
batch_size: 64
lr: 3e-4
weight_decay: 5e-4
warmup_epochs: 5
label_smoothing: 0.1
optimizer: AdamW
lr_scheduler: Warmup + Cosine Decay
grad_clip: 1.0
early_stop_patience: 30
epochs: 100 (实际早停在 epoch ~30)
seed: 2024
```

## 🏗️ 模型架构

LateFusionModel:
- text encoder: Linear(text_dim=768, 2*hidden=640) → LayerNorm → GELU → Dropout → Linear(hidden=320) → LayerNorm → GELU → Dropout
- audio encoder: 同上 (audio_dim=74)
- vision encoder: 同上 (vision_dim=35)
- fusion: Linear(3*hidden=960, hidden=320) → LayerNorm → GELU → Dropout
- shared head: Linear(hidden, hidden/2=160) → LayerNorm → GELU → Dropout
- cls_head: Linear(hidden/2, 3)
- reg_head: Linear(hidden/2, 1)

参数量: ~1.05M
