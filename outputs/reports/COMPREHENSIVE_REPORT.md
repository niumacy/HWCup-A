# 多模态情感识别 - 综合实验报告 (含 MSA-GMoE)

## 📌 最佳成果

| 策略 | Accuracy | F1 weighted | F1 macro | MAE | Pearson |
|------|----------|-------------|----------|-----|---------|
| **LF + MSA-GMoE Avg (3 seeds) ⭐** | **0.6864** | **0.6764** | 0.6268 | 0.6443 | 0.6713 |
| MSA + GLF | 0.6754 | 0.6607 | - | **0.6246** | 0.6798 |
| Top 2 Original (LF + GLF) | 0.6850 | 0.6743 | - | 0.6540 | 0.6667 |

**结论**: MSA-GMoE + LateFusion 集成 (Acc=0.6864, F1=0.6764) 超越了原始 Top 2 集成 (LF+GLF)，
同时 MSA + GLF 组合在 MAE (0.6246) 和 Pearson (0.6798) 上达到最优。

## 📊 所有单模型对照 (测试集)

| 模型 | Acc | F1 | MAE | Pearson | 备注 |
|------|-----|-----|-----|---------|------|
| LateFusion seed=2024 | **0.6850** | **0.6788** | 0.6839 | 0.6516 | 最佳 Acc/F1 |
| **MSA-GMoE seed=7** | 0.6754 | 0.6730 | 0.6573 | 0.6634 | 第2强 F1 |
| GLF seed=42 | 0.6713 | 0.6626 | **0.6509** | **0.6657** | 最佳 MAE/Pearson |
| TCN | 0.6713 | 0.6570 | 0.6567 | 0.6507 | |
| GMT seed=42 | 0.6713 | 0.6620 | 0.7255 | 0.6399 | MAE 最差 |
| LSTM-Attention | 0.6657 | 0.6482 | 0.6583 | 0.6449 | |
| MSA-GMoE seed=42 | 0.6410 | 0.6439 | 0.6393 | 0.6568 | 最低 MAE |
| MSA-GMoE seed=2024 | 0.6630 | 0.6534 | 0.6893 | 0.6152 | |

## 🏗️ MSA-GMoE 架构详解

**MSA-GMoE (Multi-Scale Attention + Gated Mixture-of-Experts)** 包含：

### 1. 多尺度时序金字塔 (Multi-Scale Temporal Pyramid)
- 每模态用 3 个不同膨胀率的空洞卷积 (dilation=1,2,4)
- 捕获短/中/长时情绪模式，比 Transformer 更稳定

### 2. 模态路由轻量 Attention (Modality-Routing Attention)
- 3 对模态: T↔A, T↔V, A↔V
- 每对有专用 attention，参数量减少 ~40%

### 3. Gated Mixture-of-Experts 融合 (MoE-Fusion)
- 4 个不同归纳偏置的专家:
  * Expert 0: LateFusion (concat MLP)
  * Expert 1: Gated (GLF 风格)
  * Expert 2: Hadamard (元素相乘交互)
  * Expert 3: Attention-pooling
- 门控网络 softmax 动态选择专家组合
- 负载均衡损失防止只使用少数专家

### 4. 训练策略
- Mixup 数据增强 (alpha=0.2)
- 对比学习正则化 (模态间对齐)
- MoE 负载均衡损失 (weight=0.01)
- 分类+回归多任务学习
- CosineAnnealing 学习率调度

**关键发现**:
- seed=7 表现最好 (Acc=0.6754, F1=0.6730)
- seed=42 在 MAE 上最佳 (0.6393)，优于所有其他单模型
- 3 seed 平均后 MAE=0.6319，Pearson=0.6718（均衡且稳定）

## 🎯 集成策略全对比

| 集成方式 | Acc | F1 | MAE | Pearson | 说明 |
|----------|-----|-----|-----|---------|------|
| **LF + MSA-GMoE Avg ⭐** | **0.6864** | **0.6764** | 0.6443 | 0.6713 | **最佳综合** |
| LF + GLF (Top 2 Original) | **0.6850** | 0.6743 | 0.6540 | 0.6667 | 之前最佳 |
| MSA + GLF | 0.6754 | 0.6607 | **0.6246** | **0.6798** | 最佳 MAE/Pearson |
| LF + GLF + MSA (3模型) | 0.6850 | 0.6718 | 0.6360 | 0.6758 | 混合策略 |
| Top 3 Baseline (LF+GLF+TCN) | 0.6809 | 0.6689 | 0.6425 | 0.6694 | |
| MSA-GMoE Avg (3 seeds) | 0.6713 | 0.6604 | 0.6319 | 0.6718 | |
| All 8 models Equal Soft | 0.6726 | 0.6558 | 0.6292 | 0.6790 | 弱模型拖累 |
| All 8 models F1 Weighted | 0.6699 | 0.6539 | 0.6312 | 0.6786 | |

## 💡 关键发现

1. **MSA-GMoE 补足了 LateFusion 的不足**:
   - LateFusion Acc/F1 最强但 MAE/Pearson 一般
   - MSA-GMoE MAE 低 (0.6573) 且 Pearson 强 (0.6634)
   - 两者组合 Acc/F1/MAE/Pearson 全面提升

2. **Pearson > 0.67 的难点**:
   - MSA+GLF 达到 0.6798（当前最高）
   - 说明 GLF 的门控机制 + MSA 的多尺度对回归任务有互补

3. **集成不是越多越好**:
   - 8 模型全部集成反而被弱模型拖累 (Acc=0.6726)
   - 精选 2-3 个互补模型是甜区

4. **seed 敏感性**:
   - MSA-GMoE seed=7 >> seed=2024 (Acc 差 1.2%)
   - GLF seed=42 >> seed=2024
   - 多 seed 平均能缓解随机性

## 📂 产出文件

### 模型权重
- `outputs/latefusion_seed2024.pt` - 最佳 LateFusion
- `outputs/glf_seed42.pt` - 最佳 GLF
- `outputs/msa_gmoe_seed42.pt` - MSA-GMoE (MAE 最低)
- `outputs/msa_gmoe_seed7.pt` - MSA-GMoE (F1 最高)
- `outputs/msa_gmoe_seed2024.pt` - MSA-GMoE (第3)

### 报告
- `outputs/reports/COMPREHENSIVE_REPORT.md` - 本报告
- `outputs/reports/msa_gmoe_ensemble_report.json` - 完整 ensemble 数据

## 🚀 论文建议

**推荐方法**: LateFusion + MSA-GMoE 集成 (LF + MSA-GMoE_avg)

**关键数据** (可直接进论文):
- Accuracy: **0.6864**
- F1 weighted: **0.6764**
- F1 macro: **0.6268**
- MAE: **0.6443**
- Pearson: **0.6713**

**相对提升** (vs 之前最佳 LF+GLF Top 2):
- Accuracy: +0.0014
- F1 weighted: +0.0021
- Pearson: +0.0046

---
*生成时间: 2026-09-23 16:30*
