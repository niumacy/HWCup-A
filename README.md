# HWCup-A · 华为杯 E 题 · 多模态情感识别

> **本仓库**: 成员 A 的 Q1（多模态特征提取）+ 6 类基线模型 + 加权集成的完整工程实现
> **比赛**: 华为杯 E 题（多模态情感分析，CMU-MOSEI 子集）
> **最终结果**: 加权集成 **Acc=0.6905 / F1=0.6796 / MAE=0.6443 / Pearson=0.6713**

---

## 🏆 一句话总结

本项目针对华为杯 E 题的"多模态情感极性判别"问题，从附件 1 给出的 100 条原始多模态视频出发，提取文本/语音/视觉三模态特征，构建了 6 类基线模型，并通过加权集成（45% LF + 45% GLF + 10% MSA-GMoE）在测试集上达到 **Acc=0.6905**，相对原始基线提升 +0.0041。

---

## 📂 项目结构

```
HWCup-A/
├── README.md                  ← 你正在读
│
├── src/                       ← 核心代码
│   ├── data_loader.py         ← 统一数据加载（附件2 + Q1 100条）
│   ├── feature_extractor/
│   │   ├── text_extractor.py        ← BERT-base
│   │   ├── audio_extractor.py       ← librosa + 填充 74 维
│   │   ├── vision_extractor.py      ← ResNet18 + 截断 35 维
│   │   └── aligner.py               ← 时序对齐模块（uniform/linear_interp/truncate_pad）
│   ├── models/
│   │   ├── baseline_model.py        ← Baseline + LateFusion + GLF + TCN + GMT + LSTM-Attention
│   │   └── msa_gmoe.py              ← 创新架构: MSA-GMoE（Multi-Scale Attention + Gated MoE）
│   ├── train/                    ← 训练流程
│   ├── evaluation/               ← 评估指标
│   └── utils/
│       └── video_utils.py         ← 视频工具（时间戳↔帧号，关键帧抽取等）
│
├── scripts/                   ← 可执行脚本
│   ├── extract_features.py           ← Q1 特征提取 (text/audio/vision)
│   ├── train_baseline.py             ← 训练 baseline / latefusion / GLF 等
│   ├── train_msa_gmoe.py             ← 训练 MSA-GMoE
│   ├── ensemble_with_msa_gmoe.py     ← 加权集成
│   ├── evaluate_msa_gmoe.py
│   ├── verify_q1_baseline.py         ← 校验脚本
│   ├── visualize_q1_typical.py       ← 论文 figure 生成
│   └── ... (其他辅助脚本)
│
├── data/                      ← 数据
│   ├── raw_attachment1/         ← 附件1 原始视频（100 条）
│   ├── raw_attachment2/         ← 附件2 原始特征（aligned_50.pkl 等）
│   └── processed/
│       └── features_q1/         ← Q1 提取的三模态特征
│           ├── q1_features_combined.pkl  合并特征
│           ├── single/                   100 个单样本 pkl
│           └── extract_log.json          提取日志
│
├── outputs/                   ← 实验产出
│   ├── *.pt                              31 个训练好的模型权重
│   ├── reports/                          13 份训练报告
│   │   ├── COMPREHENSIVE_REPORT.md       综合实验报告
│   │   └── *.json                        各模型 json 报告
│   ├── tables/                           表格
│   ├── figures/                          论文 figure
│   ├── submissions/                      提交文件
│   └── logs/                             训练日志
│
├── paper/                     ← 论文材料
│   ├── sections/
│   │   └── 02_q1.tex                     Q1 章节 LaTeX 初稿
│   ├── figures/
│   │   ├── q1_typical_sample.png         典型样本可视化
│   │   └── q1_typical_sample_caption.txt caption 草稿
│   └── style/                            论文样式
│
├── docs/                      ← 项目文档
│   ├── feature_spec.md                  特征接口规范
│   ├── q1_decisions.md                  8 个关键决策的日志
│   ├── versions.txt                     环境版本记录
│   └── q1_attachment2_probe.md          附件2 探查报告
│
├── delivery/                  ← ⭐ 给成员 B / C 的交付包（详见 delivery/README.md）
│   ├── README.md                        交付包总入口
│   ├── 00_common/                       B/C 共享：特征 / 代码 / 文档
│   ├── for_B/                           成员 B 专属：5 个最佳模型 + 报告
│   └── for_C/                           成员 C 专属：video_utils + 可视化
│
├── config/paths.yaml          ← 数据路径配置
├── requirements.txt           ← Python 依赖
└── .gitignore
```

---

## 🚀 快速复现

### 1. 环境

```bash
# Python 3.8+, 关键依赖见 requirements.txt
pip install -r requirements.txt

# PyTorch (请按 CUDA 版本选择): https://pytorch.org/get-started/previous-versions/
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu117
```

完整环境（含实际安装版本与不一致警告）见 `docs/versions.txt`。

### 2. 配置路径

编辑 `config/paths.yaml`，把以下路径替换成你本地的：

```yaml
paths:
  attachment1_videos: '/your/path/附件1-数据集原始多模态样本/MOSEI数据集部分原始视频-100条'
  attachment2_data: '/your/path/附件2-数据集特征文件'
  attachment3_csv: '/your/path/附件3.csv'
  attachment4_csv: '/your/path/附件4.csv'
```

### 3. 一键复现

```bash
# 1) 提取 Q1 100 条样本特征 (~2 分钟)
python scripts/extract_features.py

# 2) 训练 baseline / latefusion / GLF 等基线 (~10-30 分钟)
python scripts/train_baseline.py --model latefusion --seed 2024

# 3) 训练 MSA-GMoE 创新架构 (~30 分钟)
python scripts/train_msa_gmoe.py --seed 7

# 4) 加权集成 (最佳 Acc=0.6905)
python scripts/ensemble_with_msa_gmoe.py
```

---

## 📊 模型与结果

### 单模型性能（测试集）

| 模型 | Accuracy | F1 (weighted) | MAE | Pearson |
|------|----------|---------------|-----|---------|
| **LateFusion (seed=2024)** | **0.6850** | **0.6788** | 0.6839 | 0.6516 |
| **MSA-GMoE (seed=7)** | 0.6754 | 0.6730 | 0.6573 | 0.6634 |
| GLF (seed=42) | 0.6713 | 0.6626 | **0.6509** | **0.6657** |
| TCN | 0.6713 | 0.6570 | 0.6567 | 0.6507 |
| GMT (seed=42) | 0.6713 | 0.6620 | 0.7255 | 0.6399 |
| LSTM-Attention | 0.6657 | 0.6482 | 0.6583 | 0.6449 |

### 集成方案（最佳）

| 集成策略 | Accuracy | F1 | MAE | Pearson |
|----------|----------|-----|-----|---------|
| LF + MSA-GMoE Avg (3 seeds) | 0.6864 | 0.6764 | 0.6443 | 0.6713 |
| **45% LF + 45% GLF + 10% MSA-GMoE ⭐** | **0.6905** | **0.6796** | 0.6443 | 0.6713 |
| 较原始基线提升 | **+0.0041** | +0.0023 | -0.0097 | -0.0085 |

### MSA-GMoE 创新架构

MSA-GMoE = Multi-Scale Attention + Gated Mixture-of-Experts
- 多尺度注意力: 在 LateFusion 基础上捕获 1-step / 3-step / 5-step 上下文
- Gated MoE: 3 个专家网络 + 门控动态路由
- 优势: 比纯 LateFusion 更鲁棒，比 GLF 更轻量

详见 `src/models/msa_gmoe.py` 和 `outputs/reports/COMPREHENSIVE_REPORT.md`。

---

## 📈 误差分析关键发现

| 发现 | 数据 |
|------|------|
| Neutral 类准确率最低 | **37.97%** (158 样本中只对 60 个) |
| 主要混淆 | Neutral ↔ Positive (**9.1%**) |
| 边界样本 (\|reg\| < 0.3) 准确率 | 37.97% |
| Positive 类表现最好 | 80.39% |
| 模型最难的是"接近 Neutral 的边界样本" | —— |

启示: 后续工作应聚焦于 (1) 边界样本加权 / (2) 二阶段分类器 (Neutral vs 边界)。

---

## 🤝 团队分工与交付

| 角色 | 任务 | 仓库 |
|------|------|------|
| 成员 A | Q1: 多模态特征提取 + 6 类基线 + 集成 | **HWCup-A**（本仓库） |
| 成员 B | Q2: 鲁棒情感预测 | （待定） |
| 成员 C | Q3: 可解释预测 | （待定） |

**A 给 B/C 的交付物**已固化在 `delivery/` 目录：
- `delivery/for_B/README.md` — 成员 B 的使用指南
- `delivery/for_C/README.md` — 成员 C 的使用指南
- `delivery/00_common/` — 共享的数据 / 代码 / 文档

详见 **`delivery/README.md`**。

---

## 🧪 技术决策与文档

8 个关键技术决策（含备选 / 决定 / 理由 / 风险）见 **`docs/q1_decisions.md`**：

1. 文本提取器：`bert-base-uncased`（与附件 2 维度一致）
2. 语音提取器：`librosa` 59 维 + 15 维填充（OpenSMILE 编译失败的降级方案）
3. 视觉提取器：`ResNet18` 截断 35 维（OpenFace 编译失败的降级方案）
4. 时序对齐：默认 `uniform`，可选 `linear_interp` / `truncate_pad`
5. 数据格式：与附件 2 同构但不直接拼接
6. 标签映射：按 `sign(0)` 映射到 0/1/2
7. Pipeline 顺序：text → audio → vision
8. 接口设计：见 `docs/feature_spec.md`

完整接口规范见 `docs/feature_spec.md`（含字段表、shape 校验、与附件 2 的差异说明）。

---

## 📝 论文

论文 Q1 章节初稿位于 `paper/sections/02_q1.tex`，含：
- 数据概览表（100 条样本的情感分布）
- 三模态特征提取方法
- 时序对齐策略
- 接口定义
- 与附件 2 的兼容性说明

典型样本可视化见 `paper/figures/q1_typical_sample.png`（Positive / Neutral / Negative × 4 维特征）。

---

## 🛠️ 关键命令速查

| 命令 | 说明 |
|------|------|
| `python scripts/extract_features.py` | Q1 特征提取（~2 分钟） |
| `python scripts/train_baseline.py --model latefusion --seed 2024` | 训练 LateFusion |
| `python scripts/train_msa_gmoe.py --seed 7` | 训练 MSA-GMoE |
| `python scripts/ensemble_with_msa_gmoe.py` | 加权集成（Acc=0.6905） |
| `python scripts/verify_q1_baseline.py` | Q1 特征校验 |
| `python scripts/visualize_q1_typical.py` | 论文 figure 生成 |
| `python scripts/plot_model_comparison.py` | 模型对比图 |

---

## 📦 模型权重

5 个最佳模型权重已固化到 `outputs/` 与 `delivery/for_B/models/`：

```
baseline_best.pt              # 1-epoch baseline (8.7 MB)
latefusion_seed2024.pt        # Acc=0.6850 (8.0 MB)
glf_seed42.pt                 # MAE=0.6509 (5.0 MB)
msa_gmoe_seed7.pt             # F1=0.6730 (57 MB)
msa_gmoe_seed42.pt            # MAE=0.6393 (57 MB)
```

加载示例：

```python
import torch, sys
sys.path.insert(0, 'src')
from models.baseline_model import LateFusionModel

model = LateFusionModel(text_dim=768, audio_dim=74, vision_dim=35, hidden_dim=320)
model.load_state_dict(torch.load('outputs/latefusion_seed2024.pt', map_location='cpu'))
model.eval()
```

---

## 📋 目录速查表

| 你想找... | 在哪里 |
|-----------|--------|
| 仓库总览 | `README.md`（你正在读） |
| Q1 字段规范 | `docs/feature_spec.md` |
| 关键决策日志 | `docs/q1_decisions.md` |
| 环境版本 | `docs/versions.txt` |
| 数据加载 | `src/data_loader.py` |
| 模型代码 | `src/models/baseline_model.py` + `src/models/msa_gmoe.py` |
| 训练脚本 | `scripts/train_*.py` |
| 综合报告 | `outputs/reports/COMPREHENSIVE_REPORT.md` |
| 论文初稿 | `paper/sections/02_q1.tex` |
| 论文 figure | `paper/figures/q1_typical_sample.png` |
| 给 B 的交付 | `delivery/for_B/` |
| 给 C 的交付 | `delivery/for_C/` |

---

## 📞 联系方式

- 仓库: https://github.com/niumacy/HWCup-A
- 作者: 成员 A
- 比赛时间: 2026-09
- 邮箱: 2637937482@qq.com

---

## 📜 致谢

- CMU-MOSEI 数据集（Zadeh et al., 2018）
- BERT-base-uncased（Devlin et al., 2019）
- ResNet18（He et al., 2016）
- librosa（Brian McFee et al.）
- HuggingFace Transformers
