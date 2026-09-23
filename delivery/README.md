# Q1 交付包（成员 A → 成员 B / 成员 C）

> **生成时间**: 2026-09-23
> **生成人**: 成员 A
> **接收人**: 成员 B（Q2 鲁棒预测）、成员 C（Q3 可解释预测）
> **内容**: Q1 多模态特征提取与时序对齐的全部交付物

---

## 📁 目录结构

```
delivery/
├── README.md                              ← 本文件（你正在读）
│
├── 00_common/                             ← B 和 C 都需要的内容
│   ├── data/                              ← Q1 提取的三模态特征
│   │   ├── q1_features_combined.pkl       ← 合并特征（17 MB）
│   │   ├── q1_feature_summary.csv         ← 100 条样本特征摘要表
│   │   ├── extract_log.json               ← 提取日志
│   │   └── single/                        ← 100 个单样本 pkl（~18 MB）
│   ├── code/                              ← 通用代码
│   │   ├── requirements.txt
│   │   ├── src/
│   │   │   ├── data_loader.py             ← 统一数据加载接口（必看）
│   │   │   └── feature_extractor/
│   │   │       └── aligner.py             ← 时序对齐模块
│   │   └── scripts/
│   │       └── verify_q1_baseline.py      ← 校验脚本（必跑）
│   └── docs/                              ← 文档
│       ├── README.md                      ← 文档导航
│       ├── feature_spec.md                ← 字段说明（必看）
│       ├── q1_decisions.md                ← 关键决策记录
│       ├── versions.txt                   ← 环境版本记录
│       ├── q1_attachment2_probe.md        ← 附件2 探查报告
│       └── 02_q1.tex                      ← 论文 Q1 章节初稿
│
├── for_B/                                 ← B 专属（Q2 鲁棒预测）
│   ├── README.md                          ← B 的使用指南（必读）
│   ├── code/
│   │   └── baseline_model.py              ← Baseline + LateFusion + GLF + TCN 等
│   ├── models/                            ← 5 个最佳模型权重
│   │   ├── baseline_best.pt               ← 1 epoch baseline
│   │   ├── latefusion_seed2024.pt         ← Acc=0.685 最佳单模型
│   │   ├── glf_seed42.pt                  ← MAE=0.651 最强回归
│   │   ├── msa_gmoe_seed7.pt              ← F1=0.673 创新架构
│   │   └── msa_gmoe_seed42.pt             ← MAE=0.639 最低
│   └── reports/                           ← 13 份训练报告
│       ├── COMPREHENSIVE_REPORT.md        ← 综合报告（必读）
│       ├── final_optimized_result.json    ← 当前最优集成结果
│       └── ...                            ← 其他单模型报告
│
└── for_C/                                 ← C 专属（Q3 可解释预测）
    ├── README.md                          ← C 的使用指南（必读）
    ├── code/
    │   ├── video_utils.py                 ← 视频工具（必用）
    │   └── visualize_q1_typical.py        ← 典型样本可视化脚本
    ├── figures/
    │   ├── q1_typical_sample.png          ← 论文 figure
    │   └── q1_typical_sample_caption.txt  ← caption
    └── extra_data/
        └── video_metadata.csv             ← 100 条视频的 fps/时长/分辨率
```

---

## 🚀 快速开始

### 给成员 B

```bash
cd delivery/for_B
cat README.md
# 重点看 COMPREHENSIVE_REPORT.md 了解当前最优结果
```

### 给成员 C

```bash
cd delivery/for_C
cat README.md
# 重点看 video_utils.py 的 API 文档和可视化脚本
```

---

## 📊 当前最优结果（截至 2026-09-23）

| 策略 | Accuracy | F1 weighted | MAE | Pearson |
|------|----------|-------------|-----|---------|
| **加权集成 (45% LF + 45% GLF + 10% MSA-GMoE)** | **0.6905** | **0.6796** | 0.6443 | 0.6713 |
| LateFusion 单模型 (seed=2024) | 0.6850 | 0.6788 | 0.6839 | 0.6516 |
| GLF 单模型 (seed=42) | 0.6713 | 0.6626 | **0.6509** | **0.6657** |
| MSA-GMoE 单模型 (seed=7) | 0.6754 | 0.6730 | 0.6573 | 0.6634 |

详见 `for_B/reports/COMPREHENSIVE_REPORT.md`。

---

## 📌 重要兼容性提示

⚠️ **不要把 Q1 的 100 条特征直接拼接到附件 2 训练**
- 原因：text_bert 维度顺序相反、audio/vision 来源不同
- 正确用法：用附件 2 训练，用 Q1 做推理 / 案例分析 / 可解释性可视化

⚠️ **Q1 audio 用 librosa + 填充（74维），附件 2 可能是 OpenSMILE（74维）**
- 数值分布不同，但下游模型只学相对关系，不影响使用

详见 `00_common/docs/feature_spec.md` 第 5 节。

---

## 📞 联系方式

成员 A 的全部 Q1 产出已在此交付包内。
如有疑问或需要新增导出物，请联系成员 A。
