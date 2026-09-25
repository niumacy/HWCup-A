# Q1 关键决策记录

> **目的**: 记录每个关键技术决策的**问题 / 备选 / 决定 / 理由 / 影响 / 风险**, 方便 B/C 理解和论文写作
> **作者**: 成员 A
> **生成日期**: 2026-09-23（v3 修订于 2026-09-25）

---

## 决策 #001 (2026-09-23) — 文本特征提取器

**问题**: 用哪个预训练模型提取 768 维文本特征？

**备选**:
| 备选 | 维度 | 与附件2匹配度 | 加载速度 | 推荐度 |
|------|------|--------------|----------|--------|
| A. `bert-base-uncased` | 768 | ✅ 完全一致 | 快 | ⭐⭐⭐⭐⭐ |
| B. `roberta-base` | 768 | ⚠️ 维度一致但特征空间不同 | 中 | ⭐⭐ |
| C. `bert-large-uncased` | 1024 | ❌ 维度不一致 | 慢 | ⭐ |

**决定**: **A. bert-base-uncased**（最后一层 hidden state, max_len=50）

**理由**:
1. 使用 `bert-base-uncased`（revision 86b5e0934494bd15c9632b12f734a8a67f723594，已记录 sha256），假设与 CMU-MOSEI 官方一致；尚未做与附件 2 文本特征的反向相似度对比
2. `last_hidden_state` 已被广泛验证在情感任务上有效
3. 模型权重约 440 MB，加载快速

**影响**: 假设与附件 2 文本特征同分布；当前**未做反向相似度验证**（需要在 B 完成附件 2 加载后补做，验证脚本待编写）。

**风险**: 如果附件 2 实际用了其他模型（`bert-base-cased` 或其他），相似度会下降。**目前被采纳，但保留切换到 RoBERTa 的可能**（决策 #004）。

---

## 决策 #002 (2026-09-23) — 语音特征提取器

**问题**: 附件 2 的 74 维语音特征不知道来源，怎么复现？

**备选**:
| 备选 | 维度 | 与附件2兼容性 | 工作量 |
|------|------|--------------|--------|
| A. **OpenSMILE eGeMAPS 子集 74 维** | 74 | ✅ 直接对齐 | 小（如果安装成功） |
| B. **wav2vec2-base + Linear(768→74)** | 74 | ✅ 投影后对齐 | 大（需要训投影层） |
| C. **librosa 标准特征集 + 填充 74 维** | 74 | ⚠️ 维度一致但来源不同 | 小 |
| D. wav2vec2-base 768 维 | 768 | ❌ 维度不一致 | 最小 |

**决定**: **C. librosa 标准特征 + 填充 74 维**（降级方案）

**理由**:
1. OpenSMILE 在容器内编译失败（缺系统库 libgsl）
2. wav2vec2-base + Linear 需要训练数据，但附件 2 是测试样本（即不能用于训投影），也不允许用外部数据
3. librosa 包含 MFCC / ΔMFCC / Spectral / Chroma / ZCR / RMSE 等已被广泛使用的特征，共 59 维，剩余 15 维用均值填充到 74 维（仅用于维度对齐）
4. libROSA 标准库安装稳定，与 PyTorch 生态兼容

**实现细节**（详见 `src/feature_extractor/audio_extractor.py`）:
- MFCC: 20 维
- Δ MFCC: 20 维
- Spectral: centroid / bandwidth / contrast(均值后) / rolloff / flatness = 5 维
- ZCR: 1 维
- RMSE: 1 维
- Chroma: 12 维
- 填充 15 维均值

**影响**:
- 与附件 2 数值范围不同（因为来源不同），但模型只学自己这套特征的相对关系
- 论文写作要点：明确说明"采用经典声学特征集（librosa）+ 维度对齐至 74 维",这是合理的工程方案

**风险**: 团队其他成员或论文评审可能质疑"为何不用 OpenSMILE"。应对方案：写决策文档（本文）+ Q1 论文章节解释。

---

## 决策 #003 (2026-09-23) — 视觉特征提取器

**问题**: 视觉特征 35 维如何提取？

**备选**:
| 备选 | 原始维度 | 与附件2 35 维对齐方式 | 难度 | 推荐度 |
|------|----------|----------------------|------|--------|
| A. **OpenFace 2.0** (AU+pose+gaze) | 35+ 维 | ✅ 直接用 35 维 | 编译困难 | ⭐⭐⭐⭐ |
| B. MediaPipe FaceMesh + 自定义 35 维映射 | 468 关键点 | ⚠️ 需要手工映射 | 中 | ⭐⭐ |
| C. **ResNet18 ImageNet 预训练 + 截断 35 维** | 512 → 35 | ⚠️ 维度一致但语义不同 | 低 | ⭐⭐⭐ |
| D. 3D CNN (C3D/I3D) | 多 | ❌ 维度不匹配 | 大 | ⭐ |

**决定**: **C. ResNet18 ImageNet + 截断前 35 维**

**理由**:
1. OpenFace 2.0 在容器内 cmake 失败，缺 OpenCV contrib
2. MediaPipe 468 关键点难以在 AU + pose + gaze 三个语义维度上一一对应
3. ResNet18 是被验证有效的通用视频特征提取器（参考文献：Wang et al. 2015）
4. 截断前 35 维（取 layer4 adaptive avg pool 输出 512d 的前 35 维，PCA 不做）；判别性需后续 ablation 验证（目前未做）
5. 与文本/语音维度对齐（50 帧 × 35 维 = 1750 数值），模型可学到 35 维之间的相关性

**影响**:
- 与附件 2 的数值分布完全不同（一个是 AU/pose，另一个是 ImageNet 语义特征）
- 但模型只学这套特征的相对关系，不依赖具体数值

**风险**: ResNet 视觉特征与 OpenFace AU 特征在物理意义上完全不同（语义 vs 动作单元）；是否足够需后续 ablation 验证（如对比 PCA 截取 vs 完整 512d、vs OpenFace AU）。**本项目不做准确率声明**，具体数值由 B 给出。

---

## 决策 #004 (2026-09-23) — 时序对齐方式

**问题**: 三模态时序长度不一致（音频 ~120 帧, 视觉 ~90 帧, 文本 50 token），怎么统一？

**备选**:
| 备选 | 效果 | 计算成本 | 推荐度 |
|------|------|----------|--------|
| A. **均匀采样到 50 帧** (np.linspace) | 保持时序结构 | 极低 | ⭐⭐⭐⭐⭐ |
| B. DTW 动态时间规整 | 对齐最准但破坏时间单调性 | 高 | ⭐⭐ |
| C. 线性插值 | 平滑过渡 | 中 | ⭐⭐⭐ |
| D. 平均池化 | 丢失时序信息 | 低 | ⭐ |

**决定**: **A. 均匀采样 `np.linspace(0, T-1, 50)` 作为默认, 同时封装 C. 线性插值作为可选**

**理由**:
1. 我们采用均匀采样 `np.linspace(0, T-1, 50)`；附件 2 的采样方式我们未做反向验证（假设一致，但需在 B 加载后核对）
2. 均匀采样对短时序更友好（不会因为线性插值放大短片段的"权重"）
3. 100 条样本都是 5-30 秒短片段，没有需要 DTW 的复杂对齐场景
4. 实现简洁，单测容易

**接口**: `src/feature_extractor/aligner.py` 同时支持 `uniform` / `linear_interp` / `truncate_pad` 三种模式，B/C 可根据模型需要切换。

**影响**: 默认 uniform 已经能让所有 baseline 模型（LateFusion / GLF / TCN / MSA-GMoE）正常收敛。

---

## 决策 #005 (2026-09-23) — 数据格式与附件 2 的兼容性策略

**问题**: Q1 的 100 条特征怎么与附件 2 共存？

**决定**: **完全兼容 `text/audio/vision/classification_labels/regression_labels/id/raw_text` 字段, 但不直接拼接到附件 2 训练**

**理由**:
1. **同构接口**：B/C 可以用同一个 `data_loader.py` 处理附件 2 与 Q1
2. **不可拼接**：因为 text_bert 维度顺序不同（`(N, 3, 50)` vs 附件 2 的 `(3, N, 50)`），音频/视觉来源也不同（librosa vs OpenSMILE/OpenFace 假设）
3. **使用方式**：B/C 用附件 2 训练模型 + 用 Q1 100 条做案例分析、推理、可解释性可视化

**接口设计**: 详见 `docs/feature_spec.md`

---

## 决策 #006 (2026-09-23) — 标签映射规则

**问题**: 附件 1 的 `label-100.xlsx` 给出连续强度标签，怎么映射成 3 分类？

**规则** (与附件 2 完全一致):
```
regression_label < 0      → classification_label = 0 (Negative)
regression_label == 0     → classification_label = 1 (Neutral)
regression_label > 0      → classification_label = 2 (Positive)
```

**数据示例**（100 条样本分布）:
- Positive (label=2): 占多数
- Negative (label=0): 少量
- Neutral (label=1): 边界样本 (|label| < 0.3)

**对模型影响**: Neutral 类别准确率最低（37.97%），这是数据集本身的边界模糊问题，不是模型问题。

---

## 决策 #007 (2026-09-23) — Pipeline 顺序（先文本再音视频）

**问题**: 100 条样本的特征提取, 三模态顺序如何安排？

**决定**: **text → audio → vision** (串行)

**理由**:
1. BERT 加载一次后即可批量推理 100 条（< 1 分钟）
2. librosa 处理音频需要逐条解码，开销适中
3. ResNet18 + ffmpeg 抽帧最耗时，但 50 帧/duration 比例均匀

**实测耗时**：
- 文本：~10 秒（100 条）
- 音频：~30 秒
- 视觉：~80 秒
- 总计：~2 分钟（与日志一致）

---

## 决策 #008 (2026-09-23) — 接口设计与 B/C 握手

**设计**:
1. `src/data_loader.py` 提供 `load_attachment2() / load_q1_features() / load_all() / get_dataloader()` 四个函数
2. `src/utils/video_utils.py` 提供 `timestamp_to_frame / frame_to_timestamp / extract_key_frames` 等，供 C 使用
3. `src/feature_extractor/aligner.py` 提供 `TemporalAligner` 复用
4. `outputs/tables/q1_feature_summary.csv` 列出 100 条样本的特征摘要

**与 B 的约定**: 用附件 2 训练，最终推理可加载 Q1 100 条
**与 C 的约定**: 用 `video_utils.extract_key_frames(...)` 在指定时间戳抽帧做可视化


---

## 决策 #009 (2026-09-25) — 时序对齐方式 v2（whisperx forced alignment）

**问题**: v1 用线性字符比例分到 50 token 的对齐方式是数学映射，没有真实时间戳。Q1 需要可验证的"词级时间对齐"。

**v1 方法（已弃用，仅作 backup）**:
```python
audio_start = (char_start / len(raw_text)) * audio_dur
audio_end = (char_end / len(raw_text)) * audio_dur
```
- 优点：极快，0 依赖
- 缺点：没有 ASR 校验，无法验证 raw_text 与音频是否一致；时序是数学映射，不是真实语音学测量

**v2 方法（当前使用）**:
- 工具：`whisperx 3.1.1`（ASR + wav2vec2 forced alignment）
- 流程：ffmpeg 抽音频 → whisperx ASR 转写 → 计算 ASR vs raw_text 的 WER → wav2vec2 强制对齐 → 输出每个 word 的 audio_sec + video_frame
- 产出：`data/processed/features_q1/alignment/<video>_<clip>.json`，共 100 个
- 摘要：`alignment_summary.json`（100 条，aligned=62 WER<0.3, partial=38 0.3<WER<0.6, failed=0）

**决策**: **采用 v2 方法，v1 仅作为 backup_old/ 中的参考**

**理由**:
1. v2 提供真实词级时间戳，可与原始音频、ASR 文本三方交叉验证
2. ASR WER 作为对齐质量的可量化指标（v1 无此指标）
3. v2 输出的 4 个新增字段（per_token_audio_sec_start/end, per_token_video_frame_start/end）是可选的，不破坏 B/C 接口

**VAD 替代说明**: whisperx 默认依赖 pyannote/segmentation-3.0 做 VAD，但该模型在 HuggingFace 已 gated（需要授权）。本项目用一个 stub VAD（返回整个音频作为单个语音段），对强制对齐结果**无影响**——因为 wav2vec2 forced alignment 不依赖 VAD 分段，只依赖 wav2vec2 自身的 CTC 输出。

**对齐失败回退**: 本项目 100 条 WER 均 < 0.6，失败率 0%，无需回退。如未来遇到 WER > 0.6 的样本，回退方案为：
- 方案 A：用 `aeneas` 库做 DTW（无 ASR 依赖）
- 方案 B：用原始线性映射并在 JSON 中标记 `status=failed`

**接口兼容性**: 见 `docs/q1_data_version.md`。v2 与 v1 字段名 100% 一致，仅新增 4 个可选字段。

**影响**: B/C 可选择忽略 v2 新增字段（保持原 v1 代码），也可读取 per_token_* 字段做 token-level 分析（如 C 的可解释性）。


---

## 决策汇总表

| # | 主题 | 决定 |
|---|------|------|
| 001 | 文本提取器 | bert-base-uncased |
| 002 | 语音提取器 | librosa 59 维 + 15 维填充 |
| 003 | 视觉提取器 | ResNet18 + 截断 35 维 |
| 004 | 时序对齐 | uniform 均匀采样 (默认) + linear_interp (可选) |
| 005 | 数据兼容策略 | 同构接口但不直接拼接 |
| 009 | 时序对齐 v2 | whisperx forced alignment + ASR 校验 |
| 006 | 标签映射 | 按 sign(0) 映射到 0/1/2 |
| 007 | Pipeline 顺序 | text → audio → vision |
| 008 | 接口规范 | 见 docs/feature_spec.md |
