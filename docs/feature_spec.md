# Q1 特征接口规范文档

> **读者对象**：成员 B（Q2 鲁棒预测）、成员 C（Q3 可解释预测）  
> **作者**：成员 A  
> **版本**：v2.0（与 alignment v2 同步）  
> **最后更新**：2026-09-25

本文档定义了 100 条原始样本（附件 1）的三模态特征文件格式与字段语义。所有接口与附件 2 的 `aligned_50.pkl` **保持完全兼容**，可直接通过 `src/data_loader.py` 加载。

---

## 1. 文件位置与命名

```
data/processed/features_q1/
├── single/                              # 100 个单样本 pkl
│   ├── q1_-3g5yACwYnA_13.pkl
│   ├── q1_-3g5yACwYnA_3.pkl
│   └── ...
├── q1_features_combined.pkl            # 合并文件，供 B/C 一行调用
└── extract_log.json                     # 提取日志（含时长、版本号）
```

| 类型 | 格式 | 示例 |
|------|------|------|
| 单样本 | `q1_{video_id}_{clip_id}.pkl` | `q1_-3g5yACwYnA_13.pkl` |
| 合并 | `q1_features_combined.pkl` | 全部 100 条 |
| 提取日志 | `extract_log_YYYYMMDD_HHMMSS.json` | `extract_log.json` |

---

## 2. 单样本 pkl 字段说明

加载示例：
```python
import pickle
with open('q1_-3g5yACwYnA_13.pkl', 'rb') as f:
    feat = pickle.load(f)
```

| 字段 | 类型 | 形状 | dtype | 说明 |
|------|------|------|-------|------|
| `id` | `str` | - | - | 格式 `video_id$_$clip_id`, 例 `'-3g5yACwYnA$_$13'` |
| `video_id` | `str` | - | - | YouTube 视频 ID, 例 `-3g5yACwYnA` |
| `clip_id` | `int` | - | - | 该视频内的片段编号, 例 `13` |
| `raw_text` | `str` | - | - | 原始英文转写文本（来自附件 1） |
| `text` | `np.ndarray` | `(50, 768)` | `float32` | BERT-base 最后一层 hidden states |
| `text_bert` | `np.ndarray` | `(3, 50)` | `int64` | `[input_ids, attention_mask, token_type_ids]` |
| `audio` | `np.ndarray` | `(50, 74)` | `float32` | 音频特征（已对齐到 50 帧） |
| `vision` | `np.ndarray` | `(50, 35)` | `float32` | 视觉特征（已对齐到 50 帧） |
| `annotation` | `str` | - | - | 三分类标签名: `'Negative'` / `'Neutral'` / `'Positive'` |
| `classification_label` | `int` | - | - | 数字标签: `0` (Neg) / `1` (Neu) / `2` (Pos) |
| `regression_label` | `float` | - | - | 连续强度, 范围 `[-3, 3]` |
| `modalities` | `dict` | - | - | 每模态提取状态, 例 `{'text': 'success', 'audio': 'success', 'vision': 'success'}` |

### 2.1 字段语义详解

#### text — 文本特征 (50, 768)
- **来源**：HuggingFace `bert-base-uncased`（最后一层 hidden state）
- **Preprocessing**：`padding='max_length'`, `truncation=True`, `max_length=50`
- **dtype**：`float32`
- **归一化**：未做归一化（HuggingFace 默认输出）

#### text_bert — 文本词袋 (3, 50)
- 顺序固定：`[input_ids, attention_mask, token_type_ids]`
- 可用于重新跑 BERT 或在其他模型里 fine-tune

#### audio — 语音特征 (50, 74)
- **提取器**：`librosa` + 手动 74 维填充（详见 `docs/q1_decisions.md` 决策 #002）
- **组成**：MFCC (20) + ΔMFCC (20) + Spectral (5) + ZCR (1) + RMSE (1) + Chroma (12) + 均值填充 (15)
- **采样率**：16 kHz
- **对齐方式**：`np.linspace(0, T-1, 50)` 等距采样

#### vision — 视觉特征 (50, 35)
- **提取器**：`torchvision.models.resnet18`（ImageNet 预训练） + 截断投影到 35 维
- **帧采样**：均匀采样到 50 帧 + ResNet18 提取每帧 512 维 → 取前 35 维
- **替代**：OpenFace 2.0 编译困难，改用 ResNet18（详见 `docs/q1_decisions.md` 决策 #003）

#### classification_label 与 regression_label
- 与附件 2 的转换规则一致：
  - `regression < 0` → `classification_label = 0` (Negative)
  - `regression == 0` → `classification_label = 1` (Neutral)
  - `regression > 0` → `classification_label = 2` (Positive)
- 标签来源于附件 1 的 `label-100.xlsx`

---

## 3. 合并 pkl 接口（`q1_features_combined.pkl`）

直接喂给 `src/data_loader.py` 即可，与附件 2 `aligned_50.pkl` 同构。

```python
{
    'all': {
        'id': list[str],                       # 100 个
        'raw_text': list[str],                 # 100 个
        'text': (100, 50, 768) float32,
        'text_bert': (100, 3, 50) int64,
        'audio': (100, 50, 74) float32,
        'vision': (100, 50, 35) float32,
        'classification_labels': (100,) int64,
        'regression_labels': (100,) float32,
        'annotation': list[str],

        # === v2 新增字段（可选） ===
        'per_token_audio_sec_start': (100, 50) float32,   # 每个 token 对应音频起秒
        'per_token_audio_sec_end':   (100, 50) float32,   # 每个 token 对应音频止秒
        'per_token_video_frame_start': (100, 50) int32,   # 每个 token 对应视频帧起
        'per_token_video_frame_end':   (100, 50) int32,   # 每个 token 对应视频帧止
    }
}

# v2 字段说明：
#   - 这些字段来自 whisperx forced alignment（见 alignment/<video>_<clip>.json）
#   - 索引 [i, 0] = [CLS]，[i, 49] = [PAD]
#   - 实际词数：text_length_effective = alignment.tokens 中非 PAD 数（5-49）
#   - B/C 可忽略这些字段（保持原 v1 代码），也可读取做 token-level 分析
```

---

## 4. 加载方式（最简）

```python
import sys
sys.path.insert(0, '${REPO_ROOT}')  # ${REPO_ROOT} = 仓库根目录（脱敏后）

from src.data_loader import load_q1_features, get_dataloader

# 1) 加载字典
data = load_q1_features('data/processed/features_q1/q1_features_combined.pkl')
all_data = data['all']
print(all_data['text'].shape)   # (100, 50, 768)

# 2) 转 DataLoader（与附件 2 同样用法）
loader = get_dataloader(all_data, batch_size=64, shuffle=True)
for batch in loader:
    text = batch['text']    # (B, 50, 768)
    audio = batch['audio']  # (B, 50, 74)
    vision = batch['vision'] # (B, 50, 35)
    cls = batch['cls_label']
    reg = batch['reg_label']
    break
```

### 4.1 与附件 2 一起加载

```python
from src.data_loader import load_all

both = load_all(
    version='aligned',
    q1_extra_path='data/processed/features_q1/q1_features_combined.pkl',
)
# both = {
#     'train': <附件2 训练集>,
#     'valid': <附件2 验证集>,
#     'test':  <附件2 测试集>,
#     'q1':    {'all': <成员A 100条>}
# }
```

---

## 5. 与附件 2 接口的差异说明（重要！）

| 维度 | 附件 2 | 成员 A Q1 | 兼容性 |
|------|--------|-----------|--------|
| text 维度 | (N, 50, 768) | (N, 50, 768) | ✅ 完全一致 |
| text_bert 维度 | (3, N, 50) ⚠️ 文档顺序 | (N, 3, 50) | ⚠️ 顺序相反 |
| audio 维度 | (N, 50, 74) | (N, 50, 74) | ✅ 完全一致 |
| vision 维度 | (N, 50, 35) | (N, 50, 35) | ✅ 完全一致 |
| 分类标签 | 0/1/2 | 0/1/2 | ✅ 完全一致 |
| 回归标签 | [-3, 3] | [-3, 3] | ✅ 完全一致 |
| audio 来源 | OpenSMILE 子集 (假设) | librosa + 填充 | ⚠️ 来源不同 |
| vision 来源 | OpenFace 2.0 (假设) | ResNet18 截断 | ⚠️ 来源不同 |
| 数值范围 | 实测需对齐 | 实测有差异 | ⚠️ 不能直接拼接到附件 2 |

> ⚠️ **关键警告**：成员 A 的 Q1 特征不能直接拼接到附件 2 训练（因为提取器不同），但 B/C 可以用附件 2 训练模型，再用 Q1 特征做**推理 / 案例分析 / 可解释性可视化**。

> ⚠️ `text_bert` 的维度顺序差异：附件 2 是 `(3, N, 50)`，本项目是 `(N, 3, 50)`。B/C 调用前请确认自己模型的输入顺序。

---

## 6. 校验代码

```python
# 必跑：验证你的下游代码能加载 Q1 特征
import pickle, numpy as np

with open('data/processed/features_q1/q1_features_combined.pkl','rb') as f:
    d = pickle.load(f)['all']

assert d['text'].shape == (100, 50, 768)
assert d['text_bert'].shape == (100, 3, 50)
assert d['audio'].shape == (100, 50, 74)
assert d['vision'].shape == (100, 50, 35)
assert len(d['id']) == 100
assert set(d['classification_labels'].tolist()).issubset({0, 1, 2})
assert d['regression_labels'].min() >= -3.0 and d['regression_labels'].max() <= 3.0
print('✅ Q1 接口兼容性验证通过')
```

---

## 7. 异常处理约定

- **特征为全零**：表示该模态提取失败（视频损坏 / ffmpeg 错误）。通过 `sample['modalities']['audio']` 检查状态。
- **维度不匹配**：首先确认 `is_aligned` 参数（默认 `True` 即 `q1_features_combined.pkl`），如果想用 `unaligned` 版本，请加载 `unaligned_*.pkl`（本项目未生成）。
- **找不到样本**：使用 `from src.data_loader import load_label_xlsx` 配合 `locate_video_file` 重新定位。

---

## 8. 后续扩展接口（未实现但预留）

| 函数 | 用途 | 当前状态 |
|------|------|----------|
| `load_unaligned_features()` | 加载未对齐版本，保留原始 T | ❌ 未生成（需要重新提取） |
| `attach_provenance(path)` | 把 Q1 的 provenance 注入到附件 2 元数据 | ❌ 未实现 |
| `merge_modalities_for_member_c(...)` | 为 C 生成"特征到原始帧"的反向映射 | ⚠️ 部分在 `src/utils/video_utils.py` |

---

## 9. 联系方式

如接口变更或发现问题，请联系成员 A 或在本文件追加 CHANGELOG。

---

## 10. v2 时间对齐说明（2026-09-25 升级）

### 10.1 新增文件

```
data/processed/features_q1/
├── alignment/                          # v2 新增：100 个词级对齐 JSON
│   ├── -3g5yACwYnA_13.json
│   ├── ...
│   └── (共 100 个)
├── alignment_summary.json             # v2 新增：100 条对齐汇总
├── sync_cases/                         # v2 新增：3 个对照案例三件套
│   ├── case01_normal/
│   ├── case02_visual_missing/
│   └── case03_text_sparse/
├── backup_old/                         # v2 新增：v1 备份
│   ├── q1_features_all.pkl
│   └── extract_log_v1.json
├── q1_features_all_v2.pkl             # v2 新增：合并特征实体
├── q1_features_combined.pkl           # 软链接 → q1_features_all_v2.pkl
└── ...
```

### 10.2 alignment JSON 结构

每个 `alignment/<video>_<clip>.json`：
```json
{
  "video_id": "-3g5yACwYnA",
  "clip_id": 13,
  "sample_id": "-3g5yACwYnA$_$13",
  "asr_verification": {
    "raw_text": "...",
    "asr_text": "...",
    "wer": 0.2,
    "match_ok": true
  },
  "audio_duration_sec": 5.4,
  "video_fps": 47.3,
  "alignment_method": "whisperx-base + wav2vec2",
  "tokens": [
    {"idx": 0, "token": "[CLS]", "audio_start_sec": 0.0, "audio_end_sec": 0.0, ...},
    {"idx": 1, "token": "They've", "audio_start_sec": 0.0, "audio_end_sec": 0.47, ...},
    ... 50 个 token
  ],
  "status": "aligned"  // or "partial" / "failed"
}
```

### 10.3 摘要表新增列

`outputs/tables/q1_feature_summary.csv`（v2）：
- 新增：`text_length_effective`, `audio_length_effective`, `vision_length_effective`
- 新增：`asr_match_rate`, `alignment_status`, `alignment_wer`
- 新增：`audio_duration_sec`, `video_fps`
- 新增：`source_video_path_anonymized`

### 10.4 兼容性

v2 字段名与 v1 **100% 一致**，仅新增 4 个 per_token 可选字段。B/C 代码无须改动。

回退方法（如果 B/C 不兼容）：
```bash
cd data/processed/features_q1
rm q1_features_combined.pkl
cp backup_old/q1_features_all.pkl .
```

### 10.5 alignment 统计

| 指标 | 值 |
|------|-----|
| 总样本 | 100 |
| aligned (WER<0.3) | 62 |
| partial (0.3<WER<0.6) | 38 |
| failed | 0 |
| 平均 ASR 匹配率 | 0.606 |
| 平均 ASR WER | 0.394 |
