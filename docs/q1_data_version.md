# Q1 数据版本说明

> **目的**：让 B/C 团队明确当前 `q1_features_combined.pkl` 的两个版本区别，确保切换无感知。

---

## v1（commit ca08b27 — 旧版）

| 项 | 值 |
|----|-----|
| 文件名 | `q1_features_combined.pkl` |
| 备份位置 | `data/processed/features_q1/backup_old/q1_features_all.pkl` |
| 提取方法 | 线性字符映射（`audio_dur * char_idx / len(raw_text)`） |
| 时间对齐精度 | **无真实对齐**，仅按字符数等比例分 |
| `text_length` | 全部为 50（无 effective 区分） |
| `audio_length` | 50（按 50 帧存储） |
| `vision_length` | 50 |
| extract_log | `extractor_versions: {}`（无版本记录） |

## v2（当前推荐）

| 项 | 值 |
|----|-----|
| 文件名 | `q1_features_all_v2.pkl` |
| 软链接 | `q1_features_combined.pkl -> q1_features_all_v2.pkl`（默认指向 v2） |
| 提取方法 | **whisperx 强制对齐 + ASR 二次校验** |
| 时间对齐精度 | **词级**（wav2vec2-base 对齐，每个 token 的 audio_sec/frame） |
| 新增字段 | `per_token_audio_sec_start`, `per_token_audio_sec_end`, `per_token_video_frame_start`, `per_token_video_frame_end` |
| `text_length_effective` | 从 `alignment/tokens` 中非 PAD 计数（实际 7-18） |
| `audio_length_effective` | 基于 `audio_duration_sec` 计算 |
| `vision_length_effective` | 基于 `video_fps × audio_duration_sec` 计算 |
| extract_log | `extractor_versions` 含 sha256（见 `data/processed/features_q1/extract_log.json`） |

---

## 字段兼容性

v2 的所有 v1 字段**完全保留**，新增的 4 个 per_token 字段为可选：

```python
# v1 代码无须改动
data = pickle.load(open('q1_features_combined.pkl', 'rb'))
text = data['all']['text']      # (100, 50, 768)
audio = data['all']['audio']    # (100, 50, 74)
vision = data['all']['vision']  # (100, 50, 35)
labels = data['all']['classification_labels']  # (100,)

# v2 新增（可选使用）
audio_sec_start = data['all']['per_token_audio_sec_start']  # (100, 50)
audio_sec_end   = data['all']['per_token_audio_sec_end']
video_frame_start = data['all']['per_token_video_frame_start']  # (100, 50)
video_frame_end   = data['all']['per_token_video_frame_end']
```

---

## 如何回退到 v1

```bash
# 在 data/processed/features_q1/ 下
rm q1_features_combined.pkl
cp backup_old/q1_features_all.pkl .
```

---

## 选择建议

- **优先使用 v2**：因为含真实词级对齐，可复现性高
- **仅在 v2 与 B/C 模型不兼容时回退**：目前 v2 字段与 v1 完全兼容，无须改动

---

## alignment 文件清单

```
data/processed/features_q1/alignment/
├── -3g5yACwYnA_13.json      # 每条样本一个 JSON
├── -3g5yACwYnA_2.json
├── ...
└── (共 100 个)

alignment_summary.json        # 100 条对齐的 WER 汇总
```

每个 JSON 包含：
- `asr_verification.raw_text` / `asr_text` / `wer`
- `audio_duration_sec` / `video_fps`
- `tokens[].audio_start_sec/end` / `video_frame_start/end`
- `tokens[].asr_word` / `asr_confidence`
- `status` = `aligned` (WER<0.3) / `partial` (WER<0.6)

---

**维护者**：成员 A  
**最后更新**：2026-09-25
