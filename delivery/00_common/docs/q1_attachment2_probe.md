# 附件2探查报告

## 数据集规模
- train: 3395 samples
- valid: 728 samples
- test: 727 samples

## 特征维度确认
| 模态 | 形状 | dtype | 推断来源 |
|------|------|-------|----------|
| text | (N, 50, 768) | float32 | BERT-base-uncased hidden state |
| audio | (N, 50, 74) | float64 | OpenSMILE eGeMAPS 子集 |
| vision | (N, 50, 35) | float64 | OpenFace 2.0 |

## 语音特征异常发现
- audio 最大值为 500.0（非正常值）
- 建议：提取后对 audio 做 clip 到合理范围（如 ±100）
- 标签映射：Negative→0, Neutral→1, Positive→2
