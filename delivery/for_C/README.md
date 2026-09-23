# 给成员 C 的交付包（Q3 可解释预测）

> **接收人**: 成员 C
> **目标**: Q3 - 可解释情感预测（时间戳到原始帧的映射）
> **数据基础**: 100 条 Q1 样本（带原始视频 + 三模态特征）

---

## 🎯 你应该优先看的 5 个文件

| 顺序 | 文件 | 说明 |
|------|------|------|
| 1 | `code/video_utils.py` | **核心工具**，所有可解释性可视化都依赖它 |
| 2 | `figures/q1_typical_sample.png` | 论文 figure 模板（参考布局） |
| 3 | `figures/q1_typical_sample_caption.txt` | caption 草稿 |
| 4 | `../00_common/docs/feature_spec.md` | 字段说明（必看） |
| 5 | `../00_common/code/src/feature_extractor/aligner.py` | 时序对齐（用于 timestep → 秒数映射）|

---

## 📂 目录结构说明

```
for_C/
├── code/                              ← 工具代码
│   ├── video_utils.py                 ← 必用：7 类工具函数
│   └── visualize_q1_typical.py        ← 论文 figure 生成脚本（可复用）
├── figures/                           ← 论文 figure
│   ├── q1_typical_sample.png          ← 166 KB 高清 figure
│   └── q1_typical_sample_caption.txt  ← caption
└── extra_data/                        ← 额外数据
    └── video_metadata.csv             ← 100 条视频的 fps/时长/分辨率
                                          (省去你调用 ffmpeg 的时间)
```

---

## 🔧 5 分钟上手

### 核心 API（video_utils.py）

```python
import sys
sys.path.insert(0, 'code')
from video_utils import (
    timestamp_to_frame,          # 秒 → 帧号
    frame_to_timestamp,          # 帧号 → 秒
    get_video_metadata,          # 读视频元数据
    extract_key_frames,          # 均匀抽关键帧
    extract_frame_at_timestamp,  # 在指定秒抽一帧
    map_modality_timestep_to_original_time,   # 时序对齐时间步 → 原始秒数
    map_original_time_to_modality_timestep,   # 反向
    locate_video_file,           # sample_id → 视频路径
)
```

### 快速示例：可视化"模型在 25/50 时间步触发了情感"

```python
import sys, os
sys.path.insert(0, 'code')
from video_utils import (
    locate_video_file, extract_frame_at_timestamp,
    frame_to_timestamp
)
from src.feature_extractor.aligner import TemporalAligner  # 在 00_common/code

VIDEO_BASE = '/mnt/data2/home/chenyu/HuaWeiCup/E题数据/附件1-数据集原始多模态样本/MOSEI数据集部分原始视频-100条'

# 1. 根据 sample_id 定位视频
sample_id = '-3g5yACwYnA$_$13'
video_path = locate_video_file(sample_id, VIDEO_BASE)

# 2. 读元数据
meta = get_video_metadata(video_path)
duration, fps, total = meta['duration'], meta['fps'], meta['total_frames']
# duration=8.767s, fps=30, total=261

# 3. 假设模型给出"关键 timestep"=25 (50 步对齐)
aligner = TemporalAligner(target_len=50)
sec = aligner.timestep_to_original_time(25, original_length=120, video_duration_sec=duration)
# → ~4.0 秒

# 4. 在 4.0 秒抽帧
frame = extract_frame_at_timestamp(video_path, sec)
# frame shape = (720, 1280, 3) BGR

# 5. 抽前后 3 秒的多帧做对比
key_frames = extract_key_frames(video_path, n_frames=5, mode='uniform')
for kf in key_frames:
    print(kf['timestamp'], kf['image'].shape)
```

### 使用预先生成的元数据 CSV

```python
import pandas as pd
meta_df = pd.read_csv('extra_data/video_metadata.csv')
row = meta_df[meta_df['id'] == '-3g5yACwYnA$_$13'].iloc[0]
print(row['duration_sec'], row['fps'], row['total_frames'])
# 8.767 30.0 261
```

### 复用可视化脚本

```bash
# 直接跑就能产出新的 figure
python3 code/visualize_q1_typical.py --out my_figure.png
# (目前脚本不带参数，编辑脚本顶部指定输出路径即可)
```

---

## 🎨 已生成的论文 Figure

**`figures/q1_typical_sample.png`** （166 KB）

布局: 3 列 × 4 行
- 列: Positive / Neutral / Negative 各一个典型样本
- 行: token L2 norm / 文本特征热力图(768→16) / 音频 74 维 / 视觉 35 维

caption 草稿在 `figures/q1_typical_sample_caption.txt`。

---

## 📊 可用样本的统计

| 类别 | 样本数 |
|------|--------|
| Positive (label=2) | 57 |
| Neutral (label=1) | 25 |
| Negative (label=0) | 18 |

每个样本的视频时长分布在 `video_metadata.csv` 里:
- duration: 5 ~ 30 秒
- fps: 全部 30
- 视频分辨率: 1280×720（部分可能不同）

---

## ⚠️ 重要提示

1. **视频路径在 metadata.csv 里硬编码**，如附件 1 路径变了需要重新生成：
   ```bash
   python3 -c "
   import sys, pandas as pd, os
   sys.path.insert(0, 'code')
   from video_utils import get_video_metadata
   df = pd.read_csv('../00_common/data/q1_feature_summary.csv')
   VIDEO_BASE = '<NEW_PATH>'
   ...
   "
   ```

2. **时间映射的精度**：
   - 视频原始帧数 30 fps × 时长秒数
   - 对齐 50 时间步后，每步对应原视频 (T-1)/49 帧
   - 例: 8 秒视频对应 timestep=25 ≈ 4 秒 ± 0.16 秒

3. **可视化建议**：
   - 用 `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` 转换给 matplotlib
   - 用 `cv2.putText` 在关键帧上叠加 timestep 编号
   - 论文 figure 推荐 150 dpi + 16:9 宽屏

---

## 📋 已尝试的可解释性相关工作

| 方向 | 状态 |
|------|------|
| 关键帧抽取（均匀） | ✅ 已实现 `extract_key_frames` |
| timestep → 原始秒数映射 | ✅ 已实现 `timestep_to_original_time` |
| 文本 token → 时间区间 | ✅ 已实现 `build_text_token_time_alignment` |
| ResNet18 中间特征可视化 | ✅ 在 visualize_q1_typical.py |
| 注意力热力图 | ❌ 当前 baseline 无 attention，可选扩展 |

---

## 🚀 建议的可解释性方向

1. **时间局部化**: 模型给出"哪一帧最影响情感" → 用 `extract_key_frames` 可视化
2. **模态权重可视化**: 在 LateFusion 模型上做 modality importance (gradient-based)
3. **错误案例分析**: 对 Q1 100 条里被误判的样本做"反事实"分析
4. **论文 figure**: 把 `q1_typical_sample.png` 当模板，画 6~8 个有代表性的样本

---

## 📞 联系方式

如需新增导出物或接口变更，请联系成员 A。
