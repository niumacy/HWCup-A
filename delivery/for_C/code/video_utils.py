"""
视频工具模块
供成员C (Q3 可解释预测) 使用，用于：
- 时间戳 ↔ 帧号 互转
- 关键帧提取（用于可解释性可视化）
- 视频元数据获取
- 文本 token 时间对齐（供 C 把模型给出的"关键秒数"映射回原始视频帧）

所有函数与 ffmpeg/opencv 一致，可在不依赖 GPU 的环境下调用。
"""
from __future__ import annotations

import os
import json
import subprocess
from typing import List, Tuple, Dict, Optional, Union

import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


# ────────────────────────────────────────────────────────────────────────────
# 1. 时间戳 ↔ 帧号 互转
# ────────────────────────────────────────────────────────────────────────────

def timestamp_to_frame(timestamp_sec: float, fps: float, total_frames: Optional[int] = None) -> int:
    """
    把"秒数"映射到对应的视频帧号（按帧率就近向下取整）

    Args:
        timestamp_sec: 秒数（允许浮点）
        fps: 视频帧率 (frame per second)
        total_frames: 可选总帧数，超出范围时进行 clip

    Returns:
        帧号 (int)
    """
    if fps <= 0:
        raise ValueError(f"fps 必须 > 0, 当前 {fps}")
    if timestamp_sec < 0:
        timestamp_sec = 0.0
    frame_idx = int(timestamp_sec * fps)
    if total_frames is not None:
        frame_idx = min(frame_idx, max(total_frames - 1, 0))
    return frame_idx


def frame_to_timestamp(frame_idx: int, fps: float) -> float:
    """
    把帧号映射回秒数

    Args:
        frame_idx: 帧号 (int)
        fps: 视频帧率

    Returns:
        秒数 (float)
    """
    if fps <= 0:
        raise ValueError(f"fps 必须 > 0, 当前 {fps}")
    return float(frame_idx) / float(fps)


def frame_range_to_timestamps(start_frame: int, end_frame: int, fps: float) -> Tuple[float, float]:
    """把一帧区间映射回秒数区间"""
    return frame_to_timestamp(start_frame, fps), frame_to_timestamp(end_frame, fps)


# ────────────────────────────────────────────────────────────────────────────
# 2. 视频元数据
# ────────────────────────────────────────────────────────────────────────────

def get_video_metadata(video_path: str) -> Dict[str, Union[int, float, str, None]]:
    """
    用 ffprobe 读取视频元数据（时长、帧率、分辨率、编码格式等）

    Args:
        video_path: 视频文件路径

    Returns:
        dict: {
            'duration': float（秒）,
            'fps': float,
            'total_frames': int (估算值),
            'width': int,
            'height': int,
            'codec': str,
            'video_path': str,
            'error': str (失败时存在)
        }
    """
    if not os.path.exists(video_path):
        return {'video_path': video_path, 'error': f'file not found: {video_path}'}

    try:
        # 优先 ffprobe (比 cv2 更精确)
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json',
            '-show_format', '-show_streams', video_path
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        info = json.loads(res.stdout)

        # 主视频流
        video_stream = None
        for stream in info.get('streams', []):
            if stream.get('codec_type') == 'video':
                video_stream = stream
                break

        if video_stream is None:
            return {'video_path': video_path, 'error': 'no video stream found'}

        # 帧率（r_frame_rate 是分数形式 "30/1"）
        fps_str = video_stream.get('avg_frame_rate', '30/1')
        if '/' in fps_str:
            num, den = fps_str.split('/')
            fps = float(num) / float(den) if float(den) > 0 else 30.0
        else:
            fps = float(fps_str)

        duration = float(info.get('format', {}).get('duration', 0))
        total_frames = int(video_stream.get('nb_frames', int(duration * fps)))

        return {
            'video_path': video_path,
            'duration': duration,
            'fps': fps,
            'total_frames': total_frames,
            'width': int(video_stream.get('width', 0)),
            'height': int(video_stream.get('height', 0)),
            'codec': video_stream.get('codec_name', 'unknown'),
        }
    except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError) as e:
        # fallback 到 OpenCV
        if HAS_CV2:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return {'video_path': video_path, 'error': f'cannot open: {e}'}
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = total_frames / fps if fps > 0 else 0.0
            cap.release()
            return {
                'video_path': video_path,
                'duration': duration,
                'fps': fps,
                'total_frames': total_frames,
                'width': width,
                'height': height,
                'codec': 'unknown (cv2 fallback)',
            }
        return {'video_path': video_path, 'error': f'ffprobe failed: {e}'}


# ────────────────────────────────────────────────────────────────────────────
# 3. 关键帧提取（供 Q3 可视化）
# ────────────────────────────────────────────────────────────────────────────

def extract_key_frames(video_path: str, n_frames: int = 5,
                       mode: str = 'uniform',
                       return_timestamps: bool = True) -> List[Dict]:
    """
    均匀或基于重要性抽取关键帧

    Args:
        video_path: 视频文件路径
        n_frames: 抽几帧
        mode: 'uniform' (等距) | 'center' (中点周围) | 'first_last' (头尾+中间)
        return_timestamps: 是否在每个帧的 dict 里加 timestamp 字段

    Returns:
        list of dict: [
            {'frame_idx': int, 'timestamp': float, 'image': np.ndarray[H,W,3] BGR},
            ...
        ]
    """
    if not HAS_CV2:
        raise RuntimeError("需要 opencv-python，请安装: pip install opencv-python")
    if not os.path.exists(video_path):
        raise FileNotFoundError(video_path)

    meta = get_video_metadata(video_path)
    if 'error' in meta:
        raise RuntimeError(f"无法读取视频: {meta['error']}")

    total_frames = meta['total_frames']
    fps = meta['fps']

    if mode == 'uniform':
        indices = np.linspace(0, total_frames - 1, n_frames).astype(int).tolist()
    elif mode == 'center':
        mid = total_frames // 2
        half = n_frames // 2
        indices = list(range(max(0, mid - half), min(total_frames, mid + half + 1)))
        # 如果超界，左右扩展
        while len(indices) < n_frames:
            if indices[0] > 0:
                indices.insert(0, indices[0] - 1)
            elif indices[-1] < total_frames - 1:
                indices.append(indices[-1] + 1)
            else:
                break
    elif mode == 'first_last':
        indices = [0, total_frames - 1]
        if n_frames >= 3:
            mid = list(np.linspace(0, total_frames - 1, n_frames).astype(int).tolist())
            indices = sorted(set(mid))
        indices = indices[:n_frames]
    else:
        raise ValueError(f"mode 必须是 'uniform'/'center'/'first_last', 当前 {mode}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"无法打开视频: {video_path}")

    results = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        item = {
            'frame_idx': int(idx),
            'image': frame,  # BGR
        }
        if return_timestamps:
            item['timestamp'] = frame_to_timestamp(idx, fps)
        results.append(item)

    cap.release()
    return results


def extract_frame_at_timestamp(video_path: str, timestamp_sec: float) -> Optional[np.ndarray]:
    """
    在指定秒数处抽取一帧

    Args:
        video_path: 视频路径
        timestamp_sec: 秒数

    Returns:
        BGR 图像数组，如果失败返回 None
    """
    if not HAS_CV2:
        raise RuntimeError("需要 opencv-python")
    meta = get_video_metadata(video_path)
    if 'error' in meta:
        return None

    frame_idx = timestamp_to_frame(timestamp_sec, meta['fps'], meta['total_frames'])
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


# ────────────────────────────────────────────────────────────────────────────
# 4. 文本 ↔ 时序对齐（供 C 把"重要时间点"映射回原文 token）
# ────────────────────────────────────────────────────────────────────────────

def map_modality_timestep_to_original_time(
    timestep_idx: int,
    modality_total_steps: int,
    video_duration_sec: float,
) -> float:
    """
    把多模态特征的第 i 时间步映射回原始视频的秒数
    与 src/feature_extractor/aligner.py 的均匀对齐方式一致

    适用: text/audio/vision 任意一个（已经 align 到 modality_total_steps）
    其中 modality_total_steps=50 (本项目约定)

    Args:
        timestep_idx: 当前时间步 (0-indexed)
        modality_total_steps: 总步数（50）
        video_duration_sec: 原始视频时长（秒）

    Returns:
        对应的视频秒数
    """
    if modality_total_steps <= 1:
        return 0.0
    ratio = float(timestep_idx) / float(modality_total_steps - 1)
    return ratio * video_duration_sec


def map_original_time_to_modality_timestep(
    timestamp_sec: float,
    video_duration_sec: float,
    modality_total_steps: int = 50,
) -> int:
    """把视频秒数映射到多模态时间步（与上述相反）"""
    if video_duration_sec <= 0 or modality_total_steps <= 1:
        return 0
    ratio = max(0.0, min(1.0, timestamp_sec / video_duration_sec))
    return int(round(ratio * (modality_total_steps - 1)))


# ────────────────────────────────────────────────────────────────────────────
# 5. 把 50 帧 token 边界映射到原始字符级时间戳 (供 C 可视化"哪句话引发情感")
# ────────────────────────────────────────────────────────────────────────────

def build_text_token_time_alignment(
    raw_text: str,
    tokenizer,
    modality_total_steps: int = 50,
    fallback_duration_sec: Optional[float] = None,
) -> List[Dict]:
    """
    把文本的 [CLS] tok_1 tok_2 ... [SEP] [PAD]... 映射到 (modality_total_steps=50) 时间步
    给 C 用于"哪个 token 触发了情感极性"的可解释性分析

    Args:
        raw_text: 原始英文文本
        tokenizer: HuggingFace tokenizer (BERT/RoBERTa 等)
        modality_total_steps: 50
        fallback_duration_sec: 文本对齐的总时长（默认用 1 秒/句，不准确但足够演示）

    Returns:
        list of dict, len = modality_total_steps:
            {
                'timestep': int,
                'token_indices': list[int],  # 此时间步对应的 BERT token 索引
                'tokens': list[str],         # 解码后的 token 字符串
                'time_range_sec': (start, end),
            }
    """
    enc = tokenizer(
        raw_text,
        padding='max_length',
        truncation=True,
        max_length=modality_total_steps,
        return_tensors='pt',
    )
    input_ids = enc['input_ids'][0].tolist()
    total_tokens = len(input_ids)
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    # 简单均分（更严谨的可以强制对齐，但 50 步默认 = 50 token）
    step_to_tokens: Dict[int, List[int]] = {i: [] for i in range(modality_total_steps)}
    for tok_idx in range(total_tokens):
        step = min(modality_total_steps - 1, tok_idx * modality_total_steps // total_tokens)
        step_to_tokens[step].append(tok_idx)

    duration = fallback_duration_sec if fallback_duration_sec is not None else max(1.0, len(raw_text) / 10.0)
    sec_per_step = duration / modality_total_steps

    result = []
    for step in range(modality_total_steps):
        tok_idxs = step_to_tokens[step]
        result.append({
            'timestep': step,
            'token_indices': tok_idxs,
            'tokens': [tokens[i] for i in tok_idxs],
            'time_range_sec': (step * sec_per_step, (step + 1) * sec_per_step),
        })
    return result


# ────────────────────────────────────────────────────────────────────────────
# 6. 批量工具：用 sample_id 找到原始视频文件
# ────────────────────────────────────────────────────────────────────────────

def locate_video_file(
    sample_id: str,
    video_base_dir: str,
    extensions: Tuple[str, ...] = ('.mp4', '.avi', '.mov'),
) -> Optional[str]:
    """
    根据 sample_id (格式 video_id$_$clip_id) 找到原始视频路径

    Args:
        sample_id: e.g. '-3g5yACwYnA$_$13'
        video_base_dir: 视频根目录
        extensions: 接受的扩展名

    Returns:
        视频绝对路径 or None
    """
    try:
        video_id, clip_id = sample_id.split('$_$')
    except ValueError:
        return None
    for ext in extensions:
        path = os.path.join(video_base_dir, video_id, f'{clip_id}{ext}')
        if os.path.exists(path):
            return path
    return None


# ────────────────────────────────────────────────────────────────────────────
# 7. 命令行调试入口
# ────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='视频工具自检')
    parser.add_argument('--video', type=str, required=True, help='测试视频路径')
    parser.add_argument('--n_frames', type=int, default=5)
    args = parser.parse_args()

    meta = get_video_metadata(args.video)
    print("=" * 60)
    print(f"视频元数据: {args.video}")
    print("=" * 60)
    for k, v in meta.items():
        print(f"  {k}: {v}")

    print("\n关键帧抽样:")
    frames = extract_key_frames(args.video, n_frames=args.n_frames, mode='uniform')
    for f in frames:
        print(f"  frame_idx={f['frame_idx']:>5}  timestamp={f['timestamp']:.3f}s  shape={f['image'].shape}")

    # 测试时间戳映射
    print("\n时间戳映射测试:")
    print(f"  1.5s → frame {timestamp_to_frame(1.5, meta['fps'], meta['total_frames'])}")
    print(f"  frame 30 → {frame_to_timestamp(30, meta['fps']):.3f}s")
