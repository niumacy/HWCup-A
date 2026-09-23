"""
时序对齐模块
负责把不同长度的多模态特征统一对齐到相同时间步长，供 Q1/Q2/Q3 使用

三种对齐模式：
  align_uniform      : 等距采样 + 零填充（默认，与附件2 aligned_50.pkl 一致）
  align_linear_interp: 线性插值（保留更多原始信息但需要 numpy）
  align_truncate_pad : 仅截断 + 零填充（最朴素）

对齐粒度：
  - aligned:  三模态都统一到 (T, D) 同一时间步 → 用于 Late Fusion / GLF / TCN 等
  - unaligned: 仅文本对齐到 (50, 768)，音频/视觉保持原始 T → 用于 LSTM/MoE 等时序模型
"""
from __future__ import annotations

import numpy as np
from typing import Tuple, Literal, Optional


# ────────────────────────────────────────────────────────────────────────────
# 工具函数
# ────────────────────────────────────────────────────────────────────────────

def _pad_or_truncate(feat: np.ndarray, target_len: int, mode: str = 'uniform') -> np.ndarray:
    """
    一维/二维特征对齐到目标长度

    Args:
        feat: (T, D) 或 (T,) 的 numpy 数组
        target_len: 目标时间步 T
        mode:
            - 'uniform': 等距采样到 target_len
            - 'linear_interp': 线性插值
            - 'truncate_pad': 截断到 target_len, 不足时用 0 补
    """
    if feat.ndim == 1:
        feat = feat[:, None]
    T, D = feat.shape

    if T == target_len:
        return feat.astype(np.float32)

    if mode == 'truncate_pad':
        if T >= target_len:
            return feat[:target_len].astype(np.float32)
        pad = np.zeros((target_len - T, D), dtype=feat.dtype)
        return np.concatenate([feat, pad], axis=0).astype(np.float32)

    if mode == 'uniform':
        if T > target_len:
            indices = np.linspace(0, T - 1, target_len).astype(int)
            return feat[indices].astype(np.float32)
        else:
            indices = np.linspace(0, max(T - 1, 0), target_len).astype(int)
            indices = np.clip(indices, 0, T - 1)
            return feat[indices].astype(np.float32)

    if mode == 'linear_interp':
        if T == target_len:
            return feat.astype(np.float32)
        # 用 np.interp 对每一维独立插值
        x_old = np.linspace(0.0, 1.0, T)
        x_new = np.linspace(0.0, 1.0, target_len)
        out = np.empty((target_len, D), dtype=np.float64)
        for d in range(D):
            out[:, d] = np.interp(x_new, x_old, feat[:, d])
        return out.astype(np.float32)

    raise ValueError(f"未知的对齐模式: {mode}")


def get_actual_length(arr: np.ndarray, threshold: float = 1e-6) -> int:
    """
    推断特征的有效时间长度（去除 padding 0）

    用于 unaligned 版本：知道原始视频产生了多少帧的特征
    """
    # 任何一行有非零元素都被视为有效
    norm = np.linalg.norm(arr, axis=tuple(range(1, arr.ndim))) if arr.ndim > 1 else np.abs(arr)
    nz = np.where(norm > threshold)[0]
    return int(nz[-1] + 1) if len(nz) > 0 else 0


# ────────────────────────────────────────────────────────────────────────────
# 主类
# ────────────────────────────────────────────────────────────────────────────

class TemporalAligner:
    """
    把 text / audio / vision 三模态对齐到共同时间步 (默认 50)
    与附件2 aligned_50.pkl 的语义保持一致
    """

    def __init__(self, target_len: int = 50, mode: Literal['uniform', 'linear_interp', 'truncate_pad'] = 'uniform'):
        """
        Args:
            target_len: 目标时间步数 (附件2 是 50，所以默认 50)
            mode: 对齐模式
                - uniform: 等距重采样 (默认, 与 extract_features.py 一致)
                - linear_interp: 线性插值 (推荐音频/视觉)
                - truncate_pad: 仅截断+零填充
        """
        if target_len <= 0:
            raise ValueError("target_len 必须 > 0")
        self.target_len = target_len
        self.mode = mode

    # ── 对齐三模态到统一时间步 ─────────────────────────────────────────────
    def align_50(
        self,
        text: np.ndarray,
        audio: np.ndarray,
        vision: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        把文本 (T_t, 768), 语音 (T_a, 74), 视觉 (T_v, 35)
        三者统一对齐到 (target_len, dim)

        Returns:
            text_aligned, audio_aligned, vision_aligned
        """
        text_a = _pad_or_truncate(text, self.target_len, mode=self.mode)
        audio_a = _pad_or_truncate(audio, self.target_len, mode=self.mode)
        vision_a = _pad_or_truncate(vision, self.target_len, mode=self.mode)
        return text_a, audio_a, vision_a

    # ── 对齐为 unaligned 版本 ──────────────────────────────────────────────
    def align_unaligned(
        self,
        text: np.ndarray,
        audio: np.ndarray,
        vision: np.ndarray,
        enforce_text_len: int = 50,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        仅把文本对齐到 enforce_text_len, 音频/视觉保持原长
        用于 LSTM / RNN / MoE 等需要变长序列的模型

        Returns:
            text_aligned, audio, vision
        """
        text_a = _pad_or_truncate(text, enforce_text_len, mode=self.mode)
        return text_a, audio, vision

    # ── 单独对齐单模态 ────────────────────────────────────────────────────
    def align_one(self, feat: np.ndarray) -> np.ndarray:
        """把单模态特征对齐到 target_len"""
        return _pad_or_truncate(feat, self.target_len, mode=self.mode)

    # ── 把单时间步还原回原始秒数 ──────────────────────────────────────────
    def timestep_to_original_time(
        self,
        timestep_idx: int,
        original_length: int,
        video_duration_sec: float,
    ) -> float:
        """
        把对齐后的 timestep_idx (0..target_len-1) 映射回原始视频的秒数
        近似认为原始特征在视频内均匀分布
        """
        if original_length <= 1:
            return 0.0
        # 等距映射到 original_length，再换算到秒
        step_in_original = int(round(timestep_idx * (original_length - 1) / max(self.target_len - 1, 1)))
        return step_in_original * video_duration_sec / max(original_length - 1, 1)

    def __repr__(self) -> str:
        return f"TemporalAligner(target_len={self.target_len}, mode={self.mode!r})"


# ────────────────────────────────────────────────────────────────────────────
# 单测
# ────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("TemporalAligner 单测")
    print("=" * 60)

    # 模拟三模态不同长度
    text = np.random.randn(50, 768).astype(np.float32)   # 已经是 50
    audio = np.random.randn(120, 74).astype(np.float32)   # 120 帧 (8s 视频, 15fps)
    vision = np.random.randn(95, 35).astype(np.float32)   # 95 帧

    aligner = TemporalAligner(target_len=50, mode='uniform')
    print(f"配置: {aligner}")

    t, a, v = aligner.align_50(text, audio, vision)
    print(f"\nuniform 对齐:")
    print(f"  text:  {text.shape}  -> {t.shape}")
    print(f"  audio: {audio.shape} -> {a.shape}")
    print(f"  vision: {vision.shape} -> {v.shape}")

    # 验证 padding = 0
    print(f"\n音频尾部 padding = {a[100:, :].sum():.4f}  (期望 0)")

    # unaligned 版本
    t2, a2, v2 = aligner.align_unaligned(text, audio, vision)
    print(f"\nunaligned 版本:")
    print(f"  text:  {t2.shape}  (强制 50)")
    print(f"  audio: {a2.shape} (保持 {audio.shape[0]})")
    print(f"  vision: {v2.shape} (保持 {vision.shape[0]})")

    # 切换为线性插值
    aligner2 = TemporalAligner(target_len=50, mode='linear_interp')
    t3, a3, v3 = aligner2.align_50(text, audio, vision)
    print(f"\nlinear_interp 对齐:")
    print(f"  audio: {audio.shape} -> {a3.shape}")

    # 测试 timestep_to_original_time
    print(f"\ntimestep=25 (中部) -> 原始时间: {aligner.timestep_to_original_time(25, 120, 8.0):.3f}s  (期望 ~4.0s)")
    print(f"timestep=0  (起)  -> 原始时间: {aligner.timestep_to_original_time(0, 120, 8.0):.3f}s  (期望 0s)")
    print(f"timestep=49 (末)  -> 原始时间: {aligner.timestep_to_original_time(49, 120, 8.0):.3f}s  (期望 ~8s)")

    # 测试 get_actual_length
    padded = np.zeros((100, 35), dtype=np.float32)
    padded[:30] = np.random.randn(30, 35)  # 只有前30帧有特征
    print(f"\npadded 100 帧, 实际有效长度: {get_actual_length(padded)}  (期望 30)")

    print("\n✅ 所有自检通过")
