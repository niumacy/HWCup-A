"""
音频特征提取器
使用 librosa 提取标准声学特征，维度与附件2音频(74维)对齐

策略:
  1. 提取 MFCC(20) + Delta-MFCC(20) + Spectral(7) + ZCR(1) + RMSE(1) + Chroma(12) = 61维
  2. 对齐到 50 帧 (与附件2 aligned_50 一致)
  3. 最终 audio 形状: (50, 74) float32
  4. 缺失的13维用均值填充（仅用于补齐维度，不影响模型学习）

对于 Q1 论文说明:
  采用 librosa 提取标准声学特征，MFCC 捕捉韵律和语义信息，
  Spectral 特征捕捉音色和共振峰特性，Chroma 捕捉音高特征，
  该特征集在多模态情感分析中被广泛使用。
"""
import numpy as np
import librosa
import soundfile as sf
import os
from typing import Union, Optional


# 特征维度配置
FEATURE_DIMS = {
    'mfcc': 20,       # MFCC
    'delta_mfcc': 20, # Delta MFCC
    'spectral': 5,    # spectral centroid, bandwidth, contrast(mean), rolloff, flatness
    'zcr': 1,         # zero crossing rate
    'rmse': 1,        # RMSE energy
    'chroma': 12,     # chroma
}
TOTAL_FEATURES = sum(FEATURE_DIMS.values())  # 59维
TARGET_DIM = 74  # 与附件2对齐


class AudioFeatureExtractor:
    """
    从音频文件或波形提取标准声学特征

    输出形状: (num_frames, 74) float32
    - 时间帧数: 默认 50 帧（与附件2 aligned_50 对齐）
    - 特征维度: 74（MFCC+Delta+光谱+韵律+色度）
    """

    def __init__(self,
                 sr: int = 16000,
                 n_fft: int = 512,
                 hop_length: int = 256,
                 n_mfcc: int = 20,
                 n_mels: int = 128,
                 target_frames: int = 50):
        """
        Args:
            sr: 采样率 (Hz)
            n_fft: FFT 窗口大小
            hop_length: 帧移
            n_mfcc: MFCC 系数数量
            n_mels: mel 滤波器组数量
            target_frames: 目标时间帧数（对齐用）
        """
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.target_frames = target_frames

    def extract_from_file(self, audio_path: str) -> np.ndarray:
        """从音频文件提取特征"""
        # 加载音频
        y, sr = librosa.load(audio_path, sr=self.sr, mono=True)
        return self.extract_from_waveform(y, sr)

    def extract_from_waveform(self,
                              y: np.ndarray,
                              sr: Optional[int] = None) -> np.ndarray:
        """从波形数据提取特征"""
        if sr is None:
            sr = self.sr

        features = []

        # 1. MFCC (20维)
        mfcc = librosa.feature.mfcc(
            y=y, sr=sr, n_fft=self.n_fft,
            hop_length=self.hop_length, n_mfcc=self.n_mfcc
        )
        features.append(mfcc)

        # 2. Delta MFCC (20维)
        delta_mfcc = librosa.feature.delta(mfcc)
        features.append(delta_mfcc)

        # 3. 光谱特征 (7维)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length)
        spectral_flatness = librosa.feature.spectral_flatness(y=y, n_fft=self.n_fft, hop_length=self.hop_length)

        # 取均值让 spectral_contrast 维度与其它一致
        spectral_contrast_mean = spectral_contrast.mean(axis=0, keepdims=True)

        spectral_feat = np.concatenate([
            spectral_centroid,
            spectral_bandwidth,
            spectral_contrast_mean,
            spectral_rolloff,
            spectral_flatness,
        ], axis=0)
        features.append(spectral_feat)

        # 4. 过零率 (1维)
        zcr = librosa.feature.zero_crossing_rate(y, hop_length=self.hop_length)
        features.append(zcr)

        # 5. RMSE 能量 (1维)
        rmse = librosa.feature.rms(y=y, hop_length=self.hop_length)
        features.append(rmse)

        # 6. Chroma (12维)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length)
        features.append(chroma)

        # 拼接
        feat = np.concatenate(features, axis=0)  # (61, T)
        assert feat.shape[0] == TOTAL_FEATURES, f"期望{TOTAL_FEATURES}维, 实际{feat.shape[0]}维"

        # 填充/截断到 target_frames
        feat = self._align_to_target(feat, self.target_frames)

        # 扩展到 74 维 (附加13维均值)
        if feat.shape[1] < TARGET_DIM:
            pad_width = TARGET_DIM - feat.shape[1]
            pad = np.full((feat.shape[0], pad_width), feat.mean(axis=1, keepdims=True).flatten()[0])
            # 用各特征的均值填充
            feat_padded = np.zeros((TARGET_DIM, feat.shape[1]))
            feat_padded[:feat.shape[0], :] = feat
            # 13个填充维用整体均值
            feat_padded[feat.shape[0]:, :] = feat.mean()
            feat = feat_padded

        return feat.astype(np.float32)  # (74, T)

    def _align_to_target(self, feat: np.ndarray, target: int) -> np.ndarray:
        """对齐到目标帧数"""
        T = feat.shape[1]
        if T >= target:
            # 均匀采样 + 截断
            indices = np.linspace(0, T - 1, target).astype(int)
            return feat[:, indices]
        else:
            # 用线性插值填充
            indices = np.linspace(0, T - 1, target).astype(int)
            indices = np.clip(indices, 0, T - 1)
            return feat[:, indices]

    def extract_for_sample(self, audio_path: str) -> dict:
        """
        提取单条样本的音频特征（兼容附件2格式）
        """
        feat = self.extract_from_file(audio_path)  # (74, T)
        return {
            'audio': feat.T if feat.shape[0] == self.target_frames else feat,  # (T, 74)
        }


def extract_audio_from_video(video_path: str,
                             temp_dir: str = '/tmp',
                             sr: int = 16000) -> np.ndarray:
    """
    从视频文件提取音频并返回波形

    使用 ffmpeg 提取音频, librosa 加载
    """
    import subprocess

    temp_audio = os.path.join(temp_dir, f'audio_{os.getpid()}.wav')

    # ffmpeg 提取音频
    cmd = [
        'ffmpeg', '-y', '-i', video_path,
        '-vn', '-acodec', 'pcm_s16le',
        '-ar', str(sr), '-ac', '1',
        temp_audio
    ]
    subprocess.run(cmd, capture_output=True, check=True)

    y, sr = librosa.load(temp_audio, sr=sr, mono=True)

    try:
        os.remove(temp_audio)
    except:
        pass

    return y, sr
