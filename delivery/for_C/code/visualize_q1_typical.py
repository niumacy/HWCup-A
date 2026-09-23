"""
生成 Q1 典型样本可视化图
选 1 个 Positive + 1 个 Neutral + 1 个 Negative 样本
绘制 3 模态 (text / audio / vision) 的对齐可视化
"""
import os
import sys
import pickle
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.video_utils import extract_key_frames


def load_combined():
    """加载合并 pkl"""
    pkl_path = 'data/processed/features_q1/q1_features_combined.pkl'
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)['all']


def pick_typical_samples(data, video_base_dir):
    """从 100 条样本里挑 1 Pos + 1 Neu + 1 Neg,且视频存在"""
    cls = np.array(data['classification_labels'])
    ann = np.array(data['annotation'])

    picked = []
    for target_label, target_ann in [(2, 'Positive'), (1, 'Neutral'), (0, 'Negative')]:
        idx_pool = np.where(cls == target_label)[0]
        for idx in idx_pool:
            sample_id = data['id'][idx]
            vid_id, clip_id = sample_id.split('$_$')
            video_path = os.path.join(video_base_dir, vid_id, f'{clip_id}.mp4')
            if os.path.exists(video_path):
                picked.append((idx, target_ann, sample_id, video_path))
                break
    return picked


def heatmap_feature(arr, ax, title, ylabel='dim'):
    """画特征热力图 (T x D)"""
    im = ax.imshow(arr.T, aspect='auto', cmap='viridis', origin='lower',
                   interpolation='nearest')
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('time step')
    ax.set_ylabel(ylabel)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def plot_audio_waveform(y, ax, title):
    """画音频波形"""
    t = np.linspace(0, len(y) / 16000, len(y))
    ax.plot(t, y, color='#1f77b4', linewidth=0.5)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('time (s)')
    ax.set_ylabel('amplitude')
    ax.set_xlim(t[0], t[-1])
    ax.grid(True, alpha=0.3)


def main():
    data = load_combined()
    video_base_dir = '/mnt/data2/home/chenyu/HuaWeiCup/E题数据/附件1-数据集原始多模态样本/MOSEI数据集部分原始视频-100条'

    picked = pick_typical_samples(data, video_base_dir)
    print(f"已选中 {len(picked)} 个典型样本:")
    for idx, ann, sid, vp in picked:
        print(f"  - {ann}: id={sid}, 视频={os.path.basename(vp)}")

    fig = plt.figure(figsize=(16, 12), dpi=110)
    gs = gridspec.GridSpec(4, 3, height_ratios=[1.0, 0.9, 0.9, 1.1], hspace=0.55, wspace=0.35)

    # 标题
    fig.suptitle('Q1 Typical Sample Visualization: Text / Audio / Vision Alignment (50-step)',
                 fontsize=14, fontweight='bold', y=0.995)

    for col, (idx, ann, sample_id, video_path) in enumerate(picked):
        text = data['text'][idx]            # (50, 768)
        audio = data['audio'][idx]          # (50, 74)
        vision = data['vision'][idx]        # (50, 35)
        raw_text = data['raw_text'][idx]
        reg_label = data['regression_labels'][idx]

        # 第 0 行: 文本 (token 强度按 L2 norm 可视化)
        ax_text = fig.add_subplot(gs[0, col])
        token_strength = np.linalg.norm(text, axis=-1)  # (50,)
        ax_text.bar(np.arange(50), token_strength, color='#2ca02c', alpha=0.85)
        ax_text.set_title(f'{ann} (y={reg_label:+.2f})', fontsize=11, fontweight='bold')
        ax_text.set_xlabel('token step (0..49)')
        ax_text.set_ylabel('L2 norm')
        ax_text.set_xlim(-1, 50)

        # 第 1 行: 文本特征热力图 (降维到 16 维通过 PCA-like 切片)
        ax_text_h = fig.add_subplot(gs[1, col])
        # 取前 16 维
        im = ax_text_h.imshow(text[:, :16].T, aspect='auto', cmap='viridis',
                              origin='lower', interpolation='nearest')
        ax_text_h.set_title(f'text feature heatmap (768→16)', fontsize=9)
        ax_text_h.set_xlabel('time step')
        ax_text_h.set_ylabel('dim 0..15')
        plt.colorbar(im, ax=ax_text_h, fraction=0.046, pad=0.04)

        # 第 2 行: 语音特征热力图
        ax_audio = fig.add_subplot(gs[2, col])
        im = ax_audio.imshow(audio.T, aspect='auto', cmap='magma',
                             origin='lower', interpolation='nearest')
        ax_audio.set_title(f'audio feature (74-dim)', fontsize=9)
        ax_audio.set_xlabel('time step')
        ax_audio.set_ylabel('dim 0..73')
        plt.colorbar(im, ax=ax_audio, fraction=0.046, pad=0.04)

        # 第 3 行: 视觉特征热力图
        ax_vision = fig.add_subplot(gs[3, col])
        im = ax_vision.imshow(vision.T, aspect='auto', cmap='cividis',
                              origin='lower', interpolation='nearest')
        ax_vision.set_title(f'vision feature (35-dim)', fontsize=9)
        ax_vision.set_xlabel('time step')
        ax_vision.set_ylabel('dim 0..34')
        plt.colorbar(im, ax=ax_vision, fraction=0.046, pad=0.04)

    out_dir = 'paper/figures'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'q1_typical_sample.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n✅ 已保存: {out_path}  ({os.path.getsize(out_path)/1024:.1f} KB)")

    # 同时输出 raw_text 预览到 caption 文件
    caption_path = os.path.join(out_dir, 'q1_typical_sample_caption.txt')
    with open(caption_path, 'w', encoding='utf-8') as f:
        f.write("Q1 Typical Sample Visualization - Caption\n")
        f.write("=" * 60 + "\n\n")
        for idx, ann, sample_id, video_path in picked:
            raw_text = data['raw_text'][idx]
            reg_label = data['regression_labels'][idx]
            cls_label = int(data['classification_labels'][idx])
            f.write(f"[{ann}] id={sample_id}\n")
            f.write(f"  regression_label = {reg_label:+.4f}  (classification_label = {cls_label})\n")
            f.write(f"  raw_text: \"{raw_text}\"\n")
            f.write(f"  video: {os.path.basename(video_path)}\n\n")
    print(f"✅ 已保存 caption: {caption_path}")


if __name__ == '__main__':
    main()
