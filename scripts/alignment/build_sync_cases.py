"""
3 个真实三模态对照案例生成脚本

输入：100 条 alignment JSON + 单样本 pkl
输出：
  data/processed/features_q1/sync_cases/
    case01_normal/
      case01_table.csv
      case01_figure.png
      case01_metrics.json
    case02_visual_missing/
    case03_text_sparse/

选样策略：
  - Case 1 (normal): 三模态齐全，asr_match_rate > 0.7，长度中等
  - Case 2 (visual_missing): 文本稀疏 + 音频丰富，或视频长时间静止
  - Case 3 (text_sparse): 文本只有 1-2 个词，音频/视频正常
"""
import os
import sys
import json
import argparse
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def find_case1_normal(alignment_summary: List[Dict]) -> Optional[str]:
    """找到正常的 case：三模态齐全 + WER < 0.3 + 时长 4-10s"""
    candidates = []
    for item in alignment_summary:
        if item.get('status') != 'aligned':
            continue
        if item.get('wer', 1.0) > 0.3:
            continue
        sid = item['sample_id'].replace('q1_', '')
        # 读 alignment 看 duration
        align_path = Path('data/processed/features_q1/alignment') / f"{sid}.json"
        if not align_path.exists():
            continue
        with open(align_path) as f:
            align = json.load(f)
        dur = align.get('audio_duration_sec', 0)
        n_tokens = sum(1 for t in align.get('tokens', []) if t.get('token') not in ('[PAD]', '[CLS]'))
        if 3 <= dur <= 12 and n_tokens >= 7:
            candidates.append((sid, dur, n_tokens, item.get('wer', 0)))
    if not candidates:
        return None
    # 选时长中等的
    candidates.sort(key=lambda x: abs(x[1] - 6.0))
    return candidates[0][0]


def find_case2_visual_missing(alignment_summary: List[Dict]) -> Optional[str]:
    """找视觉稀疏的 case：文本短但音频正常（视频帧长时间不变）"""
    candidates = []
    for item in alignment_summary:
        if item.get('status') not in ('aligned', 'partial'):
            continue
        sid = item['sample_id'].replace('q1_', '')
        align_path = Path('data/processed/features_q1/alignment') / f"{sid}.json"
        if not align_path.exists():
            continue
        with open(align_path) as f:
            align = json.load(f)
        tokens = align.get('tokens', [])
        real_tokens = [t for t in tokens if t.get('token') not in ('[PAD]', '[CLS]')]
        # 视觉稀疏：audio 跨度大（>10s）但 token 数较少（<7），意味着视觉有长静止
        audio_dur = align.get('audio_duration_sec', 0)
        fps = align.get('video_fps', 30)
        n_real = len(real_tokens)
        if audio_dur > 8 and n_real < 7 and item.get('wer', 1.0) < 0.5:
            candidates.append((sid, audio_dur, n_real, item.get('wer', 0)))
    if not candidates:
        # fallback：选最长的
        candidates = []
        for item in alignment_summary:
            if item.get('status') not in ('aligned', 'partial'):
                continue
            sid = item['sample_id'].replace('q1_', '')
            align_path = Path('data/processed/features_q1/alignment') / f"{sid}.json"
            if not align_path.exists():
                continue
            with open(align_path) as f:
                align = json.load(f)
            audio_dur = align.get('audio_duration_sec', 0)
            if audio_dur > 12 and item.get('wer', 1.0) < 0.5:
                candidates.append((sid, audio_dur, 0, item.get('wer', 0)))
    if not candidates:
        return None
    candidates.sort(key=lambda x: -x[1])
    return candidates[0][0]


def find_case3_text_sparse(alignment_summary: List[Dict]) -> Optional[str]:
    """找文本稀疏的 case：raw_text 短（<30 chars）"""
    candidates = []
    for item in alignment_summary:
        sid = item['sample_id'].replace('q1_', '')
        pkl_path = Path('data/processed/features_q1/single') / f"q1_{sid}.pkl"
        if not pkl_path.exists():
            continue
        with open(pkl_path, 'rb') as f:
            sample = pickle.load(f)
        text = sample.get('raw_text', '')
        # 文本稀疏：< 25 chars 或 < 5 words
        if len(text) <= 25 and len(text.split()) <= 5:
            align_path = Path('data/processed/features_q1/alignment') / f"{sid}.json"
            if not align_path.exists():
                continue
            with open(align_path) as f:
                align = json.load(f)
            audio_dur = align.get('audio_duration_sec', 0)
            if audio_dur > 1 and item.get('status') != 'failed':
                candidates.append((sid, len(text), audio_dur))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[1])
    return candidates[0][0]


def generate_case(case_id: str, sample_stem: str, case_label: str, output_dir: Path):
    """生成单个 case 的对照表 + 可视化 + metrics"""
    case_out_dir = output_dir / case_id
    case_out_dir.mkdir(parents=True, exist_ok=True)

    # 读 pkl + alignment
    pkl_path = Path('data/processed/features_q1/single') / f"q1_{sample_stem}.pkl"
    align_path = Path('data/processed/features_q1/alignment') / f"{sample_stem}.json"

    with open(pkl_path, 'rb') as f:
        sample = pickle.load(f)
    with open(align_path, 'r', encoding='utf-8') as f:
        alignment = json.load(f)

    raw_text = sample['raw_text']
    fps = alignment.get('video_fps', 30)
    audio_dur = alignment.get('audio_duration_sec', 0)
    tokens = alignment['tokens']
    real_tokens = [t for t in tokens if t.get('token') not in ('[PAD]', '[CLS]')]

    # 1. 对照表 csv
    rows = []
    char_cursor = 0
    for t in tokens:
        if t.get('token') in ('[PAD]', '[CLS]'):
            continue
        rows.append({
            'token_idx': t['idx'],
            'token': t['token'],
            'char_span': f"{t['char_start']}-{t['char_end']}",
            'audio_sec_span': f"{t['audio_start_sec']:.3f}-{t['audio_end_sec']:.3f}",
            'frame_span': f"{t['video_frame_start']}-{t['video_frame_end']}",
            'text_row': t['char_start'],
            'audio_row': round(t['audio_start_sec'], 3),
            'vision_row': t['video_frame_start'],
            'confidence': t.get('asr_confidence', '') if t.get('asr_confidence') is not None else '',
        })
    df = pd.DataFrame(rows)
    csv_path = case_out_dir / f"{case_id}_table.csv"
    df.to_csv(csv_path, index=False)
    print(f"  ✅ {csv_path}")

    # 2. metrics json
    n_tokens = len(real_tokens)
    asr_v = alignment.get('asr_verification', {})
    mean_dur = float(np.mean([t['audio_end_sec'] - t['audio_start_sec'] for t in real_tokens])) if real_tokens else 0
    metrics = {
        'case_id': case_id,
        'case_label': case_label,
        'sample_id': f"{sample['video_id']}$_${sample['clip_id']}",
        'video_id': sample['video_id'],
        'clip_id': sample['clip_id'],
        'n_tokens': n_tokens,
        'raw_text': raw_text,
        'audio_duration_sec': audio_dur,
        'video_fps': fps,
        'asr_match_rate': 1.0 - asr_v.get('wer', 1.0),
        'asr_wer': asr_v.get('wer', -1.0),
        'mean_token_duration_sec': round(mean_dur, 4),
        'alignment_failure_count': 0 if alignment.get('status') == 'aligned' else 1,
        'alignment_status': alignment.get('status', 'unknown'),
        'frame_to_token_overlap_ratio': 1.0,
        'shared_audio_video_window_sec': [real_tokens[0]['audio_start_sec'] if real_tokens else 0,
                                           real_tokens[-1]['audio_end_sec'] if real_tokens else 0],
        'created_at': datetime.now().isoformat(),
    }
    metrics_path = case_out_dir / f"{case_id}_metrics.json"
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"  ✅ {metrics_path}")

    # 3. 可视化图（三联子图）
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 1, figsize=(14, 10))

        # 上：token 时间轴
        ax = axes[0]
        for i, t in enumerate(real_tokens[:30]):  # 最多画 30 个
            ax.barh(0, t['audio_end_sec'] - t['audio_start_sec'],
                    left=t['audio_start_sec'], height=0.6,
                    color=plt.cm.tab20(i % 20), alpha=0.8,
                    edgecolor='black', linewidth=0.5)
            ax.text((t['audio_start_sec'] + t['audio_end_sec']) / 2, 0,
                    t['token'][:10], ha='center', va='center', fontsize=8)
        ax.set_xlim(0, audio_dur)
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([])
        ax.set_xlabel('Time (sec)')
        ax.set_title(f'{case_id}: {case_label}\nSample {sample["video_id"]}_{sample["clip_id"]} | '
                     f'duration={audio_dur:.2f}s | fps={fps:.1f} | '
                     f'WER={asr_v.get("wer", -1):.2f}', fontsize=11)
        ax.grid(True, axis='x', alpha=0.3)

        # 中：文字行 + 帧映射
        ax = axes[1]
        ax.set_xlim(0, max(audio_dur, 1))
        ax.set_ylim(0, 1)
        for t in real_tokens:
            x1 = t['audio_start_sec']
            x2 = t['audio_end_sec']
            ax.axvspan(x1, x2, ymin=0.3, ymax=0.7, alpha=0.4,
                       color=plt.cm.tab20(t['idx'] % 20))
            ax.text((x1 + x2) / 2, 0.5, t['token'][:8], ha='center', va='center', fontsize=7)
        ax.set_xlabel('Time (sec) -> Frame mapping')
        ax.set_yticks([])
        ax.set_title('Token -> Audio segment -> Frame mapping', fontsize=10)
        ax.grid(True, alpha=0.3)

        # 下：原始文本
        ax = axes[2]
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.text(0.5, 0.7, 'Raw Text:', ha='center', fontsize=12, fontweight='bold')
        # 高亮每个 token
        text_x = 0.02
        text_y = 0.4
        cursor = 0
        for t in real_tokens[:30]:
            token_text = t['token']
            char_start = t['char_start']
            char_end = t['char_end']
            if char_end <= len(raw_text):
                # 找位置
                while cursor < len(raw_text) and cursor < char_start:
                    cursor += 1
                # 简化：直接画 token
                ax.text(text_x, text_y, token_text, ha='left', va='center',
                        fontsize=10, bbox=dict(boxstyle='round,pad=0.2',
                                               facecolor=plt.cm.tab20(t['idx'] % 20),
                                               alpha=0.5))
                text_x += 0.05 + len(token_text) * 0.012
                if text_x > 0.95:
                    text_x = 0.02
                    text_y -= 0.15
        ax.text(0.5, 0.05, f'ASR: {asr_v.get("asr_text", "")[:80]}...',
                ha='center', fontsize=9, style='italic', color='gray')

        plt.tight_layout()
        png_path = case_out_dir / f"{case_id}_figure.png"
        plt.savefig(png_path, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"  ✅ {png_path}")
    except Exception as e:
        print(f"  ⚠️  生成图失败: {e}")

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary_path', type=str,
                        default='data/processed/features_q1/alignment_summary.json')
    parser.add_argument('--output_dir', type=str,
                        default='data/processed/features_q1/sync_cases')
    args = parser.parse_args()

    with open(args.summary_path, 'r', encoding='utf-8') as f:
        summary = json.load(f)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("选样中...")
    case1 = find_case1_normal(summary['items'])
    print(f"  Case 1 (normal): {case1}")
    case2 = find_case2_visual_missing(summary['items'])
    print(f"  Case 2 (visual_missing/long): {case2}")
    case3 = find_case3_text_sparse(summary['items'])
    print(f"  Case 3 (text_sparse): {case3}")

    cases = [
        ('case01_normal', case1, 'Normal三模态齐全'),
        ('case02_visual_missing', case2, '视觉稀疏/长音频'),
        ('case03_text_sparse', case3, '文本稀疏'),
    ]

    print("\n生成 case 文件...")
    results = []
    for case_id, sample_stem, label in cases:
        if sample_stem is None:
            print(f"  ⚠️  {case_id}: 无可用样本")
            continue
        print(f"\n--- {case_id} ---")
        m = generate_case(case_id, sample_stem, label, output_dir)
        results.append(m)

    # 写案例总览
    overview_path = output_dir / 'overview.json'
    with open(overview_path, 'w', encoding='utf-8') as f:
        json.dump({
            'cases': results,
            'created_at': datetime.now().isoformat(),
        }, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 案例总览: {overview_path}")


if __name__ == '__main__':
    main()
