"""
合并特征 v2 生成 + 摘要表（含 effective length）

读取：
  - data/processed/features_q1/single/*.pkl（100 条）
  - data/processed/features_q1/alignment/*.json（100 条词级对齐）

产出：
  - q1_features_all_v2.pkl（合并特征，字段名与 v1 一致，新增可选 per_token 字段）
  - q1_features_combined.pkl -> q1_features_all_v2.pkl（软链接）
  - outputs/tables/q1_feature_summary.csv（含 effective length）
  - data/processed/features_q1/backup_old/（旧版本备份）
"""
import os
import sys
import json
import pickle
import shutil
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd


def compute_effective_lengths(sample: dict, alignment: dict) -> Dict:
    """计算三模态的有效长度（基于 mask / token records 而非 dim 是否为零）"""
    # 文本有效长度 = alignment 中非 [PAD] 的 token 数（不含 [CLS]）
    tokens = alignment.get('tokens', [])
    real_text_tokens = sum(1 for t in tokens if t.get('token', '') not in ('[PAD]', '[CLS]'))
    text_eff = real_text_tokens

    # 音频有效长度：使用 audio_duration_sec * sample_rate / hop_length ≈ 帧数
    # 这里我们用 alignment tokens 中每个 token 的有效区间估算
    audio_dur = alignment.get('audio_duration_sec', 0.0)
    fps = alignment.get('video_fps', 30.0)
    # 真实音频帧数：duration * fps
    sr = 16000
    hop_length = 256
    audio_eff_raw = int(audio_dur * sr / hop_length) if audio_dur > 0 else 0
    vision_eff_raw = int(audio_dur * fps) if audio_dur > 0 else 0

    return {
        'text_length_effective': real_text_tokens,
        'audio_length_effective': audio_eff_raw,
        'vision_length_effective': vision_eff_raw,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--single_dir', type=str,
                        default='data/processed/features_q1/single')
    parser.add_argument('--alignment_dir', type=str,
                        default='data/processed/features_q1/alignment')
    parser.add_argument('--output_dir', type=str,
                        default='data/processed/features_q1')
    parser.add_argument('--summary_csv', type=str,
                        default='outputs/tables/q1_feature_summary.csv')
    parser.add_argument('--old_combined', type=str,
                        default='data/processed/features_q1/q1_features_combined.pkl')
    args = parser.parse_args()

    single_dir = Path(args.single_dir)
    alignment_dir = Path(args.alignment_dir)
    output_dir = Path(args.output_dir)
    summary_csv = Path(args.summary_csv)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)

    # 1. 备份旧版本
    backup_dir = output_dir / 'backup_old'
    backup_dir.mkdir(exist_ok=True)
    if Path(args.old_combined).exists():
        backup_path = backup_dir / 'q1_features_all.pkl'
        if not backup_path.exists():
            shutil.copy2(args.old_combined, backup_path)
            print(f"已备份旧合并特征到 {backup_path}")
    # 备份旧 extract_log.json
    old_log = output_dir / 'extract_log.json'
    if old_log.exists():
        backup_log = backup_dir / 'extract_log_v1.json'
        if not backup_log.exists():
            shutil.copy2(old_log, backup_log)
            print(f"已备份旧日志到 {backup_log}")

    # 2. 加载所有单样本 pkl + 对齐 JSON
    pkl_files = sorted(single_dir.glob('q1_*.pkl'))
    print(f"加载 {len(pkl_files)} 条样本...")

    all_ids = []
    all_raw_text = []
    all_text = []
    all_text_bert = []
    all_audio = []
    all_vision = []
    all_cls_labels = []
    all_reg_labels = []
    all_annotations = []

    # 新增字段（per_token）
    all_per_token_audio_start = []
    all_per_token_audio_end = []
    all_per_token_video_start = []
    all_per_token_video_end = []

    summary_rows = []

    for pkl_path in pkl_files:
        with open(pkl_path, 'rb') as f:
            sample = pickle.load(f)

        sample_id = sample['id']
        stem = sample_id.replace('$_$', '_')
        align_path = alignment_dir / f"{stem}.json"
        if not align_path.exists():
            print(f"  ⚠️  对齐文件缺失: {align_path}")
            continue

        with open(align_path, 'r', encoding='utf-8') as f:
            alignment = json.load(f)

        # 收集合并字段
        all_ids.append(sample_id)
        all_raw_text.append(sample['raw_text'])
        all_text.append(sample['text'].astype(np.float32))
        all_text_bert.append(sample['text_bert'].astype(np.int64))
        all_audio.append(sample['audio'].astype(np.float32))
        all_vision.append(sample['vision'].astype(np.float32))
        all_cls_labels.append(sample['classification_label'])
        all_reg_labels.append(sample['regression_label'])
        all_annotations.append(sample['annotation'])

        # 新增 per_token 字段（len=50）
        tokens = alignment['tokens']
        audio_start = [t['audio_start_sec'] for t in tokens]
        audio_end = [t['audio_end_sec'] for t in tokens]
        video_start = [t['video_frame_start'] for t in tokens]
        video_end = [t['video_frame_end'] for t in tokens]
        all_per_token_audio_start.append(audio_start)
        all_per_token_audio_end.append(audio_end)
        all_per_token_video_start.append(video_start)
        all_per_token_video_end.append(video_end)

        # 计算有效长度
        eff = compute_effective_lengths(sample, alignment)
        asr_v = alignment.get('asr_verification', {})

        # 摘要表行
        row = {
            'sample_id': sample_id,
            'video_id': sample['video_id'],
            'clip_id': sample['clip_id'],
            'annotation': sample['annotation'],
            'regression_label': sample['regression_label'],
            'classification_label': sample['classification_label'],
            'text_length_storage': 50,
            'text_length_effective': eff['text_length_effective'],
            'text_dim': 768,
            'text_nonzero_ratio': float(np.mean(sample['text'] != 0)),
            'audio_length_storage': sample['audio'].shape[0],
            'audio_length_effective': eff['audio_length_effective'],
            'audio_dim': 74,
            'audio_nonzero_ratio': float(np.mean(sample['audio'] != 0)),
            'vision_length_storage': sample['vision'].shape[0],
            'vision_length_effective': eff['vision_length_effective'],
            'vision_dim': 35,
            'vision_nonzero_ratio': float(np.mean(sample['vision'] != 0)),
            'raw_text_preview': sample['raw_text'][:80],
            'extract_status': 'success',
            # 新增
            'asr_match_rate': 1.0 - asr_v.get('wer', 0.0),
            'alignment_status': alignment.get('status', 'unknown'),
            'alignment_wer': asr_v.get('wer', -1.0),
            'audio_duration_sec': alignment.get('audio_duration_sec', 0.0),
            'video_fps': alignment.get('video_fps', 30.0),
            'source_video_path_anonymized': f"data/raw_attachment1/MOSEI数据集部分原始视频-100条/{sample['video_id']}/{sample['clip_id']}.mp4",
            'source_clip_start_sec': audio_start[1] if len(audio_start) > 1 else 0,
            'source_clip_end_sec': audio_end[15] if len(audio_end) > 15 else 0,
        }
        summary_rows.append(row)

    # 3. 生成合并特征 v2
    combined_data = {
        'all': {
            'id': all_ids,
            'raw_text': all_raw_text,
            'text': np.stack(all_text, axis=0).astype(np.float32),
            'text_bert': np.stack(all_text_bert, axis=0).astype(np.int64),
            'audio': np.stack(all_audio, axis=0).astype(np.float32),
            'vision': np.stack(all_vision, axis=0).astype(np.float32),
            'classification_labels': np.array(all_cls_labels, dtype=np.int64),
            'regression_labels': np.array(all_reg_labels, dtype=np.float32),
            'annotation': all_annotations,
            # 新增可选字段
            'per_token_audio_sec_start': np.array(all_per_token_audio_start, dtype=np.float32),
            'per_token_audio_sec_end': np.array(all_per_token_audio_end, dtype=np.float32),
            'per_token_video_frame_start': np.array(all_per_token_video_start, dtype=np.int32),
            'per_token_video_frame_end': np.array(all_per_token_video_end, dtype=np.int32),
        }
    }

    v2_path = output_dir / 'q1_features_all_v2.pkl'
    with open(v2_path, 'wb') as f:
        pickle.dump(combined_data, f)
    print(f"✅ v2 合并特征已保存: {v2_path}")
    print(f"   text: {combined_data['all']['text'].shape}")
    print(f"   audio: {combined_data['all']['audio'].shape}")
    print(f"   vision: {combined_data['all']['vision'].shape}")

    # 4. 软链接 q1_features_combined.pkl -> q1_features_all_v2.pkl
    link_path = output_dir / 'q1_features_combined.pkl'
    if link_path.exists() or link_path.is_symlink():
        link_path.unlink()
    os.symlink('q1_features_all_v2.pkl', str(link_path))
    print(f"✅ 软链接已建立: {link_path} -> q1_features_all_v2.pkl")

    # 5. 摘要表
    df = pd.DataFrame(summary_rows)
    df.to_csv(summary_csv, index=False)
    print(f"✅ 摘要表已保存: {summary_csv}")
    print(f"   总行数: {len(df)}")
    print(f"   text_length_effective 范围: [{df['text_length_effective'].min()}, {df['text_length_effective'].max()}]")
    print(f"   audio_length_effective 范围: [{df['audio_length_effective'].min()}, {df['audio_length_effective'].max()}]")
    print(f"   vision_length_effective 范围: [{df['vision_length_effective'].min()}, {df['vision_length_effective'].max()}]")
    print(f"   alignment_status 分布: {df['alignment_status'].value_counts().to_dict()}")
    print(f"   mean asr_match_rate: {df['asr_match_rate'].mean():.3f}")


if __name__ == '__main__':
    main()
