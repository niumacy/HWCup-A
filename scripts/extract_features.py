"""
全量100条样本特征提取主脚本
从附件1的原始视频提取文本、语音、视觉三模态特征
"""
import os
import sys
import json
import pickle
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm

# 必须在 import transformers 之前设置镜像
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.feature_extractor.text_extractor import TextFeatureExtractor
from src.feature_extractor.audio_extractor import AudioFeatureExtractor, extract_audio_from_video
from src.feature_extractor.vision_extractor import VisionFeatureExtractor


def find_video_path(video_id: str, clip_id: int, base_dir: str) -> str:
    """根据 video_id 和 clip_id 定位视频文件路径"""
    return os.path.join(base_dir, video_id, f'{clip_id}.mp4')


def extract_all_features(label_xlsx: str,
                         video_base_dir: str,
                         output_dir: str,
                         target_frames: int = 50,
                         device: str = 'cuda',
                         modalities: list = ['text', 'audio', 'vision']):
    """
    全量提取100条样本的三模态特征

    Args:
        label_xlsx: label-100.xlsx 路径
        video_base_dir: 视频根目录 (附件1)
        output_dir: 输出目录
        target_frames: 时间帧数
        device: 'cuda'/'cpu'
        modalities: 要提取的模态列表
    """
    # 读取标签
    df = pd.read_excel(label_xlsx)
    print(f"📋 读取标签: {len(df)} 条样本")
    print(f"   情感分布: {df['annotation'].value_counts().to_dict()}")

    # 准备输出目录
    single_dir = os.path.join(output_dir, 'single')
    os.makedirs(single_dir, exist_ok=True)

    # 初始化提取器
    if 'text' in modalities:
        print("\n🔤 初始化文本提取器 (BERT-base-uncased)...")
        text_ext = TextFeatureExtractor(device=device, max_seq_len=target_frames)
    if 'audio' in modalities:
        print("\n🔊 初始化音频提取器 (librosa)...")
        audio_ext = AudioFeatureExtractor(target_frames=target_frames)
    if 'vision' in modalities:
        print("\n🎬 初始化视觉提取器 (ResNet18)...")
        vision_ext = VisionFeatureExtractor(device=device, target_frames=target_frames)

    # 提取日志
    log = {
        'start_time': datetime.now().isoformat(),
        'total_samples': len(df),
        'success_count': 0,
        'failed_samples': [],
        'modalities': modalities,
        'extractor_versions': {},
    }

    # 存储合并数据
    all_ids = []
    all_raw_text = []
    all_text = []
    all_text_bert = []
    all_audio = []
    all_vision = []
    all_cls_labels = []
    all_reg_labels = []
    all_annotations = []

    # 遍历所有样本
    print("\n" + "=" * 70)
    print("开始提取特征...")
    print("=" * 70)

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="提取进度"):
        video_id = row['video_id']
        clip_id = row['clip_id']
        text = row['text']
        label = row['label']
        annotation = row['annotation']

        sample_id = f"{video_id}$_${clip_id}"
        video_path = find_video_path(video_id, clip_id, video_base_dir)

        sample_result = {
            'id': sample_id,
            'video_id': video_id,
            'clip_id': clip_id,
            'raw_text': text,
            'annotation': annotation,
            'regression_label': label,
            'classification_label': 0 if label < 0 else (1 if label == 0 else 2),
            'modalities': {},
        }

        # 1. 文本特征
        if 'text' in modalities:
            try:
                text_feat = text_ext.extract(text)
                sample_result['text'] = text_feat['text'].astype(np.float32)  # (50, 768)
                sample_result['text_bert'] = text_feat['text_bert'].astype(np.int64)  # (3, 50)
                sample_result['modalities']['text'] = 'success'
            except Exception as e:
                sample_result['modalities']['text'] = f'failed: {str(e)}'
                sample_result['text'] = np.zeros((target_frames, 768), dtype=np.float32)
                sample_result['text_bert'] = np.zeros((3, target_frames), dtype=np.int64)

        # 2. 音频特征
        if 'audio' in modalities:
            try:
                if os.path.exists(video_path):
                    y, sr = extract_audio_from_video(video_path)
                    audio_feat = audio_ext.extract_from_waveform(y, sr)  # (74, T)
                    sample_result['audio'] = audio_feat.T.astype(np.float32)  # (T, 74)
                    sample_result['modalities']['audio'] = 'success'
                else:
                    raise FileNotFoundError(f"视频文件不存在: {video_path}")
            except Exception as e:
                sample_result['modalities']['audio'] = f'failed: {str(e)}'
                sample_result['audio'] = np.zeros((target_frames, 74), dtype=np.float32)

        # 3. 视觉特征
        if 'vision' in modalities:
            try:
                if os.path.exists(video_path):
                    frames = vision_ext.extract_frames_from_video(video_path, num_frames=target_frames)
                    vision_feat = vision_ext.extract_features_from_frames(frames)  # (50, 35)
                    sample_result['vision'] = vision_feat.astype(np.float32)
                    sample_result['modalities']['vision'] = 'success'
                else:
                    raise FileNotFoundError(f"视频文件不存在: {video_path}")
            except Exception as e:
                sample_result['modalities']['vision'] = f'failed: {str(e)}'
                sample_result['vision'] = np.zeros((target_frames, 35), dtype=np.float32)

        # 保存单个样本
        sample_file = os.path.join(single_dir, f'q1_{video_id}_{clip_id}.pkl')
        with open(sample_file, 'wb') as f:
            pickle.dump(sample_result, f)

        # 收集到合并数据
        if 'text' in modalities:
            all_ids.append(sample_id)
            all_raw_text.append(text)
            all_text.append(sample_result['text'])
            all_text_bert.append(sample_result['text_bert'])
            all_audio.append(sample_result['audio'])
            all_vision.append(sample_result['vision'])
            all_cls_labels.append(sample_result['classification_label'])
            all_reg_labels.append(label)
            all_annotations.append(annotation)

        # 检查是否成功
        all_success = all(v == 'success' for v in sample_result['modalities'].values())
        if all_success:
            log['success_count'] += 1
        else:
            log['failed_samples'].append({
                'id': sample_id,
                'modalities': sample_result['modalities'],
            })

    # 生成合并文件
    combined_path = os.path.join(output_dir, 'q1_features_combined.pkl')
    combined_data = {
        'all': {
            'id': all_ids,
            'raw_text': all_raw_text,
            'text': np.stack(all_text, axis=0).astype(np.float32),       # (N, 50, 768)
            'text_bert': np.stack(all_text_bert, axis=0).astype(np.int64), # (N, 3, 50)
            'audio': np.stack(all_audio, axis=0).astype(np.float32),     # (N, 50, 74)
            'vision': np.stack(all_vision, axis=0).astype(np.float32),   # (N, 50, 35)
            'classification_labels': np.array(all_cls_labels, dtype=np.int64),
            'regression_labels': np.array(all_reg_labels, dtype=np.float32),
            'annotation': all_annotations,
        }
    }

    with open(combined_path, 'wb') as f:
        pickle.dump(combined_data, f)

    log['end_time'] = datetime.now().isoformat()
    log['combined_path'] = combined_path
    log['single_dir'] = single_dir

    # 保存日志
    log_path = os.path.join(output_dir, 'extract_log.json')
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(log, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 提取完成:")
    print(f"   总样本数: {len(df)}")
    print(f"   成功样本数: {log['success_count']}")
    print(f"   失败样本数: {len(log['failed_samples'])}")
    print(f"   合并文件: {combined_path}")
    print(f"   单样本目录: {single_dir}")
    print(f"   特征形状: text={combined_data['all']['text'].shape}, "
          f"audio={combined_data['all']['audio'].shape}, "
          f"vision={combined_data['all']['vision'].shape}")
    print(f"   日志文件: {log_path}")

    return combined_data, log


def main():
    parser = argparse.ArgumentParser(description='全量特征提取')
    parser.add_argument('--label', type=str,
                        default='/mnt/data2/home/chenyu/HuaWeiCup/E题数据/附件1-数据集原始多模态样本/MOSEI数据集部分原始视频-100条/label-100.xlsx')
    parser.add_argument('--video_dir', type=str,
                        default='/mnt/data2/home/chenyu/HuaWeiCup/E题数据/附件1-数据集原始多模态样本/MOSEI数据集部分原始视频-100条')
    parser.add_argument('--output_dir', type=str,
                        default='/mnt/data2/home/chenyu/HuaWeiCup/huaweibei-E/data/processed/features_q1')
    parser.add_argument('--target_frames', type=int, default=50)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--modalities', type=str, default='text,audio,vision')
    args = parser.parse_args()

    extract_all_features(
        label_xlsx=args.label,
        video_base_dir=args.video_dir,
        output_dir=args.output_dir,
        target_frames=args.target_frames,
        device=args.device,
        modalities=args.modalities.split(','),
    )


if __name__ == '__main__':
    main()
