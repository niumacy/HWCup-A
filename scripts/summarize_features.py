"""
生成特征摘要表
输出: outputs/tables/q1_feature_summary.csv
"""
import pickle
import numpy as np
import pandas as pd
import json


def generate_summary(q1_pkl_path: str, label_xlsx_path: str, output_csv_path: str):
    """生成 Q1 特征摘要表"""

    # 加载生成的特征
    with open(q1_pkl_path, 'rb') as f:
        q1_data = pickle.load(f)

    # 加载原始标签
    df_label = pd.read_excel(label_xlsx_path)

    # 收集摘要信息
    summaries = []
    data = q1_data['all']
    for i in range(len(data['id'])):
        sample_id = data['id'][i]
        # 解析 id
        video_id, clip_id = sample_id.split('$_$')

        text_arr = np.array(data['text'][i])
        audio_arr = np.array(data['audio'][i])
        vision_arr = np.array(data['vision'][i])

        # 找到对应的原始标签
        label_row = df_label[(df_label['video_id'] == video_id) &
                              (df_label['clip_id'] == int(clip_id))].iloc[0]

        summary = {
            'sample_id': sample_id,
            'video_id': video_id,
            'clip_id': int(clip_id),
            'annotation': data['annotation'][i],
            'regression_label': float(data['regression_labels'][i]),
            'classification_label': int(data['classification_labels'][i]),
            'text_length': len(text_arr),
            'text_dim': text_arr.shape[1] if text_arr.ndim > 1 else 0,
            'text_nonzero_ratio': float((text_arr != 0).mean()),
            'audio_length': len(audio_arr),
            'audio_dim': audio_arr.shape[1] if audio_arr.ndim > 1 else 0,
            'audio_nonzero_ratio': float((audio_arr != 0).mean()),
            'vision_length': len(vision_arr),
            'vision_dim': vision_arr.shape[1] if vision_arr.ndim > 1 else 0,
            'vision_nonzero_ratio': float((vision_arr != 0).mean()),
            'raw_text_preview': data['raw_text'][i][:80] + ('...' if len(data['raw_text'][i]) > 80 else ''),
            'extract_status': 'success',
        }
        summaries.append(summary)

    # 创建 DataFrame
    df = pd.DataFrame(summaries)

    # 保存
    df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    print(f"✅ 摘要表已保存: {output_csv_path}")
    print(f"   共 {len(df)} 条样本")
    print(f"\n📊 数据概览:")
    print(f"   文本形状: {df['text_dim'].iloc[0]} 维 × {df['text_length'].iloc[0]} 帧")
    print(f"   音频形状: {df['audio_dim'].iloc[0]} 维 × {df['audio_length'].iloc[0]} 帧")
    print(f"   视觉形状: {df['vision_dim'].iloc[0]} 维 × {df['vision_length'].iloc[0]} 帧")
    print(f"\n🏷️  情感分布:")
    print(df['annotation'].value_counts())

    return df


if __name__ == '__main__':
    q1_pkl = '/mnt/data2/home/chenyu/HuaWeiCup/huaweibei-E/data/processed/features_q1/q1_features_combined.pkl'
    label_xlsx = '/mnt/data2/home/chenyu/HuaWeiCup/E题数据/附件1-数据集原始多模态样本/MOSEI数据集部分原始视频-100条/label-100.xlsx'
    output_csv = '/mnt/data2/home/chenyu/HuaWeiCup/huaweibei-E/outputs/tables/q1_feature_summary.csv'

    df = generate_summary(q1_pkl, label_xlsx, output_csv)
    print(f"\n前 5 行预览:")
    print(df.head().to_string())
