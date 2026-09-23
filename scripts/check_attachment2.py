"""探查附件2特征文件的统计特性，用于确定提取工具选型"""
import pickle
import numpy as np
import os

def probe_pkl(pkl_path, output_path=None):
    """全面探查附件2特征文件的内部结构"""

    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    lines = []
    lines.append("=" * 70)
    lines.append("附件2 探查报告 (aligned_50.pkl)")
    lines.append("=" * 70)

    for split in ['train', 'valid', 'test']:
        d = data[split]
        lines.append(f"\n{'─' * 30} {split.upper()} ({len(d['id'])} samples) {'─' * 30}")

        # 基本信息
        lines.append(f"\n  字段列表: {list(d.keys())}")

        # 标签分布
        if 'classification_labels' in d:
            labels = np.array(d['classification_labels'])
            lines.append(f"\n  分类标签分布 (0=Neg, 1=Neu, 2=Pos):")
            names = ['Negative', 'Neutral', 'Positive']
            for i, name in enumerate(names):
                count = (labels == i).sum()
                pct = count / len(labels) * 100
                lines.append(f"    {name}({i}): {count} ({pct:.1f}%)")

        # 回归标签统计
        if 'regression_labels' in d:
            reg = np.array(d['regression_labels'])
            lines.append(f"\n  回归标签统计:")
            lines.append(f"    min: {reg.min():.3f}, max: {reg.max():.3f}, mean: {reg.mean():.3f}, std: {reg.std():.3f}")

        # 各模态特征统计
        for field in ['text', 'audio', 'vision']:
            if field in d:
                arr = np.array(d[field])
                lines.append(f"\n  【{field}】")
                lines.append(f"    形状: {arr.shape}  dtype: {arr.dtype}")
                flat = arr.reshape(-1)
                lines.append(f"    全局: min={flat.min():.4f}, max={flat.max():.4f}, mean={flat.mean():.4f}, std={flat.std():.4f}")
                lines.append(f"    非零比例: {(flat != 0).mean():.4f}")

                # 逐样本统计 (最后一个维度)
                if arr.ndim == 3:
                    # 各维度的mean over time
                    per_sample = arr.mean(axis=1)  # (N, D)
                    lines.append(f"    样本均值: mean={per_sample.mean():.4f}, std={per_sample.std():.4f}")
                    # 各样本的L2范数
                    l2_norms = np.linalg.norm(arr.reshape(arr.shape[0], -1), axis=1)
                    lines.append(f"    样本L2: mean={l2_norms.mean():.2f}, std={l2_norms.std():.2f}")

    # ── 关键推断 ──────────────────────────────────────────────────────────────
    lines.append("\n" + "=" * 70)
    lines.append("📌 关键推断")
    lines.append("=" * 70)

    # 文本特征推断
    text = np.array(data['train']['text'])
    lines.append(f"\n  【文本特征】")
    lines.append(f"    维度: {text.shape[-1]} → 符合 BERT-base-uncased (768d)")
    lines.append(f"    值域: [{text.min():.3f}, {text.max():.3f}] → BERT hidden state 典型值域")
    # 检查是否像 BERT (值集中在某个范围)
    lines.append(f"    建议工具: transformers.BertModel (bert-base-uncased)")

    # 语音特征推断
    audio = np.array(data['train']['audio'])
    audio_flat = audio.reshape(-1, audio.shape[-1])
    nonzero_cols = (audio_flat.std(axis=0) > 1e-6).sum()
    lines.append(f"\n  【语音特征】")
    lines.append(f"    维度: {audio.shape[-1]}  非零维度数: {nonzero_cols}")
    lines.append(f"    值域: [{audio.min():.4f}, {audio.max():.4f}]")
    lines.append(f"    均值: {audio.mean():.4f}, 标准差: {audio.std():.4f}")
    if audio.shape[-1] == 74:
        lines.append(f"    ✅ 74维 → 与 OpenSMILE eGeMAPS 子集高度吻合")
        lines.append(f"    建议工具: opensmile.FeatureSet.eGeMAPSv02 (取其 LLD 子集)")
    elif audio.shape[-1] == 88:
        lines.append(f"    88维 → 可能是完整的 OpenSMILE eGeMAPS LLD")
    elif audio.shape[-1] == 768:
        lines.append(f"    768维 → 可能是 wav2vec2 特征 (需要 Linear 投影到 74)")
    else:
        lines.append(f"    未知维度: {audio.shape[-1]}")

    # 视觉特征推断
    vision = np.array(data['train']['vision'])
    lines.append(f"\n  【视觉特征】")
    lines.append(f"    维度: {vision.shape[-1]}  值域: [{vision.min():.4f}, {vision.max():.4f}]")
    if vision.shape[-1] == 35:
        lines.append(f"    ✅ 35维 → OpenFace 2.0 的 AU (17) + pose (6) + gaze (6) + 其它")
        lines.append(f"    建议工具: OpenFace 2.0 FeatureExtraction")
    else:
        lines.append(f"    未知维度: {vision.shape[-1]}")

    # 标签映射验证
    lines.append(f"\n  【标签映射】")
    if 'classification_labels' in data['train'] and 'regression_labels' in data['train']:
        reg = np.array(data['train']['regression_labels'])
        cls = np.array(data['train']['classification_labels'])
        # Negative → 0
        neg_ok = (cls[reg < 0] == 0).all()
        # Neutral → 1
        neu_ok = (cls[reg == 0] == 1).all()
        # Positive → 2
        pos_ok = (cls[reg > 0] == 2).all()
        lines.append(f"    Negative (reg<0) → 0: {'✅' if neg_ok else '❌'}")
        lines.append(f"    Neutral  (reg=0) → 1: {'✅' if neu_ok else '❌'}")
        lines.append(f"    Positive (reg>0) → 2: {'✅' if pos_ok else '❌'}")

    # 输出
    output = "\n".join(lines)
    print(output)

    # 保存到文件
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(output)
        print(f"\n✅ 报告已保存到: {output_path}")

    return output


if __name__ == '__main__':
    import sys
    pkl_path = sys.argv[1] if len(sys.argv) > 1 else 'huaweibei-E/data/raw_attachment2/aligned_50.pkl'
    output_path = sys.argv[2] if len(sys.argv) > 2 else 'huaweibei-E/docs/q1_attachment2_probe.md'
    probe_pkl(pkl_path, output_path)
