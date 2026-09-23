"""
文本特征提取器
使用 BERT-base-uncased 提取 768 维文本特征
与附件2 aligned_50.pkl 的 text/text_bert 字段完全兼容
"""
import torch
import numpy as np
from transformers import BertModel, BertTokenizer
from typing import Union, List
import os


class TextFeatureExtractor:
    """
    从原始英文文本提取 BERT 768 维特征

    输出与附件2完全对齐:
        - text: (T, 768) float32, T=max_seq_len
        - text_bert: (3, T) int64, [input_ids, attention_mask, token_type_ids]
    """

    def __init__(self,
                 model_name: str = 'bert-base-uncased',
                 max_seq_len: int = 50,
                 device: str = None,
                 cache_dir: str = None):
        """
        Args:
            model_name: HuggingFace 模型名
            max_seq_len: 最大序列长度（与附件2保持一致为50）
            device: 'cuda'/'cpu', None则自动选择
            cache_dir: 权重缓存目录
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.max_seq_len = max_seq_len
        print(f"[TextExtractor] 加载模型: {model_name} -> {self.device}")

        self.tokenizer = BertTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
        self.model = BertModel.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            output_hidden_states=False  # 只取最后一层，节省显存
        )
        self.model.eval()
        self.model.to(self.device)

    def extract(self, raw_text: Union[str, List[str]],
                 return_cls: bool = False) -> dict:
        """
        从原始文本提取特征

        Args:
            raw_text: 单条文本或文本列表
            return_cls: 是否返回 [CLS] 向量（用于分类）

        Returns:
            dict with keys:
                text: (N, T, 768) float32  # N=batch or 1
                text_bert: (N, 3, T) int64
                cls: (N, 768) float32 (if return_cls=True)
        """
        is_single = isinstance(raw_text, str)
        if is_single:
            raw_text = [raw_text]

        # Tokenize
        encoded = self.tokenizer(
            raw_text,
            padding='max_length',
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors='pt'
        )

        # text_bert: [input_ids, attention_mask, token_type_ids] → (N, 3, T)
        text_bert = torch.stack([
            encoded['input_ids'],
            encoded['attention_mask'],
            encoded['token_type_ids']
        ], dim=1).long().numpy()

        # 推理
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with torch.no_grad():
            outputs = self.model(**encoded)

        # last_hidden_state: (N, T, 768)
        text_feat = outputs.last_hidden_state.cpu().numpy().astype(np.float32)

        result = {
            'text': text_feat,
            'text_bert': text_bert,
        }

        if return_cls:
            result['cls'] = outputs.last_hidden_state[:, 0, :].cpu().numpy().astype(np.float32)

        if is_single:
            # 单条返回时去掉 batch 维
            result = {
                'text': result['text'][0],
                'text_bert': result['text_bert'][0],
            }
            if return_cls:
                result['cls'] = result['cls'][0]

        return result

    def extract_for_sample(self, raw_text: str) -> dict:
        """
        提取单条样本（兼容附件2字段格式）
        返回可直接写入 pkl 的字段
        """
        feat = self.extract(raw_text, return_cls=True)
        return {
            'raw_text': raw_text,
            'text': feat['text'],         # (50, 768)
            'text_bert': feat['text_bert'],  # (3, 50)
        }

    def batch_extract(self, texts: List[str], batch_size: int = 32) -> dict:
        """
        批量提取（内存友好）
        """
        all_text = []
        all_text_bert = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            feat = self.extract(batch)
            all_text.append(feat['text'])
            all_text_bert.append(feat['text_bert'])

        return {
            'text': np.concatenate(all_text, axis=0).astype(np.float32),
            'text_bert': np.concatenate(all_text_bert, axis=0).astype(np.int64),
        }


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """计算两个向量的余弦相似度"""
    a_flat = a.flatten()
    b_flat = b.flatten()
    return np.dot(a_flat, b_flat) / (np.linalg.norm(a_flat) * np.linalg.norm(b_flat))


def verify_with_attachment2(extractor: TextFeatureExtractor,
                             pkl_path: str,
                             sample_idx: int = 0) -> float:
    """
    用附件2的 raw_text 重新跑 BERT,验证与附件2 text 的相似度
    期望 > 0.95 表示特征提取器与附件2来源一致
    """
    import pickle

    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    train_data = data['train']
    raw_text = train_data['raw_text'][sample_idx]
    ref_text_feat = train_data['text'][sample_idx]

    print(f"\n[验证] 样本 {sample_idx}: \"{raw_text[:80]}...\"")
    print(f"[验证] 参考特征 shape: {ref_text_feat.shape}")

    my_feat = extractor.extract(raw_text)['text']
    sim = cosine_similarity(my_feat, ref_text_feat)

    print(f"[验证] 余弦相似度: {sim:.6f}")
    if sim > 0.95:
        print(f"[验证] ✅ 相似度 > 0.95, 特征提取器与附件2一致")
    elif sim > 0.80:
        print(f"[验证] ⚠️ 相似度 0.80~0.95, 可能存在差异（如预训练权重版本）")
    else:
        print(f"[验证] ❌ 相似度 < 0.80, 请检查模型配置")

    return sim


if __name__ == '__main__':
    import argparse
    import pickle

    parser = argparse.ArgumentParser(description='文本特征提取')
    parser.add_argument('--pkl_path', type=str,
                        default='/mnt/data2/home/chenyu/HuaWeiCup/E题数据/附件2-数据集特征文件/aligned_50.pkl',
                        help='附件2 pkl 路径（用于验证）')
    parser.add_argument('--test', action='store_true', help='仅运行验证测试')
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    extractor = TextFeatureExtractor(device=device)

    if args.test:
        # 验证: 从附件2取10条样本,对比相似度
        with open(args.pkl_path, 'rb') as f:
            data = pickle.load(f)

        train_data = data['train']
        sims = []
        for i in range(10):
            raw_text = train_data['raw_text'][i]
            ref_feat = train_data['text'][i]
            my_feat = extractor.extract(raw_text)['text']
            sim = cosine_similarity(my_feat, ref_feat)
            sims.append(sim)
            print(f"  样本 {i}: 相似度 {sim:.4f}")

        print(f"\n平均相似度: {np.mean(sims):.4f}")
        print(f"最低相似度: {np.min(sims):.4f}")
        print(f"最高相似度: {np.max(sims):.4f}")
    else:
        print("[TextExtractor] 已初始化, 使用 extractor.extract(text) 提取特征")
