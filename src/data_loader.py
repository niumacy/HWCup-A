"""
统一数据加载接口
支持: 附件2 + 成员A自生成的Q1特征
"""
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Optional, Literal


def load_attachment2(pkl_path: str, split: Optional[Literal['train', 'valid', 'test']] = None):
    """
    加载附件2特征文件

    Args:
        pkl_path: aligned_50.pkl 或 unaligned_50.pkl 路径
        split: 可选，只加载某个划分

    Returns:
        data: dict with keys 'train'/'valid'/'test', each is a dict with:
            - id: list[str]
            - raw_text: list[str]
            - text: (N, 50, 768) float32
            - text_bert: (3, N, 50) int64
            - audio: (N, T, 74) float64
            - vision: (N, T, 35) float64
            - classification_labels: (N,) int64  (0=Neg, 1=Neu, 2=Pos)
            - regression_labels: (N,) float32  ([-3, 3])
    """
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    if split is not None:
        return data[split]
    return data


def load_q1_features(pkl_path: str):
    """
    加载成员A自生成的Q1特征文件
    与附件2接口兼容
    """
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data


def get_dataloader(data_dict: dict, batch_size: int = 32,
                   shuffle: bool = True,
                   modalities: tuple = ('text', 'audio', 'vision')):
    """
    创建 PyTorch DataLoader

    Args:
        data_dict: 附件2格式的 dict
        batch_size: 批大小
        shuffle: 是否打乱
        modalities: 要加载的模态 tuple

    Returns:
        PyTorch DataLoader
    """
    from torch.utils.data import Dataset, DataLoader
    import torch

    class MultimodalDataset(Dataset):
        def __init__(self, data_dict, modalities):
            self.N = len(data_dict['id'])
            self.modalities = modalities
            self.data = data_dict

            # 转换为 numpy array（附件2有些是 list）
            self._prepare()

        def _prepare(self):
            # text: (N, 50, 768)
            self.text = np.array(self.data['text']).astype(np.float32)
            # audio: (N, T, 74)
            self.audio = np.array(self.data['audio']).astype(np.float32)
            # vision: (N, T, 35)
            self.vision = np.array(self.data['vision']).astype(np.float32)
            # labels
            self.cls_labels = np.array(self.data['classification_labels']).astype(np.int64)
            self.reg_labels = np.array(self.data['regression_labels']).astype(np.float32)

        def __len__(self):
            return self.N

        def __getitem__(self, idx):
            item = {
                'id': self.data['id'][idx],
                'raw_text': self.data['raw_text'][idx],
                'cls_label': self.cls_labels[idx],
                'reg_label': self.reg_labels[idx],
            }
            if 'text' in self.modalities:
                item['text'] = torch.from_numpy(self.text[idx])
            if 'audio' in self.modalities:
                item['audio'] = torch.from_numpy(self.audio[idx])
            if 'vision' in self.modalities:
                item['vision'] = torch.from_numpy(self.vision[idx])
            return item

    dataset = MultimodalDataset(data_dict, modalities)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def load_label_xlsx(xlsx_path: str) -> pd.DataFrame:
    """加载标签 Excel 文件"""
    return pd.read_excel(xlsx_path)


def merge_q1_with_attachment2(q1_data: dict, att2_data: dict) -> dict:
    """
    将Q1生成的100条特征合并到附件2格式中
    Q1的100条全部放在 'all' 键下
    """
    return {
        'all': q1_data,
        'train': att2_data.get('train', {}),
        'valid': att2_data.get('valid', {}),
        'test': att2_data.get('test', {}),
    }


# ── 快捷函数 ───────────────────────────────────────────────────────────────────

def load_all(version: Literal['aligned', 'unaligned'] = 'aligned',
             q1_extra_path: Optional[str] = None,
             data_root: str = 'huaweibei-E/data/raw_attachment2'):
    """
    一行加载所有数据的快捷函数

    Args:
        version: 'aligned' → aligned_50.pkl, 'unaligned' → unaligned_50.pkl
        q1_extra_path: 可选，Q1生成的额外特征路径
        data_root: 附件2根目录

    Returns:
        dict with 'train', 'valid', 'test', 可选 'q1'
    """
    fname = f'{version}_50.pkl'
    att2 = load_attachment2(f'{data_root}/{fname}')

    if q1_extra_path:
        q1 = load_q1_features(q1_extra_path)
        att2['q1'] = q1

    return att2
