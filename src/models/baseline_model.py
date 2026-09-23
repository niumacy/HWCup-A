"""
增强版多模态情感预测模型
支持:
1. Late Fusion (基线)
2. LSTM + Cross-Attention (增强)
3. Self-Attention Fusion (Transformer风格)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal
import math


# ──────────────────────────────────────────────────────────────────────────────
# 1. 基线 Late Fusion (原版升级)
# ──────────────────────────────────────────────────────────────────────────────

class LateFusionBaseline(nn.Module):
    """Late Fusion 多模态基线模型 - 增强版"""

    def __init__(self,
                 text_dim: int = 768,
                 audio_dim: int = 74,
                 vision_dim: int = 35,
                 hidden_dim: int = 512,
                 num_classes: int = 3,
                 dropout: float = 0.4):
        super().__init__()

        # 各模态编码器 - 升级版 (LayerNorm + 2层 MLP)
        def make_encoder(input_dim):
            return nn.Sequential(
                nn.Linear(input_dim, hidden_dim * 2),
                nn.LayerNorm(hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        self.text_enc = make_encoder(text_dim)
        self.audio_enc = make_encoder(audio_dim)
        self.vision_enc = make_encoder(vision_dim)

        # 融合
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 输出头 (共享底层)
        self.shared_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.cls_head = nn.Linear(hidden_dim // 2, num_classes)
        self.reg_head = nn.Linear(hidden_dim // 2, 1)

    def forward(self, text, audio, vision):
        if text.dim() == 3:
            text = text.mean(dim=1)
            audio = audio.mean(dim=1)
            vision = vision.mean(dim=1)

        t = self.text_enc(text)
        a = self.audio_enc(audio)
        v = self.vision_enc(vision)
        fused = self.fusion(torch.cat([t, a, v], dim=-1))

        shared = self.shared_head(fused)
        cls_logits = self.cls_head(shared)
        reg_pred = self.reg_head(shared).squeeze(-1)
        return cls_logits, reg_pred


# ──────────────────────────────────────────────────────────────────────────────
# 2. LSTM + Cross-Attention Fusion (高级)
# ──────────────────────────────────────────────────────────────────────────────

class LSTMCrossAttention(nn.Module):
    """
    LSTM 处理时序 + Cross-Attention 融合
    - 每模态用 LSTM 处理
    - 各模态间通过 cross-attention 交换信息
    - 最终拼接 + 融合
    """

    def __init__(self,
                 text_dim: int = 768,
                 audio_dim: int = 74,
                 vision_dim: int = 35,
                 hidden_dim: int = 256,
                 lstm_layers: int = 1,
                 num_heads: int = 4,
                 num_classes: int = 3,
                 dropout: float = 0.3):
        super().__init__()

        # 各模态用 LSTM 处理时序
        self.text_lstm = nn.LSTM(text_dim, hidden_dim, num_layers=lstm_layers,
                                  batch_first=True, dropout=dropout if lstm_layers > 1 else 0,
                                  bidirectional=True)
        self.audio_lstm = nn.LSTM(audio_dim, hidden_dim, num_layers=lstm_layers,
                                   batch_first=True, dropout=dropout if lstm_layers > 1 else 0,
                                   bidirectional=True)
        self.vision_lstm = nn.LSTM(vision_dim, hidden_dim, num_layers=lstm_layers,
                                    batch_first=True, dropout=dropout if lstm_layers > 1 else 0,
                                    bidirectional=True)

        # 投影到统一维度 (BiLSTM 输出是 2*hidden_dim)
        self.text_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.audio_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.vision_proj = nn.Linear(hidden_dim * 2, hidden_dim)

        # Cross-Attention 模块 (text attends to audio+vision)
        self.text_cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.audio_cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.vision_cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.attn_norm1 = nn.LayerNorm(hidden_dim)
        self.attn_norm2 = nn.LayerNorm(hidden_dim)
        self.attn_norm3 = nn.LayerNorm(hidden_dim)

        # Self-Attention 融合层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.fusion_attn = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.fusion_dropout = nn.Dropout(dropout)

        # 输出头
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self.reg_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, text, audio, vision):
        """
        text: (B, T, 768)
        audio: (B, T, 74)
        vision: (B, T, 35)
        """
        # 1. LSTM 处理时序
        t_out, _ = self.text_lstm(text)   # (B, T, 2*H)
        a_out, _ = self.audio_lstm(audio)
        v_out, _ = self.vision_lstm(vision)

        # 2. 投影
        T = self.text_proj(t_out)  # (B, T, H)
        A = self.audio_proj(a_out)
        V = self.vision_proj(v_out)

        # 3. Cross-Attention
        # text attends to audio + vision
        T_cross, _ = self.text_cross_attn(T, torch.cat([A, V], dim=1), torch.cat([A, V], dim=1))
        T = self.attn_norm1(T + T_cross)

        # audio attends to text + vision
        A_cross, _ = self.audio_cross_attn(A, torch.cat([T, V], dim=1), torch.cat([T, V], dim=1))
        A = self.attn_norm2(A + A_cross)

        # vision attends to text + audio
        V_cross, _ = self.vision_cross_attn(V, torch.cat([T, A], dim=1), torch.cat([T, A], dim=1))
        V = self.attn_norm3(V + V_cross)

        # 4. Self-Attention 融合
        combined = torch.cat([T, A, V], dim=1)  # (B, 3T, H)
        fused = self.fusion_attn(combined)  # (B, 3T, H)
        fused = self.fusion_dropout(fused)

        # 5. 池化
        T_fused = fused[:, :T.size(1)].mean(dim=1)  # (B, H)
        A_fused = fused[:, T.size(1):T.size(1)+A.size(1)].mean(dim=1)
        V_fused = fused[:, T.size(1)+A.size(1):].mean(dim=1)

        all_fused = torch.cat([T_fused, A_fused, V_fused], dim=-1)

        cls_logits = self.cls_head(all_fused)
        reg_pred = self.reg_head(all_fused).squeeze(-1)

        return cls_logits, reg_pred


# ──────────────────────────────────────────────────────────────────────────────
# 3. Temporal Convolutional Network (TCN) Fusion
# ──────────────────────────────────────────────────────────────────────────────

class TCNBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel_size,
                                padding=padding, dilation=dilation)
        self.norm1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size,
                                padding=padding, dilation=dilation)
        self.norm2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.conv1(x)[..., :residual.size(-1)]
        x = self.norm1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x)[..., :residual.size(-1)]
        x = self.norm2(x)
        x = self.relu(x + residual)
        x = self.dropout(x)
        return x


class TCNFusion(nn.Module):
    """TCN 时序 + 跨模态 attention"""

    def __init__(self,
                 text_dim=768, audio_dim=74, vision_dim=35,
                 hidden_dim=128, num_classes=3, dropout=0.3):
        super().__init__()

        # 投影到统一维度
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.vision_proj = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # TCN 处理各模态时序
        self.text_tcn = nn.Sequential(
            TCNBlock(hidden_dim, dilation=1, dropout=dropout),
            TCNBlock(hidden_dim, dilation=2, dropout=dropout),
        )
        self.audio_tcn = nn.Sequential(
            TCNBlock(hidden_dim, dilation=1, dropout=dropout),
            TCNBlock(hidden_dim, dilation=2, dropout=dropout),
        )
        self.vision_tcn = nn.Sequential(
            TCNBlock(hidden_dim, dilation=1, dropout=dropout),
            TCNBlock(hidden_dim, dilation=2, dropout=dropout),
        )

        # 融合
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 输出头
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self.reg_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, text, audio, vision):
        # text: (B, T, 768), audio: (B, T, 74), vision: (B, T, 35)
        T = self.text_proj(text)   # (B, T, H)
        A = self.audio_proj(audio)
        V = self.vision_proj(vision)

        # Conv1d 需要 (B, C, T)
        T = T.transpose(1, 2)
        A = A.transpose(1, 2)
        V = V.transpose(1, 2)

        T = self.text_tcn(T).transpose(1, 2)
        A = self.audio_tcn(A).transpose(1, 2)
        V = self.vision_tcn(V).transpose(1, 2)

        # 全局池化
        T_feat = T.mean(dim=1)
        A_feat = A.mean(dim=1)
        V_feat = V.mean(dim=1)

        # 融合
        fused = self.fusion(torch.cat([T_feat, A_feat, V_feat], dim=-1))
        cls_logits = self.cls_head(fused)
        reg_pred = self.reg_head(fused).squeeze(-1)
        return cls_logits, reg_pred


# ──────────────────────────────────────────────────────────────────────────────
# 模型工厂
# ──────────────────────────────────────────────────────────────────────────────

def build_model(model_type: Literal['baseline', 'latefusion', 'attention', 'lstmattn', 'tcn'] = 'latefusion',
                **kwargs) -> nn.Module:
    """快速构建模型"""
    model_type = model_type.lower()
    if model_type in ('baseline', 'latefusion'):
        return LateFusionBaseline(**kwargs)
    elif model_type in ('attention', 'lstmattn'):
        return LSTMCrossAttention(**kwargs)
    elif model_type == 'tcn':
        return TCNFusion(**kwargs)
    elif model_type in ('gmt', 'gated'):
        return GatedMultimodalTransformer(**kwargs)
    elif model_type in ('glf', 'gatedlf', 'gated_late'):
        return GatedLateFusion(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. Gated Late Fusion (GLF) - 简化版门控融合
# ══════════════════════════════════════════════════════════════════════════════

class GatedLateFusion(nn.Module):
    """
    简化版: Late Fusion + 门控机制
    只用 gating 没有 cross-attention, 适合小数据集
    """

    def __init__(self,
                 text_dim: int = 768,
                 audio_dim: int = 74,
                 vision_dim: int = 35,
                 hidden_dim: int = 256,
                 num_classes: int = 3,
                 dropout: float = 0.4):
        super().__init__()

        # 模态编码器
        self.text_enc = nn.Sequential(
            nn.Linear(text_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.audio_enc = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.vision_enc = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # 门控网络
        # 输入: 三个模态拼接 -> 输出3个门控值 (0~1)
        self.gate_network = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )

        # 门值缩放因子 (可学习), 让门控值平均接近1
        self.gate_temperature = nn.Parameter(torch.ones(1))

        # 共享融合层
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 输出头 (共享底层)
        self.shared_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.cls_head = nn.Linear(hidden_dim // 2, num_classes)
        self.reg_head = nn.Linear(hidden_dim // 2, 1)

    def forward(self, text, audio, vision):
        if text.dim() == 3:
            text = text.mean(dim=1)
            audio = audio.mean(dim=1)
            vision = vision.mean(dim=1)

        t = self.text_enc(text)
        a = self.audio_enc(audio)
        v = self.vision_enc(vision)

        # 拼接后计算门控值
        concat = torch.cat([t, a, v], dim=-1)  # (B, 3H)
        gates = torch.sigmoid(self.gate_network(concat) * self.gate_temperature)  # (B, 3)

        g_t = gates[:, 0:1]
        g_a = gates[:, 1:2]
        g_v = gates[:, 2:3]

        # 门控加权
        t_gated = t * g_t
        a_gated = a * g_a
        v_gated = v * g_v

        # 拼接 + 融合
        fused = self.fusion(torch.cat([t_gated, a_gated, v_gated], dim=-1))

        shared = self.shared_head(fused)
        cls_logits = self.cls_head(shared)
        reg_pred = self.reg_head(shared).squeeze(-1)
        return cls_logits, reg_pred


# 添加 GLF 模型到 build_model
def _build_glf(**kwargs):
    return GatedLateFusion(**kwargs)


# ── GMT 模型类保留原定义 ──

class GatedMultimodalUnit(nn.Module):
    """
    门控多模态单元
    学习每个模态的重要性权重 (0~1)
    并对每个模态特征做残差加权融合
    """

    def __init__(self, text_dim: int, audio_dim: int, vision_dim: int, hidden_dim: int):
        super().__init__()
        # 各模态的置信度门
        # 输入是拼接后的全局信息,输出各模态的门控值
        self.gate_fc = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),  # 3个模态的门
        )
        # 激活函数 (Sigmoid 让值在 0~1)
        self.sigmoid = nn.Sigmoid()

        # 用于校准门控值
        self.gate_bias = nn.Parameter(torch.zeros(3))  # [text_gate, audio_gate, vision_gate]

    def forward(self, t: torch.Tensor, a: torch.Tensor, v: torch.Tensor) -> tuple:
        """
        Args:
            t: (B, H) text 特征
            a: (B, H) audio 特征
            v: (B, H) vision 特征
        Returns:
            (g_t, g_a, g_v): 各模态的门控值 (B, 1)
        """
        concat = torch.cat([t, a, v], dim=-1)  # (B, 3H)
        raw_gates = self.gate_fc(concat) + self.gate_bias  # (B, 3)
        gates = self.sigmoid(raw_gates)  # (B, 3)
        return gates[:, 0:1], gates[:, 1:2], gates[:, 2:3]  # 各 (B, 1)


class CrossModalAttention(nn.Module):
    """
    跨模态注意力
    以 query 模态为基准, attend 到 key/value 模态
    """

    def __init__(self, query_dim: int, kv_dim: int, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.key_proj = nn.Linear(kv_dim, hidden_dim)
        self.value_proj = nn.Linear(kv_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query: (B, T_q, D_q) 或 (B, D_q) [单个向量]
            key_value: (B, T_kv, D_kv) 或 (B, D_kv)
        Returns:
            attended: (B, T_q, hidden_dim) 或 (B, hidden_dim)
        """
        is_1d = query.dim() == 2
        if is_1d:
            query = query.unsqueeze(1)   # (B, 1, D)
        if key_value.dim() == 2:
            key_value = key_value.unsqueeze(1)  # (B, 1, D)

        Q = self.query_proj(query)
        K = self.key_proj(key_value)
        V = self.value_proj(key_value)

        out, _ = self.attn(Q, K, V)
        out = self.norm(query + self.dropout(out))
        out = out.squeeze(1) if is_1d else out
        return out


class GatedMultimodalTransformer(nn.Module):
    """
    Gated Cross-Modal Transformer (GMT)

    核心设计:
    1. 各模态独立编码 (带 LayerNorm)
    2. 跨模态注意力: Text ← Audio, Text ← Vision (文本为核心)
    3. 门控融合: 学习文本/语音/视觉的动态权重
    4. 自注意力层: 融合后的表示做 self-attention
    5. 多任务输出: 分类 + 回归

    参考文献:
    - Multimodal Transformer (Tsai et al., 2019)
    - Modality Gate (MISA, Hazarika et al., 2020)
    - Gated Multimodal Fusion (Arevalo et al., 2020)
    """

    def __init__(self,
                 text_dim: int = 768,
                 audio_dim: int = 74,
                 vision_dim: int = 35,
                 hidden_dim: int = 256,
                 num_heads: int = 4,
                 num_layers: int = 2,
                 num_classes: int = 3,
                 dropout: float = 0.35,
                 use_gated_fusion: bool = True,
                 use_temporal: bool = True):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.use_gated_fusion = use_gated_fusion
        self.use_temporal = use_temporal

        # ── 1. 模态编码器 ──────────────────────────────────────────────
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.vision_proj = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # ── 2. 时序处理 (可选: 用 TransformerEncoder) ──────────────────
        if use_temporal:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True,  # Pre-LN 更稳定
            )
            self.text_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.audio_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.vision_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        else:
            # 时序 → 均值池化
            pass

        # ── 3. 跨模态注意力 ──────────────────────────────────────────
        # 以文本为 query, 语音和视觉为 key/value
        self.text_from_audio = CrossModalAttention(
            hidden_dim, hidden_dim, hidden_dim, num_heads, dropout
        )
        self.text_from_vision = CrossModalAttention(
            hidden_dim, hidden_dim, hidden_dim, num_heads, dropout
        )

        # 语音和视觉也互相 attend
        self.audio_from_text = CrossModalAttention(
            hidden_dim, hidden_dim, hidden_dim, num_heads, dropout
        )
        self.audio_from_vision = CrossModalAttention(
            hidden_dim, hidden_dim, hidden_dim, num_heads, dropout
        )
        self.vision_from_text = CrossModalAttention(
            hidden_dim, hidden_dim, hidden_dim, num_heads, dropout
        )
        self.vision_from_audio = CrossModalAttention(
            hidden_dim, hidden_dim, hidden_dim, num_heads, dropout
        )

        # ── 4. 门控融合 ──────────────────────────────────────────────
        if use_gated_fusion:
            self.gated_unit = GatedMultimodalUnit(
                text_dim, audio_dim, vision_dim, hidden_dim
            )

        # ── 5. 融合后的自注意力 ──────────────────────────────────────
        fusion_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.fusion_attn = nn.TransformerEncoder(fusion_layer, num_layers=1)

        # ── 6. 融合投影 ──────────────────────────────────────────────
        self.fusion_proj = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ── 7. 输出头 ──────────────────────────────────────────────
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self.reg_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.dropout_final = nn.Dropout(dropout)

    def forward(self, text: torch.Tensor, audio: torch.Tensor, vision: torch.Tensor):
        """
        Args:
            text: (B, T, 768) 或 (B, 768)
            audio: (B, T, 74) 或 (B, 74)
            vision: (B, T, 35) 或 (B, 35)
        Returns:
            cls_logits: (B, 3)
            reg_pred: (B,)
        """
        is_sequence = text.dim() == 3

        if is_sequence:
            # ── 时序处理 ──
            # 投影
            T = self.text_proj(text)    # (B, T, H)
            A = self.audio_proj(audio)  # (B, T, H)
            V = self.vision_proj(vision)  # (B, T, H)

            # 逐帧的 cross-attention
            # Text attends to Audio & Vision
            T_from_A = self.text_from_audio(T, A)   # (B, T, H)
            T_from_V = self.text_from_vision(T, V)   # (B, T, H)
            T = (T + T_from_A + T_from_V) / 3

            # Audio attends to Text & Vision
            A_from_T = self.audio_from_text(A, T)
            A_from_V = self.audio_from_vision(A, V)
            A = (A + A_from_T + A_from_V) / 3

            # Vision attends to Text & Audio
            V_from_T = self.vision_from_text(V, T)
            V_from_A = self.vision_from_audio(V, A)
            V = (V + V_from_T + V_from_A) / 3

            # 时序自注意力
            T = self.text_encoder(T)
            A = self.audio_encoder(A)
            V = self.vision_encoder(V)

            # 全局池化
            T = T.mean(dim=1)  # (B, H)
            A = A.mean(dim=1)
            V = V.mean(dim=1)

        else:
            # ── 单向量处理 ──
            T = self.text_proj(text)
            A = self.audio_proj(audio)
            V = self.vision_proj(vision)

            # 直接做 cross-attention (1D → unsqueeze → squeeze)
            T_from_A = self.text_from_audio(T, A)
            T_from_V = self.text_from_vision(T, V)
            T = (T + T_from_A + T_from_V) / 3

            A_from_T = self.audio_from_text(A, T)
            A_from_V = self.audio_from_vision(A, V)
            A = (A + A_from_T + A_from_V) / 3

            V_from_T = self.vision_from_text(V, T)
            V_from_A = self.vision_from_audio(V, A)
            V = (V + V_from_T + V_from_A) / 3

        # ── 门控融合 ──────────────────────────────────────────────
        if self.use_gated_fusion:
            g_t, g_a, g_v = self.gated_unit(T, A, V)
            # 门控加权
            T = T * g_t
            A = A * g_a
            V = V * g_v

        # 拼接融合
        fused = torch.cat([T, A, V], dim=-1)  # (B, 3H)
        fused = self.fusion_proj(fused)  # (B, H)
        fused = self.dropout_final(fused)

        # ── 自注意力 ──────────────────────────────────────────────
        fused = fused.unsqueeze(1)  # (B, 1, H)
        fused = self.fusion_attn(fused)  # (B, 1, H)
        fused = fused.squeeze(1)  # (B, H)

        # ── 输出 ──────────────────────────────────────────────────
        cls_logits = self.cls_head(fused)
        reg_pred = self.reg_head(fused).squeeze(-1)

        return cls_logits, reg_pred
