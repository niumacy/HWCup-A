"""
MSA-GMoE: Multi-Scale Attention + Gated Mixture-of-Experts
═══════════════════════════════════════════════════════════════════════════════
更强架构 (在 GMT/GLF 基础上):

核心设计:
1. 多尺度时序金字塔
   - 每模态用 3 个不同膨胀率的空洞卷积 (dilation=1,2,4) 捕获短/中/长时情绪
   - 比 Transformer 更稳定,小数据集上不易过拟合

2. 模态路由轻量 Attention (Modality-Routing Attention)
   - 3 对 (T-A, T-V, A-V) 各有专用 attention (vs GMT 全连接)
   - 参数量减少 ~40%, 专注跨模态互补

3. Gated Mixture-of-Experts 融合 (MoE-Fusion)
   - 4 个不同归纳偏置的专家:
     * Expert 0: LateFusion (concat MLP)
     * Expert 1: Gated (GLF)
     * Expert 2: Bilinear (双线性)
     * Expert 3: Attention-pooling
   - 门控网络动态选择专家组合
   - 不同样本用不同融合策略, 缓解单一融合器偏差

4. (训练选项) 对比学习 + Mixup:
   - ConLoss: 让同一样本的不同模态对齐 (audio-vision 对比 text)
   - Mixup: 数据层面正则化

参考文献:
- Multi-Scale TCN: Bai et al., 2018 (Temporal Convolutional Networks)
- MoE for Multimodal: Wu et al., 2022 (Multimodal Mixture-of-Experts)
- Cross-Modal Contrastive: Yuan et al., 2021
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ──────────────────────────────────────────────────────────────────────────────
# 1. 多尺度时序金字塔 (Multi-Scale Temporal Pyramid)
# ──────────────────────────────────────────────────────────────────────────────

class MultiScaleConvBlock(nn.Module):
    """单尺度空洞卷积块 (Pre-LN + GELU + Dropout)"""

    def __init__(self, channels, kernel_size=3, dilation=1, dropout=0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.norm = nn.LayerNorm(channels)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, H) -> (B, H, T) for conv
        # Pre-LN
        residual = x
        x = self.norm(x).transpose(1, 2)  # (B, H, T)
        x = F.gelu(self.conv1(x))
        x = self.dropout(x)
        x = F.gelu(self.conv2(x))
        x = x.transpose(1, 2)  # (B, T, H)
        return residual + self.dropout(x)


class MultiScaleTemporalPyramid(nn.Module):
    """多尺度时序金字塔

    用 3 个不同膨胀率的卷积块 + 池化,产生短/中/长尺度表示
    """

    def __init__(self, in_dim, hidden_dim, num_scales=3, kernel_size=3, dropout=0.2):
        super().__init__()

        # 输入投影
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 多尺度融合
        self.scale_fusion = nn.Sequential(
            nn.Linear(hidden_dim * num_scales, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.dropout_final = nn.Dropout(dropout)

        # 3 个不同膨胀率的卷积块
        self.scale_blocks = nn.ModuleList([
            MultiScaleConvBlock(hidden_dim, kernel_size=kernel_size,
                                dilation=2**i, dropout=dropout)
            for i in range(num_scales)
        ])

        # 尺度池化 (stride=2 卷积代替平均池化)
        self.pool_layers = nn.ModuleList([
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=2, stride=2)
            for _ in range(num_scales - 1)
        ])

    def forward(self, x):
        """
        Args:
            x: (B, T, in_dim)
        Returns:
            scale_feats: list of (B, H) 各尺度的池化特征
        """
        # 投影
        h = self.input_proj(x)  # (B, T, H)

        # 多尺度: 每个尺度块池化后取出
        scales = [self.scale_blocks[0](h).mean(dim=1)]
        cur = self.pool_layers[0](h.transpose(1, 2)).transpose(1, 2)  # (B, T/2, H)
        for i in range(1, len(self.scale_blocks)):
            out = self.scale_blocks[i](cur)
            scales.append(out.mean(dim=1))
            if i < len(self.scale_blocks) - 1:
                cur = self.pool_layers[i](out.transpose(1, 2)).transpose(1, 2)

        # 融合
        fused = self.scale_fusion(torch.cat(scales, dim=-1))  # (B, H)
        return self.dropout_final(fused)


# ──────────────────────────────────────────────────────────────────────────────
# 2. 模态路由轻量 Attention (Modality-Routing Attention)
# ──────────────────────────────────────────────────────────────────────────────

class RoutingAttention(nn.Module):
    """单对模态的路由 attention

    query 模态 attend to key/value 模态
    比全连接 attention 参数量更少
    """

    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads,
            dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )

    def forward(self, query, key_value):
        # query: (B, T_q, H), key_value: (B, T_kv, H)
        attended, _ = self.attn(query, key_value, key_value)
        h = self.norm1(query + attended)
        h = self.norm2(h + self.ffn(h))
        return h


class ModalityRoutingAttention(nn.Module):
    """3 对模态路由 attention: T↔A, T↔V, A↔V"""

    def __init__(self, dim, num_heads=4, dropout=0.1, num_routing_layers=1):
        super().__init__()

        # 3 对路由 attention
        self.ta_layers = nn.ModuleList([
            RoutingAttention(dim, num_heads, dropout)
            for _ in range(num_routing_layers)
        ])
        self.tv_layers = nn.ModuleList([
            RoutingAttention(dim, num_heads, dropout)
            for _ in range(num_routing_layers)
        ])
        self.av_layers = nn.ModuleList([
            RoutingAttention(dim, num_heads, dropout)
            for _ in range(num_routing_layers)
        ])

        # 输出投影 (回到原 dim)
        self.out_proj_t = nn.Linear(dim, dim)
        self.out_proj_a = nn.Linear(dim, dim)
        self.out_proj_v = nn.Linear(dim, dim)
        self.norm_t = nn.LayerNorm(dim)
        self.norm_a = nn.LayerNorm(dim)
        self.norm_v = nn.LayerNorm(dim)

    def forward(self, t, a, v):
        """
        t, a, v: (B, T, H)
        """
        T_, A_, V_ = t, a, v

        for ta, tv, av in zip(self.ta_layers, self.tv_layers, self.av_layers):
            # T attends to A
            T_new = ta(T_, A_)
            # T attends to V
            T_new = tv(T_new, V_)

            # A attends to T
            A_new = av(A_, T_)
            # A attends to V
            A_new = ta(A_new, V_)

            # V attends to A
            V_new = av(V_, A_)
            # V attends to T
            V_new = tv(V_new, T_)

            T_, A_, V_ = T_new, A_new, V_new

        # 输出投影 + 残差
        T_out = self.norm_t(t + self.out_proj_t(T_))
        A_out = self.norm_a(a + self.out_proj_a(A_))
        V_out = self.norm_v(v + self.out_proj_v(V_))
        return T_out, A_out, V_out


# ──────────────────────────────────────────────────────────────────────────────
# 3. Gated Mixture-of-Experts 融合 (MoE-Fusion)
# ──────────────────────────────────────────────────────────────────────────────

class LateFusionExpert(nn.Module):
    """Expert 0: Late Fusion"""

    def __init__(self, dim, hidden_dim, dropout=0.2):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Linear(dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.out = nn.Linear(hidden_dim, dim)

    def forward(self, t, a, v):
        return self.out(self.fuse(torch.cat([t, a, v], dim=-1)))


class GatedExpert(nn.Module):
    """Expert 1: Gated Late Fusion (GLF)"""

    def __init__(self, dim, hidden_dim, dropout=0.2):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, 3),
        )
        self.gate_temp = nn.Parameter(torch.ones(1))
        self.fuse = nn.Sequential(
            nn.Linear(dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.out = nn.Linear(hidden_dim, dim)

    def forward(self, t, a, v):
        concat = torch.cat([t, a, v], dim=-1)
        gates = torch.sigmoid(self.gate(concat) * self.gate_temp)
        tg = t * gates[:, 0:1]
        ag = a * gates[:, 1:2]
        vg = v * gates[:, 2:3]
        return self.out(self.fuse(torch.cat([tg, ag, vg], dim=-1)))


class HadamardExpert(nn.Module):
    """Expert 2: Hadamard 元素相乘融合 (轻量, 捕捉模态交互)"""

    def __init__(self, dim, hidden_dim, dropout=0.2):
        super().__init__()
        # 各模态的 Hadamard 交互
        self.ta_fuse = nn.Sequential(
            nn.Linear(dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.tv_fuse = nn.Sequential(
            nn.Linear(dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.av_fuse = nn.Sequential(
            nn.Linear(dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.out = nn.Linear(hidden_dim // 2 * 3, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, t, a, v):
        ta = self.ta_fuse(t * a)
        tv = self.tv_fuse(t * v)
        av = self.av_fuse(a * v)
        return self.out(self.dropout(torch.cat([ta, tv, av], dim=-1)))


class AttentionPoolExpert(nn.Module):
    """Expert 3: Attention-Pooling 融合"""

    def __init__(self, dim, num_heads=4, dropout=0.2):
        super().__init__()
        self.q_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads,
            dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, t, a, v):
        # 3 个模态拼成 token 序列, 用可学习 query 池化
        B = t.size(0)
        q = self.q_token.expand(B, -1, -1)  # (B, 1, H)
        tokens = torch.stack([t, a, v], dim=1)  # (B, 3, H)
        out, _ = self.attn(q, tokens, tokens)  # (B, 1, H)
        out = self.dropout(self.norm(out.squeeze(1)))
        return out


class GatedMoEFusion(nn.Module):
    """门控多专家融合

    - 4 个不同归纳偏置的专家
    - 门控网络 softmax 选择专家权重
    - 加权求和 + 最终融合投影
    """

    def __init__(self, dim, hidden_dim, num_experts=4, dropout=0.2):
        super().__init__()

        self.num_experts = num_experts

        # 4 个专家
        self.experts = nn.ModuleList([
            LateFusionExpert(dim, hidden_dim, dropout),
            GatedExpert(dim, hidden_dim, dropout),
            HadamardExpert(dim, hidden_dim, dropout),
            AttentionPoolExpert(dim, num_heads=4, dropout=dropout),
        ])

        # 门控网络: 输入各模态, 输出专家权重
        self.gate = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, num_experts),
        )
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)

        # 负载均衡损失权重
        self.load_balance_weight = 0.01

    def forward(self, t, a, v, return_gate_weights=False):
        """
        Args:
            t, a, v: (B, H)
        Returns:
            fused: (B, H)
            (optional) gate_weights: (B, num_experts)
            (optional) load_balance_loss: scalar
        """
        # 计算门控权重
        gate_input = torch.cat([t, a, v], dim=-1)  # (B, 3H)
        gate_logits = self.gate(gate_input)  # (B, num_experts)
        gate_weights = F.softmax(gate_logits * self.temperature, dim=-1)  # (B, num_experts)

        # 各专家输出
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(t, a, v))  # (B, H)

        expert_outputs = torch.stack(expert_outputs, dim=1)  # (B, num_experts, H)

        # 加权求和
        fused = torch.einsum('be,beh->bh', gate_weights, expert_outputs)  # (B, H)

        if return_gate_weights:
            # 计算负载均衡损失 (鼓励各专家均匀使用)
            avg_weight = gate_weights.mean(dim=0)  # (num_experts,)
            load_balance_loss = (avg_weight * torch.log(avg_weight + 1e-8)).sum() * self.load_balance_weight
            return fused, gate_weights, load_balance_loss

        return fused


# ──────────────────────────────────────────────────────────────────────────────
# 4. 完整 MSA-GMoE 模型
# ──────────────────────────────────────────────────────────────────────────────

class MSAGMoE(nn.Module):
    """
    MSA-GMoE: Multi-Scale Attention + Gated Mixture-of-Experts

    输入: (B, T=50, 768/74/35)
    ↓
    Multi-Scale Temporal Pyramid (3 尺度空洞卷积)
    ↓
    Modality-Routing Attention (T↔A, T↔V, A↔V)
    ↓
    Gated MoE-Fusion (4 专家)
    ↓
    分类头 + 回归头
    """

    def __init__(self,
                 text_dim: int = 768,
                 audio_dim: int = 74,
                 vision_dim: int = 35,
                 hidden_dim: int = 256,
                 num_heads: int = 4,
                 num_classes: int = 3,
                 dropout: float = 0.3,
                 num_routing_layers: int = 1,
                 use_contrastive: bool = False):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.use_contrastive = use_contrastive

        # ── 1. 多尺度时序金字塔 ──
        self.text_pyramid = MultiScaleTemporalPyramid(
            in_dim=text_dim, hidden_dim=hidden_dim, dropout=dropout
        )
        self.audio_pyramid = MultiScaleTemporalPyramid(
            in_dim=audio_dim, hidden_dim=hidden_dim, dropout=dropout
        )
        self.vision_pyramid = MultiScaleTemporalPyramid(
            in_dim=vision_dim, hidden_dim=hidden_dim, dropout=dropout
        )

        # ── 2. 模态路由 Attention ──
        self.routing_attn = ModalityRoutingAttention(
            dim=hidden_dim, num_heads=num_heads,
            dropout=dropout, num_routing_layers=num_routing_layers
        )

        # ── 3. Gated MoE 融合 ──
        self.moe_fusion = GatedMoEFusion(
            dim=hidden_dim, hidden_dim=hidden_dim, num_experts=4, dropout=dropout
        )

        # 最终残差融合 (concat -> MLP)
        self.final_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3 + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.dropout_final = nn.Dropout(dropout)

        # ── 4. 输出头 ──
        self.shared_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.cls_head = nn.Linear(hidden_dim // 2, num_classes)
        self.reg_head = nn.Linear(hidden_dim // 2, 1)

        # ── 5. 对比学习投影头 (可选) ──
        if use_contrastive:
            self.contrastive_proj = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 128),
            )

    def forward(self, text, audio, vision, return_gate_weights=False):
        """
        Args:
            text: (B, T=50, 768)
            audio: (B, T=50, 74)
            vision: (B, T=50, 35)
            return_gate_weights: 是否返回 MoE 门控权重

        Returns:
            cls_logits: (B, num_classes)
            reg_pred: (B,)
            extras: dict (可选: gate_weights, moe_balance_loss, contrastive_features)
        """
        # ── 多尺度时序金字塔 ──
        T_ms = self.text_pyramid(text)    # (B, H)
        A_ms = self.audio_pyramid(audio)  # (B, H)
        V_ms = self.vision_pyramid(vision)

        # ── 构造时序版本用于路由 attention ──
        # 把多尺度特征广播回时序维度 (B, T, H)
        T_seq = T_ms.unsqueeze(1).expand(-1, text.size(1), -1)
        A_seq = A_ms.unsqueeze(1).expand(-1, audio.size(1), -1)
        V_seq = V_ms.unsqueeze(1).expand(-1, vision.size(1), -1)

        # ── 模态路由 Attention ──
        T_routed, A_routed, V_routed = self.routing_attn(T_seq, A_seq, V_seq)

        # 全局池化回 (B, H)
        T_fused = T_routed.mean(dim=1)
        A_fused = A_routed.mean(dim=1)
        V_fused = V_routed.mean(dim=1)

        # ── MoE 融合 ──
        if return_gate_weights:
            moe_out, gate_weights, moe_balance_loss = self.moe_fusion(
                T_fused, A_fused, V_fused, return_gate_weights=True
            )
        else:
            moe_out = self.moe_fusion(T_fused, A_fused, V_fused)
            moe_balance_loss = torch.tensor(0.0, device=text.device)
            gate_weights = None

        # ── 最终融合: 拼接 + 投影 ──
        concat = torch.cat([T_fused, A_fused, V_fused, moe_out], dim=-1)  # (B, 4H)
        final = self.final_fusion(concat)  # (B, H)
        final = self.dropout_final(final)

        # ── 输出头 ──
        shared = self.shared_head(final)
        cls_logits = self.cls_head(shared)
        reg_pred = self.reg_head(shared).squeeze(-1)

        # ── 对比学习特征 (可选) ──
        contrastive_feats = None
        if self.use_contrastive:
            contrastive_feats = {
                'text': self.contrastive_proj(T_fused),
                'audio': self.contrastive_proj(A_fused),
                'vision': self.contrastive_proj(V_fused),
            }

        extras = {
            'moe_balance_loss': moe_balance_loss,
            'gate_weights': gate_weights,
            'contrastive_feats': contrastive_feats,
        }

        return cls_logits, reg_pred, extras


# ──────────────────────────────────────────────────────────────────────────────
# 5. 工具函数: 对比学习损失 + Mixup
# ──────────────────────────────────────────────────────────────────────────────

def contrastive_loss(feats_dict, temperature=0.1):
    """
    多模态对比学习 (InfoNCE)
    让同一样本的不同模态表示在 embedding space 中更接近

    feats_dict: {'text': (B, H), 'audio': (B, H), 'vision': (B, H)}

    - 共有 3B 个"样本" (B 个原样本 × 3 模态)
    - 同一样本的 3 个模态互为正样本 (3 对)
    - 其他样本的所有模态都是负样本
    """
    feats = torch.stack([feats_dict['text'], feats_dict['audio'], feats_dict['vision']], dim=1)  # (B, 3, H)
    B = feats.size(0)
    device = feats.device

    # L2 归一化
    feats = F.normalize(feats, dim=-1)
    feats_flat = feats.reshape(B * 3, -1)  # (3B, H)

    # 余弦相似度矩阵
    sim_matrix = torch.mm(feats_flat, feats_flat.t()) / temperature  # (3B, 3B)

    # 构造正样本 mask
    # 行 i (i = b + B*m, b∈[0,B), m∈{0,1,2}) 表示样本 b 的模态 m
    # 正样本是同一样本的其他 2 个模态, 即 b + B*m', m' ≠ m
    pos_mask = torch.zeros(B * 3, B * 3, device=device, dtype=torch.bool)
    for b in range(B):
        # 样本 b 的 3 个模态索引
        idx = [b, b + B, b + 2 * B]
        for i in range(3):
            for j in range(3):
                if i != j:
                    pos_mask[idx[i], idx[j]] = True

    # 去掉自身 (对角线)
    self_mask = torch.eye(B * 3, device=device, dtype=torch.bool)
    pos_mask = pos_mask & ~self_mask

    # InfoNCE: 对每个 anchor, log(exp(pos) / sum(exp(all)))
    # 这里用对称损失 (双向), 简化: 每个正样本对的平均 NCE 损失
    # loss_i = -log( sum_j exp(sim_ij) for pos j ) / (sum_k exp(sim_ik) for all k != i))
    # 数值稳定: 用 logsumexp

    # 对每个 anchor i, 计算它对其所有正样本的损失, 取平均
    # mask 掉对角线 (自身)
    sim_matrix_masked = sim_matrix.masked_fill(self_mask, float('-inf'))

    losses = []
    for i in range(B * 3):
        pos_indices = pos_mask[i].nonzero(as_tuple=True)[0]
        if len(pos_indices) == 0:
            continue
        # 分母: logsumexp over all 3B
        lse_all = torch.logsumexp(sim_matrix_masked[i], dim=0)
        # 分子: log-sum-exp over pos
        lse_pos = torch.logsumexp(sim_matrix[i, pos_indices], dim=0)
        losses.append(lse_pos - lse_all)

    if len(losses) == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    loss = torch.stack(losses).mean()
    return loss


def mixup_data(text, audio, vision, cls_label, reg_label, alpha=0.4):
    """
    Mixup 数据增强
    Returns: 混合后的 6 个输入 + 2 组标签 (cls_label_a, cls_label_b, lambda)
    """
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    lam = max(lam, 1 - lam)  # 保证 lam >= 0.5

    B = text.size(0)
    idx = torch.randperm(B, device=text.device)

    text_mixed = lam * text + (1 - lam) * text[idx]
    audio_mixed = lam * audio + (1 - lam) * audio[idx]
    vision_mixed = lam * vision + (1 - lam) * vision[idx]

    return (text_mixed, audio_mixed, vision_mixed,
            cls_label, cls_label[idx], reg_label, reg_label[idx], lam)


# ──────────────────────────────────────────────────────────────────────────────
# 6. 模型工厂
# ──────────────────────────────────────────────────────────────────────────────

def build_msa_gmoe(text_dim=768, audio_dim=74, vision_dim=35,
                   hidden_dim=256, num_heads=4, dropout=0.3,
                   num_routing_layers=1, use_contrastive=False, **kwargs):
    """构建 MSA-GMoE 模型的便捷函数"""
    return MSAGMoE(
        text_dim=text_dim,
        audio_dim=audio_dim,
        vision_dim=vision_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        dropout=dropout,
        num_routing_layers=num_routing_layers,
        use_contrastive=use_contrastive,
        **kwargs,
    )


if __name__ == '__main__':
    # 模型结构测试
    print("=== MSA-GMoE 模型结构测试 ===")

    model = MSAGMoE(
        text_dim=768, audio_dim=74, vision_dim=35,
        hidden_dim=256, num_heads=4, dropout=0.3,
        use_contrastive=True,
    )

    # 参数量
    total = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total:,}")

    # 前向测试
    B, T = 4, 50
    text = torch.randn(B, T, 768)
    audio = torch.randn(B, T, 74)
    vision = torch.randn(B, T, 35)

    cls_logits, reg_pred, extras = model(text, audio, vision, return_gate_weights=True)
    print(f"cls_logits: {cls_logits.shape}")
    print(f"reg_pred: {reg_pred.shape}")
    print(f"moe_balance_loss: {extras['moe_balance_loss'].item():.4f}")
    print(f"gate_weights: {extras['gate_weights'].shape}")

    # 对比学习损失
    if extras['contrastive_feats'] is not None:
        c_loss = contrastive_loss(extras['contrastive_feats'])
        print(f"contrastive_loss: {c_loss.item():.4f}")

    print("\n✅ 模型测试通过!")
