"""这个文件负责实现物理约束下的因果天气跨注意力，并与 causal_conv.py、weather_injection_block.py 协同工作。"""

import math

import torch
import torch.nn as nn

from models.weather.causal_conv import normalize_temporal_mask


# 构造显式下三角因果掩码。
def build_causal_mask(seq_len, device):
    """生成 [1, 1, 1, T, T] 形式的下三角因果掩码。"""
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
    return causal_mask.view(1, 1, 1, seq_len, seq_len)


# 物理约束下的因果跨注意力模块。
class PhysicsConstrainedCausalCrossAttention(nn.Module):
    """使用主干隐状态查询全局天气表征，并显式施加因果约束。"""

    # 初始化跨注意力模块。
    def __init__(self, model_dim, num_heads, dropout=0.0, use_causal_mask=True):
        """构建多头投影层与输出映射。"""
        super().__init__()
        if model_dim % num_heads != 0:
            raise ValueError(f'model_dim={model_dim} 不能被 num_heads={num_heads} 整除。')

        self.model_dim = int(model_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.model_dim // self.num_heads
        self.use_causal_mask = bool(use_causal_mask)

        self.query_proj = nn.Linear(self.model_dim, self.model_dim)
        self.key_proj = nn.Linear(self.model_dim, self.model_dim)
        self.value_proj = nn.Linear(self.model_dim, self.model_dim)
        self.output_proj = nn.Linear(self.model_dim, self.model_dim)
        self.attn_dropout = nn.Dropout(dropout)

    # 根据因果性和可选时间掩码生成最终注意力掩码。
    def _build_attention_mask(self, batch_size, seq_len, optional_mask, device, dtype):
        """构建 Key 侧可见性与因果下三角共同作用的掩码。"""
        if self.use_causal_mask:
            attn_mask = build_causal_mask(seq_len, device)
        else:
            attn_mask = torch.ones(1, 1, 1, seq_len, seq_len, device=device, dtype=torch.bool)

        weather_mask = normalize_temporal_mask(optional_mask, batch_size, seq_len, device, dtype)
        key_mask = weather_mask.squeeze(-1).to(dtype=torch.bool).view(batch_size, 1, 1, 1, seq_len)
        attn_mask = attn_mask & key_mask
        return attn_mask

    # 执行跨注意力前向传播。
    def forward(self, h_main, weather_hidden, optional_mask=None):
        """将天气表征以多头跨注意力方式注入主干隐状态。"""
        if h_main.dim() != 4:
            raise ValueError(f'h_main 形状必须为 [B, T, N, D]，实际为 {tuple(h_main.shape)}')
        if weather_hidden.dim() != 3:
            raise ValueError(f'weather_hidden 形状必须为 [B, T, D]，实际为 {tuple(weather_hidden.shape)}')

        batch_size, seq_len, node_count, model_dim = h_main.shape
        if weather_hidden.shape[0] != batch_size or weather_hidden.shape[1] != seq_len or weather_hidden.shape[2] != model_dim:
            raise ValueError(
                f'weather_hidden 形状应与 h_main 在 B/T/D 上一致，实际为 {tuple(weather_hidden.shape)}，'
                f'而 h_main 为 {tuple(h_main.shape)}'
            )

        query_source = h_main.permute(0, 2, 1, 3)
        query = self.query_proj(query_source).view(batch_size, node_count, seq_len, self.num_heads, self.head_dim)
        query = query.permute(0, 1, 3, 2, 4)

        key = self.key_proj(weather_hidden).view(batch_size, seq_len, self.num_heads, self.head_dim)
        key = key.permute(0, 2, 1, 3)
        value = self.value_proj(weather_hidden).view(batch_size, seq_len, self.num_heads, self.head_dim)
        value = value.permute(0, 2, 1, 3)

        attn_logits = torch.einsum('bnhtd,bhsd->bnhts', query, key) / math.sqrt(self.head_dim)
        attn_mask = self._build_attention_mask(batch_size, seq_len, optional_mask, h_main.device, h_main.dtype)

        masked_logits = attn_logits.masked_fill(~attn_mask, torch.finfo(attn_logits.dtype).min)
        attn_weights = torch.softmax(masked_logits, dim=-1)
        attn_weights = attn_weights * attn_mask.to(dtype=attn_weights.dtype)
        attn_denominator = attn_weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        attn_weights = attn_weights / attn_denominator
        attn_weights = self.attn_dropout(attn_weights)

        context = torch.einsum('bnhts,bhsd->bnhtd', attn_weights, value)
        context = context.permute(0, 1, 3, 2, 4).contiguous().view(batch_size, node_count, seq_len, model_dim)
        context = context.permute(0, 2, 1, 3)
        context = self.output_proj(context)
        return context, attn_weights
