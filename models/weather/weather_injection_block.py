"""这个文件负责封装天气编码与天气注入总模块，并与 CSP-TimeFilter 主干输出 H_main 在接口层对接。"""

import torch
import torch.nn as nn

from models.weather.causal_conv import CausalConv1dWeatherEncoder
from models.weather.causal_cross_attention import PhysicsConstrainedCausalCrossAttention


# 基于拼接的天气消融融合模块。
class ConcatFusionWeatherProjector(nn.Module):
    """将天气编码与主干隐状态直接拼接后做线性投影。"""

    # 初始化拼接融合模块。
    def __init__(self, model_dim, dropout=0.0):
        """构建拼接后的线性投影。"""
        super().__init__()
        self.proj = nn.Linear(model_dim * 2, model_dim)
        self.dropout = nn.Dropout(dropout)

    # 执行拼接融合。
    def forward(self, h_main, h_weather):
        """将 [B, T, D] 天气编码扩展到节点维并与主干状态融合。"""
        weather_tokens = h_weather.unsqueeze(2).expand(-1, -1, h_main.shape[2], -1)
        fused = torch.cat([h_main, weather_tokens], dim=-1)
        fused = self.proj(fused)
        fused = self.dropout(fused)
        return fused


# 物理约束下的因果天气注入总模块。
class PhysicsConstrainedCausalWeatherInjection(nn.Module):
    """将全局天气序列以因果、滞后、节点异质的方式注入主干隐状态。"""

    # 初始化天气注入总模块。
    def __init__(
        self,
        model_dim,
        weather_dim=3,
        num_heads=4,
        weather_hidden_dim=None,
        kernel_size=3,
        dilations=(1, 2),
        dropout=0.0,
        ablation_mode='causal_attn',
    ):
        """构建天气编码器、跨注意力和消融开关。"""
        super().__init__()
        valid_modes = {'causal_attn', 'vanilla_attn', 'concat_fusion'}
        if ablation_mode not in valid_modes:
            raise ValueError(f'ablation_mode 必须属于 {sorted(valid_modes)}，实际为 {ablation_mode!r}')

        self.ablation_mode = ablation_mode
        self.weather_encoder = CausalConv1dWeatherEncoder(
            input_dim=weather_dim,
            model_dim=model_dim,
            hidden_dim=weather_hidden_dim,
            kernel_size=kernel_size,
            dilations=dilations,
            dropout=dropout,
        )

        if self.ablation_mode == 'concat_fusion':
            self.weather_fusion = ConcatFusionWeatherProjector(model_dim=model_dim, dropout=dropout)
            self.weather_attention = None
        else:
            self.weather_attention = PhysicsConstrainedCausalCrossAttention(
                model_dim=model_dim,
                num_heads=num_heads,
                dropout=dropout,
                use_causal_mask=(self.ablation_mode == 'causal_attn'),
            )
            self.weather_fusion = None

        self.output_norm = nn.LayerNorm(model_dim)

    # 执行天气注入。
    def forward(self, h_main, weather_seq, optional_mask=None):
        """输入主干隐状态和天气序列，输出天气增强后的隐状态与可解释权重。"""
        if h_main.dim() != 4:
            raise ValueError(f'h_main 形状必须为 [B, T, N, D]，实际为 {tuple(h_main.shape)}')
        if weather_seq.dim() != 3:
            raise ValueError(f'weather_seq 形状必须为 [B, T, Cw]，实际为 {tuple(weather_seq.shape)}')
        if h_main.shape[0] != weather_seq.shape[0] or h_main.shape[1] != weather_seq.shape[1]:
            raise ValueError(
                f'h_main 与 weather_seq 的 B/T 维度必须一致，实际为 h_main={tuple(h_main.shape)}，weather_seq={tuple(weather_seq.shape)}'
            )

        h_weather = self.weather_encoder(weather_seq, optional_mask=optional_mask)
        if self.ablation_mode == 'concat_fusion':
            weather_delta = self.weather_fusion(h_main, h_weather)
            attn_weights = None
        else:
            weather_delta, attn_weights = self.weather_attention(h_main, h_weather, optional_mask=optional_mask)

        h_exo = self.output_norm(h_main + weather_delta)
        return h_exo, attn_weights
