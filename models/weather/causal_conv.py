"""这个文件负责实现严格因果的天气卷积编码器，并与 weather_injection_block.py 配合为天气注入模块提供缓变外生表征。"""

import torch
import torch.nn as nn


# 将可选时间掩码统一整理成 [B, T, 1] 形式。
def normalize_temporal_mask(optional_mask, batch_size, seq_len, device, dtype):
    """将缺失值或 padding 掩码转换为统一形状。"""
    if optional_mask is None:
        return torch.ones(batch_size, seq_len, 1, device=device, dtype=dtype)

    mask = optional_mask.to(device=device, dtype=dtype)
    if mask.dim() == 2:
        mask = mask.unsqueeze(-1)
    elif mask.dim() == 3 and mask.shape[-1] != 1:
        mask = mask.amin(dim=-1, keepdim=True)
    elif mask.dim() != 3:
        raise ValueError(f'optional_mask 维度不合法，期望 2 维或 3 维，实际为 {tuple(mask.shape)}')

    if mask.shape[0] != batch_size or mask.shape[1] != seq_len:
        raise ValueError(f'optional_mask 形状与输入不匹配，期望 ({batch_size}, {seq_len}, *)，实际为 {tuple(mask.shape)}')
    return mask


# 单层严格因果卷积块，用于模拟天气累积与滞后作用。
class CausalConv1dBlock(nn.Module):
    """带有残差连接的严格因果一维卷积块。"""

    # 初始化因果卷积块。
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        """构建卷积、残差映射和非线性层。"""
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.dilation = int(dilation)
        self.left_padding = (self.kernel_size - 1) * self.dilation
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            padding=self.left_padding,
        )
        self.residual_proj = nn.Identity() if in_channels == out_channels else nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    # 前向计算因果卷积块输出。
    def forward(self, x):
        """执行卷积并裁剪未来信息。"""
        residual = self.residual_proj(x)
        out = self.conv(x)
        if self.left_padding > 0:
            out = out[..., :-self.left_padding]
        out = self.activation(out)
        out = self.dropout(out)
        return out + residual


# 严格因果卷积天气编码器，用于把原始天气序列编码成缓变表征。
class CausalConv1dWeatherEncoder(nn.Module):
    """将全局天气序列编码成满足因果性和滞后性的隐表示。"""

    # 初始化天气编码器。
    def __init__(
        self,
        input_dim,
        model_dim,
        hidden_dim=None,
        kernel_size=3,
        dilations=(1, 2),
        dropout=0.0,
    ):
        """构建至少两层的严格因果卷积编码器。"""
        super().__init__()
        if len(dilations) < 2:
            raise ValueError('CausalConv1dWeatherEncoder 至少需要两层因果卷积。')

        hidden_dim = model_dim if hidden_dim is None else int(hidden_dim)
        self.input_proj = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        self.blocks = nn.ModuleList()
        in_channels = hidden_dim
        for dilation in dilations:
            self.blocks.append(
                CausalConv1dBlock(
                    in_channels=in_channels,
                    out_channels=hidden_dim,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
            in_channels = hidden_dim

        self.output_proj = nn.Conv1d(hidden_dim, model_dim, kernel_size=1)
        self.output_norm = nn.LayerNorm(model_dim)

    # 编码天气序列。
    def forward(self, weather_seq, optional_mask=None):
        """将 [B, T, Cw] 编码为 [B, T, D]。"""
        if weather_seq.dim() != 3:
            raise ValueError(f'weather_seq 形状必须为 [B, T, Cw]，实际为 {tuple(weather_seq.shape)}')

        batch_size, seq_len, _ = weather_seq.shape
        mask = normalize_temporal_mask(optional_mask, batch_size, seq_len, weather_seq.device, weather_seq.dtype)
        channel_mask = mask.transpose(1, 2)

        x = weather_seq * mask
        x = x.transpose(1, 2)
        x = self.input_proj(x)
        x = x * channel_mask

        for block in self.blocks:
            x = block(x)
            x = x * channel_mask

        x = self.output_proj(x)
        x = x.transpose(1, 2)
        x = self.output_norm(x)
        x = x * mask
        return x
