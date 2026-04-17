"""这个文件负责实现爆破事件的物理解析编码，并与 step_response_gate.py、blast_injection_block.py 协同工作。"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# 将正初始化值映射回 softplus 之前的参数空间。
def inverse_softplus(value):
    """把正数初始化值转换为 softplus 原参数。"""
    if value <= 0:
        raise ValueError(f'初始化值必须大于 0，实际为 {value}')
    return value + math.log(-math.expm1(-value))


# 解析计算节点级连续爆破扰动。
class BlastAnalyticEncoder(nn.Module):
    """根据爆破位置、爆破时刻和爆破强度解析计算 e_it。"""

    # 初始化解析编码器。
    def __init__(
        self,
        init_sigma_b=120.0,
        init_gamma_b=0.1,
        max_distance_sq=1e8,
        max_time_delta=1e6,
        max_intensity=1e6,
        eps=1e-6,
    ):
        """构建具有可学习正参数的空间衰减与时间衰减编码器。"""
        super().__init__()
        self.raw_sigma_b = nn.Parameter(torch.tensor(inverse_softplus(float(init_sigma_b)), dtype=torch.float32))
        self.raw_gamma_b = nn.Parameter(torch.tensor(inverse_softplus(float(init_gamma_b)), dtype=torch.float32))
        self.max_distance_sq = float(max_distance_sq)
        self.max_time_delta = float(max_time_delta)
        self.max_intensity = float(max_intensity)
        self.eps = float(eps)

    # 将原始参数映射为正数。
    def _positive_parameter(self, raw_param):
        """使用 softplus 保证解析参数恒为正值。"""
        return F.softplus(raw_param) + self.eps

    # 对输入形状做基本检查。
    def _validate_inputs(self, node_coords, blast_locs, blast_times, blast_intensity, target_times, optional_static):
        """检查解析计算所需张量的维度与关键对齐关系。"""
        if node_coords.dim() != 2 or node_coords.shape[-1] != 3:
            raise ValueError(f'node_coords 形状必须为 [N, 3]，实际为 {tuple(node_coords.shape)}')
        if blast_locs.dim() != 3 or blast_locs.shape[-1] != 3:
            raise ValueError(f'blast_locs 形状必须为 [B, K, 3]，实际为 {tuple(blast_locs.shape)}')
        if blast_times.dim() != 2:
            raise ValueError(f'blast_times 形状必须为 [B, K]，实际为 {tuple(blast_times.shape)}')
        if blast_intensity.dim() != 2:
            raise ValueError(f'blast_intensity 形状必须为 [B, K]，实际为 {tuple(blast_intensity.shape)}')
        if target_times.dim() != 2:
            raise ValueError(f'target_times 形状必须为 [B, T]，实际为 {tuple(target_times.shape)}')
        if blast_locs.shape[:2] != blast_times.shape or blast_times.shape != blast_intensity.shape:
            raise ValueError(
                'blast_locs、blast_times、blast_intensity 的 [B, K] 维度必须一致，'
                f'实际分别为 {tuple(blast_locs.shape)}、{tuple(blast_times.shape)}、{tuple(blast_intensity.shape)}'
            )
        if blast_locs.shape[0] != target_times.shape[0]:
            raise ValueError(
                f'blast_locs 与 target_times 的 batch 维度必须一致，实际为 {blast_locs.shape[0]} 和 {target_times.shape[0]}'
            )
        if optional_static is not None and optional_static.dim() not in {2, 3}:
            raise ValueError(f'optional_static 仅支持 [N, C] 或 [B, N, C]，实际为 {tuple(optional_static.shape)}')

    # 执行解析编码前向传播。
    def forward(self, node_coords, blast_locs, blast_times, blast_intensity, target_times, optional_static=None):
        """利用广播一次性计算 [B, T, N, 1] 形式的连续爆破扰动。"""
        self._validate_inputs(node_coords, blast_locs, blast_times, blast_intensity, target_times, optional_static)

        compute_dtype = node_coords.dtype if node_coords.is_floating_point() else torch.float32
        device = blast_locs.device
        node_coords = node_coords.to(device=device, dtype=compute_dtype)
        blast_locs = blast_locs.to(device=device, dtype=compute_dtype)
        blast_times = blast_times.to(device=device, dtype=compute_dtype)
        blast_intensity = blast_intensity.to(device=device, dtype=compute_dtype)
        target_times = target_times.to(device=device, dtype=compute_dtype)

        batch_size, event_count, _ = blast_locs.shape
        node_count = node_coords.shape[0]
        seq_len = target_times.shape[1]

        sigma_b = self._positive_parameter(self.raw_sigma_b)
        gamma_b = self._positive_parameter(self.raw_gamma_b)

        node_coords_view = node_coords.view(1, node_count, 1, 3)
        blast_locs_view = blast_locs.view(batch_size, 1, event_count, 3)
        distance_sq = ((node_coords_view - blast_locs_view) ** 2).sum(dim=-1)
        distance_sq = distance_sq.clamp_min(0.0).clamp_max(self.max_distance_sq)
        spatial_term = torch.exp(-distance_sq / (2.0 * sigma_b.square()))

        delta_t = target_times.unsqueeze(-1) - blast_times.unsqueeze(1)
        causal_mask = delta_t >= 0
        delta_t = delta_t.clamp_min(0.0).clamp_max(self.max_time_delta)
        temporal_term = torch.exp(-gamma_b * delta_t)

        safe_intensity = blast_intensity.clamp_min(0.0).clamp_max(self.max_intensity)
        contributions = (
            safe_intensity.view(batch_size, 1, 1, event_count)
            * spatial_term.view(batch_size, 1, node_count, event_count)
            * temporal_term.view(batch_size, seq_len, 1, event_count)
            * causal_mask.to(dtype=compute_dtype).view(batch_size, seq_len, 1, event_count)
        )

        e_it = contributions.sum(dim=-1).clamp_min(0.0).clamp_max(self.max_intensity)
        return e_it.unsqueeze(-1)
