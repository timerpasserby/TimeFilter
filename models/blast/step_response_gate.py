"""这个文件负责实现爆破阶跃响应门控与旁路残差注入，并与 blast_analytic_encoder.py、blast_injection_block.py 协同工作。"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# 将解析扰动映射为门控和待注入特征增量。
class StepResponseGate(nn.Module):
    """把 e_it 转换为门控开度 g_t 与旁路增量 delta_H_blast。"""

    # 初始化阶跃响应门控。
    def __init__(self, model_dim, hidden_dim=None, dropout=0.0):
        """构建单调门控与特征增量映射层。"""
        super().__init__()
        hidden_dim = int(hidden_dim or model_dim)
        self.gate_weight = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.gate_bias = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.value_proj = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, model_dim),
            nn.Dropout(dropout),
        )

    # 执行门控与增量映射。
    def forward(self, e_it):
        """输入 [B, T, N, 1] 解析扰动，输出标量门控与 D 维增量。"""
        if e_it.dim() != 4 or e_it.shape[-1] != 1:
            raise ValueError(f'e_it 形状必须为 [B, T, N, 1]，实际为 {tuple(e_it.shape)}')

        positive_weight = F.softplus(self.gate_weight)
        g_t = torch.sigmoid(positive_weight * e_it + self.gate_bias)
        value_source = torch.log1p(e_it.clamp_min(0.0))
        delta_h_blast = self.value_proj(value_source)
        return g_t, delta_h_blast


# 将爆破增量以旁路残差形式注入上游隐状态。
class BypassResidualInjection(nn.Module):
    """执行 H_final = H_exo + g_t * delta_H_blast 的旁路融合。"""

    # 初始化旁路残差注入器。
    def __init__(self, dropout=0.0, use_gate=True):
        """配置是否启用门控以及旁路 dropout。"""
        super().__init__()
        self.use_gate = bool(use_gate)
        self.dropout = nn.Dropout(dropout)

    # 执行旁路注入。
    def forward(self, h_exo, g_t, delta_h_blast):
        """根据是否启用门控，把爆破增量残差注入到 H_exo。"""
        if h_exo.dim() != 4:
            raise ValueError(f'h_exo 形状必须为 [B, T, N, D]，实际为 {tuple(h_exo.shape)}')
        if delta_h_blast.dim() != 4:
            raise ValueError(f'delta_h_blast 形状必须为 [B, T, N, D]，实际为 {tuple(delta_h_blast.shape)}')
        if g_t.dim() != 4:
            raise ValueError(f'g_t 形状必须为 [B, T, N, 1] 或 [B, T, N, D]，实际为 {tuple(g_t.shape)}')
        if h_exo.shape[:3] != delta_h_blast.shape[:3] or h_exo.shape[:3] != g_t.shape[:3]:
            raise ValueError(
                'h_exo、g_t、delta_h_blast 的 [B, T, N] 维度必须一致，'
                f'实际分别为 {tuple(h_exo.shape)}、{tuple(g_t.shape)}、{tuple(delta_h_blast.shape)}'
            )
        if g_t.shape[-1] not in {1, h_exo.shape[-1]}:
            raise ValueError(f'g_t 最后一维必须为 1 或 D={h_exo.shape[-1]}，实际为 {g_t.shape[-1]}')
        if delta_h_blast.shape[-1] != h_exo.shape[-1]:
            raise ValueError(
                f'delta_h_blast 的最后一维必须与 h_exo 一致，实际为 {delta_h_blast.shape[-1]} 和 {h_exo.shape[-1]}'
            )

        applied_delta = g_t * delta_h_blast if self.use_gate else delta_h_blast
        applied_delta = self.dropout(applied_delta)
        h_final = h_exo + applied_delta
        return h_final, applied_delta
