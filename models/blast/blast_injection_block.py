"""这个文件负责封装爆破解析编码、阶跃门控与旁路注入总模块，并与天气模块输出 H_exo 在接口层对接。"""

import torch
import torch.nn as nn

from models.blast.blast_analytic_encoder import BlastAnalyticEncoder
from models.blast.step_response_gate import BypassResidualInjection, StepResponseGate


# 使用 GRU 对解析爆破扰动做替代编码，仅用于消融实验。
class GRUBlastAblation(nn.Module):
    """把 e_it 送入 GRU，并输出可注入主路径的 D 维特征增量。"""

    # 初始化 GRU 消融分支。
    def __init__(self, model_dim, hidden_dim=None, dropout=0.0):
        """构建仅用于消融的 GRU 编码器。"""
        super().__init__()
        hidden_dim = int(hidden_dim or model_dim)
        self.gru = nn.GRU(input_size=1, hidden_size=hidden_dim, batch_first=True)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, model_dim),
            nn.Dropout(dropout),
        )

    # 执行 GRU 消融前向传播。
    def forward(self, e_it):
        """将 [B, T, N, 1] 的解析扰动转为 [B, T, N, D] 的消融增量。"""
        if e_it.dim() != 4 or e_it.shape[-1] != 1:
            raise ValueError(f'e_it 形状必须为 [B, T, N, 1]，实际为 {tuple(e_it.shape)}')

        batch_size, seq_len, node_count, _ = e_it.shape
        gru_input = torch.log1p(e_it.clamp_min(0.0))
        gru_input = gru_input.permute(0, 2, 1, 3).contiguous().view(batch_size * node_count, seq_len, 1)
        gru_output, _ = self.gru(gru_input)
        delta_h_blast = self.output_proj(gru_output)
        delta_h_blast = delta_h_blast.view(batch_size, node_count, seq_len, -1).permute(0, 2, 1, 3).contiguous()
        return delta_h_blast


# 物理先验驱动的爆破瞬态注入总模块。
class PhysicsInformedStepResponseBlastInjection(nn.Module):
    """先解析计算 e_it，再以旁路门控方式将爆破瞬态扰动注入 H_exo。"""

    # 初始化爆破注入总模块。
    def __init__(
        self,
        model_dim,
        hidden_dim=None,
        init_sigma_b=120.0,
        init_gamma_b=0.1,
        dropout=0.0,
        mode='main',
    ):
        """构建主模型分支与两种消融分支。"""
        super().__init__()
        valid_modes = {'main', 'wo_gate', 'gru_blast'}
        if mode not in valid_modes:
            raise ValueError(f'mode 必须属于 {sorted(valid_modes)}，实际为 {mode!r}')

        self.mode = mode
        self.analytic_encoder = BlastAnalyticEncoder(
            init_sigma_b=init_sigma_b,
            init_gamma_b=init_gamma_b,
        )

        if self.mode == 'gru_blast':
            self.step_gate = None
            self.gru_blast = GRUBlastAblation(model_dim=model_dim, hidden_dim=hidden_dim, dropout=dropout)
            self.bypass_injection = BypassResidualInjection(dropout=dropout, use_gate=False)
        else:
            self.step_gate = StepResponseGate(model_dim=model_dim, hidden_dim=hidden_dim, dropout=dropout)
            self.gru_blast = None
            self.bypass_injection = BypassResidualInjection(dropout=dropout, use_gate=(self.mode == 'main'))

    # 对主输入做形状检查。
    def _validate_inputs(self, h_exo, node_coords, blast_locs, blast_times, blast_intensity, target_times):
        """检查 H_exo 与爆破解析输入的关键维度关系。"""
        if h_exo.dim() != 4:
            raise ValueError(f'h_exo 形状必须为 [B, T, N, D]，实际为 {tuple(h_exo.shape)}')
        if node_coords.dim() != 2 or node_coords.shape[-1] != 3:
            raise ValueError(f'node_coords 形状必须为 [N, 3]，实际为 {tuple(node_coords.shape)}')
        if blast_locs.dim() != 3 or blast_locs.shape[-1] != 3:
            raise ValueError(f'blast_locs 形状必须为 [B, K, 3]，实际为 {tuple(blast_locs.shape)}')
        if blast_times.dim() != 2 or blast_intensity.dim() != 2 or target_times.dim() != 2:
            raise ValueError(
                'blast_times、blast_intensity、target_times 必须分别为 [B, K]、[B, K]、[B, T]，'
                f'实际为 {tuple(blast_times.shape)}、{tuple(blast_intensity.shape)}、{tuple(target_times.shape)}'
            )
        if h_exo.shape[0] != blast_locs.shape[0] or h_exo.shape[0] != target_times.shape[0]:
            raise ValueError(
                f'h_exo、blast_locs、target_times 的 batch 维度必须一致，实际为 {h_exo.shape[0]}、{blast_locs.shape[0]}、{target_times.shape[0]}'
            )
        if h_exo.shape[1] != target_times.shape[1]:
            raise ValueError(f'h_exo 的时间长度必须与 target_times 一致，实际为 {h_exo.shape[1]} 和 {target_times.shape[1]}')
        if h_exo.shape[2] != node_coords.shape[0]:
            raise ValueError(f'h_exo 的节点数必须与 node_coords 一致，实际为 {h_exo.shape[2]} 和 {node_coords.shape[0]}')
        if blast_locs.shape[:2] != blast_times.shape or blast_times.shape != blast_intensity.shape:
            raise ValueError(
                'blast_locs、blast_times、blast_intensity 的 [B, K] 维度必须一致，'
                f'实际分别为 {tuple(blast_locs.shape)}、{tuple(blast_times.shape)}、{tuple(blast_intensity.shape)}'
            )

    # 执行 GRU 消融分支。
    def _forward_gru_mode(self, e_it):
        """计算仅用于消融实验的 GRU-blast 增量。"""
        return self.gru_blast(e_it)

    # 执行爆破注入前向传播。
    def forward(self, h_exo, node_coords, blast_locs, blast_times, blast_intensity, target_times, optional_static=None):
        """输入天气增强隐状态与爆破日志，输出最终融合特征及解释量。"""
        self._validate_inputs(h_exo, node_coords, blast_locs, blast_times, blast_intensity, target_times)

        e_it = self.analytic_encoder(
            node_coords=node_coords,
            blast_locs=blast_locs,
            blast_times=blast_times,
            blast_intensity=blast_intensity,
            target_times=target_times,
            optional_static=optional_static,
        )

        if self.mode == 'gru_blast':
            g_t = torch.ones_like(e_it)
            delta_h_blast = self._forward_gru_mode(e_it)
        else:
            g_t, delta_h_blast = self.step_gate(e_it)
            if self.mode == 'wo_gate':
                g_t = torch.ones_like(g_t)

        h_final, _ = self.bypass_injection(h_exo, g_t, delta_h_blast)
        return h_final, e_it, g_t, delta_h_blast
