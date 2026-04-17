"""这个脚本负责演示爆破瞬态注入模块的最小前向流程，并与 models/blast/ 下的实现协同验证。"""

import os
import sys

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.blast.blast_injection_block import PhysicsInformedStepResponseBlastInjection


# 运行三种模式的最小前向演示。
def main():
    """构造随机输入并打印主模型与消融分支的输出形状。"""
    torch.manual_seed(23)

    batch_size, seq_len, node_count, model_dim, event_count = 2, 8, 5, 16, 3
    h_exo = torch.randn(batch_size, seq_len, node_count, model_dim)
    node_coords = torch.randn(node_count, 3)
    blast_locs = torch.randn(batch_size, event_count, 3)
    blast_times = torch.tensor([[0, 2, 5], [1, 3, 6]], dtype=torch.float32)
    blast_intensity = torch.tensor([[1.0, 2.0, 1.5], [1.2, 1.8, 2.5]], dtype=torch.float32)
    target_times = torch.arange(seq_len, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1)

    for mode in ('main', 'wo_gate', 'gru_blast'):
        module = PhysicsInformedStepResponseBlastInjection(
            model_dim=model_dim,
            hidden_dim=32,
            mode=mode,
            dropout=0.0,
        )
        module.eval()
        h_final, e_it, g_t, delta_h_blast = module(
            h_exo=h_exo,
            node_coords=node_coords,
            blast_locs=blast_locs,
            blast_times=blast_times,
            blast_intensity=blast_intensity,
            target_times=target_times,
        )
        print(
            f'mode={mode} H_exo={tuple(h_exo.shape)} H_final={tuple(h_final.shape)} '
            f'e_it={tuple(e_it.shape)} g_t={tuple(g_t.shape)} delta_H_blast={tuple(delta_h_blast.shape)}'
        )


if __name__ == '__main__':
    main()
