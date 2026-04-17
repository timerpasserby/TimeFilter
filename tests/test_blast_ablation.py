"""这个测试文件负责验证爆破模块的消融分支可以独立前向运行。"""

import os
import sys
import unittest

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.blast.blast_injection_block import PhysicsInformedStepResponseBlastInjection


# 验证 wo_gate 与 gru_blast 两种消融模式。
class TestBlastAblation(unittest.TestCase):
    """测试两种爆破消融模式的独立运行能力。"""

    # 准备公共输入。
    def setUp(self):
        """构造适用于两种消融的输入张量。"""
        torch.manual_seed(43)
        self.h_exo = torch.randn(2, 5, 3, 12)
        self.node_coords = torch.randn(3, 3)
        self.blast_locs = torch.randn(2, 4, 3)
        self.blast_times = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.float32)
        self.blast_intensity = torch.tensor([[1.0, 2.0, 1.5, 0.8], [1.1, 1.9, 2.3, 1.4]], dtype=torch.float32)
        self.target_times = torch.arange(5, dtype=torch.float32).unsqueeze(0).repeat(2, 1)

    # 检查 wo_gate 消融可正常前向。
    def test_wo_gate_ablation_forward(self):
        """验证去门控版本的输出形状。"""
        module = PhysicsInformedStepResponseBlastInjection(
            model_dim=12,
            hidden_dim=24,
            mode='wo_gate',
            dropout=0.0,
        )
        module.eval()

        h_final, e_it, g_t, delta_h_blast = module(
            h_exo=self.h_exo,
            node_coords=self.node_coords,
            blast_locs=self.blast_locs,
            blast_times=self.blast_times,
            blast_intensity=self.blast_intensity,
            target_times=self.target_times,
        )
        self.assertEqual(tuple(h_final.shape), (2, 5, 3, 12))
        self.assertEqual(tuple(e_it.shape), (2, 5, 3, 1))
        self.assertEqual(tuple(g_t.shape), (2, 5, 3, 1))
        self.assertEqual(tuple(delta_h_blast.shape), (2, 5, 3, 12))
        self.assertTrue(torch.allclose(g_t, torch.ones_like(g_t)))

    # 检查 gru_blast 消融可正常前向。
    def test_gru_blast_ablation_forward(self):
        """验证 GRU 替代版本的输出形状。"""
        module = PhysicsInformedStepResponseBlastInjection(
            model_dim=12,
            hidden_dim=20,
            mode='gru_blast',
            dropout=0.0,
        )
        module.eval()

        h_final, e_it, g_t, delta_h_blast = module(
            h_exo=self.h_exo,
            node_coords=self.node_coords,
            blast_locs=self.blast_locs,
            blast_times=self.blast_times,
            blast_intensity=self.blast_intensity,
            target_times=self.target_times,
        )
        self.assertEqual(tuple(h_final.shape), (2, 5, 3, 12))
        self.assertEqual(tuple(e_it.shape), (2, 5, 3, 1))
        self.assertEqual(tuple(g_t.shape), (2, 5, 3, 1))
        self.assertEqual(tuple(delta_h_blast.shape), (2, 5, 3, 12))
        self.assertTrue(torch.allclose(g_t, torch.ones_like(g_t)))


if __name__ == '__main__':
    unittest.main()
