"""这个测试文件负责验证爆破瞬态注入模块的输出形状是否符合接口约定。"""

import os
import sys
import unittest

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.blast.blast_injection_block import PhysicsInformedStepResponseBlastInjection


# 验证主模型模式下的输出张量形状。
class TestBlastShapes(unittest.TestCase):
    """测试 H_final、e_it、g_t 和 delta_H_blast 的形状。"""

    # 构造公共测试输入。
    def setUp(self):
        """准备一组合法的随机输入。"""
        torch.manual_seed(31)
        self.h_exo = torch.randn(2, 6, 4, 16)
        self.node_coords = torch.randn(4, 3)
        self.blast_locs = torch.randn(2, 3, 3)
        self.blast_times = torch.tensor([[0, 2, 4], [1, 3, 5]], dtype=torch.float32)
        self.blast_intensity = torch.tensor([[1.0, 2.0, 1.5], [0.5, 1.2, 2.4]], dtype=torch.float32)
        self.target_times = torch.arange(6, dtype=torch.float32).unsqueeze(0).repeat(2, 1)

    # 验证主模型模式的输出形状。
    def test_main_mode_output_shapes(self):
        """检查主模型返回的所有关键张量形状。"""
        module = PhysicsInformedStepResponseBlastInjection(
            model_dim=16,
            hidden_dim=32,
            mode='main',
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

        self.assertEqual(tuple(h_final.shape), (2, 6, 4, 16))
        self.assertEqual(tuple(e_it.shape), (2, 6, 4, 1))
        self.assertEqual(tuple(g_t.shape), (2, 6, 4, 1))
        self.assertEqual(tuple(delta_h_blast.shape), (2, 6, 4, 16))


if __name__ == '__main__':
    unittest.main()
