"""这个测试文件负责验证未来爆破事件不会影响过去输出。"""

import os
import sys
import unittest

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.blast.blast_injection_block import PhysicsInformedStepResponseBlastInjection


# 验证爆破模块的严格因果性。
class TestBlastCausality(unittest.TestCase):
    """测试只修改未来爆破事件时，过去输出保持不变。"""

    # 构造共享测试输入。
    def setUp(self):
        """准备一个便于检查未来泄露的样例。"""
        torch.manual_seed(37)
        self.h_exo = torch.randn(1, 6, 3, 8)
        self.node_coords = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        self.blast_locs = torch.tensor([[[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]], dtype=torch.float32)
        self.blast_times = torch.tensor([[1.0, 5.0]], dtype=torch.float32)
        self.blast_intensity = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        self.target_times = torch.arange(6, dtype=torch.float32).unsqueeze(0)

    # 检查未来爆破不会影响过去时刻的输出。
    def test_future_blast_does_not_affect_past_outputs(self):
        """只改未来爆破时，过去时间步的输出应保持一致。"""
        module = PhysicsInformedStepResponseBlastInjection(
            model_dim=8,
            hidden_dim=16,
            init_sigma_b=3.0,
            init_gamma_b=0.2,
            mode='main',
            dropout=0.0,
        )
        module.eval()

        future_changed_locs = self.blast_locs.clone()
        future_changed_times = self.blast_times.clone()
        future_changed_intensity = self.blast_intensity.clone()
        future_changed_locs[:, 1, :] = torch.tensor([20.0, 10.0, 5.0], dtype=torch.float32)
        future_changed_times[:, 1] = 5.0
        future_changed_intensity[:, 1] = 9.0

        original_outputs = module(
            h_exo=self.h_exo,
            node_coords=self.node_coords,
            blast_locs=self.blast_locs,
            blast_times=self.blast_times,
            blast_intensity=self.blast_intensity,
            target_times=self.target_times,
        )
        changed_outputs = module(
            h_exo=self.h_exo,
            node_coords=self.node_coords,
            blast_locs=future_changed_locs,
            blast_times=future_changed_times,
            blast_intensity=future_changed_intensity,
            target_times=self.target_times,
        )

        past_slice = slice(0, 5)
        self.assertTrue(torch.allclose(original_outputs[0][:, past_slice], changed_outputs[0][:, past_slice], atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(original_outputs[1][:, past_slice], changed_outputs[1][:, past_slice], atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(original_outputs[2][:, past_slice], changed_outputs[2][:, past_slice], atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(original_outputs[3][:, past_slice], changed_outputs[3][:, past_slice], atol=1e-6, rtol=1e-6))


if __name__ == '__main__':
    unittest.main()
