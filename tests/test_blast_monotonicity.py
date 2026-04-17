"""这个测试文件负责验证爆破解析扰动满足距离衰减、强度单调和时间衰减。"""

import os
import sys
import unittest

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.blast.blast_analytic_encoder import BlastAnalyticEncoder


# 验证解析扰动的基础物理单调性。
class TestBlastMonotonicity(unittest.TestCase):
    """测试 e_it 在距离、强度和时间维度上的单调关系。"""

    # 初始化解析编码器。
    def setUp(self):
        """准备一个固定参数的解析编码器。"""
        torch.manual_seed(41)
        self.encoder = BlastAnalyticEncoder(init_sigma_b=2.0, init_gamma_b=0.5)
        self.encoder.eval()

    # 验证距离越远解析扰动越小。
    def test_distance_monotonicity(self):
        """固定爆破强度和时间差时，近节点扰动不小于远节点。"""
        node_coords = torch.tensor([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype=torch.float32)
        blast_locs = torch.tensor([[[0.0, 0.0, 0.0]]], dtype=torch.float32)
        blast_times = torch.tensor([[0.0]], dtype=torch.float32)
        blast_intensity = torch.tensor([[2.0]], dtype=torch.float32)
        target_times = torch.tensor([[1.0]], dtype=torch.float32)

        e_it = self.encoder(node_coords, blast_locs, blast_times, blast_intensity, target_times)
        near_value = e_it[0, 0, 0, 0].item()
        far_value = e_it[0, 0, 1, 0].item()
        self.assertGreaterEqual(near_value + 1e-8, far_value)

    # 验证爆破强度越大解析扰动越大。
    def test_intensity_monotonicity(self):
        """固定距离和时间差时，增大强度不应降低 e_it。"""
        node_coords = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
        blast_locs = torch.tensor([[[0.0, 0.0, 0.0]]], dtype=torch.float32)
        blast_times = torch.tensor([[0.0]], dtype=torch.float32)
        low_intensity = torch.tensor([[1.0]], dtype=torch.float32)
        high_intensity = torch.tensor([[3.0]], dtype=torch.float32)
        target_times = torch.tensor([[2.0]], dtype=torch.float32)

        e_low = self.encoder(node_coords, blast_locs, blast_times, low_intensity, target_times)
        e_high = self.encoder(node_coords, blast_locs, blast_times, high_intensity, target_times)
        self.assertGreaterEqual(e_high.item() + 1e-8, e_low.item())

    # 验证同一爆破贡献会随时间衰减。
    def test_temporal_decay(self):
        """固定节点与强度时，离爆破时刻越远单次贡献应越小。"""
        node_coords = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
        blast_locs = torch.tensor([[[0.0, 0.0, 0.0]]], dtype=torch.float32)
        blast_times = torch.tensor([[0.0]], dtype=torch.float32)
        blast_intensity = torch.tensor([[2.0]], dtype=torch.float32)
        target_times = torch.tensor([[0.0, 1.0, 3.0]], dtype=torch.float32)

        e_it = self.encoder(node_coords, blast_locs, blast_times, blast_intensity, target_times)
        value_t0 = e_it[0, 0, 0, 0].item()
        value_t1 = e_it[0, 1, 0, 0].item()
        value_t3 = e_it[0, 2, 0, 0].item()
        self.assertGreaterEqual(value_t0 + 1e-8, value_t1)
        self.assertGreaterEqual(value_t1 + 1e-8, value_t3)


if __name__ == '__main__':
    unittest.main()
