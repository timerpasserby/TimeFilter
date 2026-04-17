"""这个测试文件负责验证天气注入模块的输出形状与注意力可解释性接口。"""

import os
import sys
import unittest

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.weather.weather_injection_block import PhysicsConstrainedCausalWeatherInjection


# 验证天气模块 shape 与注意力返回结果。
class TestWeatherShapes(unittest.TestCase):
    """测试天气注入模块的基础 shape 行为。"""

    # 构造测试输入。
    def setUp(self):
        """准备公共测试张量。"""
        torch.manual_seed(7)
        self.batch_size = 2
        self.seq_len = 6
        self.node_count = 5
        self.model_dim = 16
        self.weather_dim = 3
        self.num_heads = 4
        self.h_main = torch.randn(self.batch_size, self.seq_len, self.node_count, self.model_dim)
        self.weather_seq = torch.randn(self.batch_size, self.seq_len, self.weather_dim)

    # 验证主模块输出形状和注意力权重维度。
    def test_weather_injection_output_shape(self):
        """检查 H_exo 与 attn_weights 的形状。"""
        module = PhysicsConstrainedCausalWeatherInjection(
            model_dim=self.model_dim,
            weather_dim=self.weather_dim,
            num_heads=self.num_heads,
            ablation_mode='causal_attn',
            dropout=0.0,
        )
        module.eval()

        h_exo, attn_weights = module(self.h_main, self.weather_seq)
        self.assertEqual(tuple(h_exo.shape), (self.batch_size, self.seq_len, self.node_count, self.model_dim))
        self.assertIsNotNone(attn_weights)
        self.assertEqual(tuple(attn_weights.shape), (self.batch_size, self.node_count, self.num_heads, self.seq_len, self.seq_len))

    # 验证注意力权重能用于后续热力图解释。
    def test_attention_weights_interpretability_shape(self):
        """检查注意力权重最后两维是否为时间到时间的矩阵。"""
        module = PhysicsConstrainedCausalWeatherInjection(
            model_dim=self.model_dim,
            weather_dim=self.weather_dim,
            num_heads=self.num_heads,
            ablation_mode='causal_attn',
            dropout=0.0,
        )
        module.eval()

        _, attn_weights = module(self.h_main, self.weather_seq)
        self.assertEqual(attn_weights.dim(), 5)
        self.assertEqual(attn_weights.shape[-1], self.seq_len)
        self.assertEqual(attn_weights.shape[-2], self.seq_len)


if __name__ == '__main__':
    unittest.main()
