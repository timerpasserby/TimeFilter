"""这个测试文件负责验证天气模块的消融版本可以独立运行。"""

import os
import sys
import unittest

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.weather.weather_injection_block import PhysicsConstrainedCausalWeatherInjection


# 验证天气模块消融版本的独立前向能力。
class TestWeatherAblation(unittest.TestCase):
    """测试 concat 与 vanilla attention 两种消融版本。"""

    # 构造测试输入。
    def setUp(self):
        """准备公共输入张量。"""
        torch.manual_seed(19)
        self.h_main = torch.randn(2, 7, 3, 16)
        self.weather_seq = torch.randn(2, 7, 3)

    # 验证 concat fusion 消融版本可单独前向。
    def test_concat_fusion_ablation_forward(self):
        """检查 concat fusion 的输出形状。"""
        module = PhysicsConstrainedCausalWeatherInjection(
            model_dim=16,
            weather_dim=3,
            num_heads=4,
            ablation_mode='concat_fusion',
            dropout=0.0,
        )
        module.eval()

        h_exo, attn_weights = module(self.h_main, self.weather_seq)
        self.assertEqual(tuple(h_exo.shape), (2, 7, 3, 16))
        self.assertIsNone(attn_weights)

    # 验证 vanilla attention 消融版本可单独前向。
    def test_vanilla_attention_ablation_forward(self):
        """检查 vanilla attention 的输出形状。"""
        module = PhysicsConstrainedCausalWeatherInjection(
            model_dim=16,
            weather_dim=3,
            num_heads=4,
            ablation_mode='vanilla_attn',
            dropout=0.0,
        )
        module.eval()

        h_exo, attn_weights = module(self.h_main, self.weather_seq)
        self.assertEqual(tuple(h_exo.shape), (2, 7, 3, 16))
        self.assertEqual(tuple(attn_weights.shape), (2, 3, 4, 7, 7))


if __name__ == '__main__':
    unittest.main()
