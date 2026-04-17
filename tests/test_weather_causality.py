"""这个测试文件负责验证天气注入模块的严格因果性，确保未来天气不会泄露到过去输出。"""

import os
import sys
import unittest

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.weather.weather_injection_block import PhysicsConstrainedCausalWeatherInjection


# 验证未来天气变化不会影响过去时间步输出。
class TestWeatherCausality(unittest.TestCase):
    """测试因果天气注入模块的时间因果性。"""

    # 构造测试输入。
    def setUp(self):
        """准备因果性测试张量。"""
        torch.manual_seed(11)
        self.batch_size = 2
        self.seq_len = 8
        self.node_count = 4
        self.model_dim = 12
        self.h_main = torch.randn(self.batch_size, self.seq_len, self.node_count, self.model_dim)
        self.weather_seq = torch.randn(self.batch_size, self.seq_len, 3)

    # 验证改动未来天气后过去输出保持不变。
    def test_future_weather_does_not_change_past_output(self):
        """检查严格因果性。"""
        module = PhysicsConstrainedCausalWeatherInjection(
            model_dim=self.model_dim,
            weather_dim=3,
            num_heads=3,
            ablation_mode='causal_attn',
            dropout=0.0,
        )
        module.eval()

        cutoff = 4
        altered_weather = self.weather_seq.clone()
        altered_weather[:, cutoff + 1:, :] = altered_weather[:, cutoff + 1:, :] + 100.0

        h_exo_a, attn_a = module(self.h_main, self.weather_seq)
        h_exo_b, attn_b = module(self.h_main, altered_weather)

        self.assertTrue(torch.allclose(h_exo_a[:, :cutoff + 1], h_exo_b[:, :cutoff + 1], atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(attn_a[:, :, :, :cutoff + 1], attn_b[:, :, :, :cutoff + 1], atol=1e-6, rtol=1e-6))


if __name__ == '__main__':
    unittest.main()
