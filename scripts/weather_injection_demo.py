"""这个脚本用于演示天气注入模块的最小前向调用，并与 models/weather 下的实现保持一致。"""

import os
import sys

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.weather.weather_injection_block import PhysicsConstrainedCausalWeatherInjection


# 运行最小天气注入前向示例。
def main():
    """分别演示主模型和两种消融模式的前向调用。"""
    torch.manual_seed(42)
    batch_size, seq_len, node_count, model_dim, weather_dim = 2, 10, 6, 16, 3
    h_main = torch.randn(batch_size, seq_len, node_count, model_dim)
    weather_seq = torch.randn(batch_size, seq_len, weather_dim)

    for mode in ['causal_attn', 'vanilla_attn', 'concat_fusion']:
        module = PhysicsConstrainedCausalWeatherInjection(
            model_dim=model_dim,
            weather_dim=weather_dim,
            num_heads=4,
            ablation_mode=mode,
            dropout=0.0,
        )
        module.eval()
        h_exo, attn_weights = module(h_main, weather_seq)
        print(
            f'mode={mode} '
            f'H_main={tuple(h_main.shape)} '
            f'H_exo={tuple(h_exo.shape)} '
            f'attn_shape={None if attn_weights is None else tuple(attn_weights.shape)}'
        )


if __name__ == '__main__':
    main()
