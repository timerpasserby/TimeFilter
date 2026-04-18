"""这个测试文件负责验证天气模块与爆破模块接入 TimeFilter 主干后的基础可运行性。"""

import os
import sys
import tempfile
import unittest
from types import SimpleNamespace

import pandas as pd
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.TimeFilter import Model


# 构建与实验入口一致的静态掩码。
def build_timefilter_mask(seq_len, c_out, patch_len):
    """生成 TimeFilter 训练时使用的静态区域掩码。"""
    total_tokens = seq_len * c_out // patch_len
    patch_count = seq_len // patch_len
    dtype = torch.float32
    masks = []
    for index in range(total_tokens):
        same_node = ((torch.arange(total_tokens) % patch_count == index % patch_count)
                     & (torch.arange(total_tokens) != index)).to(dtype)
        same_patch = ((torch.arange(total_tokens) >= index // patch_count * patch_count)
                      & (torch.arange(total_tokens) < index // patch_count * patch_count + patch_count)
                      & (torch.arange(total_tokens) != index)).to(dtype)
        others = torch.ones(total_tokens, dtype=dtype) - same_node - same_patch
        others[index] = 0.0
        masks.append(torch.stack([same_node, same_patch, others], dim=0))
    return torch.stack(masks, dim=0)


# 验证主干在不同外生模块配置下都能顺利前向和反向。
class TestTimeFilterExogenousIntegration(unittest.TestCase):
    """测试 TimeFilter 与天气、爆破模块的集成行为。"""

    # 构造公共输入与临时坐标文件。
    def setUp(self):
        """准备模型配置、图掩码和 dummy 输入。"""
        torch.manual_seed(11)
        self.batch_size = 2
        self.seq_len = 8
        self.pred_len = 2
        self.node_count = 4
        self.model_dim = 8
        self.patch_len = 4
        self.num_patches = self.seq_len // self.patch_len
        self.masks = build_timefilter_mask(self.seq_len, self.node_count, self.patch_len)
        self.batch_x = torch.randn(self.batch_size, self.seq_len, self.node_count)

        self.temp_dir = tempfile.TemporaryDirectory()
        self.coords_path = os.path.join(self.temp_dir.name, 'coords.csv')
        coords_frame = pd.DataFrame({
            'node_id': [0, 1, 2, 3],
            'grid_x': [0.0, 10.0, 20.0, 30.0],
            'grid_y': [0.0, 5.0, 10.0, 15.0],
            'grid_z': [1.0, 2.0, 3.0, 4.0],
        })
        coords_frame.to_csv(self.coords_path, index=False)

    # 清理临时文件。
    def tearDown(self):
        """释放测试中创建的临时目录。"""
        self.temp_dir.cleanup()

    # 生成一组最小可运行的模型配置。
    def _build_config(self, use_weather_module=False, use_blast_module=False):
        """根据开关生成 TimeFilter 配置。"""
        return SimpleNamespace(
            task_name='long_term_forecast',
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            c_out=self.node_count,
            d_model=self.model_dim,
            d_ff=16,
            patch_len=self.patch_len,
            alpha=0.1,
            top_p=0.5,
            pos=1,
            n_heads=4,
            e_layers=1,
            dropout=0.0,
            enc_in=self.node_count,
            use_csp_adapter=False,
            csp_debug=False,
            coords_path=self.coords_path,
            spatial_dim=16,
            rff_dim=16,
            rff_sigma=10.0,
            spatial_hidden_dim=16,
            learnable_z_scale=True,
            init_z_scale=1.0,
            prompt_alpha=0.1,
            learnable_prompt_alpha=False,
            physical_mask_radius=120.0,
            physical_mask_self_loop=True,
            use_weather_module=use_weather_module,
            weather_dim=3,
            weather_hidden_dim=0,
            weather_kernel_size=3,
            weather_dilations=[1, 2],
            weather_ablation_mode='causal_attn',
            use_blast_module=use_blast_module,
            blast_hidden_dim=0,
            blast_mode='main',
            blast_init_sigma_b=120.0,
            blast_init_gamma_b=0.1,
            exo_debug=False,
        )

    # 验证不开天气和爆破时，baseline 路径仍然可用。
    def test_baseline_forward_without_exogenous_inputs(self):
        """检查 baseline 路径在不传 extra_inputs 时仍可正常前向。"""
        model = Model(self._build_config(use_weather_module=False, use_blast_module=False))
        model.eval()

        outputs, moe_loss, debug_info = model(self.batch_x, self.masks, is_training=False, return_debug=True)
        self.assertEqual(tuple(outputs.shape), (self.batch_size, self.pred_len, self.node_count))
        self.assertTrue(torch.isfinite(outputs).all())
        self.assertFalse(debug_info['use_weather_module'])
        self.assertFalse(debug_info['use_blast_module'])
        self.assertTrue(torch.isfinite(moe_loss))

    # 验证天气和爆破一起接入后，forward/backward 都能跑通。
    def test_forward_backward_with_weather_and_blast(self):
        """检查天气与爆破接入主干后的完整前反向过程。"""
        model = Model(self._build_config(use_weather_module=True, use_blast_module=True))
        model.train()

        extra_inputs = {
            'weather_seq': torch.randn(self.batch_size, self.seq_len, 3),
            'weather_mask': torch.ones(self.batch_size, self.seq_len, 1),
            'blast_locs': torch.tensor(
                [
                    [[0.0, 0.0, 0.0], [15.0, 5.0, 2.0], [25.0, 10.0, 3.0]],
                    [[5.0, 2.0, 1.0], [18.0, 8.0, 2.5], [28.0, 12.0, 3.5]],
                ],
                dtype=torch.float32,
            ),
            'blast_times': torch.tensor([[1.0, 3.0, 6.0], [0.0, 2.0, 5.0]], dtype=torch.float32),
            'blast_intensity': torch.tensor([[1.0, 0.6, 1.2], [0.5, 1.1, 0.8]], dtype=torch.float32),
            'patch_times': torch.tensor([[3.0, 7.0], [3.0, 7.0]], dtype=torch.float32),
        }

        outputs, moe_loss, debug_info = model(
            self.batch_x,
            self.masks,
            is_training=True,
            extra_inputs=extra_inputs,
            return_debug=True,
        )
        loss = outputs.mean() + 0.01 * moe_loss
        loss.backward()

        self.assertEqual(tuple(outputs.shape), (self.batch_size, self.pred_len, self.node_count))
        self.assertTrue(torch.isfinite(outputs).all())
        self.assertIn('weather_patch_shape', debug_info)
        self.assertIn('blast_e_it_shape', debug_info)
        self.assertEqual(debug_info['weather_patch_shape'], [self.batch_size, self.num_patches, 3])
        self.assertEqual(debug_info['blast_e_it_shape'], [self.batch_size, self.num_patches, self.node_count, 1])
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad))


if __name__ == '__main__':
    unittest.main()
