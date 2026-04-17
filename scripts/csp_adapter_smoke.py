"""这个脚本用于验证 TimeFilter 的 CSPAdapter 接入是否可运行，并和 models/TimeFilter.py、models/csp_adapter.py、data_provider/data_loader.py 配合完成最小 smoke 检查。"""

import argparse
import json
import os
import subprocess
import sys
import time
from types import SimpleNamespace

import torch
from torch import optim

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data_provider.data_factory import data_provider
from models.TimeFilter import Model


# 构造 smoke 测试所需的最小配置对象。
def build_args(use_csp_adapter=True, csp_debug=False):
    """构造适用于 radar 数据 smoke 测试的参数集合。"""
    return SimpleNamespace(
        task_name='long_term_forecast',
        data='custom',
        root_path='./dataset/radar',
        data_path='sim_radar_hourly_displacement.csv',
        features='M',
        target='node_0',
        freq='h',
        checkpoints='./checkpoints/',
        seq_len=96,
        label_len=48,
        pred_len=12,
        seasonal_patterns='Monthly',
        inverse=False,
        top_k=5,
        num_kernels=6,
        enc_in=1000,
        dec_in=1000,
        c_out=1000,
        d_model=32,
        n_heads=4,
        e_layers=1,
        d_layers=1,
        d_ff=64,
        moving_avg=25,
        factor=1,
        pos=1,
        distil=True,
        dropout=0.1,
        embed='timeF',
        activation='gelu',
        output_attention=False,
        channel_independence=1,
        decomp_method='moving_avg',
        use_norm=0,
        down_sampling_layers=0,
        down_sampling_window=1,
        down_sampling_method='avg',
        patch_len=96,
        alpha=0.1,
        top_p=0.5,
        use_csp_adapter=use_csp_adapter,
        coords_path='./dataset/radar/sim_nodes_static.csv',
        spatial_dim=64,
        rff_dim=64,
        rff_sigma=50.0,
        spatial_hidden_dim=128,
        learnable_z_scale=True,
        init_z_scale=1.0,
        prompt_alpha=0.1,
        learnable_prompt_alpha=False,
        physical_mask_radius=120.0,
        physical_mask_self_loop=True,
        csp_debug=csp_debug,
        num_workers=0,
        itr=1,
        train_epochs=1,
        batch_size=1,
        patience=1,
        learning_rate=5e-4,
        des='csp_smoke',
        loss='MSE',
        lradj='cosine',
        use_amp=False,
        use_gpu=False,
        gpu=0,
        use_multi_gpu=False,
        devices='0',
        p_hidden_dims=[128, 128],
        p_hidden_layers=2,
        use_dtw=False,
        augmentation_ratio=0,
        seed=2,
        jitter=False,
        scaling=False,
        permutation=False,
        randompermutation=False,
        magwarp=False,
        timewarp=False,
        windowslice=False,
        windowwarp=False,
        rotation=False,
        spawner=False,
        dtwwarp=False,
        shapedtwwarp=False,
        wdba=False,
        discdtw=False,
        discsdtw=False,
        extra_tag='',
    )


# 按 TimeFilter 的图分区方式构建默认掩码。
def build_masks(seq_len, c_out, patch_len, device):
    """构建 TimeFilter 训练时使用的区域掩码。"""
    dtype = torch.float32
    total_tokens = seq_len * c_out // patch_len
    patch_count = seq_len // patch_len
    masks = []
    for index in range(total_tokens):
        same_node = ((torch.arange(total_tokens) % patch_count == index % patch_count)
                     & (torch.arange(total_tokens) != index)).to(dtype).to(device)
        same_patch = ((torch.arange(total_tokens) >= index // patch_count * patch_count)
                      & (torch.arange(total_tokens) < index // patch_count * patch_count + patch_count)
                      & (torch.arange(total_tokens) != index)).to(dtype).to(device)
        others = torch.ones(total_tokens, dtype=dtype, device=device) - same_node - same_patch
        others[index] = 0.0
        masks.append(torch.stack([same_node, same_patch, others], dim=0))
    return torch.stack(masks, dim=0)


# 记录单次前向和反向的耗时与显存统计。
def get_runtime_stats(device, forward_seconds, backward_seconds):
    """整理单次前后向的运行统计。"""
    stats = {
        'device': str(device),
        'forward_seconds': round(float(forward_seconds), 4),
        'backward_seconds': round(float(backward_seconds), 4),
    }
    if device.type == 'cuda':
        stats['peak_cuda_memory_mb'] = round(torch.cuda.max_memory_allocated(device) / 1024 / 1024, 2)
    else:
        stats['peak_cuda_memory_mb'] = 0.0
    return stats


# 从 Git 当前提交中读取原始 baseline 模型，用来对照验证关闭 CSP 后的兼容性。
def load_head_baseline_model_class():
    """读取 Git HEAD 版本的 TimeFilter Model 类。"""
    source_code = subprocess.check_output(
        ['git', 'show', 'HEAD:models/TimeFilter.py'],
        text=True,
    )
    # 兼容当前主干返回第三个 debug_info 的情况，避免旧模型加载后解包失败。
    source_code = source_code.replace(
        "        x, moe_loss = self.backbone(x, masks, self.alpha, is_training)\n",
        "        backbone_output = self.backbone(x, masks, self.alpha, is_training)\n"
        "        x, moe_loss = backbone_output[:2]\n",
    )
    namespace = {}
    exec(source_code, namespace)
    return namespace['Model']


# 执行一次 dummy forward/backward，用来验证 shape、开关兼容和数值稳定性。
def run_single_batch_case(device, use_csp_adapter, enable_debug):
    """运行单个随机 batch 的前后向检查。"""
    torch.manual_seed(2026)
    args = build_args(use_csp_adapter=use_csp_adapter, csp_debug=enable_debug)
    model = Model(args).to(device)
    masks = build_masks(args.seq_len, args.c_out, args.patch_len, device)
    batch_x = torch.randn(1, args.seq_len, args.c_out, device=device)
    batch_y = torch.randn(1, args.pred_len, args.c_out, device=device)

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

    forward_start = time.perf_counter()
    prediction, moe_loss, debug_info = model(batch_x, masks, is_training=True, return_debug=True)
    forward_seconds = time.perf_counter() - forward_start

    loss = torch.nn.functional.mse_loss(prediction, batch_y) + 0.05 * moe_loss
    backward_start = time.perf_counter()
    loss.backward()
    backward_seconds = time.perf_counter() - backward_start

    result = {
        'use_csp_adapter': use_csp_adapter,
        'prediction_shape': list(prediction.shape),
        'loss': round(float(loss.item()), 6),
        'moe_loss': round(float(moe_loss.detach().item()), 6),
        'output_has_nan': bool(debug_info['output_has_nan']),
        'output_has_inf': bool(debug_info['output_has_inf']),
        'coords_shape': debug_info.get('coords_shape'),
        'spatial_embed_shape': debug_info.get('spatial_embed_shape'),
        'patch_token_shape': debug_info.get('patch_token_shape'),
        'prompted_token_shape': debug_info.get('prompted_token_shape'),
        'physical_mask_shape': debug_info.get('physical_mask_shape'),
        'adjacency_shape': debug_info.get('adjacency_shape'),
        'routing_mask_shape': debug_info.get('routing_mask_shape'),
        'radius_mask_active_ratio': round(float(debug_info.get('radius_mask_active_ratio', 0.0)), 6),
        'routing_mask_active_ratio': round(float(debug_info.get('routing_mask_active_ratio', 0.0)), 6),
        'token_norm_before': round(float(debug_info.get('token_norm_before', 0.0)), 6),
        'token_norm_after': round(float(debug_info.get('token_norm_after', 0.0)), 6),
        'adj_before_mask_mean': round(float(debug_info.get('adj_before_mask_mean', 0.0)), 6),
        'adj_after_mask_mean': round(float(debug_info.get('adj_after_mask_mean', 0.0)), 6),
        'adj_softmax_mean': round(float(debug_info.get('adj_softmax_mean', 0.0)), 6),
    }
    result.update(get_runtime_stats(device, forward_seconds, backward_seconds))
    return result


# 用真实 radar 数据跑几步训练，确认 loss 会变化且优化器可正常更新。
def run_dataset_smoke(device, steps, enable_debug):
    """在 radar 数据上执行少量训练步数的 smoke 检查。"""
    torch.manual_seed(2026)
    args = build_args(use_csp_adapter=True, csp_debug=enable_debug)
    train_dataset, train_loader = data_provider(args, 'train')
    model = Model(args).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.MSELoss()
    masks = build_masks(args.seq_len, args.c_out, args.patch_len, device)

    losses = []
    step_records = []
    for step_index, batch in enumerate(train_loader):
        batch_x, batch_y, _, _ = batch
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)

        optimizer.zero_grad()
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)

        forward_start = time.perf_counter()
        prediction, moe_loss, debug_info = model(batch_x, masks, is_training=True, return_debug=True)
        forward_seconds = time.perf_counter() - forward_start

        prediction = prediction[:, -args.pred_len:, :]
        target = batch_y[:, -args.pred_len:, :]
        loss = criterion(prediction, target) + 0.05 * moe_loss

        backward_start = time.perf_counter()
        loss.backward()
        backward_seconds = time.perf_counter() - backward_start
        optimizer.step()

        loss_value = float(loss.item())
        losses.append(loss_value)
        record = {
            'step': step_index,
            'loss': round(loss_value, 6),
            'radius_mask_active_ratio': round(float(debug_info['radius_mask_active_ratio']), 6),
            'adj_after_mask_mean': round(float(debug_info['adj_after_mask_mean']), 6),
        }
        record.update(get_runtime_stats(device, forward_seconds, backward_seconds))
        step_records.append(record)

        if step_index + 1 >= steps:
            break

    if not losses:
        raise RuntimeError('smoke 训练没有取到任何 batch。')

    return {
        'train_size': len(train_dataset),
        'steps': step_records,
        'loss_start': round(losses[0], 6),
        'loss_end': round(losses[-1], 6),
        'loss_changed': bool(max(losses) - min(losses) > 1e-6),
        'all_finite': bool(all(torch.isfinite(torch.tensor(losses)).tolist())),
    }


# 对照 Git 中的原始 baseline，检查关闭 CSP 后是否保持原始行为。
def run_switch_compatibility_test(device):
    """对比当前 baseline 路径和 Git HEAD 中原始模型的输出差异。"""
    baseline_model_class = load_head_baseline_model_class()
    args = build_args(use_csp_adapter=False, csp_debug=False)
    masks = build_masks(args.seq_len, args.c_out, args.patch_len, device)
    batch_x = torch.randn(1, args.seq_len, args.c_out, device=device)

    torch.manual_seed(2026)
    current_model = Model(args).to(device)
    torch.manual_seed(2026)
    original_model = baseline_model_class(args).to(device)

    current_model.eval()
    original_model.eval()
    with torch.no_grad():
        current_prediction, current_moe = current_model(batch_x, masks, is_training=False)
        original_prediction, original_moe = original_model(batch_x, masks, is_training=False)

    prediction_diff = float((current_prediction - original_prediction).abs().max().item())
    current_moe_value = float(current_moe.detach().cpu().item())
    original_moe_value = float(original_moe.detach().cpu().item())
    moe_diff = float(abs(current_moe_value - original_moe_value))
    return {
        'prediction_max_abs_diff': round(prediction_diff, 8),
        'moe_loss_abs_diff': round(moe_diff, 8),
        'compatible': bool(prediction_diff < 1e-6 and moe_diff < 1e-6),
    }


# 统一执行全部 smoke 检查。
def run_all_checks(device, steps, enable_debug):
    """串行运行本次任务要求的最小检查集合。"""
    baseline_case = run_single_batch_case(device, use_csp_adapter=False, enable_debug=False)
    csp_case = run_single_batch_case(device, use_csp_adapter=True, enable_debug=enable_debug)
    compatibility_case = run_switch_compatibility_test(device)
    dataset_case = run_dataset_smoke(device, steps=steps, enable_debug=False)

    if baseline_case['prediction_shape'] != [1, 12, 1000]:
        raise AssertionError(f"baseline 输出 shape 异常: {baseline_case['prediction_shape']}")
    if csp_case['prediction_shape'] != [1, 12, 1000]:
        raise AssertionError(f"CSP 输出 shape 异常: {csp_case['prediction_shape']}")
    if csp_case['physical_mask_shape'] != [1, 1, 1000, 1000]:
        raise AssertionError(f"CSP 物理掩码 shape 异常: {csp_case['physical_mask_shape']}")
    if baseline_case['output_has_nan'] or baseline_case['output_has_inf']:
        raise AssertionError('baseline 路径存在 NaN/Inf。')
    if csp_case['output_has_nan'] or csp_case['output_has_inf']:
        raise AssertionError('CSP 路径存在 NaN/Inf。')
    if not compatibility_case['compatible']:
        raise AssertionError(f"baseline 兼容性检查失败: {compatibility_case}")
    if not dataset_case['loss_changed']:
        raise AssertionError('smoke 训练 loss 没有变化。')
    if not dataset_case['all_finite']:
        raise AssertionError('smoke 训练 loss 出现非有限值。')

    return {
        'dummy_shape_test': {
            'baseline_prediction_shape': baseline_case['prediction_shape'],
            'csp_prediction_shape': csp_case['prediction_shape'],
        },
        'switch_compatibility_test': {
            'baseline_has_mask': baseline_case['physical_mask_shape'] is not None,
            'csp_has_mask': csp_case['physical_mask_shape'] is not None,
            **compatibility_case,
        },
        'forward_backward_test': {
            'baseline': baseline_case,
            'csp': csp_case,
        },
        'short_train_smoke_test': dataset_case,
    }


# 解析命令行并运行 smoke 测试。
def main():
    """作为脚本入口运行 CSPAdapter smoke 检查。"""
    parser = argparse.ArgumentParser(description='CSPAdapter smoke test')
    parser.add_argument('--device', type=str, default='cpu', help='运行设备，支持 cpu 或 cuda')
    parser.add_argument('--steps', type=int, default=3, help='真实数据 smoke 训练步数')
    parser.add_argument('--debug', action='store_true', help='开启一次 CSP debug 输出')
    args = parser.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('当前环境不可用 CUDA，请改用 --device cpu。')

    device = torch.device(args.device)
    results = run_all_checks(device=device, steps=args.steps, enable_debug=args.debug)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
