"""这个脚本用于对比 baseline 与 CSP 的预测结果，并输出空间热力图与代表节点曲线。"""

import argparse
import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

matplotlib.use('Agg')
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


# 读取单个结果目录中的 pred/true 文件。
def load_result_arrays(result_dir):
    """从结果目录加载预测和真值数组。"""
    pred_path = Path(result_dir) / 'pred.npy'
    true_path = Path(result_dir) / 'true.npy'
    if not pred_path.exists() or not true_path.exists():
        raise FileNotFoundError(f'结果目录缺少 pred.npy 或 true.npy: {result_dir}')
    pred = np.load(pred_path)
    true = np.load(true_path)
    if pred.shape != true.shape:
        raise ValueError(f'pred 与 true 形状不一致: {pred.shape} vs {true.shape}')
    if pred.ndim != 3:
        raise ValueError(f'当前脚本只支持 [样本, 预测步长, 节点] 三维数组，实际为 {pred.shape}')
    return pred, true


# 根据前缀自动匹配一个结果目录。
def find_result_dir_by_prefix(results_root, prefix):
    """在 results 根目录下按前缀匹配唯一目录。"""
    candidates = sorted([item for item in os.listdir(results_root) if item.startswith(prefix)])
    if not candidates:
        raise FileNotFoundError(f'未找到前缀为 {prefix!r} 的结果目录')
    if len(candidates) > 1:
        raise RuntimeError(f'前缀 {prefix!r} 匹配到多个目录，请手动指定路径: {candidates}')
    return os.path.join(results_root, candidates[0])


# 计算逐节点 MAE 与 MSE。
def compute_node_metrics(pred, true):
    """计算每个节点在全测试集上的 MAE 和 MSE。"""
    abs_err = np.abs(pred - true)
    sq_err = (pred - true) ** 2
    node_mae = abs_err.mean(axis=(0, 1))
    node_mse = sq_err.mean(axis=(0, 1))
    return node_mae, node_mse


# 计算逐节点的相对改进率。
def compute_improve_rate(base_metric, csp_metric):
    """基于 baseline 指标计算 CSP 的相对改进率。"""
    denominator = np.maximum(base_metric, 1e-9)
    return (base_metric - csp_metric) / denominator


# 绘制节点空间散点图。
def plot_spatial_delta(coords, delta_values, output_path, title):
    """按节点坐标绘制改进热力散点图。"""
    fig, axis = plt.subplots(figsize=(8, 6))
    scatter = axis.scatter(
        coords[:, 0],
        coords[:, 1],
        c=delta_values,
        cmap='RdYlGn',
        s=18,
        alpha=0.85,
    )
    axis.set_title(title)
    axis.set_xlabel('grid_x')
    axis.set_ylabel('grid_y')
    color_bar = fig.colorbar(scatter, ax=axis)
    color_bar.set_label('baseline - csp')
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


# 绘制三维节点空间散点图。
def plot_spatial_delta_3d(coords, delta_values, output_path, title, elev=24.0, azim=38.0):
    """按节点三维坐标绘制改进热力散点图。"""
    fig = plt.figure(figsize=(8.4, 6.6))
    axis = fig.add_subplot(111, projection='3d')
    scatter = axis.scatter(
        coords[:, 0],
        coords[:, 1],
        coords[:, 2],
        c=delta_values,
        cmap='RdYlGn',
        s=16,
        alpha=0.9,
        depthshade=False,
    )
    axis.set_title(title)
    axis.set_xlabel('grid_x')
    axis.set_ylabel('grid_y')
    axis.set_zlabel('grid_z')
    axis.view_init(elev=elev, azim=azim)
    color_bar = fig.colorbar(scatter, ax=axis, pad=0.08, shrink=0.75)
    color_bar.set_label('baseline - csp')
    fig.tight_layout()
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


# 绘制某个节点的预测曲线对比图。
def plot_node_curve_compare(base_pred, csp_pred, true, node_index, output_path, title):
    """绘制指定节点在全测试样本上的预测与真值对比曲线。"""
    flat_true = true[:, :, node_index].reshape(-1)
    flat_base = base_pred[:, :, node_index].reshape(-1)
    flat_csp = csp_pred[:, :, node_index].reshape(-1)
    x_axis = np.arange(flat_true.shape[0])

    fig, axis = plt.subplots(figsize=(12, 4.6))
    axis.plot(x_axis, flat_true, label='GroundTruth', color='tab:orange', linewidth=1.6)
    axis.plot(x_axis, flat_base, label='Baseline', color='tab:blue', linewidth=1.2, alpha=0.85)
    axis.plot(x_axis, flat_csp, label='CSP', color='tab:green', linewidth=1.2, alpha=0.85)
    axis.set_title(title)
    axis.set_xlabel('flattened forecast index')
    axis.set_ylabel('displacement')
    axis.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


# 从改进排序里挑代表性节点。
def select_representative_nodes(delta_metric):
    """选择最优、居中、最差三个代表节点。"""
    ordered_idx = np.argsort(delta_metric)
    worst = int(ordered_idx[0])
    median = int(ordered_idx[len(ordered_idx) // 2])
    best = int(ordered_idx[-1])
    return {'best': best, 'median': median, 'worst': worst}


# 组装逐节点统计表。
def build_node_summary(base_mae, csp_mae, base_mse, csp_mse):
    """构建节点级对比统计表。"""
    delta_mae = base_mae - csp_mae
    delta_mse = base_mse - csp_mse
    mae_rate = compute_improve_rate(base_mae, csp_mae)
    mse_rate = compute_improve_rate(base_mse, csp_mse)
    summary = pd.DataFrame({
        'node_id': np.arange(base_mae.shape[0]),
        'mae_baseline': base_mae,
        'mae_csp': csp_mae,
        'mae_delta': delta_mae,
        'mae_improve_rate': mae_rate,
        'mse_baseline': base_mse,
        'mse_csp': csp_mse,
        'mse_delta': delta_mse,
        'mse_improve_rate': mse_rate,
    })
    return summary.sort_values(by='mae_delta', ascending=False).reset_index(drop=True)


# 主流程：读取结果、计算统计并输出图表与表格。
def run_report(args):
    """执行 CSP 生效可视化报告流程。"""
    results_root = args.results_root
    if args.baseline_dir:
        baseline_dir = args.baseline_dir
    else:
        baseline_prefix = f'long_term_forecast_baseline_long_pl{args.pred_len}_'
        baseline_dir = find_result_dir_by_prefix(results_root, baseline_prefix)

    if args.csp_dir:
        csp_dir = args.csp_dir
    else:
        csp_prefix = f'long_term_forecast_csp_long_pl{args.pred_len}_'
        csp_dir = find_result_dir_by_prefix(results_root, csp_prefix)

    base_pred, base_true = load_result_arrays(baseline_dir)
    csp_pred, csp_true = load_result_arrays(csp_dir)
    if base_pred.shape != csp_pred.shape:
        raise ValueError(f'baseline 与 csp 的 pred 形状不一致: {base_pred.shape} vs {csp_pred.shape}')
    if not np.allclose(base_true, csp_true):
        print('警告: baseline 与 csp 的 true.npy 不完全一致，报告仍将继续。')

    coords_frame = pd.read_csv(args.coords_path)
    coord_columns = ['grid_x', 'grid_y', 'grid_z']
    if not all(column in coords_frame.columns for column in coord_columns):
        raise ValueError(f'坐标文件缺少列 {coord_columns}: {args.coords_path}')
    coords = coords_frame[coord_columns].values
    if coords.shape[0] != base_pred.shape[-1]:
        raise ValueError(f'节点数不一致: coords={coords.shape[0]} vs pred_nodes={base_pred.shape[-1]}')

    base_mae, base_mse = compute_node_metrics(base_pred, base_true)
    csp_mae, csp_mse = compute_node_metrics(csp_pred, csp_true)
    summary = build_node_summary(base_mae, csp_mae, base_mse, csp_mse)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / 'node_metric_compare.csv', index=False)
    summary.head(args.top_k).to_csv(out_dir / 'top_improved_nodes.csv', index=False)
    summary.tail(args.top_k).to_csv(out_dir / 'top_worsened_nodes.csv', index=False)

    delta_mae = summary.set_index('node_id')['mae_delta'].reindex(np.arange(base_pred.shape[-1])).values
    delta_mse = summary.set_index('node_id')['mse_delta'].reindex(np.arange(base_pred.shape[-1])).values
    plot_spatial_delta(coords, delta_mae, out_dir / 'spatial_mae_delta.png', '节点 MAE 改进热力图 (baseline - csp)')
    plot_spatial_delta(coords, delta_mse, out_dir / 'spatial_mse_delta.png', '节点 MSE 改进热力图 (baseline - csp)')
    if args.plot_3d:
        plot_spatial_delta_3d(
            coords,
            delta_mae,
            out_dir / 'spatial_mae_delta_3d.png',
            '节点 MAE 改进三维热力图 (baseline - csp)',
            elev=args.view_elev,
            azim=args.view_azim,
        )
        plot_spatial_delta_3d(
            coords,
            delta_mse,
            out_dir / 'spatial_mse_delta_3d.png',
            '节点 MSE 改进三维热力图 (baseline - csp)',
            elev=args.view_elev,
            azim=args.view_azim,
        )

    representatives = select_representative_nodes(delta_mae)
    for tag, node_id in representatives.items():
        node_row = summary[summary['node_id'] == node_id].iloc[0]
        title = (
            f'节点 {node_id} | {tag} | '
            f'MAE: {node_row.mae_baseline:.4f}->{node_row.mae_csp:.4f} | '
            f'MSE: {node_row.mse_baseline:.4f}->{node_row.mse_csp:.4f}'
        )
        plot_node_curve_compare(
            base_pred,
            csp_pred,
            base_true,
            node_id,
            out_dir / f'curve_{tag}_node_{node_id}.png',
            title,
        )

    global_base_mae = float(np.abs(base_pred - base_true).mean())
    global_csp_mae = float(np.abs(csp_pred - csp_true).mean())
    global_base_mse = float(((base_pred - base_true) ** 2).mean())
    global_csp_mse = float(((csp_pred - csp_true) ** 2).mean())
    improved_ratio = float((delta_mae > 0).mean())

    print('================ CSP 生效可视化报告 ================')
    print(f'baseline_dir: {baseline_dir}')
    print(f'csp_dir: {csp_dir}')
    print(f'output_dir: {out_dir}')
    print(f'全局 MAE: {global_base_mae:.6f} -> {global_csp_mae:.6f} (delta={global_base_mae - global_csp_mae:.6f})')
    print(f'全局 MSE: {global_base_mse:.6f} -> {global_csp_mse:.6f} (delta={global_base_mse - global_csp_mse:.6f})')
    print(f'MAE 改善节点占比: {improved_ratio:.4%}')
    print(f'代表节点: best={representatives["best"]}, median={representatives["median"]}, worst={representatives["worst"]}')


# 构建命令行参数。
def build_parser():
    """定义脚本参数。"""
    parser = argparse.ArgumentParser(description='CSPAdapter 生效可视化报告')
    parser.add_argument('--results_root', type=str, default='./results', help='结果根目录')
    parser.add_argument('--baseline_dir', type=str, default='', help='baseline 结果目录，留空时按 pred_len 自动匹配')
    parser.add_argument('--csp_dir', type=str, default='', help='csp 结果目录，留空时按 pred_len 自动匹配')
    parser.add_argument('--pred_len', type=int, default=12, help='自动匹配目录时使用的预测步长')
    parser.add_argument('--coords_path', type=str, default='./dataset/radar/sim_nodes_static.csv', help='节点坐标文件路径')
    parser.add_argument('--output_dir', type=str, default='./outputs/csp_effect_report', help='报告输出目录')
    parser.add_argument('--top_k', type=int, default=20, help='导出前后 k 个节点')
    parser.add_argument('--plot_3d', type=int, default=1, help='是否输出三维空间热力图，1=输出，0=关闭')
    parser.add_argument('--view_elev', type=float, default=24.0, help='三维图观察仰角')
    parser.add_argument('--view_azim', type=float, default=38.0, help='三维图观察方位角')
    return parser


if __name__ == '__main__':
    arguments = build_parser().parse_args()
    run_report(arguments)
