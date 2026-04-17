# 这个脚本负责将雷达、天气、爆破数据分别处理并独立保存，避免跨源数据合并导致雷达特征被污染。

import os
import pandas as pd


def ensure_output_dir(output_dir):
    """确保输出目录存在。"""
    os.makedirs(output_dir, exist_ok=True)


def load_radar_data(input_path):
    """读取雷达位移数据并统一时间列名。"""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"雷达输入文件不存在: {input_path}")

    radar_df = pd.read_csv(input_path)
    if 'report_time' not in radar_df.columns:
        raise ValueError("雷达文件缺少 report_time 列，无法处理。")

    radar_df = radar_df.rename(columns={'report_time': 'date'})
    radar_df['date'] = pd.to_datetime(radar_df['date'])
    return radar_df


def build_pure_radar_frame(radar_df):
    """构建纯雷达数据，只保留时间列和1000个节点列。"""
    node_cols = sorted(
        [col for col in radar_df.columns if col.startswith('node_')],
        key=lambda x: int(x.split('_')[1])
    )
    if len(node_cols) != 1000:
        raise ValueError(f"雷达节点列数量异常，期望1000列，实际{len(node_cols)}列。")

    pure_radar_df = radar_df[['date'] + node_cols].copy()
    return pure_radar_df


def load_weather_data(input_path):
    """读取天气数据并统一时间列名。"""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"天气输入文件不存在: {input_path}")

    weather_df = pd.read_csv(input_path)
    if 'report_time' not in weather_df.columns:
        raise ValueError("天气文件缺少 report_time 列，无法处理。")

    weather_df = weather_df.rename(columns={'report_time': 'date'})
    weather_df['date'] = pd.to_datetime(weather_df['date'])
    return weather_df


def load_and_aggregate_blast_data(input_path):
    """读取爆破日志并按小时聚合强度。"""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"爆破输入文件不存在: {input_path}")

    blast_df = pd.read_csv(input_path)
    if 'timestamp' not in blast_df.columns or 'intensity' not in blast_df.columns:
        raise ValueError("爆破文件缺少 timestamp 或 intensity 列，无法处理。")

    blast_df = blast_df.rename(columns={'timestamp': 'date'})
    blast_df['date'] = pd.to_datetime(blast_df['date'])
    blast_agg = blast_df.groupby('date', as_index=False)['intensity'].sum()
    blast_agg = blast_agg.rename(columns={'intensity': 'blast_intensity'})
    return blast_agg


def save_outputs(radar_df, weather_df, blast_df, output_dir):
    """将三类数据分别保存为独立CSV文件。"""
    radar_out = os.path.join(output_dir, 'radar.csv')
    weather_out = os.path.join(output_dir, 'weather.csv')
    blast_out = os.path.join(output_dir, 'blast.csv')

    radar_df.to_csv(radar_out, index=False)
    weather_df.to_csv(weather_out, index=False)
    blast_df.to_csv(blast_out, index=False)

    return radar_out, weather_out, blast_out


def main():
    """执行分离式数据处理主流程。"""
    output_dir = 'dataset/radar_sim'
    ensure_output_dir(output_dir)

    print("1. 加载原始数据...")
    radar_df = load_radar_data('sim_radar_hourly_displacement.csv')
    weather_df = load_weather_data('sim_weather.csv')
    blast_df = load_and_aggregate_blast_data('sim_blast_logs.csv')

    print("2. 生成纯雷达数据（仅时间列 + 1000节点列）...")
    pure_radar_df = build_pure_radar_frame(radar_df)

    print("3. 分别保存雷达/天气/爆破文件（不做merge）...")
    radar_out, weather_out, blast_out = save_outputs(
        pure_radar_df, weather_df, blast_df, output_dir
    )

    print("数据处理完毕！")
    print(f"- 雷达数据: {radar_out}，样本数={len(pure_radar_df)}，列数={len(pure_radar_df.columns)}")
    print(f"- 天气数据: {weather_out}，样本数={len(weather_df)}，列数={len(weather_df.columns)}")
    print(f"- 爆破数据: {blast_out}，样本数={len(blast_df)}，列数={len(blast_df.columns)}")


if __name__ == '__main__':
    main()
