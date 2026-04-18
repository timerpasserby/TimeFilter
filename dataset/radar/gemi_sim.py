# 这个脚本负责生成雷达边坡模拟数据、三维节点坐标和爆破日志，供 TimeFilter 等模型训练使用。

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def simulate_mine_slope_data(num_points=1000, start_date='2024-04-01', end_date='2024-10-01'):
    """生成包含空间拓扑、滞后效应和非线性耦合的边坡模拟数据。"""
    print("开始生成时间轴...")
    time_index = pd.date_range(start=start_date, end=end_date, freq='h')
    n_steps = len(time_index)

    # 1. 空间坐标与区域划分
    print("构建空间拓扑与活跃区...")
    np.random.seed(42)
    coords_x = np.random.uniform(0, 500, num_points)
    coords_y = np.random.uniform(0, 500, num_points)

    # 生成三维地形：西高东低的单面高陡边坡
    slope_angle = np.radians(35)
    coords_z = 300 - coords_x * np.tan(slope_angle) + np.random.normal(0, 3.0, num_points)
    coords_z = np.clip(coords_z, a_min=0, a_max=None)

    active_center = np.array([250, 250])
    active_radius = 110
    distances = np.sqrt((coords_x - active_center[0])**2 + (coords_y - active_center[1])**2)
    sensitivity = np.exp(-0.5 * (distances / (active_radius / 2))**2)
    # 给非活跃区保留基础响应，避免天气和爆破只在极少数点位出现。
    geo_transfer = 0.25 + 0.75 * sensitivity
    base_creep = 0.03 + 0.22 * geo_transfer

    # 2. 生成气象数据
    print("生成气象动力学特征...")
    days_from_start = np.arange(n_steps) / 24.0
    seasonal_temp = 15 + 15 * np.sin(np.pi * days_from_start / 180)
    daily_temp = 5 * np.sin(2 * np.pi * days_from_start - np.pi / 2)
    temperature = seasonal_temp + daily_temp + np.random.normal(0, 1, n_steps)

    humidity = np.clip(80 - 1.5 * daily_temp + np.random.normal(0, 5, n_steps), 20, 100)

    rain_prob = 0.02 + 0.05 * np.exp(-0.5 * ((days_from_start - 90) / 30) ** 2)
    rainfall = np.zeros(n_steps)
    for i in range(n_steps):
        if np.random.rand() < rain_prob[i]:
            rainfall[i] = np.random.exponential(scale=5)
            if i + 1 < n_steps:
                rain_prob[i + 1] += 0.3

    weather_df = pd.DataFrame({
        'report_time': time_index,
        'temperature': temperature,
        'humidity': humidity,
        'rainfall': rainfall
    })

    # 3. 生成爆破日志 (离散事件) - 三维物理修正版
    print("生成爆破事件日志(受控于三维地形)...")
    n_blasts = int((n_steps / 24) / 7 * 2)
    blast_indices = np.random.choice(np.arange(24, n_steps - 24), n_blasts, replace=False)
    blast_indices.sort()

    blast_logs = []
    blast_intensity_series = np.zeros(n_steps)

    # 必须与生成监测点时的地形参数保持绝对一致
    slope_angle = np.radians(35)
    base_elevation = 300

    for idx in blast_indices:
        intensity = np.random.randint(1, 6)
        blast_intensity_series[idx] = intensity

        # 1. 生成平面的 X, Y 坐标
        blast_x = np.random.uniform(100, 400)
        blast_y = np.random.uniform(100, 400)

        # 2. 地貌方程约束：计算该 (X, Y) 对应的地表高程 Z
        surface_z = base_elevation - blast_x * np.tan(slope_angle)

        # 3. 物理震源深度模拟：爆破点位于地表以下 5 到 15 米的炮孔内
        hole_depth = np.random.uniform(5.0, 15.0)
        blast_z = np.clip(surface_z - hole_depth, a_min=0, a_max=None)

        blast_logs.append({
            'timestamp': time_index[idx],
            'location_x': blast_x,
            'location_y': blast_y,
            'location_z': blast_z,
            'intensity': intensity,
            'type': np.random.choice(['预裂爆破', '抛掷爆破', '松动爆破'])
        })

    blast_df = pd.DataFrame(blast_logs)

    # 4. 生成位移特征
    print("计算时空动力学响应与位移张量...")
    displacement_matrix = np.zeros((n_steps, num_points))

    rain_effect_kernel = np.exp(-np.arange(72) / 24)
    rain_cumulative = np.convolve(rainfall, rain_effect_kernel, mode='full')[:n_steps]

    T_f = 4000
    acceleration_start = T_f - 240
    C_param = 2.0

    label_matrix = np.zeros(n_steps, dtype=np.int8)
    ttf_matrix = np.full(n_steps, -1, dtype=np.int32)

    for t in range(1, n_steps):
        noise = np.random.normal(0, 0.01, num_points)

        if t >= T_f:
            stage_label = 2
            ttf_label = -1
            creep = np.zeros(num_points)
        elif t > acceleration_start:
            stage_label = 1
            ttf_label = T_f - t
            accelerated_vel = C_param / (T_f - t + 0.1)
            creep = base_creep + 1.8 * accelerated_vel * geo_transfer
        else:
            stage_label = 0
            ttf_label = -1
            creep = base_creep * (1 + 0.1 * np.random.randn())

        label_matrix[t] = stage_label
        ttf_matrix[t] = ttf_label

        # 连续降雨和高湿度会显著降低边坡抗剪强度，因此同时考虑累积降雨、当前降雨和湿度软化。
        humidity_softening = np.clip((humidity[t] - 70.0) / 20.0, 0.0, 1.5)
        wet_threshold_boost = 1.0
        if rain_cumulative[t] > 20:
            wet_threshold_boost += 0.6
        if rain_cumulative[t] > 40:
            wet_threshold_boost += 0.8

        rain_response = (
            0.12 * rain_cumulative[t] * geo_transfer +
            0.18 * rainfall[t] * (0.3 + geo_transfer) +
            0.08 * humidity_softening * geo_transfer
        ) * wet_threshold_boost

        blast_response = np.zeros(num_points)
        if blast_intensity_series[t] > 0:
            intensity = blast_intensity_series[t]
            blast_event = blast_logs[np.where(blast_indices == t)[0][0]]
            blast_x = blast_event['location_x']
            blast_y = blast_event['location_y']
            blast_z = blast_event.get('location_z', 0.0)
            blast_dist = np.sqrt(
                (coords_x - blast_x) ** 2 +
                (coords_y - blast_y) ** 2 +
                (coords_z - blast_z) ** 2
            )

            # 降低距离衰减速度并放大基础传播项，让非核心区也能看见爆破扰动。
            blast_impact = intensity * 2.2 * np.exp(-blast_dist / 220) * (0.25 + geo_transfer)
            coupling_factor = 1.0 + 0.10 * rain_cumulative[t] + 0.35 * humidity_softening
            blast_response = blast_impact * coupling_factor

        if t > 5:
            recent_blasts = blast_intensity_series[t - 5:t]
            decay_response = np.sum(recent_blasts * np.array([0.05, 0.10, 0.18, 0.28, 0.45])) * (0.2 + geo_transfer)
        else:
            decay_response = 0

        displacement_matrix[t] = creep + rain_response + blast_response + decay_response + noise

    # 5. 格式化输出
    print("构建最终数据结构...")
    nodes_df = pd.DataFrame({
        'node_id': range(num_points),
        'grid_x': coords_x,
        'grid_y': coords_y,
        'grid_z': coords_z,
        'sensitivity': sensitivity,
    })
    nodes_df.to_csv('data/radar/sim_nodes_static.csv', index=False)

    radar_df = pd.DataFrame(displacement_matrix, columns=[f'node_{i}' for i in range(num_points)])
    radar_df.insert(0, 'report_time', time_index)
    radar_df.to_csv('data/radar/sim_radar_hourly_displacement.csv', index=False)
    weather_df.to_csv('data/radar/sim_weather.csv', index=False)
    blast_df.to_csv('data/radar/sim_blast_logs.csv', index=False)

    label_df = pd.DataFrame({
        'report_time': time_index,
        'stage_label': label_matrix,
        'ttf_hours': ttf_matrix,
    })
    label_df.to_csv('data/radar/sim_stage_labels.csv', index=False)

    print("仿真完成！已输出：\n1. sim_nodes_static.csv (节点坐标)\n2. sim_radar_hourly_displacement.csv (雷达当次位移矩阵)\n3. sim_weather.csv (气象连续特征)\n4. sim_blast_logs.csv (爆破离散日志)\n5. sim_stage_labels.csv (阶段标签与 TTF)\n6. sim_radar_features_full.csv (运动学特征)")

    append_kinematic_features(displacement_matrix, time_index, num_points)


def append_kinematic_features(displacement_matrix, time_index, num_points):
    """基于平滑微分的运动学特征计算模块。"""
    print("开始计算速度与加速度，执行降噪滤波...")

    n_steps, n_nodes = displacement_matrix.shape
    velocity_matrix = np.zeros_like(displacement_matrix)
    acceleration_matrix = np.zeros_like(displacement_matrix)

    for i in range(n_nodes):
        disp_series = displacement_matrix[:, i]
        smoothed_disp = savgol_filter(disp_series, window_length=11, polyorder=3)
        velocity = np.gradient(smoothed_disp)
        smoothed_vel = savgol_filter(velocity, window_length=7, polyorder=2)
        acceleration = np.gradient(smoothed_vel)

        velocity_matrix[:, i] = velocity
        acceleration_matrix[:, i] = acceleration

    print("构建特征多通道长表...")

    node_cols = [f'node_{i}' for i in range(num_points)]
    disp_df = pd.DataFrame(displacement_matrix, index=time_index, columns=node_cols)
    vel_df = pd.DataFrame(velocity_matrix, index=time_index, columns=node_cols)
    acc_df = pd.DataFrame(acceleration_matrix, index=time_index, columns=node_cols)

    disp_long = disp_df.stack().reset_index()
    vel_long = vel_df.stack().reset_index()
    acc_long = acc_df.stack().reset_index()

    disp_long.columns = ['report_time', 'node_id', 'displacement']
    disp_long['node_id'] = disp_long['node_id'].str.replace('node_', '', regex=False).astype(int)
    disp_long['speed'] = vel_long.iloc[:, 2].values
    disp_long['acceleration'] = acc_long.iloc[:, 2].values

    final_df = disp_long.sort_values(by=['report_time', 'node_id']).reset_index(drop=True)
    final_df.to_csv('data/radar/sim_radar_features_full.csv', index=False)
    print("生成完毕！已导出包含 [displacement, speed, acceleration] 的完整数据集 sim_radar_features_full.csv")

    return final_df


# 运行仿真器
simulate_mine_slope_data(num_points=1000)
