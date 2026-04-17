import numpy as np
import pandas as pd

np.random.seed(42)

# 生成 1024 个点的空间坐标
N_points = 1024
# 10×10 网格，每个网格中心坐标
grid_x = np.random.uniform(0, 296, N_points)  # 对应您的 grid_x 范围
grid_y = np.random.uniform(0, 229, N_points)  # 对应您的 grid_y 范围

# 分配区域标签
def assign_zone(x, y):
    # 稳定背景区：x<88, y<68
    if x < 88 and y < 68:
        return 'stable'
    # 坡脚区：x<88, y>161
    elif x < 88 and y > 161:
        return 'toe'
    # 坡顶区：x>208, y>161
    elif x > 208 and y > 161:
        return 'crest'
    # 平台区：x>208, y<68
    elif x > 208 and y < 68:
        return 'platform'
    # 坡中区：其余大部分
    else:
        # 局部活跃区：在坡中区内特定小范围
        if 118 < x < 178 and 88 < y < 138:
            return 'active'
        return 'middle'

zones = np.array([assign_zone(x, y) for x, y in zip(grid_x, grid_y)])

# 统计各区域点数
from collections import Counter
print(Counter(zones))

# 时间范围：2024-04-01 00:00 至 2024-10-01 23:00，小时级
date_rng = pd.date_range(start='2024-04-01', end='2024-10-01 23:00', freq='h')
T = len(date_rng)  # 约 4392 个小时（6个月×30.5天×24小时）

def generate_rainfall(t, T, date_rng):
    """生成小时级降雨量 (mm/h)"""
    rain = np.zeros(T)
    
    # 1. 背景小雨（随机发生，强度低）
    for i in range(T):
        if np.random.random() < 0.05:  # 5% 概率下雨
            rain[i] = np.random.exponential(0.5)  # 平均 0.5 mm/h
    
    # 2. 极端暴雨事件（集中在雨季和耦合期）
    extreme_dates = [
        ('2024-05-15', '2024-05-17', 15.0),  # 3天暴雨，峰值 15 mm/h
        ('2024-06-10', '2024-06-12', 20.0),
        ('2024-08-05', '2024-08-08', 25.0),  # 耦合期极端暴雨
        ('2024-08-20', '2024-08-22', 18.0),
    ]
    
    for start_str, end_str, peak in extreme_dates:
        start = pd.to_datetime(start_str)
        end = pd.to_datetime(end_str)
        mask = (date_rng >= start) & (date_rng <= end)
        idx = np.where(mask)[0]
        # 钟形曲线模拟降雨过程
        hours = len(idx)
        envelope = np.exp(-((np.linspace(-2, 2, hours))**2))  # 高斯包络
        rain[idx] = peak * envelope / envelope.max()
    
    # 添加小幅随机波动
    rain = np.maximum(0, rain + np.random.normal(0, 0.1, T))
    return rain

def generate_blast_log(T, date_rng):
    """生成爆破日志 DataFrame"""
    blast_events = []
    
    # 爆破施工期：2024-07-01 至 2024-07-31，每日两次
    blast_period = (date_rng >= '2024-07-01') & (date_rng <= '2024-07-31')
    for day in pd.date_range('2024-07-01', '2024-07-31', freq='D'):
        # 上午 10:00
        t1 = day + pd.Timedelta(hours=10)
        # 下午 15:00
        t2 = day + pd.Timedelta(hours=15)
        for t in [t1, t2]:
            if t in date_rng:
                blast_events.append({
                    'timestamp': t,
                    'charge_kg': np.random.uniform(50, 200),  # 装药量 50-200 kg
                    'distance_m': np.random.uniform(50, 500),  # 爆心距 50-500 m
                    'location': f'zone_{np.random.choice(["A","B","C"])}'
                })
    
    # 耦合期：少量爆破 + 降雨
    coupling_period = (date_rng >= '2024-08-01') & (date_rng <= '2024-08-15')
    for day in pd.date_range('2024-08-01', '2024-08-15', freq='D'):
        if np.random.random() < 0.3:  # 30% 概率有爆破
            t = day + pd.Timedelta(hours=np.random.choice([10, 15]))
            if t in date_rng:
                blast_events.append({
                    'timestamp': t,
                    'charge_kg': np.random.uniform(30, 150),
                    'distance_m': np.random.uniform(50, 500),
                    'location': f'zone_{np.random.choice(["A","B","C"])}'
                })
    
    return pd.DataFrame(blast_events)

def generate_temperature_humidity(T, date_rng, rain):
    """生成温度和相对湿度"""
    # 基础温度：季节趋势（4月到10月先升后降）
    days = (date_rng - date_rng[0]).days
    seasonal = 15 + 10 * np.sin(2 * np.pi * days / 365 - np.pi/2)  # 15~25°C
    
    # 日周期：白天高、夜晚低，幅度 8°C
    hour_of_day = date_rng.hour
    diurnal = 4 * np.sin(2 * np.pi * (hour_of_day - 14) / 24)
    
    temp = seasonal + diurnal + np.random.normal(0, 1, T)
    
    # 湿度：基础值 + 降雨时升高
    base_humidity = 50 + 20 * np.sin(2 * np.pi * days / 365)  # 50~70%
    # 降雨时湿度增加
    rain_effect = np.minimum(30, rain * 2)  # 降雨时额外增加，上限 30%
    humidity = base_humidity + rain_effect + np.random.normal(0, 5, T)
    humidity = np.clip(humidity, 20, 100)
    
    return temp, humidity

def compute_effective_rainfall(rain, half_life=24):
    """计算有效降雨量（考虑滞后效应）"""
    T = len(rain)
    weights = np.exp(-np.arange(72) / half_life)  # 72小时窗口
    weights = weights / weights.sum()
    
    effective = np.convolve(rain, weights, mode='same')
    return effective

def rain_displacement(effective_rain, alpha_rain, zone):
    """降雨引起的位移分量"""
    # 位移与有效降雨量成非线性关系（幂律）
    base_disp = alpha_rain * (effective_rain ** 0.8) * 0.5  # mm
    
    # 局部活跃区有额外的滞后放大效应
    if zone == 'active':
        # 累积超过阈值后加速
        cumulative = np.cumsum(effective_rain)
        threshold = 100  # 累积降雨阈值
        amplification = np.where(cumulative > threshold, 1.5, 1.0)
        base_disp = base_disp * amplification
    
    return base_disp

def blast_displacement(blast_df, T, date_rng, alpha_blast, zone):
    """爆破引起的位移分量"""
    blast_disp = np.zeros(T)
    
    for _, row in blast_df.iterrows():
        t0 = row['timestamp']
        idx0 = date_rng.get_loc(t0)
        E = row['charge_kg']
        d = row['distance_m']
        
        # 瞬时峰值位移
        peak = alpha_blast * np.sqrt(E) / (d / 100) * 0.2  # mm
        
        # 指数衰减（持续 24 小时）
        decay_hours = 24
        for dt in range(decay_hours):
            if idx0 + dt < T:
                decay = np.exp(-dt / 6)  # 衰减常数 6 小时
                blast_disp[idx0 + dt] += peak * decay
    
    return blast_disp

def creep_displacement(T, creep_rate, zone):
    """长期蠕变位移（线性累积 + 随机波动）"""
    # 基础线性趋势
    days = np.arange(T) / 24
    creep = creep_rate * days
    
    # 添加随机游走（模拟蠕变的非平稳性）
    random_walk = np.cumsum(np.random.normal(0, 0.002, T))
    
    return creep + random_walk

def compute_coupling_effect(rain_disp, blast_disp, zone):
    """计算耦合增强效应"""
    if zone in ['active', 'middle']:
        beta = 1.2  # 耦合系数
        coupling = beta * np.sqrt(np.maximum(rain_disp, 0) * np.maximum(blast_disp, 0))
        return coupling
    else:
        return 0
def generate_displacement_for_point(T, date_rng, zone, rain, effective_rain, blast_df, coef):
    """为单个监测点生成位移序列"""
    # 1. 各分量计算
    rain_disp = rain_displacement(effective_rain, coef['alpha_rain'], zone)
    blast_disp = blast_displacement(blast_df, T, date_rng, coef['alpha_blast'], zone)
    creep_disp = creep_displacement(T, coef['creep_rate'], zone)
    
    # 2. 耦合增强
    coupling = compute_coupling_effect(rain_disp, blast_disp, zone)
    
    # 3. 总位移
    total_disp = rain_disp + blast_disp + creep_disp + coupling
    
    # 4. 添加测量噪声（雷达精度约 0.1 mm）
    noise = np.random.normal(0, 0.05, T)
    total_disp = np.maximum(0, total_disp + noise)  # 位移非负
    
    return total_disp, rain_disp, blast_disp, creep_disp

import numpy as np
import pandas as pd
from tqdm import tqdm

# ========== 1. 初始化 ==========
np.random.seed(42)
date_rng = pd.date_range('2024-04-01', '2024-10-01 23:00', freq='h')
T = len(date_rng)

# ========== 2. 生成外部事件 ==========
rain = generate_rainfall(T, date_rng)
effective_rain = compute_effective_rainfall(rain)
blast_df = generate_blast_log(T, date_rng)
temp, humidity = generate_temperature_humidity(T, date_rng, rain)

# ========== 3. 生成监测点坐标和区域 ==========
N_points = 1024
grid_x = np.random.uniform(0, 296, N_points)
grid_y = np.random.uniform(0, 229, N_points)
zones = np.array([assign_zone(x, y) for x, y in zip(grid_x, grid_y)])

# ========== 4. 区域系数表 ==========
zone_coef = {
    'stable':   {'alpha_rain': 0.05, 'alpha_blast': 0.02, 'creep_rate': 0.001/24},
    'toe':      {'alpha_rain': 0.6,  'alpha_blast': 0.3,  'creep_rate': 0.005/24},
    'middle':   {'alpha_rain': 1.0,  'alpha_blast': 0.8,  'creep_rate': 0.01/24},
    'crest':    {'alpha_rain': 0.4,  'alpha_blast': 1.2,  'creep_rate': 0.008/24},
    'platform': {'alpha_rain': 0.2,  'alpha_blast': 0.1,  'creep_rate': 0.002/24},
    'active':   {'alpha_rain': 2.5,  'alpha_blast': 1.5,  'creep_rate': 0.03/24},
}

# ========== 5. 逐点生成位移 ==========
displacements = np.zeros((N_points, T))
rain_components = np.zeros((N_points, T))
blast_components = np.zeros((N_points, T))

for i in tqdm(range(N_points)):
    zone = zones[i]
    coef = zone_coef[zone]
    disp, r_comp, b_comp, c_comp = generate_displacement_for_point(
        T, date_rng, zone, rain, effective_rain, blast_df, coef
    )
    displacements[i, :] = disp
    rain_components[i, :] = r_comp
    blast_components[i, :] = b_comp

# ========== 6. 组织输出数据 ==========
# 转换为 DataFrame（长格式）
data_records = []
for i in range(N_points):
    for t in range(T):
        data_records.append({
            'timestamp': date_rng[t],
            'point_id': i,
            'grid_x': grid_x[i],
            'grid_y': grid_y[i],
            'zone': zones[i],
            'displacement': displacements[i, t],
            'rainfall': rain[t],
            'temperature': temp[t],
            'humidity': humidity[t],
            'effective_rainfall': effective_rain[t],
        })

df_main = pd.DataFrame(data_records)

# ========== 7. 保存数据 ==========
df_main.to_csv('simulated_slope_monitoring.csv', index=False)
blast_df.to_csv('simulated_blast_log.csv', index=False)

print(f"数据生成完成！")
print(f"总记录数: {len(df_main)}")
print(f"监测点数: {N_points}")
print(f"时间跨度: {date_rng[0]} 至 {date_rng[-1]}")