import pandas as pd
import numpy as np
from collections import defaultdict
import os
import warnings
warnings.filterwarnings('ignore')

class MinimalFillPEMSConverter:
    """
    最小化填充的PEMS格式转换器
    - 仅保留有完整数据的时间点和传感器
    - 避免过度插值填充
    - 严格按时间对齐
    """
    
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.df['report_time'] = pd.to_datetime(self.df['report_time'])
        self.df = self.df.sort_values('report_time').reset_index(drop=True)
        
        # 存储对齐后的信息
        self.valid_times = None
        self.valid_sensors = None
        self.time_to_idx = {}
        self.sensor_to_idx = {}
        self.T = None  # 对齐后的时间步数
        self.N = None  # 对齐后的传感器数
        self.C = 3     # 特征数
        
        # 归一化参数
        self.mean = None
        self.std = None
        
    def align_data(self, min_time_coverage=0.5, min_sensor_coverage=0.1):
        """
        数据对齐：找到满足覆盖率要求的时间点和传感器
        """
        print("开始数据对齐...")
        
        # 统计每个时间点的数据覆盖情况
        time_counts = self.df.groupby('report_time').size()
        total_possible_records = len(self.df['grid_x'].unique()) * len(self.df['grid_y'].unique())
        
        # 找到满足时间覆盖要求的时间点
        valid_times = time_counts[time_counts >= total_possible_records * min_sensor_coverage].index
        print(f"原始时间点数: {len(time_counts)}, 满足覆盖要求的时间点数: {len(valid_times)}")
        
        # 统计每个传感器的可用时间点数
        sensor_time_counts = self.df[self.df['report_time'].isin(valid_times)].groupby(['grid_x', 'grid_y']).size()
        min_required_times = len(valid_times) * min_time_coverage
        
        # 找到满足传感器覆盖要求的传感器
        valid_sensors = sensor_time_counts[sensor_time_counts >= min_required_times]
        print(f"满足覆盖要求的传感器数: {len(valid_sensors)} (总可能数: {total_possible_records})")
        
        # 选择最活跃的传感器（如果有太多的话）
        if len(valid_sensors) > 50000:  # 限制传感器数量以避免内存问题
            valid_sensors = valid_sensors.nlargest(50000)
            print(f"限制传感器数量至: {len(valid_sensors)}")
        
        self.valid_times = sorted(valid_times)
        self.valid_sensors = [(row['grid_x'], row['grid_y']) for _, row in valid_sensors.reset_index().iterrows()]
        
        # 创建映射
        self.time_to_idx = {t: i for i, t in enumerate(self.valid_times)}
        self.sensor_to_idx = {sensor: i for i, sensor in enumerate(self.valid_sensors)}
        
        self.T = len(self.valid_times)
        self.N = len(self.valid_sensors)
        
        print(f"对齐完成: T={self.T}, N={self.N}, C=3")
        print(f"时间范围: {self.valid_times[0]} 到 {self.valid_times[-1]}")
        
        return self
    
    def create_aligned_tensor(self):
        """
        创建对齐后的张量，仅填充确实缺失的数据点
        """
        print("创建对齐张量...")
        
        # 初始化张量 (T, N, C)
        self.data_raw = np.full((self.T, self.N, 3), np.nan, dtype=np.float32)
        
        # 创建一个临时字典存储所有数据
        temp_data = {}
        
        # 只处理有效的传感器和时间点
        filtered_df = self.df[
            (self.df['report_time'].isin(self.valid_times)) & 
            (self.df.set_index(['grid_x', 'grid_y']).index.isin(self.valid_sensors))
        ]
        
        print(f"有效数据记录数: {len(filtered_df)}")
        
        # 填充数据
        filled_count = 0
        for _, row in filtered_df.iterrows():
            if (row['grid_x'], row['grid_y']) in self.sensor_to_idx and \
               row['report_time'] in self.time_to_idx:
                
                t_idx = self.time_to_idx[row['report_time']]
                s_idx = self.sensor_to_idx[(row['grid_x'], row['grid_y'])]
                
                self.data_raw[t_idx, s_idx, 0] = row['deformation']
                self.data_raw[t_idx, s_idx, 1] = row['speed']
                self.data_raw[t_idx, s_idx, 2] = row['acceleration']
                filled_count += 1
        
        print(f"填充了 {filled_count} 个数据点")
        
        # 对于仍然缺失的数据点（真正缺失的），进行简单填充
        # 只对有足够相邻数据的点进行线性插值
        print("对剩余缺失值进行最小化填充...")
        remaining_missing = np.isnan(self.data_raw).sum()
        print(f"填充前剩余缺失值: {remaining_missing}")
        
        # 对每个传感器的时间序列进行简单填充
        for n in range(self.N):
            for c in range(3):
                time_series = self.data_raw[:, n, c]
                
                # 找到缺失值的位置
                missing_mask = np.isnan(time_series)
                if not missing_mask.any():
                    continue
                
                # 使用前向和后向填充
                non_missing_mask = ~missing_mask
                if non_missing_mask.sum() > 0:  # 至少有一个非缺失值
                    # 前向填充
                    time_series_ffill = pd.Series(time_series).fillna(method='ffill').values
                    # 后向填充
                    time_series_bfill = pd.Series(time_series_ffill).fillna(method='bfill').values
                    self.data_raw[:, n, c] = time_series_bfill
        
        final_missing = np.isnan(self.data_raw).sum()
        print(f"填充后剩余缺失值: {final_missing}")
        print(f"总共填充了 {remaining_missing - final_missing} 个缺失值")
        
        return self
    
    def normalize_data(self, train_ratio=0.7, val_ratio=0.15):
        """数据归一化"""
        print("开始数据归一化...")
        
        # 划分训练集
        train_end = int(self.T * train_ratio)
        
        # 计算每个特征的全局均值和标准差（使用训练集）
        train_data = self.data_raw[:train_end]
        self.mean = np.full((1, 1, 3), np.nan, dtype=np.float32)
        self.std = np.full((1, 1, 3), np.nan, dtype=np.float32)
        
        for c in range(3):
            feature_data = train_data[:, :, c]  # (T_train, N)
            valid_values = feature_data[~np.isnan(feature_data)]
            
            if len(valid_values) > 0:
                self.mean[0, 0, c] = np.mean(valid_values)
                self.std[0, 0, c] = np.std(valid_values)
                
                if self.std[0, 0, c] == 0:
                    self.std[0, 0, c] = 1.0
            else:
                print(f"警告: 特征 {c} 在训练集中没有有效值！")
                self.mean[0, 0, c] = 0.0
                self.std[0, 0, c] = 1.0
        
        # 应用归一化
        self.data_normalized = (self.data_raw - self.mean) / self.std
        
        print(f"归一化完成，均值: {self.mean.flatten()}, 标准差: {self.std.flatten()}")
        return self
    
    def split_dataset(self, train_ratio=0.7, val_ratio=0.15):
        """划分数据集"""
        train_end = int(self.T * train_ratio)
        val_end = train_end + int(self.T * val_ratio)
        
        self.train_data = self.data_normalized[:train_end]
        self.val_data = self.data_normalized[train_end:val_end]
        self.test_data = self.data_normalized[val_end:]
        
        print(f"数据集划分:")
        print(f"  训练集: {self.train_data.shape}")
        print(f"  验证集: {self.val_data.shape}")
        print(f"  测试集: {self.test_data.shape}")
        
        return self
    
    def save_as_pems_format(self, output_file):
        """保存为PEMS格式(.npz文件)"""
        # 创建PEMS格式的数据字典
        pems_data = {
            'data': self.data_normalized,  # (T, N, C)
            'train': self.train_data,      # (T_train, N, C)
            'val': self.val_data,          # (T_val, N, C)  
            'test': self.test_data,        # (T_test, N, C)
            'mean': self.mean,             # (1, 1, C)
            'std': self.std,               # (1, 1, C)
            'valid_times': self.valid_times,  # 原始时间戳
            'valid_sensors': self.valid_sensors  # 原始传感器坐标
        }
        
        # 保存为npz格式
        np.savez_compressed(output_file, **pems_data)
        
        print(f"PEMS格式数据已保存到: {output_file}")
        print(f"PEMS数据形状: {self.data_normalized.shape}")
        return self
    
    def save_metadata(self, output_dir):
        """保存元数据"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存传感器位置
        sensor_locations = []
        for i, (x, y) in enumerate(self.valid_sensors):
            sensor_locations.append([i, x, y])  # [sensor_idx, grid_x, grid_y]
        
        sensor_locations = np.array(sensor_locations)
        np.save(os.path.join(output_dir, 'sensor_locations.npy'), sensor_locations)
        
        # 保存时间信息
        time_info = {
            'time_indices': list(range(len(self.valid_times))),
            'timestamps': self.valid_times
        }
        np.save(os.path.join(output_dir, 'time_info.npy'), time_info)
        
        print(f"元数据已保存到: {output_dir}")
        return self
    
    def get_summary(self):
        """获取数据摘要"""
        total_possible = self.T * self.N * self.C
        actual_data = (~np.isnan(self.data_raw)).sum()
        
        summary = {
            'original_records': len(self.df),
            'aligned_shape': self.data_raw.shape,
            'time_steps': self.T,
            'sensors': self.N,
            'features': self.C,
            'total_possible_values': total_possible,
            'actual_values': int(actual_data),
            'data_density': round(actual_data / total_possible * 100, 2),
            'value_ranges': {
                'deformation': [
                    float(np.nanmin(self.data_raw[:, :, 0])),
                    float(np.nanmax(self.data_raw[:, :, 0]))
                ],
                'speed': [
                    float(np.nanmin(self.data_raw[:, :, 1])),
                    float(np.nanmax(self.data_raw[:, :, 1]))
                ],
                'acceleration': [
                    float(np.nanmin(self.data_raw[:, :, 2])),
                    float(np.nanmax(self.data_raw[:, :, 2]))
                ]
            },
            'time_coverage': f"{len(self.valid_times)}/{len(pd.date_range(start=self.df['report_time'].min(), end=self.df['report_time'].max(), freq='H'))}",
            'sensor_coverage': f"{self.N}/{len(self.df[['grid_x', 'grid_y']].drop_duplicates())}",
            'data_type': 'slope_monitoring_minimal_fill',
            'format': 'PEMS_style'
        }
        return summary

def convert_to_minimal_pems(csv_path, output_file, output_dir, 
                           min_time_coverage=0.5, min_sensor_coverage=0.1):
    """
    完整的最小化填充转换流程
    """
    converter = MinimalFillPEMSConverter(csv_path)
    
    # 执行完整转换流程
    result = (converter
              .align_data(min_time_coverage=min_time_coverage, 
                         min_sensor_coverage=min_sensor_coverage)
              .create_aligned_tensor()
              .normalize_data()
              .split_dataset()
              .save_as_pems_format(output_file)
              .save_metadata(output_dir))
    
    # 打印摘要
    summary = result.get_summary()
    print("\n=== 最小化填充PEMS格式转换摘要 ===")
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    # 验证保存的文件
    loaded_data = np.load(output_file)
    print(f"\n验证加载的PEMS数据:")
    print(f"  主数据形状: {loaded_data['data'].shape}")
    print(f"  训练集形状: {loaded_data['train'].shape}")
    print(f"  验证集形状: {loaded_data['val'].shape}")
    print(f"  测试集形状: {loaded_data['test'].shape}")
    print(f"  均值形状: {loaded_data['mean'].shape}")
    print(f"  标准差形状: {loaded_data['std'].shape}")
    print(f"  有效时间点数: {len(loaded_data['valid_times'])}")
    print(f"  有效传感器数: {len(loaded_data['valid_sensors'])}")
    
    return converter

# 使用示例（请替换实际的CSV路径）
if __name__ == "__main__":
    converter = convert_to_minimal_pems(
        r'/root/autodl-tmp/device_10001_sorted.csv', 
        'slope_pems_minimal.npz',
        'output_directory',
        min_time_coverage=0.3,  # 每个传感器至少需要30%的时间点有数据
        min_sensor_coverage=0.1  # 每个时间点至少需要10%的传感器有数据
    )
