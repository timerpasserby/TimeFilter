import pandas as pd
import numpy as np
from scipy.interpolate import interp1d, griddata
from sklearn.preprocessing import StandardScaler
import os
import pickle

class SpatioTemporalDataProcessor:
    """
    时空网格数据预处理器，支持多种模型输入格式
    """
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.df['report_time'] = pd.to_datetime(self.df['report_time'])
        self.df = self.df.sort_values('report_time').reset_index(drop=True)
        
        # 存储网格信息
        self.xs = None
        self.ys = None
        self.times = None
        self.H = None
        self.W = None
        self.T = None
        
        # 归一化参数
        self.scaler = StandardScaler()
        self.mean = None
        self.std = None
        
    def clean_data(self, outlier_threshold=3):
        """
        数据清洗：处理缺失值、重复值和异常值
        """
        # 检查重复值
        duplicates = self.df.duplicated(subset=['grid_x', 'grid_y', 'report_time'], keep=False)
        if duplicates.any():
            print(f"发现{duplicates.sum()}个重复记录，按时间保留最新记录")
            self.df = self.df.sort_values(['grid_x', 'grid_y', 'report_time']).drop_duplicates(
                subset=['grid_x', 'grid_y', 'report_time'], keep='last'
            )
        
        # 按网格点分组处理异常值
        numeric_cols = ['deformation', 'speed', 'acceleration']
        
        def remove_outliers(group):
            for col in numeric_cols:
                z_scores = np.abs((group[col] - group[col].mean()) / (group[col].std() + 1e-8))
                group.loc[z_scores > outlier_threshold, col] = np.nan
            return group
            
        self.df = self.df.groupby(['grid_x', 'grid_y']).apply(remove_outliers).reset_index(drop=True)
        
        # 时间序列插值填补缺失值
        self.df[numeric_cols] = self.df.groupby(['grid_x', 'grid_y'])[numeric_cols].apply(
            lambda x: x.interpolate(method='linear', limit_direction='both')
        )
        
        print(f"数据清洗完成，剩余{len(self.df)}条记录")
        return self
    
    def build_spatial_grid(self):
        """
        构建时空网格
        """
        # 获取坐标列表
        self.xs = np.sort(self.df['grid_x'].unique())
        self.ys = np.sort(self.df['grid_y'].unique())
        self.H, self.W = len(self.ys), len(self.xs)
        
        # 获取时间列表
        self.times = np.sort(self.df['report_time'].unique())
        self.T = len(self.times)
        
        print(f"构建时空网格: H={self.H}, W={self.W}, T={self.T}")
        print(f"X坐标范围: [{self.xs.min():.2f}, {self.xs.max():.2f}]")
        print(f"Y坐标范围: [{self.ys.min():.2f}, {self.ys.max():.2f}]")
        print(f"时间范围: {self.times[0]} 到 {self.times[-1]}")
        
        return self
    
    def create_tensor(self):
        """
        创建时空张量 (T, C, H, W)
        """
        # 初始化张量
        self.data_raw = np.full((self.T, 3, self.H, self.W), np.nan, dtype=np.float32)
        
        # 创建坐标映射
        x_to_idx = {x: i for i, x in enumerate(self.xs)}
        y_to_idx = {y: i for i, y in enumerate(self.ys)}
        
        # 修复：将 times 转换为与 df 中相同的类型
        self.times = pd.to_datetime(self.times)
        t_to_idx = {t: i for i, t in enumerate(self.times)}
        
        # 填充数据
        for _, row in self.df.iterrows():
            # 确保时间类型一致
            t_key = pd.Timestamp(row['report_time'])
            t = t_to_idx[t_key]
            x = x_to_idx[row['grid_x']]
            y = y_to_idx[row['grid_y']]
            
            self.data_raw[t, 0, y, x] = row['deformation']
            self.data_raw[t, 1, y, x] = row['speed']
            self.data_raw[t, 2, y, x] = row['acceleration']
        
        print(f"原始张量形状：{self.data_raw.shape}")
        return self
    
    def interpolate_missing_values(self):
        """
        插值填补缺失值
        """
        print("开始时空插值...")
        
        # 对每个通道、每个网格点沿时间插值
        for c in range(3):
            for h in range(self.H):
                for w in range(self.W):
                    series = self.data_raw[:, c, h, w]
                    if not np.isnan(series).all():  # 如果不是全为空
                        valid_idx = np.where(~np.isnan(series))[0]
                        if len(valid_idx) > 1:
                            # 时间插值
                            f = interp1d(valid_idx, series[valid_idx], 
                                       kind='linear', fill_value='extrapolate', bounds_error=False)
                            self.data_raw[:, c, h, w] = f(np.arange(self.T))
                        elif len(valid_idx) == 1:
                            # 只有一个有效值，复制
                            self.data_raw[:, c, h, w] = series[valid_idx[0]]
        
        print("时空插值完成")
        return self
    
    def normalize_data(self, train_ratio=0.7, val_ratio=0.15):
        """
        数据归一化
        """
        # 划分训练集
        train_end = int(self.T * train_ratio)
        val_end = train_end + int(self.T * val_ratio)
        
        # 计算训练集统计量
        train_data = self.data_raw[:train_end]
        self.mean = np.mean(train_data, axis=(0,2,3), keepdims=True)
        self.std = np.std(train_data, axis=(0,2,3), keepdims=True)
        
        # 归一化
        self.data_normalized = (self.data_raw - self.mean) / (self.std + 1e-8)
        
        print(f"归一化完成，均值: {self.mean.flatten()}, 标准差: {self.std.flatten()}")
        return self
    
    def split_dataset(self, train_ratio=0.7, val_ratio=0.15):
        """
        划分数据集
        """
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
    
    def save_for_cnn_models(self, output_dir):
        """
        保存为CNN基模型格式 (T, C, H, W)
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存归一化参数
        norm_params = {'mean': self.mean, 'std': self.std}
        np.save(os.path.join(output_dir, 'norm_params.npy'), norm_params)
        
        # 保存网格信息
        grid_info = {'xs': self.xs, 'ys': self.ys}
        np.save(os.path.join(output_dir, 'grid_info.npy'), grid_info)
        
        # 保存时间戳
        np.save(os.path.join(output_dir, 'time_stamps.npy'), self.times)
        
        # 保存数据集
        np.save(os.path.join(output_dir, 'train_grid.npy'), self.train_data)
        np.save(os.path.join(output_dir, 'val_grid.npy'), self.val_data)
        np.save(os.path.join(output_dir, 'test_grid.npy'), self.test_data)
        
        print(f"CNN格式数据已保存到: {output_dir}")
        return self
    
    def convert_to_timeseries_format(self):
        """
        转换为时间序列格式 (N, T, C) 用于Transformer模型
        """
        N = self.H * self.W
        # 将 (T, C, H, W) 转换为 (N, T, C)
        self.data_ts = self.data_normalized.transpose(2, 3, 0, 1).reshape(N, self.T, -1)
        
        # 按时间划分
        train_end = self.train_data.shape[0]
        val_end = train_end + self.val_data.shape[0]
        
        self.train_ts = self.data_ts[:, :train_end, :]
        self.val_ts = self.data_ts[:, train_end:val_end, :]
        self.test_ts = self.data_ts[:, val_end:, :]
        
        print(f"时间序列格式转换完成:")
        print(f"  总网格点数: {N}")
        print(f"  训练集: {self.train_ts.shape}")
        print(f"  验证集: {self.val_ts.shape}")
        print(f"  测试集: {self.test_ts.shape}")
        
        return self
    
    def save_for_transformer_models(self, output_dir):
        """
        保存为Transformer基模型格式
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存时间序列格式数据
        np.save(os.path.join(output_dir, 'train_ts.npy'), self.train_ts)
        np.save(os.path.join(output_dir, 'val_ts.npy'), self.val_ts)
        np.save(os.path.join(output_dir, 'test_ts.npy'), self.test_ts)
        
        # 保存节点坐标 (用于GNN)
        node_coords = []
        for h in range(self.H):
            for w in range(self.W):
                x_norm = (self.xs[w] - self.xs.min()) / (self.xs.max() - self.xs.min() + 1e-8)
                y_norm = (self.ys[h] - self.ys.min()) / (self.ys.max() - self.ys.min() + 1e-8)
                node_coords.append([x_norm, y_norm])
        self.node_coords = np.array(node_coords)
        np.save(os.path.join(output_dir, 'node_coords.npy'), self.node_coords)
        
        print(f"Transformer/GNN格式数据已保存到: {output_dir}")
        return self
    
    def get_summary(self):
        """
        获取数据摘要
        """
        summary = {
            'shape_raw': self.data_raw.shape,
            'shape_normalized': self.data_normalized.shape,
            'missing_values': np.isnan(self.data_raw).sum(),
            'value_ranges': {
                'deformation': [float(np.nanmin(self.data_raw[:, 0])), float(np.nanmax(self.data_raw[:, 0]))],
                'speed': [float(np.nanmin(self.data_raw[:, 1])), float(np.nanmax(self.data_raw[:, 1]))],
                'acceleration': [float(np.nanmin(self.data_raw[:, 2])), float(np.nanmax(self.data_raw[:, 2]))]
            },
            'grid_size': (self.H, self.W),
            'time_steps': self.T,
            'time_range': (str(self.times[0]), str(self.times[-1]))
        }
        return summary

def process_spatiotemporal_data(csv_path, output_dir):
    """
    完整的数据处理流程
    """
    processor = SpatioTemporalDataProcessor(csv_path)
    
    # 执行完整处理流程
    result = (processor
              .clean_data()
              .build_spatial_grid()
              .create_tensor()
              .interpolate_missing_values()
              .normalize_data()
              .split_dataset()
              .save_for_cnn_models(output_dir)
              .convert_to_timeseries_format()
              .save_for_transformer_models(output_dir))
    
    # 打印摘要
    summary = result.get_summary()
    print("\n=== 数据处理摘要 ===")
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    return processor

# 示例使用
if __name__ == "__main__":
    
    processor = process_spatiotemporal_data(r'/Users/dc/Z研究生/eedsProject/device_10001_sorted.csv', 'processed_output')
    
    # 验证输出文件
    import glob
    output_files = glob.glob('processed_output/*')
    print(f"\n生成的输出文件: {output_files}")