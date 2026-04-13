import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import os
import warnings
warnings.filterwarnings('ignore')

class SensorAggregator:
    """
    传感器聚合降维处理器
    提供多种方法减少传感器数量N，同时保留关键特征
    """
    
    def __init__(self, pems_file_path):
        # 加载PEMS格式数据
        self.pems_data = np.load(pems_file_path)
        self.data = self.pems_data['data']  # (T, N, C)
        self.train_data = self.pems_data['train']  # (T_train, N, C)
        self.val_data = self.pems_data['val']  # (T_val, N, C)
        self.test_data = self.pems_data['test']  # (T_test, N, C)
        self.mean = self.pems_data['mean']  # (1, 1, C)
        self.std = self.pems_data['std']  # (1, 1, C)
        self.valid_sensors = self.pems_data['valid_sensors']  # [(x, y), ...]
        
        self.T, self.N, self.C = self.data.shape
        self.reduced_N = None
        self.aggregation_method = None
        self.aggregation_matrix = None  # 用于将原始传感器映射到聚合传感器
        
        print(f"原始数据形状: T={self.T}, N={self.N}, C={self.C}")
    
    def spatial_clustering_aggregation(self, n_clusters=500, method='kmeans'):
        """
        基于空间坐标的聚类聚合
        将相近的传感器聚为一组，每组用代表传感器或平均值替代
        """
        print(f"使用{method}进行空间聚类聚合，目标传感器数: {n_clusters}")
        
        # 准备空间坐标数据
        coordinates = np.array([[x, y] for x, y in self.valid_sensors])
        
        if method == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        elif method == 'dbscan':
            # 估算DBSCAN参数
            distances = cdist(coordinates, coordinates)
            # 取距离的某个百分位数作为eps
            eps = np.percentile(distances[distances > 0], 5)
            clusterer = DBSCAN(eps=eps, min_samples=2)
        
        # 执行聚类
        cluster_labels = clusterer.fit_predict(coordinates)
        
        if method == 'dbscan':
            # DBSCAN可能产生噪声点(-1)，重新调整簇数
            unique_labels = set(cluster_labels)
            if -1 in unique_labels:
                unique_labels.remove(-1)
            n_clusters = len(unique_labels)
            print(f"DBSCAN实际聚类数: {n_clusters}")
        
        # 创建聚合矩阵
        self.aggregation_matrix = np.zeros((n_clusters, self.N))
        
        # 为每个聚类计算聚合权重
        for cluster_id in range(n_clusters):
            cluster_sensors = np.where(cluster_labels == cluster_id)[0]
            if len(cluster_sensors) > 0:
                # 使用平均值进行聚合
                self.aggregation_matrix[cluster_id, cluster_sensors] = 1.0 / len(cluster_sensors)
        
        self.reduced_N = n_clusters
        self.aggregation_method = f'spatial_{method}'
        
        print(f"空间聚类聚合完成: 原始N={self.N} -> 聚合后N={self.reduced_N}")
        return self
    
    def temporal_pattern_aggregation(self, n_clusters=500, n_components=50):
        """
        基于时间模式的聚类聚合
        使用PCA降维后进行聚类，找到具有相似时间模式的传感器组
        """
        print(f"使用时间模式聚类聚合，目标传感器数: {n_clusters}")
        
        # 使用训练数据的均值和方差进行标准化
        train_mean = np.mean(self.train_data, axis=0, keepdims=True)  # (1, N, C)
        train_std = np.std(self.train_data, axis=0, keepdims=True)   # (1, N, C)
        train_std = np.where(train_std == 0, 1.0, train_std)  # 避免除零
        
        # 标准化训练数据
        normalized_train = (self.train_data - train_mean) / train_std
        
        # 对每个传感器提取时间序列特征
        # 使用PCA对时间维度进行降维
        pca = PCA(n_components=min(n_components, self.train_data.shape[0]))
        # reshape为 (N, T*C) 进行PCA
        reshaped_data = normalized_train.transpose(1, 0, 2).reshape(self.N, -1)
        pca_features = pca.fit_transform(reshaped_data)  # (N, n_components)
        
        # 使用KMeans对PCA特征进行聚类
        clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = clusterer.fit_predict(pca_features)
        
        # 创建聚合矩阵
        self.aggregation_matrix = np.zeros((n_clusters, self.N))
        
        # 为每个聚类计算聚合权重
        for cluster_id in range(n_clusters):
            cluster_sensors = np.where(cluster_labels == cluster_id)[0]
            if len(cluster_sensors) > 0:
                self.aggregation_matrix[cluster_id, cluster_sensors] = 1.0 / len(cluster_sensors)
        
        self.reduced_N = n_clusters
        self.aggregation_method = 'temporal_pattern'
        
        print(f"时间模式聚类聚合完成: 原始N={self.N} -> 聚合后N={self.reduced_N}")
        print(f"PCA保留的方差比例: {np.sum(pca.explained_variance_ratio_):.4f}")
        return self
    
    def hybrid_aggregation(self, spatial_clusters=1000, temporal_clusters=500):
        """
        混合聚类：先空间聚类再时间聚类
        """
        print(f"执行混合聚类聚合: 空间聚类{spatial_clusters} -> 时间聚类{temporal_clusters}")
        
        # 第一步：空间聚类
        coordinates = np.array([[x, y] for x, y in self.valid_sensors])
        spatial_clusterer = KMeans(n_clusters=spatial_clusters, random_state=42, n_init=10)
        spatial_labels = spatial_clusterer.fit_predict(coordinates)
        
        # 对每个空间聚类内的传感器进行聚合
        spatial_aggregated_data = np.zeros((self.T, spatial_clusters, self.C))
        
        for cluster_id in range(spatial_clusters):
            cluster_sensors = np.where(spatial_labels == cluster_id)[0]
            if len(cluster_sensors) > 0:
                # 计算该聚类内传感器的平均值
                spatial_aggregated_data[:, cluster_id, :] = np.mean(
                    self.data[:, cluster_sensors, :], axis=1
                )
        
        # 第二步：对空间聚合后的数据进行时间模式聚类
        # 标准化数据
        train_data_spatial = spatial_aggregated_data[:int(0.7*self.T)]  # 使用70%作为训练数据
        scaler = StandardScaler()
        # reshape为 (spatial_clusters, T_train*C) 进行PCA
        reshaped_train = train_data_spatial.transpose(1, 0, 2).reshape(spatial_clusters, -1)
        standardized_features = scaler.fit_transform(reshaped_train)
        
        # 对标准化后的特征进行聚类
        temporal_clusterer = KMeans(n_clusters=temporal_clusters, random_state=42, n_init=10)
        temporal_labels = temporal_clusterer.fit_predict(standardized_features)
        
        # 创建最终的聚合矩阵
        self.aggregation_matrix = np.zeros((temporal_clusters, self.N))
        
        # 从空间聚类到最终聚类的映射
        for final_cluster_id in range(temporal_clusters):
            # 找到属于该时间聚类的空间聚类
            spatial_clusters_in_temporal = np.where(temporal_labels == final_cluster_id)[0]
            
            for spatial_cluster_id in spatial_clusters_in_temporal:
                # 找到该空间聚类中的原始传感器
                original_sensors = np.where(spatial_labels == spatial_cluster_id)[0]
                if len(original_sensors) > 0:
                    # 平均分配权重
                    weight = 1.0 / len(original_sensors)
                    self.aggregation_matrix[final_cluster_id, original_sensors] = weight
        
        self.reduced_N = temporal_clusters
        self.aggregation_method = 'hybrid'
        
        print(f"混合聚类聚合完成: 原始N={self.N} -> 聚合后N={self.reduced_N}")
        return self
    
    def apply_aggregation(self):
        """
        应用聚合矩阵对所有数据进行聚合
        """
        print("应用聚合矩阵...")
        
        # 修正：数据形状是 (T, N, C)，聚合矩阵是 (reduced_N, N)
        # 使用 'ij,tjc->tic' 对 N 维度进行加权求和
        aggregated_train = np.einsum('ij,tjc->tic', self.aggregation_matrix, self.train_data)
        aggregated_val = np.einsum('ij,tjc->tic', self.aggregation_matrix, self.val_data)
        aggregated_test = np.einsum('ij,tjc->tic', self.aggregation_matrix, self.test_data)
        aggregated_data = np.einsum('ij,tjc->tic', self.aggregation_matrix, self.data)
        
        self.aggregated_data = aggregated_data
        self.aggregated_train = aggregated_train
        self.aggregated_val = aggregated_val
        self.aggregated_test = aggregated_test
        
        print(f"聚合后数据形状:")
        print(f"  完整数据：{aggregated_data.shape}")
        print(f"  训练集：{aggregated_train.shape}")
        print(f"  验证集：{aggregated_val.shape}")
        print(f"  测试集：{aggregated_test.shape}")
        
        return self
    
    def save_reduced_pems(self, output_file):
        """
        保存降维后的PEMS格式数据
        """
        reduced_pems_data = {
            'data': self.aggregated_data,      # (T, reduced_N, C)
            'train': self.aggregated_train,    # (T_train, reduced_N, C)
            'val': self.aggregated_val,        # (T_val, reduced_N, C)
            'test': self.aggregated_test,      # (T_test, reduced_N, C)
            'mean': self.mean,                 # (1, 1, C) - 保持原始归一化参数
            'std': self.std,                   # (1, 1, C) - 保持原始归一化参数
            'aggregation_method': self.aggregation_method,
            'reduction_factor': self.N / self.reduced_N,
            'original_N': self.N,
            'reduced_N': self.reduced_N,
            'aggregation_matrix': self.aggregation_matrix
        }
        
        np.savez_compressed(output_file, **reduced_pems_data)
        
        print(f"降维后的PEMS数据已保存到: {output_file}")
        print(f"传感器数量减少: {self.N} -> {self.reduced_N} (减少率: {(1-self.reduced_N/self.N)*100:.2f}%)")
        return self
    
    def get_aggregation_summary(self):
        """
        获取聚合摘要
        """
        summary = {
            'original_N': self.N,
            'reduced_N': self.reduced_N,
            'reduction_factor': self.N / self.reduced_N,
            'aggregation_method': self.aggregation_method,
            'data_shape_original': self.data.shape,
            'data_shape_reduced': self.aggregated_data.shape,
            'memory_reduction': (1 - self.reduced_N/self.N) * 100
        }
        return summary

def reduce_sensors(pems_input_path, output_file, method='spatial_kmeans', **kwargs):
    """
    传感器数量减少的主函数
    """
    aggregator = SensorAggregator(pems_input_path)
    
    if method == 'spatial_kmeans':
        n_clusters = kwargs.get('n_clusters', 500)
        aggregator.spatial_clustering_aggregation(n_clusters=n_clusters, method='kmeans')
    elif method == 'spatial_dbscan':
        aggregator.spatial_clustering_aggregation(n_clusters=kwargs.get('n_clusters', 500), method='dbscan')
    elif method == 'temporal_pattern':
        n_clusters = kwargs.get('n_clusters', 500)
        n_components = kwargs.get('n_components', 50)
        aggregator.temporal_pattern_aggregation(n_clusters=n_clusters, n_components=n_components)
    elif method == 'hybrid':
        spatial_clusters = kwargs.get('spatial_clusters', 1000)
        temporal_clusters = kwargs.get('temporal_clusters', 500)
        aggregator.hybrid_aggregation(spatial_clusters=spatial_clusters, 
                                   temporal_clusters=temporal_clusters)
    
    # 应用聚合并保存
    result = aggregator.apply_aggregation().save_reduced_pems(output_file)
    
    # 打印摘要
    summary = result.get_aggregation_summary()
    print("\n=== 传感器聚合摘要 ===")
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    # 验证保存的文件
    loaded_data = np.load(output_file)
    print(f"\n验证加载的降维数据:")
    print(f"  主数据形状: {loaded_data['data'].shape}")
    print(f"  训练集形状: {loaded_data['train'].shape}")
    print(f"  聚合方法: {loaded_data['aggregation_method']}")
    print(f"  减少倍数: {loaded_data['reduction_factor']:.2f}")
    
    return aggregator

# 使用示例
if __name__ == "__main__":
    # 方法1: 空间K-means聚类 (推荐用于边坡监测)
    reducer1 = reduce_sensors(
        r'/Users/dc/Z研究生/eedsProject/slope_pems_minimal.npz',
        'slope_pems_reduced_spatial.npz',
        method='spatial_kmeans',
        n_clusters=500  # 将65992个传感器减少到500个
    )
#     
#     # 方法2: 时间模式聚类
#     reducer2 = reduce_sensors(
#         'slope_pems_minimal.npz',
#         'slope_pems_reduced_temporal.npz',
#         method='temporal_pattern',
#         n_clusters=300,
#         n_components=30
#     )
#     
#     # 方法3: 混合聚类
#     reducer3 = reduce_sensors(
#         'slope_pems_minimal.npz',
#         'slope_pems_reduced_hybrid.npz',
#         method='hybrid',
#         spatial_clusters=800,
#         temporal_clusters=400
#     )
