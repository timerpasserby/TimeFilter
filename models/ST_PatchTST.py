import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import time
import logging
from datetime import datetime
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import warnings
import argparse
import json
import pickle
from pathlib import Path
from layers.TimeFilter_layers import TimeFilter_Backbone
from layers.Embed import PatchEmbedding

warnings.filterwarnings('ignore')

def setup_logging(log_dir: str):
    """设置日志记录"""
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"patchst_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def build_physics_masks(num_grids_h, num_grids_w, num_vars=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_grids = num_grids_h * num_grids_w
    num_nodes = num_grids * num_vars
    
    # Mask 0: Self
    mask_self = torch.eye(num_nodes)
    
    # Mask 1: Physical (Intra-grid, Inter-variable)
    mask_phy = torch.zeros((num_nodes, num_nodes))
    for g in range(num_grids):
        # 找到该 grid 对应的 3 个变量的索引
        start_idx = g * num_vars
        # 让 d, v, a 互连
        mask_phy[start_idx:start_idx+3, start_idx:start_idx+3] = 1
    
    # Mask 2: Spatial (Inter-grid, Same-variable)
    mask_spatial = torch.zeros((num_nodes, num_nodes))
    # ... 这里写双重循环判断 grid 是否相邻 ...
    # 如果 grid i 和 grid j 相邻：
    # mask_spatial[i*3 + v, j*3 + v] = 1  (对每个变量 v)
    
    # 堆叠: [3, N, N]
    masks = torch.stack([mask_self, mask_phy, mask_spatial])
    return masks.to(device)

class PhysST_TimeFilter(nn.Module):
    def __init__(self, 
                 num_grids=144,      # 空间网格数 (12*12)
                 num_vars=3,         # 变量数 (位移, 速度, 加速度)
                 input_len=96,       # 输入时间长度
                 pred_len=24,        # 预测时间长度
                 patch_len=16,       # 时间 Patch 长度
                 stride=8,           # Patch 步长
                 d_model=128,        # 隐层维度
                 n_heads=4,          # 多头图注意力
                 n_layers=3,         # GraphBlock 层数
                 dropout=0.1,
                 top_p=0.5):         # MoE 的过滤阈值
        super().__init__()
        
        self.num_nodes = num_grids * num_vars
        self.pred_len = pred_len
        
        # 1. Temporal Patching (不仅是 Embedding，更是将时间维度压缩为特征)
        # 我们对每个节点独立做 Patching
        self.patch_embedding = PatchEmbedding(
            d_model, patch_len, stride, padding=0, dropout=dropout
        )
        
        # 计算 Patch 后的数量，用于确定输入给 Graph 的特征维度
        # PatchTST 的做法是将 num_time_patches 视为序列长度
        # 但为了用 TimeFilter 的图，我们将 num_time_patches 融合进特征维度 d_model
        # 或者：我们可以保持 d_model 不变，让 Graph 处理 [Batch, Num_Nodes, Num_Time_Patches * d_model]
        # 这里采用 TimeFilter 原文逻辑：节点特征 = 嵌入后的向量
        
        # 2. TimeFilter Backbone (核心图模块)
        # in_dim 传入 GraphLearner 用于生成 Gate
        self.backbone = TimeFilter_Backbone(
            hidden_dim=d_model, 
            n_vars=self.num_nodes, # 这里的 n_vars 其实是图的节点总数
            d_ff=d_model*4,
            n_heads=n_heads, 
            n_blocks=n_layers, 
            top_p=top_p, 
            dropout=dropout,
            in_dim=d_model # Gate 网络的输入维度
        )
        
        # 3. 预测头
        # 计算 Patch 数量
        self.num_time_patches = int((input_len - patch_len) / stride + 2)
        self.head = nn.Linear(d_model * self.num_time_patches, pred_len)

    def forward(self, x, masks=None):
        # x: [Batch, Input_Len, Num_Grids, Num_Vars]
        B, L, G, V = x.shape
        
        # --- 步骤 1: 维度重塑 ---
        # 我们把 (Grid, Var) 合并为单一的 "节点(Node)" 维度
        x = x.permute(0, 2, 3, 1).reshape(B, G*V, L) # [B, Nodes, L]
        
        # --- 步骤 2: Temporal Patching ---
        # PatchEmbedding 期望输入 [Batch, Time, Channels]，这里稍微 hack 一下
        # 将 Nodes 视为 Batch 维度进行并行 Patching
        x_flat = x.reshape(B * G * V, L, 1) # [B*Nodes, L, 1]
        
        # 输出: [B*Nodes, Num_Time_Patches, d_model]
        x_patch, _ = self.patch_embedding(x_flat) 
        
        # 还原维度: [B, Nodes, Num_Time_Patches, d_model]
        x_patch = x_patch.reshape(B, G * V, -1, x_patch.shape[-1])
        
        # 为了放入 Graph，我们需要融合时间 Patch 维度和 d_model 维度
        # 或者，我们可以把 Time_Patches 视为图卷积的 "Batch" 或 "Seq"
        # TimeFilter_layers 里的 GraphBlock 接受 x: [B, L, D] 其中 L 是节点数
        # 所以我们需要把 Time_Patches 维度的信息聚合，或者只取最后一个 Patch，或者 Flatten
        
        # **关键策略**: 使用 PatchTST 的思想，保留所有 Patch，但在 Graph 内部对每个 Patch 独立做图卷积
        # 变形为: [B * Num_Time_Patches, Nodes, d_model]
        num_time_patches = x_patch.shape[2]
        x_graph_in = x_patch.permute(0, 2, 1, 3).reshape(B * num_time_patches, G*V, -1)
        
        # --- 步骤 3: TimeFilter Graph Learning ---
        # 输入: [Batch_Eff, Nodes, d_model]
        # masks: 传递物理先验图 (稍后解释如何构建)
        out, moe_loss = self.backbone(x_graph_in, masks=masks)
        
        # --- 步骤 4: 预测 ---
        # 还原: [B, Num_Time_Patches, Nodes, d_model]
        out = out.reshape(B, num_time_patches, G*V, -1)
        
        # FlattenHead 需要 [B, Nodes, Num_Time_Patches * d_model]
        out = out.permute(0, 2, 1, 3).reshape(B, G*V, -1)
        
        pred = self.head(out) # [B, Nodes, Pred_Len]
        
        # 还原回 [B, Pred_Len, Num_Grids, Num_Vars]
        pred = pred.reshape(B, G, V, self.pred_len).permute(0, 3, 1, 2)
        
        return pred, moe_loss
class TemporalPatchEmbedding(nn.Module):
    """
    将时间序列转换为 Patch Embedding
    Input: [bs, seq_len, n_vars] -> Output: [bs * n_vars, num_patches, d_model]
    """
    def __init__(self, d_model, patch_len, stride, dropout):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, stride))

        # 线性投影：将一个时间 Patch (长度为 patch_len) 映射为 d_model
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        
        # 位置编码 (Positional Embedding)
        self.position_embedding = nn.Parameter(torch.randn(1, 1000, d_model)) # 预设足够大的长度
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [Batch, Time, Channels]
        # 我们先 permute 成 [Batch, Channels, Time] 以便进行 unfold
        n_vars = x.shape[2]
        x = x.permute(0, 2, 1) # [B, C, T]
        
        # 展平 Batch 和 Channel，实现 Channel Independence
        # x: [B * C, T]
        x = x.reshape(-1, x.shape[-1]) 
        
        # 填充以适应 Stride
        if self.stride > 0:
            pad_len = self.patch_len - self.stride
            if x.shape[-1] < self.patch_len:
                pad_len = self.patch_len - x.shape[-1]
            x = nn.functional.pad(x, (0, pad_len))
            
        # Unfold (滑动窗口切分)
        # x: [B*C, T_padded] -> [B*C, Num_Patches, Patch_Len]
        x = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        
        # 线性投影
        # [B*C, Num_Patches, Patch_Len] -> [B*C, Num_Patches, d_model]
        x = self.value_embedding(x)
        
        # 加位置编码
        x = x + self.position_embedding[:, :x.shape[1], :]
        return self.dropout(x), n_vars

class FlattenHead(nn.Module):
    """
    预测头：将 Transformer 的输出展平并线性映射到预测视界
    """
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x): 
        # x: [bs * nvars, num_patches, d_model]
        x = self.flatten(x)     # [bs * nvars, num_patches * d_model]
        x = self.linear(x)      # [bs * nvars, target_window]
        x = self.dropout(x)
        return x

class SpatialPatchTST(nn.Module):
    """
    优化后的空间-时间 PatchTST
    1. 空间上：保持网格划分
    2. 时间上：引入 Temporal Patching 降低复杂度
    3. 计算上：利用 Batch 并行替代 Python 循环
    """
    def __init__(self, 
                 patch_size: tuple = (12, 12),  # 空间分块大小 (仅用于记录，不直接影响模型结构)
                 input_size: int = 48,          # 历史长度 (Lookback window)
                 h: int = 12,                   # 预测长度 (Horizon)
                 features: int = 3,             # 特征数 (n_vars)
                 
                 # PatchTST 特定参数
                 time_patch_len: int = 16,      # 时间 Patch 长度 (P)
                 time_stride: int = 8,          # 时间 Patch 步长 (S)
                 
                 embed_dim: int = 128,          # d_model
                 depth: int = 3,                # Transformer 层数 (通常 PatchTST 不需要很深)
                 num_heads: int = 4,            # Attention Heads
                 mlp_ratio: float = 4.0,
                 drop_rate: float = 0.1):
        super().__init__()
        
        self.input_size = input_size
        self.h = h
        self.features = features
        self.embed_dim = embed_dim
        
        # 1. 归一化 (Reversible Instance Normalization 的简化版)
        # 这里使用 Learnable Affine 模拟 RevIN 的一部分功能
        self.affine_weight = nn.Parameter(torch.ones(1, 1, features))
        self.affine_bias = nn.Parameter(torch.zeros(1, 1, features))

        # 2. 时间 Patch Embedding
        self.patch_embedding = TemporalPatchEmbedding(
            d_model=embed_dim, 
            patch_len=time_patch_len, 
            stride=time_stride, 
            dropout=drop_rate
        )
        
        # 计算 Patch 数量用于构建 Head
        # 计算逻辑：(L - P) / S + 1 (+ padding 调整)
        # 简单起见，我们通过一次 dummy forward 获取准确的 num_patches
        dummy_input = torch.zeros(1, input_size, features) # [B, L, C]
        dummy_out, _ = self.patch_embedding(dummy_input)
        self.num_time_patches = dummy_out.shape[1]
        
        # 3. Transformer Encoder
        # 使用 PyTorch 原生 TransformerEncoderLayer，效率更高
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=int(embed_dim * mlp_ratio), 
            dropout=drop_rate, 
            activation='gelu',
            batch_first=True,
            norm_first=True # Pre-Norm 通常收敛更好
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # 4. 预测头
        # Head 输入维度 = num_time_patches * d_model
        head_nf = self.num_time_patches * embed_dim
        self.head = FlattenHead(features, head_nf, h, head_dropout=drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, input_size, num_spatial_patches, features]
        """
        B, L, N_spatial, F = x.shape
        
        # --- 1. 数据重塑与通道独立 (Channel Independence) ---
        # 原始维度: [Batch, Time, Spatial, Feat]
        # 目标: 将 Spatial 和 Feat 全部视为独立的变量 (Variates)
        # 变换为: [Batch, Time, Spatial * Feat]
        x = x.reshape(B, L, N_spatial * F)
        
        # 简单的归一化 (Instance Normalization over time)
        # 对每个样本、每个空间点、每个特征在时间维度归一化
        seq_mean = torch.mean(x, dim=1, keepdim=True)
        seq_var = torch.var(x, dim=1, keepdim=True) + 1e-5
        x = (x - seq_mean) / torch.sqrt(seq_var)
        
        # 应用可学习的 Affine 变换 (可选，模拟 RevIN)
        # 注意：这里简化处理，如果追求极致效果可引入完整的 RevIN
        
        # --- 2. Patching & Embedding ---
        # 输入: [B, L, N_spatial*F]
        # PatchEmbedding 内部会将 Batch 和 Channels 合并:
        # -> 输出: [B * (N_spatial*F), Num_Time_Patches, d_model]
        x_enc, n_vars_total = self.patch_embedding(x)
        
        # --- 3. Transformer Encoder ---
        # 此时 Batch Size 变得非常大 (B * 196 * 3)，但序列长度很短 (Num_Patches)
        # 这是计算效率提升的关键
        x_enc = self.encoder(x_enc)
        
        # --- 4. Prediction Head ---
        # -> [B * (N_spatial*F), h]
        dec_out = self.head(x_enc)
        
        # --- 5. 反重塑 (Reshape Back) ---
        # [B * N_spatial * F, h] -> [B, N_spatial * F, h]
        dec_out = dec_out.reshape(B, n_vars_total, -1)
        
        # [B, N_spatial * F, h] -> [B, h, N_spatial * F] (transpose to match time dim)
        dec_out = dec_out.permute(0, 2, 1)
        
        # --- 6. 反归一化 (De-normalization) ---
        dec_out = dec_out * torch.sqrt(seq_var) + seq_mean
        
        # 恢复空间维度: [B, h, N_spatial, F]
        dec_out = dec_out.reshape(B, self.h, N_spatial, F)
        
        return dec_out

class SpatialPatchTSTDataModule:
    """数据管理模块，处理空间分块和时间序列构建"""
    def __init__(self, 
                 data_path: str,
                 patch_size: Tuple[int, int] = (12, 12),
                 input_size: int = 48,
                 h: int = 12,
                 batch_size: int = 32,
                 val_ratio: float = 0.1,
                 test_ratio: float = 0.1):
        """
        初始化数据模块
        :param data_path: 数据CSV路径
        :param patch_size: 空间分块大小
        :param input_size: 历史时间步长度
        :param h: 预测长度
        :param batch_size: 批次大小
        :param val_ratio: 验证集比例
        :param test_ratio: 测试集比例
        """
        self.data_path = data_path
        self.patch_size = patch_size
        self.input_size = input_size
        self.h = h
        self.batch_size = batch_size
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        
        # 加载并预处理数据
        self._load_and_preprocess()
        
        # 构建时间序列
        self._build_time_series()
        
        # 划分数据集
        self._split_datasets()
        
    def _load_and_preprocess(self):
        """加载并预处理原始数据"""
        logger.info(f"Loading data from {self.data_path}...")
        df = pd.read_csv(self.data_path)
        
        # 确保时间戳排序
        df['report_time'] = pd.to_datetime(df['report_time'])
        df = df.sort_values('report_time').reset_index(drop=True)
        
        # 检查缺失值
        if df.isnull().any().any():
            logger.warning("Missing values detected. Filling with linear interpolation.")
            df = df.interpolate(method='linear')
        
        # 提取特征
        self.features = ['deformation', 'speed', 'acceleration']
        self.grid_coords = ['grid_x', 'grid_y']
        
        # 保存原始数据（用于后续评估）
        self.raw_df = df.copy()
        
        # 按时间步分组
        self.time_steps = df['report_time'].unique()
        self.num_time_steps = len(self.time_steps)
        
        logger.info(f"Total time steps: {self.num_time_steps}")
        logger.info(f"Time range: {self.time_steps[0]} to {self.time_steps[-1]}")
        
        # 空间分块参数
        self.grid_x_min, self.grid_x_max = df['grid_x'].min(), df['grid_x'].max()
        self.grid_y_min, self.grid_y_max = df['grid_y'].min(), df['grid_y'].max()
        self.patch_width = (self.grid_x_max - self.grid_x_min) / self.patch_size[0]
        self.patch_height = (self.grid_y_max - self.grid_y_min) / self.patch_size[1]
        
        # 创建空间分块
        self.patch_grid = self._create_patch_grid()
        
        logger.info(f"Space divided into {self.patch_size[0]}x{self.patch_size[1]} = {self.patch_grid.shape[0]} patches")
    
    def _create_patch_grid(self) -> np.ndarray:
        """创建空间网格划分"""
        patches = []
        for i in range(self.patch_size[0]):
            for j in range(self.patch_size[1]):
                x_min = self.grid_x_min + i * self.patch_width
                x_max = x_min + self.patch_width
                y_min = self.grid_y_min + j * self.patch_height
                y_max = y_min + self.patch_height
                patches.append((x_min, x_max, y_min, y_max))
        return np.array(patches)
    
    def _get_patch_features(self, time_step_df: pd.DataFrame) -> np.ndarray:
        """为单个时间步获取空间分块特征"""
        patch_features = np.zeros((self.patch_size[0] * self.patch_size[1], len(self.features)))
        
        for idx, (x_min, x_max, y_min, y_max) in enumerate(self.patch_grid):
            # 选择落在当前Patch内的网格点
            mask = (time_step_df['grid_x'] >= x_min) & (time_step_df['grid_x'] < x_max) & \
                   (time_step_df['grid_y'] >= y_min) & (time_step_df['grid_y'] < y_max)
            
            patch_data = time_step_df[mask]
            
            # 计算特征平均值
            if len(patch_data) > 0:
                patch_features[idx] = patch_data[self.features].mean().values
            else:
                # 如果没有点，使用零填充
                patch_features[idx] = np.zeros(len(self.features))
        
        return patch_features  # [num_patches, features]
    
    def _build_time_series(self):
        """构建时间序列数据集"""
        logger.info("Building time series dataset...")
        all_features = []
        
        # 为每个时间步计算空间分块特征
        for time_step in self.time_steps:
            time_step_df = self.raw_df[self.raw_df['report_time'] == time_step]
            patch_features = self._get_patch_features(time_step_df)
            all_features.append(patch_features)
        
        # 转换为numpy数组
        self.all_features = np.array(all_features)  # [num_time_steps, num_patches, features]
        
        # 标准化
        self.scaler = StandardScaler()
        self.all_features_scaled = self.scaler.fit_transform(
            self.all_features.reshape(-1, self.all_features.shape[2])
        ).reshape(self.all_features.shape)
        
        logger.info(f"Data shape after scaling: {self.all_features_scaled.shape}")
    
    def _split_datasets(self):
        """划分训练集、验证集和测试集"""
        logger.info("Splitting datasets...")
        total = len(self.all_features_scaled)
        val_size = int(total * self.val_ratio)
        test_size = int(total * self.test_ratio)
        train_size = total - val_size - test_size
        
        # 划分数据集
        self.train_data = self.all_features_scaled[:train_size]
        self.val_data = self.all_features_scaled[train_size:train_size+val_size]
        self.test_data = self.all_features_scaled[train_size+val_size:]
        
        logger.info(f"Train: {train_size} samples, Val: {val_size} samples, Test: {test_size} samples")
        
        # 创建时间序列数据集
        self.train_dataset = TimeSeriesDataset(
            self.train_data, 
            self.input_size, 
            self.h
        )
        self.val_dataset = TimeSeriesDataset(
            self.val_data, 
            self.input_size, 
            self.h
        )
        self.test_dataset = TimeSeriesDataset(
            self.test_data, 
            self.input_size, 
            self.h
        )
        
        # 创建数据加载器
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=4
        )
        self.val_loader = DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size,
            num_workers=4
        )
        self.test_loader = DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size,
            num_workers=4
        )
        
        logger.info("Datasets created successfully")
    
    def get_scaler(self):
        """返回标准化器"""
        return self.scaler

class TimeSeriesDataset(Dataset):
    """时间序列数据集，用于滑动窗口构建"""
    def __init__(self, data: np.ndarray, input_size: int, h: int):
        """
        :param data: [num_time_steps, num_patches, features]
        :param input_size: 历史窗口长度
        :param h: 预测长度
        """
        self.data = data
        self.input_size = input_size
        self.h = h
        self.num_samples = len(data) - input_size - h + 1
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # 输入序列: [input_size, num_patches, features]
        x = self.data[idx:idx+self.input_size]
        
        # 目标序列: [h, num_patches, features]
        y = self.data[idx+self.input_size:idx+self.input_size+self.h]
        
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def train_model(
    data_module: SpatialPatchTSTDataModule,
    output_dir: str = "results",
    epochs: int = 50,
    patience: int = 10,
    lr: float = 1e-3,
    embed_dim: int = 128,
    depth: int = 6,
    num_heads: int = 8
):
    """训练模型并保存最佳结果"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化模型
    # 修改 train_model 中的初始化部分
    model = SpatialPatchTST(
        patch_size=data_module.patch_size,
        input_size=data_module.input_size,
        h=data_module.h,
        features=len(data_module.features),
        
        # 新增参数
        time_patch_len=16,  # 比如设为16
        time_stride=8,      # 步长设为8
        
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=4.0,
        drop_rate=0.1
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # 检查模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params} total, {trainable_params} trainable")
    
    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # 设置TensorBoard
    tb_dir = os.path.join(output_dir, "tb_logs")
    os.makedirs(tb_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_dir)
    
    # 训练指标记录
    train_losses = []
    val_losses = []
    
    # 早停参数
    best_val_loss = float('inf')
    patience_counter = 0
    
    logger.info("Starting training...")
    start_time = time.time()
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        num_batches = 0
        
        for batch_idx, (x, y) in enumerate(tqdm(data_module.train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
            x, y = x.to(device), y.to(device)
            
            # 检查输入形状
            expected_input_shape = (data_module.batch_size, data_module.input_size, 
                                    data_module.patch_size[0]*data_module.patch_size[1], 
                                    len(data_module.features))
            if x.shape[1:] != expected_input_shape[1:]:
                logger.warning(f"Input shape mismatch: expected {expected_input_shape[1:]}, got {x.shape[1:]}")
                continue
            
            # 检查目标形状
            expected_target_shape = (data_module.batch_size, data_module.h, 
                                     data_module.patch_size[0]*data_module.patch_size[1], 
                                     len(data_module.features))
            if y.shape[1:] != expected_target_shape[1:]:
                logger.warning(f"Target shape mismatch: expected {expected_target_shape[1:]}, got {y.shape[1:]}")
                continue
            
            optimizer.zero_grad()
            y_hat = model(x)
            
            # 检查预测形状是否与目标形状匹配
            # y_hat: [batch, h, features]
            # y: [batch, h, num_patches, features]
            # 我们需要比较y_hat和y的平均值
            y_target = y.mean(dim=2)  # [batch, h, features]
            
            if y_hat.shape != y_target.shape:
                logger.error(f"Shape mismatch: y_hat {y_hat.shape} vs y_target {y_target.shape}")
                continue
            
            loss = criterion(y_hat, y_target)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1
            
            # 记录批次损失
            if batch_idx % 10 == 0:
                writer.add_scalar('Train/Loss', loss.item(), epoch * len(data_module.train_loader) + batch_idx)
        
        if num_batches > 0:
            avg_train_loss = train_loss / num_batches
        else:
            logger.warning("No valid batches processed in this epoch")
            continue
            
        train_losses.append(avg_train_loss)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for x, y in data_module.val_loader:
                x, y = x.to(device), y.to(device)
                
                # 检查输入形状
                if x.shape[1:] != expected_input_shape[1:]:
                    continue
                
                # 检查目标形状
                if y.shape[1:] != expected_target_shape[1:]:
                    continue
                
                y_hat = model(x)
                
                # 计算目标均值
                y_target = y.mean(dim=2)  # [batch, h, features]
                
                if y_hat.shape != y_target.shape:
                    logger.error(f"Validation shape mismatch: y_hat {y_hat.shape} vs y_target {y_target.shape}")
                    continue
                
                loss = criterion(y_hat, y_target)
                val_loss += loss.item()
                val_batches += 1
        
        if val_batches > 0:
            avg_val_loss = val_loss / val_batches
        else:
            logger.warning("No valid validation batches processed")
            continue
            
        val_losses.append(avg_val_loss)
        
        # 记录损失
        writer.add_scalar('Train/AvgLoss', avg_train_loss, epoch)
        writer.add_scalar('Val/AvgLoss', avg_val_loss, epoch)
        
        logger.info(f"Epoch {epoch+1}/{epochs}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}")
        
        # 早停机制
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            # 保存最佳模型
            best_model_path = os.path.join(output_dir, "best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'scaler': data_module.get_scaler()
            }, best_model_path)
            
            logger.info(f"New best model saved with val loss: {best_val_loss:.6f}")
        else:
            patience_counter += 1
            
        # 学习率调度
        if epoch > 0 and epoch % 5 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.9
            
        # 早停检查
        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    # 保存最终模型
    final_model_path = os.path.join(output_dir, "final_model.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': avg_train_loss,
        'val_loss': avg_val_loss,
        'scaler': data_module.get_scaler()
    }, final_model_path)
    
    # 记录训练时间
    total_time = time.time() - start_time
    logger.info(f"Training completed in {total_time:.2f} seconds")
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(output_dir, "loss_curve.png"))
    plt.close()
    
    # 保存训练指标
    metrics = {
        'train_losses': [float(loss) for loss in train_losses],  # 确保转换为Python原生类型
        'val_losses': [float(loss) for loss in val_losses],
        'best_val_loss': float(best_val_loss)
    }
    with open(os.path.join(output_dir, "training_metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    writer.close()
    return model, best_val_loss

def evaluate_model(model, data_module, output_dir: str = "results", device: str = "cpu"):
    """评估模型在测试集上的表现"""
    logger.info("Evaluating model on test set...")
    
    # 设置模型为评估模式
    model.eval()
    model = model.to(device)
    
    # 获取测试数据
    test_loader = data_module.test_loader
    
    # 收集预测和真实值
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            x, y = batch
            x, y = x.to(device), y.to(device)
            
            # 前向传播
            y_hat = model(x)
            
            # 计算目标均值
            y_target = y.mean(dim=2)  # [batch, h, features]
            
            # 检查形状
            if y_hat.shape != y_target.shape:
                logger.error(f"Evaluation shape mismatch: y_hat {y_hat.shape} vs y_target {y_target.shape}")
                continue
            
            # 保存预测和真实值
            all_preds.append(y_hat.cpu().numpy())
            all_targets.append(y_target.cpu().numpy())
    
    # 合并结果
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # 检查形状
    logger.info(f"Predictions shape: {all_preds.shape}")
    logger.info(f"Targets shape: {all_targets.shape}")
    
    # 反标准化
    scaler = data_module.get_scaler()
    
    # 重塑为 [samples*h, features] 并反标准化
    original_shape = all_preds.shape
    all_preds_flat = all_preds.reshape(-1, len(data_module.features))
    all_targets_flat = all_targets.reshape(-1, len(data_module.features))
    
    # 反标准化
    all_preds_flat_unscaled = scaler.inverse_transform(all_preds_flat)
    all_targets_flat_unscaled = scaler.inverse_transform(all_targets_flat)
    
    # 重塑回原始形状
    all_preds_unscaled = all_preds_flat_unscaled.reshape(original_shape)
    all_targets_unscaled = all_targets_flat_unscaled.reshape(original_shape)
    
    # 为每个特征计算指标
    metrics = {}
    for i, feature in enumerate(data_module.features):
        # 提取该特征的所有预测值和真实值
        feature_preds = all_preds_unscaled[:, :, i].flatten()
        feature_targets = all_targets_unscaled[:, :, i].flatten()
        
        # 计算指标
        mse = mean_squared_error(feature_targets, feature_preds)
        mae = mean_absolute_error(feature_targets, feature_preds)
        rmse = np.sqrt(mse)
        
        metrics[feature] = {
            'MSE': float(mse),  # 转换为Python原生类型
            'MAE': float(mae),
            'RMSE': float(rmse)
        }
    
    # 保存指标
    metrics_path = os.path.join(output_dir, "test_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # 打印指标
    logger.info("\nTest Metrics:")
    for feature, m in metrics.items():
        logger.info(f"{feature}: MSE={m['MSE']:.6f}, MAE={m['MAE']:.6f}, RMSE={m['RMSE']:.6f}")
    
    # 创建一个完整的预测vs真实值DataFrame
    n_samples, n_time_steps, n_features = all_preds_unscaled.shape
    all_feature_names = []
    all_pred_values = []
    all_target_values = []
    
    for i, feature in enumerate(data_module.features):
        # 为每个特征提取预测值和真实值
        feature_preds = all_preds_unscaled[:, :, i].flatten()
        feature_targets = all_targets_unscaled[:, :, i].flatten()
        
        # 确保长度一致
        assert len(feature_preds) == len(feature_targets), f"Length mismatch for {feature}: {len(feature_preds)} vs {len(feature_targets)}"
        
        # 添加特征名称
        all_feature_names.extend([feature] * len(feature_preds))
        all_pred_values.extend(feature_preds)
        all_target_values.extend(feature_targets)
    
    # 创建DataFrame
    pred_df = pd.DataFrame({
        'feature': all_feature_names,
        'pred': all_pred_values,
        'target': all_target_values
    })
    
    # 保存预测结果
    pred_df.to_csv(os.path.join(output_dir, "test_predictions.csv"), index=False)
    
    # 绘制预测vs真实值散点图
    for i, feature in enumerate(data_module.features):
        plt.figure(figsize=(8, 6))
        
        # 提取该特征的预测和真实值
        feature_mask = pred_df['feature'] == data_module.features[i]
        feature_data = pred_df[feature_mask]
        
        plt.scatter(feature_data['target'], feature_data['pred'], alpha=0.5)
        plt.plot([feature_data['target'].min(), feature_data['target'].max()], 
                 [feature_data['target'].min(), feature_data['target'].max()], 
                 'r--', lw=2)
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.title(f'{data_module.features[i]} - Predicted vs True Values')
        plt.savefig(os.path.join(output_dir, f"{data_module.features[i]}_pred_vs_true.png"))
        plt.close()
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description="空间PatchTST预测模型")
    
    # 数据路径
    parser.add_argument('--data_path', type=str, required=True, 
                        help='输入数据CSV文件路径')
    parser.add_argument('--output_dir', type=str, default='results', 
                        help='输出目录路径')
    
    # 模型参数
    parser.add_argument('--patch_size', type=int, nargs=2, default=[12, 12], 
                        help='空间分块大小 [height, width]')
    parser.add_argument('--input_size', type=int, default=48, 
                        help='历史时间步长度')
    parser.add_argument('--h', type=int, default=12, 
                        help='预测长度')
    parser.add_argument('--embed_dim', type=int, default=128, 
                        help='嵌入维度')
    parser.add_argument('--depth', type=int, default=6, 
                        help='Transformer层数')
    parser.add_argument('--num_heads', type=int, default=8, 
                        help='注意力头数')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32, 
                        help='批次大小')
    parser.add_argument('--epochs', type=int, default=50, 
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-3, 
                        help='学习率')
    parser.add_argument('--patience', type=int, default=10, 
                        help='早停耐心值')
    parser.add_argument('--val_ratio', type=float, default=0.1, 
                        help='验证集比例')
    parser.add_argument('--test_ratio', type=float, default=0.1, 
                        help='测试集比例')
    
    # 设备
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help='训练设备')
    
    args = parser.parse_args()
    
    # 设置日志
    global logger
    logger = setup_logging(args.output_dir)
    
    logger.info(f"Starting PatchTST training with args: {args}")
    
    # 检查数据文件是否存在
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data file not found: {args.data_path}")
    
    # 初始化数据模块
    logger.info("Initializing data module...")
    try:
        data_module = SpatialPatchTSTDataModule(
            data_path=args.data_path,
            patch_size=tuple(args.patch_size),
            input_size=args.input_size,
            h=args.h,
            batch_size=args.batch_size,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio
        )
    except Exception as e:
        logger.error(f"Error initializing data module: {str(e)}")
        return 1
    
    # 训练模型
    logger.info("Starting model training...")
    try:
        model, best_val_loss = train_model(
            data_module=data_module,
            output_dir=args.output_dir,
            epochs=args.epochs,
            patience=args.patience,
            lr=args.lr,
            embed_dim=args.embed_dim,
            depth=args.depth,
            num_heads=args.num_heads
        )
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        return 1
    
    # 评估模型
    logger.info("Starting model evaluation...")
    try:
        test_metrics = evaluate_model(
            model=model,
            data_module=data_module,
            output_dir=args.output_dir,
            device=args.device
        )
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        return 1
    
    logger.info("Training and evaluation completed successfully!")
    
    # 保存配置参数
    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, 'w') as f:
        # 确保所有参数都是JSON可序列化类型
        config_dict = vars(args)
        for key, value in config_dict.items():
            if isinstance(value, np.ndarray):
                config_dict[key] = value.tolist()
            elif isinstance(value, torch.Tensor):
                config_dict[key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                config_dict[key] = float(value) if isinstance(value, np.floating) else int(value)
        json.dump(config_dict, f, indent=4)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())



