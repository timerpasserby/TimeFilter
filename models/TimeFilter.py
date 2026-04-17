# 这个模型文件负责组装 TimeFilter 主干，并在最小改动下接入统一的 CSPAdapter 空间增强模块。

import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from layers.Embed import PositionalEmbedding
from layers.StandardNorm import Normalize
from layers.TimeFilter_layers import TimeFilter_Backbone
from models.csp_adapter import CSPAdapter


# 将形状转换成更易读的列表形式。
def shape_as_list(tensor):
    """把张量形状转换成 Python 列表。"""
    return list(tensor.shape)


# 检查张量里是否存在 NaN 或 Inf。
def has_invalid_value(tensor):
    """检查张量是否存在无效数值。"""
    data = tensor.detach()
    return {
        'has_nan': bool(torch.isnan(data).any().item()),
        'has_inf': bool(torch.isinf(data).any().item()),
    }


# 将一维时序切分成 patch 并映射到隐空间。
class PatchEmbed(nn.Module):
    """将每个节点的时间序列切分成 patch token。"""

    # 初始化 patch embedding 层。
    def __init__(self, dim, patch_len, stride=None, pos=True):
        """构建 patch 映射层与可选位置编码。"""
        super().__init__()
        self.patch_len = patch_len
        self.stride = patch_len if stride is None else stride
        self.patch_proj = nn.Linear(self.patch_len, dim)
        self.pos = pos
        if self.pos:
            pos_emb_theta = 10000
            self.pe = PositionalEmbedding(dim, pos_emb_theta)

    # 将节点时序切成 patch token。
    def forward(self, x):
        """将节点时序切分并映射成 token。"""
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = self.patch_proj(x)
        if self.pos:
            x += self.pe(x)
        return x


# TimeFilter 主模型，支持按需接入 CSPAdapter。
class Model(nn.Module):
    """组装 TimeFilter 主干和可选的 CSPAdapter。"""

    # 初始化模型结构、坐标缓存和可选的空间增强模块。
    def __init__(self, configs):
        """构建 TimeFilter 主干并准备可选的空间增强能力。"""
        super().__init__()

        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.n_vars = configs.c_out
        self.dim = configs.d_model
        self.d_ff = configs.d_ff
        self.patch_len = configs.patch_len
        self.stride = self.patch_len
        self.num_patches = int((self.seq_len - self.patch_len) / self.stride + 1)
        self.use_csp_adapter = bool(getattr(configs, 'use_csp_adapter', False))
        self.csp_debug = bool(getattr(configs, 'csp_debug', False))
        self.latest_debug_info = {}

        self.alpha = 0.1 if configs.alpha is None else configs.alpha
        self.top_p = 0.5 if configs.top_p is None else configs.top_p

        self.patch_embed = PatchEmbed(self.dim, self.patch_len, self.stride, configs.pos)
        self.backbone = TimeFilter_Backbone(
            self.dim,
            self.n_vars,
            self.d_ff,
            configs.n_heads,
            configs.e_layers,
            self.top_p,
            configs.dropout,
            self.seq_len * self.n_vars // self.patch_len,
        )
        self.head = nn.Linear(self.dim * self.num_patches, self.pred_len)
        self.use_RevIN = False
        self.norm = Normalize(configs.enc_in, affine=self.use_RevIN)

        default_coords = torch.empty(0, 3, dtype=torch.float32)
        if self.use_csp_adapter:
            default_coords = self._load_coords_from_path(getattr(configs, 'coords_path', ''), self.n_vars)
            self.csp_adapter = CSPAdapter(
                token_dim=self.dim,
                num_patches=self.num_patches,
                spatial_dim=configs.spatial_dim,
                rff_dim=configs.rff_dim,
                rff_sigma=configs.rff_sigma,
                spatial_hidden_dim=configs.spatial_hidden_dim,
                learnable_z_scale=configs.learnable_z_scale,
                init_z_scale=configs.init_z_scale,
                prompt_alpha=configs.prompt_alpha,
                learnable_prompt_alpha=configs.learnable_prompt_alpha,
                physical_mask_radius=configs.physical_mask_radius,
                physical_mask_self_loop=configs.physical_mask_self_loop,
                csp_debug=configs.csp_debug,
            )
        else:
            self.csp_adapter = None
        self.register_buffer('default_coords', default_coords, persistent=False)

    # 从本地文件中读取节点三维坐标。
    def _load_coords_from_path(self, coords_path, expected_nodes):
        """根据文件路径读取节点坐标并校验节点数量。"""
        if not coords_path:
            raise ValueError('use_csp_adapter=True 时必须提供 coords_path。')
        if not os.path.exists(coords_path):
            raise FileNotFoundError(f'坐标文件不存在: {coords_path}')

        if coords_path.endswith('.csv'):
            coord_frame = pd.read_csv(coords_path)
            candidate_cols = [
                ['grid_x', 'grid_y', 'grid_z'],
                ['x', 'y', 'z'],
                ['coord_x', 'coord_y', 'coord_z'],
            ]
            coords = None
            for cols in candidate_cols:
                if all(col in coord_frame.columns for col in cols):
                    coords = coord_frame[cols].values
                    break
            if coords is None:
                numeric_cols = coord_frame.select_dtypes(include=[np.number]).columns.tolist()
                numeric_cols = [col for col in numeric_cols if 'node_id' not in col.lower()]
                if len(numeric_cols) < 2:
                    raise ValueError(f'坐标文件缺少可用坐标列: {coords_path}')
                coords = coord_frame[numeric_cols[:3]].values
        elif coords_path.endswith('.npy'):
            coords = np.load(coords_path, allow_pickle=True)
        else:
            raise ValueError(f'暂不支持的坐标文件格式: {coords_path}')

        coords = np.asarray(coords, dtype=np.float32)
        if coords.ndim != 2:
            raise ValueError(f'坐标文件内容维度异常: {coords.shape}')
        if coords.shape[0] != expected_nodes:
            raise ValueError(f'坐标节点数与模型节点数不一致: {coords.shape[0]} vs {expected_nodes}')
        if coords.shape[1] == 2:
            coords = np.concatenate([coords, np.zeros((coords.shape[0], 1), dtype=np.float32)], axis=1)
        if coords.shape[1] < 2:
            raise ValueError(f'坐标维度不足，至少需要二维坐标: {coords.shape}')
        if coords.shape[1] > 3:
            coords = coords[:, :3]
        return torch.from_numpy(coords)

    # 打印一组精简的 CSP debug 日志。
    def _log_debug_info(self, debug_info):
        """在调试模式下输出关键 shape 和统计量。"""
        print(
            f"[CSPDebug] coords={debug_info.get('coords_shape')} "
            f"spatial_embed={debug_info.get('spatial_embed_shape')} "
            f"patch={debug_info.get('patch_token_shape')} "
            f"prompted={debug_info.get('prompted_token_shape')}"
        )
        if 'physical_mask_shape' in debug_info:
            print(
                f"[CSPDebug] physical_mask={debug_info.get('physical_mask_shape')} "
                f"active_ratio={debug_info.get('radius_mask_active_ratio', 0.0):.6f} "
                f"blocked_ratio={debug_info.get('radius_mask_block_ratio', 0.0):.6f}"
            )
        if 'adjacency_shape' in debug_info:
            print(
                f"[CSPDebug] adjacency={debug_info.get('adjacency_shape')} "
                f"routing={debug_info.get('routing_mask_shape')} "
                f"adj_before_mean={debug_info.get('adj_before_mask_mean', 0.0):.6f} "
                f"adj_after_mean={debug_info.get('adj_after_mask_mean', 0.0):.6f}"
            )
        print(
            f"[CSPDebug] token_norm={debug_info.get('token_norm_before', 0.0):.6f}"
            f"->{debug_info.get('token_norm_after', 0.0):.6f} "
            f"output_nan={debug_info.get('output_has_nan', False)} "
            f"output_inf={debug_info.get('output_has_inf', False)}"
        )

    # 执行 TimeFilter 的完整前向传播。
    def forward(self, x, masks, is_training=False, target=None, coords=None, return_debug=False):
        """完成归一化、patch 化、空间增强、图过滤和预测输出。"""
        batch_size, _, channel_count = x.shape
        debug_info = {
            'use_csp_adapter': self.use_csp_adapter,
            'input_shape': shape_as_list(x),
        }

        x = self.norm(x, 'norm')
        x = x.permute(0, 2, 1).reshape(batch_size, channel_count * self.seq_len)
        patch_tokens = self.patch_embed(x)
        debug_info['patch_token_shape'] = shape_as_list(patch_tokens)

        physical_mask = None
        if self.use_csp_adapter:
            coords_tensor = self.default_coords if coords is None else coords
            patch_tokens, physical_mask, adapter_debug = self.csp_adapter(patch_tokens, coords_tensor)
            debug_info.update(adapter_debug)

        backbone_debug = {} if (self.csp_debug or return_debug) else None
        patch_tokens, moe_loss, backbone_debug = self.backbone(
            patch_tokens,
            masks,
            self.alpha,
            is_training,
            physical_mask=physical_mask,
            debug_info=backbone_debug,
        )
        if backbone_debug is not None:
            debug_info.update(backbone_debug)

        x = self.head(patch_tokens.reshape(-1, self.n_vars, self.num_patches, self.dim).flatten(start_dim=-2))
        x = x.permute(0, 2, 1)
        x = self.norm(x, 'denorm')

        output_invalid = has_invalid_value(x)
        debug_info['output_shape'] = shape_as_list(x)
        debug_info['output_has_nan'] = output_invalid['has_nan']
        debug_info['output_has_inf'] = output_invalid['has_inf']
        debug_info['moe_loss'] = float(moe_loss.detach().item()) if torch.is_tensor(moe_loss) else float(moe_loss)

        self.latest_debug_info = debug_info
        if self.csp_debug:
            self._log_debug_info(debug_info)

        if return_debug:
            return x, moe_loss, debug_info
        return x, moe_loss
