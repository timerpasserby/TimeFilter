# 这个模块统一实现连续空间编码、patch 级空间提示注入和物理半径掩码，供 TimeFilter 主干调用。

import math

import torch
import torch.nn as nn


# 将张量形状转换成便于日志记录的列表。
def shape_as_list(tensor):
    """将张量形状转换成 Python 列表。"""
    return list(tensor.shape)


# 统计张量的基础数值信息，方便 debug 输出。
def summarize_tensor(tensor):
    """统计张量的均值、最大值和稀疏度。"""
    data = tensor.detach()
    return {
        'mean': float(data.mean().item()),
        'max': float(data.max().item()),
        'sparsity': float((data.abs() < 1e-8).float().mean().item()),
        'has_nan': bool(torch.isnan(data).any().item()),
        'has_inf': bool(torch.isinf(data).any().item()),
    }


# 使用连续空间坐标生成与 token 维度对齐的空间表征。
class ContinuousSpatialEncoder(nn.Module):
    """使用 RFF 和 MLP 将三维坐标映射到 token 维度。"""

    # 初始化连续空间编码器。
    def __init__(self, token_dim, spatial_dim, rff_dim, rff_sigma, spatial_hidden_dim):
        """初始化 RFF 和空间 MLP。"""
        super().__init__()
        self.token_dim = token_dim
        self.rff_pairs = max(1, rff_dim // 2)
        self.rff_sigma = max(float(rff_sigma), 1e-6)

        rff_weight = torch.randn(3, self.rff_pairs) / self.rff_sigma
        self.register_buffer('rff_weight', rff_weight, persistent=False)

        self.mlp = nn.Sequential(
            nn.Linear(self.rff_pairs * 2, spatial_hidden_dim),
            nn.GELU(),
            nn.Linear(spatial_hidden_dim, spatial_dim),
            nn.GELU(),
            nn.Linear(spatial_dim, token_dim),
        )

    # 对中心化后的坐标做连续空间编码。
    def forward(self, coords):
        """将中心化后的坐标编码成空间表征。"""
        projected = 2 * math.pi * coords @ self.rff_weight
        rff_features = torch.cat([torch.sin(projected), torch.cos(projected)], dim=-1)
        spatial_embed = self.mlp(rff_features)
        return spatial_embed


# 基于物理半径构建 patch 级别的邻接掩码。
def build_physical_radius_mask(coords, radius, num_patches, keep_self_loop=True):
    """根据节点物理距离生成 patch 级掩码。"""
    distance_matrix = torch.cdist(coords, coords, p=2)
    if radius <= 0:
        node_mask = torch.ones_like(distance_matrix, dtype=torch.bool)
    else:
        node_mask = distance_matrix <= radius

    if keep_self_loop:
        node_mask.fill_diagonal_(True)

    isolated_nodes = node_mask.sum(dim=-1) == 0
    if isolated_nodes.any():
        isolated_indices = isolated_nodes.nonzero(as_tuple=False).squeeze(-1)
        node_mask[isolated_indices, isolated_indices] = True

    patch_mask = node_mask.repeat_interleave(num_patches, dim=0)
    patch_mask = patch_mask.repeat_interleave(num_patches, dim=1)
    patch_mask = patch_mask.unsqueeze(0).unsqueeze(0)

    debug_info = {
        'physical_mask_shape': shape_as_list(patch_mask),
        'distance_matrix_shape': shape_as_list(distance_matrix),
        'radius_mask_active_ratio': float(patch_mask.float().mean().item()),
        'radius_mask_block_ratio': float(1.0 - patch_mask.float().mean().item()),
        'isolated_node_count': int(isolated_nodes.sum().item()),
    }
    return patch_mask, distance_matrix, debug_info


# 将节点空间表征扩展到 patch token 并注入原始 token。
def inject_spatial_prompt(patch_tokens, spatial_embed, num_patches, prompt_alpha):
    """把空间表征作为 prompt 注入 patch token。"""
    patch_prompt = spatial_embed.repeat_interleave(num_patches, dim=0)
    patch_prompt = patch_prompt.unsqueeze(0).expand(patch_tokens.shape[0], -1, -1)

    alpha = prompt_alpha.to(device=patch_tokens.device, dtype=patch_tokens.dtype)
    prompted_tokens = patch_tokens + alpha * patch_prompt

    debug_info = {
        'patch_token_shape': shape_as_list(patch_tokens),
        'prompted_token_shape': shape_as_list(prompted_tokens),
        'prompt_token_shape': shape_as_list(patch_prompt),
        'prompt_alpha_value': float(alpha.detach().item()),
        'token_norm_before': float(patch_tokens.detach().norm(dim=-1).mean().item()),
        'token_norm_after': float(prompted_tokens.detach().norm(dim=-1).mean().item()),
    }
    return prompted_tokens, patch_prompt, debug_info


# 统一管理空间编码、prompt 注入和物理半径约束。
class CSPAdapter(nn.Module):
    """将坐标先验和物理距离约束统一注入 TimeFilter。"""

    # 初始化 CSPAdapter 的全部子模块和配置。
    def __init__(
        self,
        token_dim,
        num_patches,
        spatial_dim,
        rff_dim,
        rff_sigma,
        spatial_hidden_dim,
        learnable_z_scale,
        init_z_scale,
        prompt_alpha,
        learnable_prompt_alpha,
        physical_mask_radius,
        physical_mask_self_loop,
        csp_debug,
    ):
        """初始化 CSPAdapter 的编码器、缩放和提示参数。"""
        super().__init__()
        self.token_dim = token_dim
        self.num_patches = num_patches
        self.physical_mask_radius = float(physical_mask_radius)
        self.physical_mask_self_loop = bool(physical_mask_self_loop)
        self.csp_debug = bool(csp_debug)

        self.encoder = ContinuousSpatialEncoder(
            token_dim=token_dim,
            spatial_dim=spatial_dim,
            rff_dim=rff_dim,
            rff_sigma=rff_sigma,
            spatial_hidden_dim=spatial_hidden_dim,
        )

        z_scale = torch.tensor(float(init_z_scale), dtype=torch.float32)
        if learnable_z_scale:
            self.z_scale = nn.Parameter(z_scale)
        else:
            self.register_buffer('z_scale', z_scale, persistent=False)

        prompt_alpha_tensor = torch.tensor(float(prompt_alpha), dtype=torch.float32)
        if learnable_prompt_alpha:
            self.prompt_alpha = nn.Parameter(prompt_alpha_tensor)
        else:
            self.register_buffer('prompt_alpha', prompt_alpha_tensor, persistent=False)

    # 将输入坐标补齐到三维并做中心化与 z 轴缩放。
    def _prepare_coords(self, coords, dtype, device):
        """将输入坐标整理成中心化后的三维张量。"""
        if coords.ndim != 2:
            raise ValueError(f'coords should be 2D, but got shape {tuple(coords.shape)}')

        coords = coords.to(device=device, dtype=dtype)
        if coords.shape[-1] == 2:
            zeros = torch.zeros(coords.shape[0], 1, device=device, dtype=dtype)
            coords = torch.cat([coords, zeros], dim=-1)
        elif coords.shape[-1] > 3:
            coords = coords[:, :3]

        if coords.shape[-1] != 3:
            raise ValueError(f'coords should have 2 or 3 columns, but got {coords.shape[-1]}')

        coords = coords - coords.mean(dim=0, keepdim=True)
        # 这里避免对参与梯度计算的坐标视图做原地改写，保证 backward 稳定。
        axis_scale = torch.ones(1, 3, device=device, dtype=dtype)
        axis_scale[:, 2] = self.z_scale.to(device=device, dtype=dtype)
        coords = coords * axis_scale
        return coords

    # 执行完整的空间增强流程并返回 debug 信息。
    def forward(self, patch_tokens, coords):
        """执行坐标编码、prompt 注入和物理掩码生成。"""
        prepared_coords = self._prepare_coords(coords, patch_tokens.dtype, patch_tokens.device)
        spatial_embed = self.encoder(prepared_coords)
        prompted_tokens, prompt_tokens, prompt_debug = inject_spatial_prompt(
            patch_tokens,
            spatial_embed,
            self.num_patches,
            self.prompt_alpha,
        )
        physical_mask, distance_matrix, mask_debug = build_physical_radius_mask(
            prepared_coords,
            self.physical_mask_radius,
            self.num_patches,
            keep_self_loop=self.physical_mask_self_loop,
        )

        debug_info = {
            'coords_shape': shape_as_list(prepared_coords),
            'spatial_embed_shape': shape_as_list(spatial_embed),
            'prompt_token_shape': shape_as_list(prompt_tokens),
            'z_scale_value': float(self.z_scale.detach().item()),
            'spatial_embed_has_nan': bool(torch.isnan(spatial_embed).any().item()),
            'spatial_embed_has_inf': bool(torch.isinf(spatial_embed).any().item()),
            'prompted_token_has_nan': bool(torch.isnan(prompted_tokens).any().item()),
            'prompted_token_has_inf': bool(torch.isinf(prompted_tokens).any().item()),
            'distance_stats': summarize_tensor(distance_matrix),
        }
        debug_info.update(prompt_debug)
        debug_info.update(mask_debug)
        return prompted_tokens, physical_mask, debug_info
