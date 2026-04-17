"""这个层文件负责 TimeFilter 的图学习与过滤主干，并支持接入物理半径约束的调试统计。"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal


# 将张量统计成便于日志输出的数值摘要。
def tensor_stats(tensor):
    """统计张量的均值、最大值和稀疏度。"""
    data = tensor.detach()
    return {
        'mean': float(data.mean().item()),
        'max': float(data.max().item()),
        'sparsity': float((data.abs() < 1e-8).float().mean().item()),
        'has_nan': bool(torch.isnan(data).any().item()),
        'has_inf': bool(torch.isinf(data).any().item()),
    }

class GCN(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        self.n_heads = n_heads

    def forward(self, adj, x):
        # adj [B, H, L, L]
        B, L, D = x.shape
        x = self.proj(x).view(B, L, self.n_heads, -1)  # [B, L, H, D_]
        adj = F.normalize(adj, p=1, dim=-1)
        x = torch.einsum("bhij,bjhd->bihd", adj, x).contiguous()  # [B, L, H, D_]
        x = x.view(B, L, -1)
        return x

###############################
# Ablation
###############################
def mask_topk_moe(adj, thre, n_vars, masks):
    #adj: [B, H, L, L], thre: [B, H, L, 3]
    if masks is None:
        B, H, L, _ = adj.shape
        N = L // n_vars
        device = adj.device
        dtype = torch.float32
        print("Masks is None!")
        masks = []
        for k in range(L):
            S = ((torch.arange(L) % N == k % N) & (torch.arange(L) != k)).to(dtype).to(device)
            T = ((torch.arange(L) >= k // N * N) & (torch.arange(L) < k // N * N + N)).to(dtype).to(device)
            ST = torch.ones(L).to(dtype).to(device) - S - T
            masks.append(torch.stack([S, T, ST], dim=0))
        # [L, 3, L]
        masks = torch.stack(masks, dim=0)

    adj_mask0 = adj * masks[:,0,:]
    adj_mask1 = adj * masks[:,1,:]
    adj_mask2 = adj * masks[:,2,:]

    adj_mask0[adj_mask0 <= thre[:, :, :, 0].unsqueeze(-1)] = 0
    adj_mask1[adj_mask1 <= thre[:, :, :, 1].unsqueeze(-1)] = 0 
    adj_mask2[adj_mask2 <= thre[:, :, :, 2].unsqueeze(-1)] = 0

    adj = adj_mask0 + adj_mask1 + adj_mask2

    return adj

def mask_topk_area(adj, n_vars, masks, alpha=0.5):
    # x: [B, H, L, L]
    B, H, L, _ = adj.shape
    N = L // n_vars
    if masks is None:
        device = adj.device
        dtype = torch.float32
        print("Masks is None!")
        masks = []
        for k in range(L):
            S = ((torch.arange(L) % N == k % N) & (torch.arange(L) != k)).to(dtype).to(device)
            T = ((torch.arange(L) >= k // N * N) & (torch.arange(L) < k // N * N + N)).to(dtype).to(device)
            ST = torch.ones(L).to(dtype).to(device) - S - T
            masks.append(torch.stack([S, T, ST], dim=0))
        # [L, 3, L]
        masks = torch.stack(masks, dim=0)
    # masks [L, 3, L]
    n0 = n_vars - 1
    n1 = N - 1
    n2 = L - n0 - n1 - 1

    adj_mask0 = adj * masks[:,0,:]
    adj_mask1 = adj * masks[:,1,:]
    adj_mask2 = adj * masks[:,2,:]

    def apply_mask_to_region(adj_mask, n):
        threshold_idx = int(n * alpha)
        sorted_values, _ = torch.sort(adj_mask, dim=-1, descending=True)
        threshold = sorted_values[:, :, :, threshold_idx]
        return adj_mask * (adj_mask >= threshold.unsqueeze(-1))

    adj_mask0 = apply_mask_to_region(adj_mask0, n0)
    adj_mask1 = apply_mask_to_region(adj_mask1, n1)
    adj_mask2 = apply_mask_to_region(adj_mask2, n2)

    adj = adj_mask0 + adj_mask1 + adj_mask2

    return adj
##########################

class mask_moe(nn.Module):
    def __init__(self, n_vars, top_p=0.5, num_experts=3, in_dim=96):
        super().__init__()
        self.num_experts = num_experts
        self.n_vars = n_vars
        self.in_dim = in_dim

        self.gate = nn.Linear(self.in_dim, num_experts, bias=False)
        self.noise = nn.Linear(self.in_dim, num_experts, bias=False)
        self.noisy_gating = True
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(2)
        self.top_p = top_p

    def cv_squared(self, x):
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def cross_entropy(self, x):
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return -torch.mul(x, torch.log(x + eps)).sum(dim=1).mean()

    def noisy_top_k_gating(self, x, is_training, noise_epsilon=1e-2):
        clean_logits = self.gate(x)
        if self.noisy_gating and is_training:
            raw_noise = self.noise(x)
            noise_stddev = ((self.softplus(raw_noise) + noise_epsilon))
            noisy_logits = clean_logits + torch.randn_like(clean_logits) * noise_stddev
            logits = noisy_logits
        else:
            logits = clean_logits

        # Convert logits to probabilities
        logits = self.softmax(logits)
        loss_dynamic = self.cross_entropy(logits)

        sorted_probs, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        mask = cumulative_probs > self.top_p

        threshold_indices = mask.long().argmax(dim=-1)
        threshold_mask = torch.nn.functional.one_hot(threshold_indices, num_classes=sorted_indices.size(-1)).bool()
        mask = mask & ~threshold_mask

        top_p_mask = torch.zeros_like(mask)
        zero_indices = (mask == 0).nonzero(as_tuple=True)
        top_p_mask[zero_indices[0], zero_indices[1], sorted_indices[zero_indices[0], zero_indices[1], zero_indices[2]]] = 1

        sorted_probs = torch.where(mask, 0.0, sorted_probs)
        loss_importance = self.cv_squared(sorted_probs.sum(0))
        lambda_2 = 0.1
        loss = loss_importance + lambda_2 * loss_dynamic

        return top_p_mask, loss

    def forward(self, x, masks=None, is_training=None, physical_mask=None, debug_info=None):
        # x [B, H, L, L]
        B, H, L, _ = x.shape
        device = x.device
        dtype = torch.float32

        mask_base = torch.eye(L, device=device, dtype=dtype).unsqueeze(0).unsqueeze(0)
        if self.top_p == 0.0:
            mask = mask_base
            if physical_mask is not None:
                mask = mask * physical_mask.to(dtype=dtype)
            if debug_info is not None:
                debug_info['routing_mask_shape'] = list(mask.shape)
                debug_info['routing_mask_active_ratio'] = float(mask.float().mean().item())
            return mask, 0.0
        
        x = x.reshape(B * H, L, L)
        gates, loss = self.noisy_top_k_gating(x, is_training)
        gates = gates.reshape(B, H, L, -1).float()
        # [B, H, L, 3]

        if masks is None:
            print("Masks is None!")
            masks = []
            N = L // self.n_vars
            for k in range(L):
                S = ((torch.arange(L) % N == k % N) & (torch.arange(L) != k)).to(dtype).to(device)
                T = ((torch.arange(L) >= k // N * N) & (torch.arange(L) < k // N * N + N)).to(dtype).to(device)
                ST = torch.ones(L).to(dtype).to(device) - S - T
                masks.append(torch.stack([S, T, ST], dim=0))
            # [L, 3, L]
            masks = torch.stack(masks, dim=0)

        mask = torch.einsum('bhli,lid->bhld', gates, masks) + mask_base
        if physical_mask is not None:
            mask = mask * physical_mask.to(dtype=dtype)
        if debug_info is not None:
            debug_info['routing_mask_shape'] = list(mask.shape)
            debug_info['routing_mask_active_ratio'] = float(mask.float().mean().item())

        return mask, loss


def mask_topk(x, alpha=0.5, largest=False):
    # B, L = x.shape[0], x.shape[-1]
    # x: [B, H, L, L]
    k = int(alpha * x.shape[-1])
    _, topk_indices = torch.topk(x, k, dim=-1, largest=largest)
    mask = torch.ones_like(x, dtype=torch.float32)
    mask.scatter_(-1, topk_indices, 0)  # 1 is topk
    return mask  # [B, H, L, L]


class GraphLearner(nn.Module):
    def __init__(self, dim, n_vars, top_p=0.5, in_dim=96):
        super().__init__()
        self.dim = dim
        self.proj_1 = nn.Linear(dim, dim)
        self.proj_2 = nn.Linear(dim, dim)
        self.n_vars = n_vars
        self.mask_moe = mask_moe(n_vars, top_p=top_p, in_dim=in_dim)

    def forward(self, x, masks=None, alpha=0.5, is_training=False, physical_mask=None, debug_info=None):
        # x: [B, H, L, D]
        adj = F.gelu(torch.einsum('bhid,bhjd->bhij', self.proj_1(x), self.proj_2(x)))
        if debug_info is not None:
            before_stats = tensor_stats(adj)
            debug_info['adjacency_shape'] = list(adj.shape)
            debug_info['adj_before_mask_mean'] = before_stats['mean']
            debug_info['adj_before_mask_max'] = before_stats['max']
            debug_info['adj_before_mask_sparsity'] = before_stats['sparsity']
            debug_info['adj_before_mask_has_nan'] = before_stats['has_nan']
            debug_info['adj_before_mask_has_inf'] = before_stats['has_inf']

        if physical_mask is not None:
            adj = adj * physical_mask.to(dtype=adj.dtype, device=adj.device)

        adj = adj * mask_topk(adj, alpha)
        mask, loss = self.mask_moe(adj, masks, is_training, physical_mask=physical_mask, debug_info=debug_info)
        adj = adj * mask

        if debug_info is not None:
            after_stats = tensor_stats(adj)
            debug_info['adj_after_mask_mean'] = after_stats['mean']
            debug_info['adj_after_mask_max'] = after_stats['max']
            debug_info['adj_after_mask_sparsity'] = after_stats['sparsity']
            debug_info['adj_after_mask_has_nan'] = after_stats['has_nan']
            debug_info['adj_after_mask_has_inf'] = after_stats['has_inf']

        return adj, loss  # [B, H, L, L]


class GraphFilter(nn.Module):
    def __init__(self, dim, n_vars, n_heads=4, scale=None, top_p=0.5, dropout=0., in_dim=96):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.scale = dim ** (-0.5) if scale is None else scale
        self.dropout = nn.Dropout(dropout)
        self.graph_learner = GraphLearner(self.dim // self.n_heads, n_vars, top_p, in_dim=in_dim)
        self.graph_conv = GCN(self.dim, self.n_heads)

    def forward(self, x, masks=None, alpha=0.5, is_training=False, physical_mask=None, debug_info=None):
        # x: [B, L, D]
        B, L, D = x.shape

        adj, loss = self.graph_learner(
            x.reshape(B, L, self.n_heads, -1).permute(0, 2, 1, 3),
            masks,
            alpha,
            is_training,
            physical_mask=physical_mask,
            debug_info=debug_info,
        )

        if physical_mask is not None:
            adj = adj.masked_fill(~physical_mask.to(dtype=torch.bool, device=adj.device), -1e4)

        adj = torch.softmax(adj, dim=-1)
        if debug_info is not None:
            softmax_stats = tensor_stats(adj)
            debug_info['adj_softmax_mean'] = softmax_stats['mean']
            debug_info['adj_softmax_max'] = softmax_stats['max']
            debug_info['adj_softmax_sparsity'] = softmax_stats['sparsity']
            debug_info['adj_softmax_has_nan'] = softmax_stats['has_nan']
            debug_info['adj_softmax_has_inf'] = softmax_stats['has_inf']
        adj = self.dropout(adj)
        out = self.graph_conv(adj, x)
        return out, loss  # [B, L, D]


class GraphBlock(nn.Module):
    def __init__(self, dim, n_vars, d_ff=None, n_heads=4, top_p=0.5, dropout=0., in_dim=96):
        super().__init__()
        self.dim = dim
        self.d_ff = dim * 4 if d_ff is None else d_ff
        self.gnn = GraphFilter(self.dim, n_vars, n_heads, top_p=top_p, dropout=dropout, in_dim=in_dim)
        self.norm1 = nn.LayerNorm(self.dim)
        self.ffn = nn.Sequential(
            nn.Linear(self.dim, self.d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_ff, self.dim),
        )
        self.norm2 = nn.LayerNorm(self.dim)

    def forward(self, x, masks=None, alpha=0.5, is_training=False, physical_mask=None, debug_info=None):
        # x: [B, L, D], time_embed: [B, time_embed_dim]
        out, loss = self.gnn(self.norm1(x), masks, alpha, is_training, physical_mask=physical_mask, debug_info=debug_info)
        x = x + out
        x = x + self.ffn(self.norm2(x))
        return x, loss


class TimeFilter_Backbone(nn.Module):
    def __init__(self, hidden_dim, n_vars, d_ff=None, n_heads=4, n_blocks=3, top_p=0.5, dropout=0., in_dim=96):
        super().__init__()
        self.dim = hidden_dim
        self.d_ff = self.dim * 2 if d_ff is None else d_ff
        # graph blocks
        self.blocks = nn.ModuleList([
            GraphBlock(self.dim, n_vars, self.d_ff, n_heads, top_p, dropout, in_dim)
            for _ in range(n_blocks)
        ])
        self.n_blocks = n_blocks

    def forward(self, x, masks=None, alpha=0.5, is_training=False, physical_mask=None, debug_info=None):
        # x: [B, N, T]
        moe_loss = 0.0
        for block in self.blocks:
            x, loss = block(x, masks, alpha, is_training, physical_mask=physical_mask, debug_info=debug_info)
            moe_loss += loss
        moe_loss /= self.n_blocks
        return x, moe_loss, debug_info  # [B, N, T]
