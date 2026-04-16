# TimeFilter 模型分析报告

本文基于当前仓库中的实现进行分析，重点对应以下源码：

- [models/TimeFilter.py](/Users/dc/Z研究生/TimeFilter/TimeFilter/models/TimeFilter.py)
- [layers/TimeFilter_layers.py](/Users/dc/Z研究生/TimeFilter/TimeFilter/layers/TimeFilter_layers.py)
- [layers/StandardNorm.py](/Users/dc/Z研究生/TimeFilter/TimeFilter/layers/StandardNorm.py)
- [exp/exp_long_term_forecasting.py](/Users/dc/Z研究生/TimeFilter/TimeFilter/exp/exp_long_term_forecasting.py)
- [run.py](/Users/dc/Z研究生/TimeFilter/TimeFilter/run.py)

这份报告的目标不是只说“它是什么模块”，而是把 TimeFilter 的数据流、张量维度、图结构含义、训练损失和实现约束都串起来，方便学习和二次改造。

## 1. 一句话概括

TimeFilter 的核心思想可以概括成一句话：

> 先把多变量时间序列按“变量维度 + 时间维度”展平，再切成 patch token，然后在 patch token 上构造带结构先验的稀疏图，最后用图过滤器逐层提炼表示并直接回归未来序列。

它不是传统的 encoder-decoder Transformer，也不是单纯的时间卷积网络，而是一个“patch token graph forecasting”模型。

---

## 2. 总体结构

从实现上看，主干可以拆成 5 段：

1. 输入归一化
2. 展平 + patch embedding
3. TimeFilter Backbone
4. 预测头
5. 反归一化

对应到源码：

- 主模型在 [models/TimeFilter.py](/Users/dc/Z研究生/TimeFilter/TimeFilter/models/TimeFilter.py)
- 图学习与过滤在 [layers/TimeFilter_layers.py](/Users/dc/Z研究生/TimeFilter/TimeFilter/layers/TimeFilter_layers.py)
- 归一化在 [layers/StandardNorm.py](/Users/dc/Z研究生/TimeFilter/TimeFilter/layers/StandardNorm.py)

可以把整个前向理解成下面这个流程：

```text
x [B, T, C]
 -> Normalize
 -> permute / flatten
 -> patchify
 -> patch projection + positional embedding
 -> GraphBlock x E
 -> reshape to variable-major blocks
 -> linear head
 -> denormalize
 -> y_hat [B, pred_len, C]
```

---

## 3. 维度总览

设：

- `B` = batch size
- `T` = 输入历史长度 `seq_len`
- `C` = 通道数 / 变量数
- `P` = patch 长度 `patch_len`
- `N = T / P` = 每个变量被切成多少个 patch，前提是 `T` 能整除 `P`
- `L = C * N = C * T / P` = patch token 总数
- `D = d_model`
- `H = n_heads`
- `d_h = D / H`

### 3.1 一个具体例子

以 `ETTh1` 脚本为例：

- `T = 96`
- `C = 7`
- `P = 2`
- `N = 48`
- `L = 7 * 48 = 336`
- 如果 `d_model = 128`，那么 `d_h = 32`

所以在图模块里，单层 adjacency 的典型尺寸是：

```text
[B, H, 336, 336]
```

以 `PEMS04` 脚本为例：

- `T = 96`
- `C = 307`
- `P = 48`
- `N = 2`
- `L = 307 * 2 = 614`
- 如果 `d_model = 512`，那么 `d_h = 128`

所以图模块里的 adjacency 尺寸大约是：

```text
[B, H, 614, 614]
```

---

## 4. 输入到输出的完整维度流

下面按 `Model.forward()` 的真实实现逐步展开。

### 4.1 输入

源码位置：

- [models/TimeFilter.py:63-80](/Users/dc/Z研究生/TimeFilter/TimeFilter/models/TimeFilter.py#L63)

输入张量：

```text
x: [B, T, C]
```

这里 `batch_x` 来自数据加载器，长序列任务里通常是历史窗口。

### 4.2 归一化

源码位置：

- [models/TimeFilter.py:67](/Users/dc/Z研究生/TimeFilter/TimeFilter/models/TimeFilter.py#L67)
- [layers/StandardNorm.py:21-68](/Users/dc/Z研究生/TimeFilter/TimeFilter/layers/StandardNorm.py#L21)

`Normalize` 会按样本、按通道，在时间维上做标准化：

```text
x: [B, T, C] -> [B, T, C]
```

注意这里不是 RevIN 那种可学习反归一化，而是一个轻量版的 instance normalization 风格实现。

它会计算：

```text
mean:  [B, 1, C]
stdev: [B, 1, C]
```

然后做：

```text
x_norm = (x - mean) / stdev
```

### 4.3 维度变换与展平

源码位置：

- [models/TimeFilter.py:68-70](/Users/dc/Z研究生/TimeFilter/TimeFilter/models/TimeFilter.py#L68)

代码逻辑：

```python
x = x.permute(0, 2, 1).reshape(-1, C*T)
```

形状变化是：

```text
[B, T, C] -> [B, C, T] -> [B, C*T]
```

这一点非常关键。

TimeFilter 不是先对每个变量单独 patch，再跨变量建图，而是先把所有变量和时间拼成一个长序列，再统一切 patch。

### 4.4 Patch Embedding

源码位置：

- [models/TimeFilter.py:11-29](/Users/dc/Z研究生/TimeFilter/TimeFilter/models/TimeFilter.py#L11)

`PatchEmbed` 的核心是：

1. `unfold` 按 `patch_len` 切块
2. 用线性层把每个 patch 投影到 `d_model`
3. 可选加入位置编码

输入输出：

```text
[B, C*T] -> unfold -> [B, L, P] -> Linear(P -> D) -> [B, L, D]
```

其中：

- `L = C*T/P`
- `P = patch_len`
- `D = d_model`

如果开启位置编码：

```text
[B, L, D] + [1, L, D] -> [B, L, D]
```

#### 这一层的直观理解

每个 token 不是“一个时间点”，而是“展平序列上的一个 patch”。

也就是说，TimeFilter 的 token 语义更接近：

- 某个变量
- 某个时间段
- 这个时间段内部的局部模式

而不是传统 Transformer 中的单点 token。

### 4.5 Backbone 输入

源码位置：

- [models/TimeFilter.py:72](/Users/dc/Z研究生/TimeFilter/TimeFilter/models/TimeFilter.py#L72)

进入 backbone 前：

```text
[B, L, D]
```

其中 `L = C*T/P`。

---

## 5. TimeFilter Backbone 逐层解析

源码位置：

- [layers/TimeFilter_layers.py:259-278](/Users/dc/Z研究生/TimeFilter/TimeFilter/layers/TimeFilter_layers.py#L259)

Backbone 本质上是 `e_layers` 个 `GraphBlock` 堆叠：

```text
GraphBlock -> GraphBlock -> ... -> GraphBlock
```

每个 block 都包含：

1. LayerNorm
2. GraphFilter
3. 残差连接
4. FFN
5. 第二个残差连接

### 5.1 GraphBlock

源码位置：

- [layers/TimeFilter_layers.py:236-256](/Users/dc/Z研究生/TimeFilter/TimeFilter/layers/TimeFilter_layers.py#L236)

输入输出：

```text
x: [B, L, D]
```

#### 第一步：LayerNorm

```text
[B, L, D] -> [B, L, D]
```

#### 第二步：GraphFilter

```text
out, loss = self.gnn(self.norm1(x), ...)
```

`out`：

```text
[B, L, D]
```

#### 第三步：残差相加

```text
x = x + out
```

形状不变，仍然是：

```text
[B, L, D]
```

#### 第四步：FFN

```text
[B, L, D] -> [B, L, d_ff] -> [B, L, D]
```

最后再加一次残差。

---

## 6. GraphFilter 详解

源码位置：

- [layers/TimeFilter_layers.py:213-233](/Users/dc/Z研究生/TimeFilter/TimeFilter/layers/TimeFilter_layers.py#L213)

GraphFilter 是整个 TimeFilter 里最核心的部分。

它做三件事：

1. 学图
2. 过滤图
3. 图卷积

### 6.1 输入切头

输入：

```text
x: [B, L, D]
```

先 reshape 成多头形式：

```text
[B, L, D] -> [B, L, H, d_h] -> [B, H, L, d_h]
```

### 6.2 GraphLearner

源码位置：

- [layers/TimeFilter_layers.py:194-210](/Users/dc/Z研究生/TimeFilter/TimeFilter/layers/TimeFilter_layers.py#L194)

GraphLearner 先做一个双线性相似度建图：

```python
adj = gelu(einsum(proj_1(x), proj_2(x)))
```

形状是：

```text
x:   [B, H, L, d_h]
adj: [B, H, L, L]
```

这里的 `adj[i, j]` 表示：

> 第 `i` 个 token 和第 `j` 个 token 的关系强度

而且它是按 head 分开的，所以每个 head 可以学出不同的图。

### 6.3 Row-wise sparsification

源码位置：

- [layers/TimeFilter_layers.py:184-191](/Users/dc/Z研究生/TimeFilter/TimeFilter/layers/TimeFilter_layers.py#L184)

`mask_topk(adj, alpha)` 的作用是稀疏化邻接矩阵。

实现上它会对每一行做 `topk`，但注意代码里 `largest=False`，所以取的是较小值对应的位置，然后把这些位置置 0。

因此更准确地说，它的效果是：

> 在每个节点的候选邻居里，先去掉一小部分较弱连接，再保留剩下的边。

这和传统“只保留最大的 K 个边”的写法不完全一样，但目的都是稀疏化图结构。

输出仍然是：

```text
[B, H, L, L]
```

### 6.4 mask_moe：结构先验 + 动态路由

源码位置：

- [layers/TimeFilter_layers.py:93-181](/Users/dc/Z研究生/TimeFilter/TimeFilter/layers/TimeFilter_layers.py#L93)

这是 TimeFilter 的第二个关键点。

`mask_moe` 不是对 feature 做 MoE，而是对图结构做 MoE 风格的路由。

#### 输入

```text
adj: [B, H, L, L]
```

#### 输出

```text
mask: [B, H, L, L]
loss: scalar
```

#### 它的本质

对每个 query token，模型会在三类结构关系里做选择：

1. `S`：same patch position across variables
2. `T`：same variable across different patches
3. `ST`：spatial-temporal residual / other relations

对应到代码中的结构掩码：

```python
S  = 同一 patch 位置，但不同变量
T  = 同一变量内的其他 patch
ST = 其它关系
```

#### 掩码的构造逻辑

`masks` 的形状是：

```text
[L, 3, L]
```

含义是：

- 第 1 维：query token id
- 第 2 维：3 种专家/关系类型
- 第 3 维：key token id

因此每个 query token 都对应一组三路专家掩码。

#### 动态路由

`mask_moe` 内部有两个线性层：

- `gate`
- `noise`

它们都把长度为 `L` 的邻接行向量映射到 3 个 expert 的概率。

训练时会加噪声，这样路由会更“软”一点，避免总是选同一个专家。

#### 最终 mask

最终通过：

```text
[B, H, L, 3] 与 [L, 3, L]
```

组合成：

```text
[B, H, L, L]
```

这一步非常像“按 query token 自适应选择图模板”。

#### 额外损失

它还会返回一个 MoE 正则项：

```text
loss = importance_loss + 0.1 * dynamic_loss
```

这个损失的作用是：

1. 避免某个 expert 被一直独占
2. 保持路由分布有一定熵，不要过早塌缩

### 6.5 图掩码后再乘回 adjacency

```python
adj = adj * mask
```

这一步把结构先验真正施加到学出来的图上。

### 6.6 softmax + dropout

在 [layers/TimeFilter_layers.py:230-231](/Users/dc/Z研究生/TimeFilter/TimeFilter/layers/TimeFilter_layers.py#L230)，邻接矩阵会再做一次 row-wise softmax：

```text
[B, H, L, L] -> softmax -> [B, H, L, L]
```

然后 dropout。

这意味着最终图是一个“归一化的、稀疏的、带结构先验的动态图”。

### 6.7 GCN

源码位置：

- [layers/TimeFilter_layers.py:6-19](/Users/dc/Z研究生/TimeFilter/TimeFilter/layers/TimeFilter_layers.py#L6)

图卷积的输入输出：

```text
adj: [B, H, L, L]
x:   [B, L, D]
out: [B, L, D]
```

这里先做线性投影：

```text
[B, L, D] -> [B, L, D]
```

再拆成多头：

```text
[B, L, D] -> [B, L, H, d_h]
```

然后做图传播：

```text
adj @ x -> [B, L, H, d_h]
```

最后拼回：

```text
[B, L, H, d_h] -> [B, L, D]
```

这里的图卷积更像“在 token 图上的多头消息传递”，而不是传统 GCN 的固定邻接矩阵传播。

---

## 7. 预测头

源码位置：

- [models/TimeFilter.py:56-79](/Users/dc/Z研究生/TimeFilter/TimeFilter/models/TimeFilter.py#L56)

Backbone 输出：

```text
[B, L, D]
```

然后模型把 token 重新整理回“变量优先”的块结构：

```text
[B, L, D]
-> [B, C, N, D]
-> flatten last two dims
-> [B, C, N*D]
```

再通过线性层映射到预测长度：

```text
[B, C, N*D] -> [B, C, pred_len]
```

最后转成：

```text
[B, pred_len, C]
```

### 一个非常重要的理解

这个 head 并不是把所有变量融合后再一次性输出未来序列。

它更像是：

> Backbone 学到跨 token 的时空表示，head 再按变量把 token 序列压缩成未来长度。

因此最终输出维度和输入变量数保持一致。

---

## 8. 训练目标

源码位置：

- [exp/exp_long_term_forecasting.py:127-135](/Users/dc/Z研究生/TimeFilter/TimeFilter/exp/exp_long_term_forecasting.py#L127)

训练时总损失是：

```text
L = MSE(pred, true) + 0.05 * moe_loss
```

其中：

- `MSE` 是预测误差
- `moe_loss` 是路由均衡正则
- `0.05` 是固定权重

这意味着模型训练目标不仅要预测准，还要让 expert 路由不要退化成单一路径。

---

## 9. 数据加载与 mask 先验

源码位置：

- [exp/exp_long_term_forecasting.py:40-52](/Users/dc/Z研究生/TimeFilter/TimeFilter/exp/exp_long_term_forecasting.py#L40)

训练脚本在初始化时会构造一个固定 `masks`。

它的逻辑和 `mask_moe` 里的 fallback 逻辑一致，本质上是根据 token 的索引关系构造三类区域：

1. 同一变量不同时间片
2. 同一时间片不同变量
3. 其余关系

这说明模型不是完全自由地学习任意图，而是被注入了明显的时空结构偏置。

---

## 10. 这个实现里最值得注意的几个点

### 10.1 它是“展平后 patch”，不是“按变量分别 patch”

这是最容易误解的一点。

很多人第一次看会以为 patch 是沿时间维对每个变量分别做的，但这里实际上是：

```text
[B, T, C] -> [B, C*T] -> patch
```

所以 token 的顺序带有强烈的“变量优先”结构。

### 10.2 结构先验来自 token 索引，而不是显式坐标图

`S/T/ST` 掩码不是从物理坐标或距离矩阵直接算出来的，而是根据展平后的索引关系定义的。

这意味着模型更偏向于：

- 在 patch 级别建模
- 用规则先验限定可连接范围
- 再交给图学习模块做细化

### 10.3 最后预测头是 per-variable 的

backbone 虽然会让 token 间交互，但最终输出时仍按变量维度整理回来。

### 10.4 代码对超参有隐含约束

下面这些条件最好满足：

1. `seq_len % patch_len == 0`
2. `enc_in == c_out`
3. `d_model % n_heads == 0`

否则 shape 很容易对不上。

### 10.5 这份实现更偏向长序列多变量预测

当前 `Exp_Long_Term_Forecast` 才是最完整的使用路径。

---

## 11. 两个配置例子怎么读

### 11.1 ETTh1 配置

脚本位置：

- [scripts/ETTh1.sh](/Users/dc/Z研究生/TimeFilter/TimeFilter/scripts/ETTh1.sh)

关键参数：

- `seq_len=96`
- `pred_len=96/192/336/720`
- `patch_len=2`
- `d_model=128`
- `d_ff=256`
- `e_layers=2`
- `n_heads=4`
- `pos=0`

对应的 token 数：

```text
N = 96 / 2 = 48
L = 7 * 48 = 336
```

这意味着：

- 每个变量被切成 48 个 patch
- 总 token 数为 336
- 每个 head 的图尺寸是 `336 x 336`

### 11.2 PEMS04 配置

脚本位置：

- [scripts/PEMS04.sh](/Users/dc/Z研究生/TimeFilter/TimeFilter/scripts/PEMS04.sh)

关键参数：

- `seq_len=96`
- `pred_len=12/24/48`
- `patch_len=48`
- `d_model=512`
- `d_ff=1024`
- `top_p=0.0`
- `use_norm=0`

对应 token 数：

```text
N = 96 / 48 = 2
L = 307 * 2 = 614
```

这说明 PEMS04 的 patch 粒度很粗，每个变量只会形成两个大 patch。

另外 `top_p=0.0` 时，`mask_moe` 直接退化成单位矩阵，只保留自环，不做动态专家路由。

---

## 12. 学习 TimeFilter 的推荐顺序

如果你要系统学习，我建议按这个顺序看：

1. 先看 [models/TimeFilter.py](/Users/dc/Z研究生/TimeFilter/TimeFilter/models/TimeFilter.py) 的 `forward`
2. 再看 [layers/TimeFilter_layers.py](/Users/dc/Z研究生/TimeFilter/TimeFilter/layers/TimeFilter_layers.py) 的 `GraphBlock`
3. 接着看 `GraphFilter` 和 `GraphLearner`
4. 然后看 `mask_moe`
5. 最后回看 `exp/exp_long_term_forecasting.py` 的 mask 构造和训练损失

这样最容易建立完整闭环。

---

## 13. 用最朴素的话总结它的工作方式

如果把 TimeFilter 讲成人话，它做的事情就是：

1. 把历史序列切成一段一段的 patch
2. 把这些 patch 当成图上的节点
3. 根据节点之间的相似性学一张图
4. 再根据“同变量 / 同时间位置 / 其他关系”这类先验，筛掉不该连的边
5. 用图卷积把信息传播几层
6. 最后直接回归到未来序列

它的优势在于：

- 计算单元是 patch，不是单点
- 图结构有先验，不完全盲学
- 能同时处理变量间关系和时间片关系

---

## 14. 你如果要继续深入，下一步最值得做的事

1. 用一个具体 batch 手工打印每一步 shape，真正跑一遍 forward
2. 把 `mask_topk` 和 `mask_moe` 的输出可视化成热力图
3. 对比 `top_p=0`、`top_p=0.5`、`top_p=1.0` 的图稀疏度
4. 对比 `pos=0` 和 `pos=1` 的效果
5. 把 patch_len 改成不同值，观察 `L` 如何变化以及性能如何变化

如果你愿意，我可以下一步继续帮你做两件很实用的事之一：

1. 直接给你写一份“带公式的逐行源码讲解版”
2. 直接给你补一个“shape 打印脚本”，让你一跑就能看到每层维度变化

