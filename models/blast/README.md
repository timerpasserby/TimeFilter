# Physics-Informed Step-Response Blast Injection

## 模块作用

这个模块用于“基于爆破时空解析与旁路门控的瞬态扰动建模”。
它不改动 `CSP-TimeFilter` 主干，也不改动天气模块，而是接在天气模块输出 `H_exo` 之后，把离散爆破事件解析成节点级连续扰动，再以旁路残差形式注入。

## 解析定义

核心解析量定义为：

```text
e_{i,t} = Σ_{k: t_k <= t} a_k * exp(-d_{i,k}^2 / (2 * sigma_b^2)) * exp(-gamma_b * (t - t_k))
```

其中：

- `d_{i,k}` 是节点 `i` 到第 `k` 次爆破位置的欧氏距离
- `a_k` 是第 `k` 次爆破强度
- `sigma_b` 是可学习的空间衰减尺度
- `gamma_b` 是可学习的时间衰减系数

当前实现中 `sigma_b` 和 `gamma_b` 都通过 `softplus` 保证为正。

## 输入输出

- `H_exo`: `[B, T, N, D]`
  含义：天气模块输出的增强隐状态
- `node_coords`: `[N, 3]`
  含义：监测点三维坐标
- `blast_locs`: `[B, K, 3]`
  含义：历史窗口内爆破位置
- `blast_times`: `[B, K]`
  含义：历史窗口内爆破时刻
- `blast_intensity`: `[B, K]`
  含义：爆破强度
- `target_times`: `[B, T]`
  含义：当前预测时间索引

- `H_final`: `[B, T, N, D]`
  含义：注入爆破瞬态后的最终隐状态
- `e_it`: `[B, T, N, 1]`
  含义：解析得到的节点级连续扰动
- `g_t`: `[B, T, N, 1]`
  含义：门控开度
- `delta_H_blast`: `[B, T, N, D]`
  含义：爆破旁路待注入的特征增量

## 因果性如何保证

1. `BlastAnalyticEncoder` 中显式计算 `target_times - blast_times`
   只有满足 `t_k <= t` 的事件才会被 `causal mask` 保留。

2. 未来爆破事件会被掩掉
   因此未来爆破不会影响当前或过去输出。

3. 测试中专门验证未来爆破不影响过去结果
   `tests/test_blast_causality.py` 会只修改未来事件，并检查过去输出保持一致。

## 门控如何实现旁路注入

主模型使用：

```python
H_final = H_exo + g_t * delta_H_blast
```

其中：

- `g_t = sigmoid(W_g(e_it))`
- `delta_H_blast = W_v(e_it)`

这样主路径仍然保留 `H_exo`，爆破扰动只通过旁路残差进入，不会把瞬态事件硬塞回主干内部。

## 三种模式如何切换

主模型：

```python
PhysicsInformedStepResponseBlastInjection(mode='main', ...)
```

去掉门控的消融：

```python
PhysicsInformedStepResponseBlastInjection(mode='wo_gate', ...)
```

GRU 替代消融：

```python
PhysicsInformedStepResponseBlastInjection(mode='gru_blast', ...)
```

## 与上游模块的接口位置

接口位置在天气模块输出之后：

```python
H_exo, _ = weather_block(H_main, weather_seq)
blast_block = PhysicsInformedStepResponseBlastInjection(...)
H_final, e_it, g_t, delta_H_blast = blast_block(
    H_exo, node_coords, blast_locs, blast_times, blast_intensity, target_times
)
```

## 如何运行测试

在项目根目录执行：

```bash
/opt/homebrew/Caskroom/miniforge/base/envs/tslib/bin/python -m unittest discover -s tests -p "test_blast_*.py"
```

## 最小 forward demo

```bash
/opt/homebrew/Caskroom/miniforge/base/envs/tslib/bin/python scripts/blast_injection_demo.py
```
