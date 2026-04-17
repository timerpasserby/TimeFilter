# Physics-Constrained Causal Weather Injection

## 模块作用

这个模块用于“基于因果天气注入的缓变外生影响建模”。
它不改动 CSP-TimeFilter 主干，只接收主干输出 `H_main`，再把全局共享天气序列以“因果、滞后、节点异质”的方式注入进去。

## 输入输出

- `H_main`: `[B, T, N, D]`
  含义：CSP-TimeFilter 主干输出的位移演化隐状态
- `weather_seq`: `[B, T, Cw]`
  含义：全局共享天气序列，默认 `Cw=3`
- `optional_mask`: `[B, T]`、`[B, T, 1]` 或 `[B, T, Cw]`
  含义：天气缺失值或 padding 掩码

- `H_exo`: `[B, T, N, D]`
  含义：天气增强后的隐状态
- `attn_weights`: `[B, N, H, T, T]`
  含义：天气跨注意力权重，便于后续画热力图

## 因果性如何保证

1. 天气序列先进入 `CausalConv1dWeatherEncoder`
   这里使用严格因果卷积，卷积采用左侧 padding，卷积后裁剪未来位置。

2. 天气跨注意力显式使用下三角 causal mask
   `PhysicsConstrainedCausalCrossAttention` 会构造 `[T, T]` 下三角掩码，确保时刻 `t` 只能看到 `0...t` 的天气。

3. 测试里专门验证未来天气不会影响过去输出
   `tests/test_weather_causality.py` 会修改未来天气值，并检查过去时间步输出保持一致。

## 与主干的接口位置

接口位置在主干输出之后：

```python
H_main = backbone_output  # [B, T, N, D]
weather_block = PhysicsConstrainedCausalWeatherInjection(...)
H_exo, attn_weights = weather_block(H_main, weather_seq, optional_mask=weather_mask)
```

这个模块不会重复做坐标编码、空间提示注入或物理半径掩码，这些内容仍由 CSP-TimeFilter 主干负责。

## 如何运行测试

在项目根目录执行：

```bash
/opt/homebrew/Caskroom/miniforge/base/envs/tslib/bin/python -m unittest discover -s tests -p "test_weather_*.py"
```

## 最小 forward demo

```bash
/opt/homebrew/Caskroom/miniforge/base/envs/tslib/bin/python scripts/weather_injection_demo.py
```

## 如何切换 w/o Causal-Attn

主模型：

```python
PhysicsConstrainedCausalWeatherInjection(ablation_mode='causal_attn', ...)
```

取消 causal mask 的 attention 消融：

```python
PhysicsConstrainedCausalWeatherInjection(ablation_mode='vanilla_attn', ...)
```

取消 attention、改为拼接融合的消融：

```python
PhysicsConstrainedCausalWeatherInjection(ablation_mode='concat_fusion', ...)
```
