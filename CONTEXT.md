# 当前正在做什么
在 `TimeFilter` 中补齐短期预测实验流程，并保持与 `CSPAdapter` 版本兼容。

# 上次停在哪个位置
2026-04-17：已按参考实现完善 `exp/exp_short_term_forecasting.py`，保留 M4 评估流程，并适配 TimeFilter 的直接预测接口。

# 近期关键决定和原因
- **统一空间模块**：连续坐标编码、patch prompt 注入、物理半径掩码合并到 `models/csp_adapter.py`，减少主干改动。
- **数据路径统一**：时序数据使用 `dataset/radar/sim_radar_hourly_displacement.csv`，坐标使用 `dataset/radar/sim_nodes_static.csv`。
- **兼容性优先**：`use_csp_adapter=False` 保持原始 baseline；`Dataset_Custom` 兼容 `report_time` 时间列；非 M4 的 `short_term_forecast` 复用通用监督预测流程。
- **实验流程对齐**：参考 `Time-Series-Library` 的长期预测脚本补齐验证、测试和结果保存能力，但不引入 encoder-decoder 接口，继续沿用 TimeFilter 原生前向方式。
- **短期流程适配**：参考 `Time-Series-Library` 的短期预测脚本补齐 M4 风格验证和结果导出，但对 TimeFilter 走直接预测前向，不再把它当成 encoder-decoder 模型调用。
