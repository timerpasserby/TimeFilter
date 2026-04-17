# 当前正在做什么
在 `TimeFilter` 中接入统一空间增强模块 `CSPAdapter`，并完成最小可运行验证。

# 上次停在哪个位置
2026-04-17：已完成 `CSPAdapter` 主干接入、真实 radar 数据 smoke 测试和调试脚本 `scripts/csp_adapter_smoke.py`。

# 近期关键决定和原因
- **统一空间模块**：连续坐标编码、patch prompt 注入、物理半径掩码合并到 `models/csp_adapter.py`，减少主干改动。
- **数据路径统一**：时序数据使用 `dataset/radar/sim_radar_hourly_displacement.csv`，坐标使用 `dataset/radar/sim_nodes_static.csv`。
- **兼容性优先**：`use_csp_adapter=False` 保持原始 baseline；`Dataset_Custom` 兼容 `report_time` 时间列；非 M4 的 `short_term_forecast` 复用通用监督预测流程。
