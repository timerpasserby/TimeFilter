# 当前正在做什么
在 `TimeFilter` 中整理当前可运行的消融实验命令脚本，并保持主干、天气模块、爆破模块的接口状态说明一致。

# 上次停在哪个位置
2026-04-17：已完成天气模块、爆破模块和对应测试；当前补充统一的消融命令脚本 `scripts/radar_ablation_pipeline.sh`。

# 近期关键决定和原因
- **统一空间模块**：连续坐标编码、patch prompt 注入、物理半径掩码合并到 `models/csp_adapter.py`，减少主干改动。
- **数据路径统一**：时序数据使用 `dataset/radar/sim_radar_hourly_displacement.csv`，坐标使用 `dataset/radar/sim_nodes_static.csv`。
- **兼容性优先**：`use_csp_adapter=False` 保持原始 baseline；`Dataset_Custom` 兼容 `report_time` 时间列；非 M4 的 `short_term_forecast` 复用通用监督预测流程。
- **实验流程对齐**：参考 `Time-Series-Library` 的长期预测脚本补齐验证、测试和结果保存能力，但不引入 encoder-decoder 接口，继续沿用 TimeFilter 原生前向方式。
- **短期流程适配**：参考 `Time-Series-Library` 的短期预测脚本补齐 M4 风格验证和结果导出，但对 TimeFilter 走直接预测前向，不再把它当成 encoder-decoder 模型调用。
- **天气模块独立**：天气缓变影响模块只消费主干输出 `H_main`，不修改主干结构，不重复实现坐标编码与空间提示。
- **因果性显式保证**：天气先过严格因果卷积，再过显式下三角掩码跨注意力，并用单元测试验证未来天气不会影响过去输出。
- **爆破模块独立**：爆破瞬态扰动模块只消费天气增强后的 `H_exo`，通过解析 `e_it` 后再做门控旁路注入，不改动主干和天气模块。
- **主模型禁用 RNN**：主爆破分支固定为 `解析建模 + gate + bypass`，`GRU` 只保留在 `gru_blast` 消融模式里。
- **命令统一整理**：当前正式接入训练入口的只有 `baseline / CSP-TimeFilter`，weather / blast 先通过 `scripts/radar_ablation_pipeline.sh` 作为模块级检查统一管理，避免误当成正式训练实验。
