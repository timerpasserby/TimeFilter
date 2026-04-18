# 当前正在做什么
已把天气模块和爆破模块正式接入 `TimeFilter` 主干，并同步补齐训练脚本、数据入口和集成测试。

# 上次停在哪个位置
2026-04-17：天气模块、爆破模块和模块级测试已经完成，但它们还没有真正接到主干训练链路里。

# 近期关键决定和原因
- **主干最小改动**：天气与爆破都只接在 `TimeFilter_Backbone` 输出后的 patch 级隐状态上，不改 backbone 主结构。
- **数据侧统一对齐**：`Dataset_Custom` 直接从 `dataset/radar/` 读取主序列、天气和爆破日志，并按同一时间窗口返回 `extra_inputs`。
- **训练入口正式打通**：`run.py`、`exp/exp_long_term_forecasting.py`、`exp/exp_short_term_forecasting.py` 已支持把天气和爆破侧信息送进模型。
- **正式消融脚本升级**：`scripts/radar_ablation_pipeline.sh` 现在不仅能跑 baseline / CSP，也能直接跑 weather / blast 主模型与消融训练。
- **结果尺度统一**：`scripts/radar_ablation_pipeline.sh` 训练命令默认增加 `--inverse`，测试导出的预测结果按原始位移尺度保存。
