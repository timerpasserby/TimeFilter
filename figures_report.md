# 图表记录

- 本次没有新增 Python 绘图输出。
- 本阶段新增的是 `CSPAdapter` 接入、smoke 测试脚本和调试日志，没有新增可视化图表。
- `exp/exp_long_term_forecasting.py` 已补回测试阶段的 `visual()` 输出能力，后续正式测试时会在 `test_results/` 下生成预测曲线 pdf。
- 天气注入模块已返回 `attn_weights`，后续可以直接据此绘制天气因果热力图；本阶段尚未生成实际热力图文件。
- 爆破注入模块已返回 `e_it` 与 `g_t`，后续可以直接绘制爆破扰动强度热力图和门控开度热力图；本阶段尚未生成实际图片文件。
- 本阶段新增的是命令整理脚本 `scripts/radar_ablation_pipeline.sh`，不产生新的图表文件。
- 本阶段新增 Windows 批处理脚本 `run_models_windows.bat` 和 `scripts/radar_ablation_pipeline_windows.bat`，不产生新的图表文件。
- `scripts/csp_adapter_smoke.py` 输出的是控制台调试统计，不生成图片文件。
- 当前未生成新的结果图文件。
