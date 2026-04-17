# 图表记录

- 本次没有新增 Python 绘图输出。
- 本阶段新增的是 `CSPAdapter` 接入、smoke 测试脚本和调试日志，没有新增可视化图表。
- `exp/exp_long_term_forecasting.py` 已补回测试阶段的 `visual()` 输出能力，后续正式测试时会在 `test_results/` 下生成预测曲线 pdf。
- 天气注入模块已返回 `attn_weights`，后续可以直接据此绘制天气因果热力图；本阶段尚未生成实际热力图文件。
- `scripts/csp_adapter_smoke.py` 输出的是控制台调试统计，不生成图片文件。
- 当前未生成新的结果图文件。
