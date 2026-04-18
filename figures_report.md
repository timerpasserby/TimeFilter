# 图表记录

- 本次没有新增 Python 绘图输出。
- 本阶段新增的是 `CSPAdapter` 接入、smoke 测试脚本和调试日志，没有新增可视化图表。
- `exp/exp_long_term_forecasting.py` 已补回测试阶段的 `visual()` 输出能力，后续正式测试时会在 `test_results/` 下生成预测曲线 pdf。
- 天气注入模块已返回 `attn_weights`，后续可以直接据此绘制天气因果热力图；本阶段尚未生成实际热力图文件。
- 爆破注入模块已返回 `e_it` 与 `g_t`，后续可以直接绘制爆破扰动强度热力图和门控开度热力图；本阶段尚未生成实际图片文件。
- 本阶段新增的是主干接入和正式训练脚本升级，包括 `scripts/radar_ablation_pipeline.sh` 与 `run_models.sh` 的新实验开关，不产生新的图表文件。
- `scripts/radar_ablation_pipeline.sh` 已默认启用 `--inverse`，后续新生成的预测图会直接使用原始位移尺度。
- `scripts/csp_adapter_smoke.py` 输出的是控制台调试统计，不生成图片文件。
- `tests/test_timefilter_exogenous_integration.py` 和真数据单 batch 冒烟只验证前后向可运行性，不生成图片文件。
- 当前未生成新的结果图文件。
