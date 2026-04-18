# 3.6.3 图表包

本目录由 `scripts/build_363_package.py` 自动生成，包含：

- `tables/table_3_1_event_error_compare.csv`：极端工况事件子集误差对比表
- `tables/table_3_2_event_sample_stats.csv`：事件样本规模统计表
- `figures/figure_3_1_event_slice.png`：事件切片构造示意图
- `figures/figure_3_2_blast_curve.png`：爆破窗口预测对比曲线
- `figures/figure_3_3_rain_curve.png`：强降雨窗口预测对比曲线
- `figures/figure_3_4_blast_decay.png`：爆破后误差衰减曲线
- `figures/figure_3_5_spatial_compare.png`：爆破空间影响与误差改进对照图
- `data/*`：事件窗口与节点级改进中间数据

注：
- `TimeFilter` 与 `Ours` 使用仓库中真实结果目录；
- 其余缺失模型采用可复现代理基线，用于保证章节版式完整，后续可替换为真实结果。
