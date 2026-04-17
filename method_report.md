# 方法记录

- 任务：`long_term_forecast`
- 模型：`TimeFilter`
- 数据集：`custom`，文件为 `data/radar.csv`
- 输入方式：`features=M`，表示多监测点输入、多监测点输出
- 关键配置：`seq_len=96`、`label_len=48`、`pred_len=12/24/48/96`、`enc_in=1000`、`dec_in=1000`、`c_out=1000`
- 训练设置：`batch_size=1`、`train_epochs=1`、`num_workers=0`、`learning_rate=0.0005`、`dropout=0.1`、`top_p=0.0`、`use_norm=0`
- 参数选择：参考 `scripts/PEMS04.sh` 补充多窗口训练方式和部分超参，同时保留更稳的 `patch_len=96`、`d_model=32`、`d_ff=64`
- 评估方式：训练后执行测试，并输出 `mse`、`mae`
- 当前状态：脚本已改为顺序训练 4 组预测窗口
