**1. 图的用途说明**
- 图1“原始二维投影点分布图”适合放在第四章“数据预处理与监测点空间分布”部分，用来说明二维投影后的监测点覆盖范围与密度。
- 图2“工程分区可视化图”适合放在第四章“工程分区与空间先验构建”部分，用来说明坡顶、坡中、坡脚、平台、背景区的空间划分思路。
- 图3“规则网格 patch 划分图”适合放在第四章“空间单元构建”部分，用来说明从原始监测点到规则 patch 的空间分块过程。
- 图4“代表性 patch 放大与统计特征示意图”适合放在第四章“空间单元特征构造”部分，用来说明 patch 内均值、标准差和有效点比例等统计特征如何形成。
- 图5“时空分块示意图”适合放在第三章“时空分块表示机制”部分，用来说明“空间 patch + 时间 block”如何组合成时空张量。
- 图6“模型输入张量结构图”适合放在第三章或第五章模型结构介绍部分，用来解释从原始输入 `X` 到 `H^{patch}` 的维度变化。

**2. Python 代码**
- 完整可运行代码已写入 [plot_thesis_spatiotemporal_figures.py](/Users/dc/Z研究生/TimeFilter/TimeFilter/scripts/plot_thesis_spatiotemporal_figures.py)。
- 关键可调参数在 [plot_thesis_spatiotemporal_figures.py:26](/Users/dc/Z研究生/TimeFilter/TimeFilter/scripts/plot_thesis_spatiotemporal_figures.py#L26) 和 [plot_thesis_spatiotemporal_figures.py:59](/Users/dc/Z研究生/TimeFilter/TimeFilter/scripts/plot_thesis_spatiotemporal_figures.py#L59)，包括 `patch_width`、`patch_height`、`time_block_hours`、`dpi`、输出目录等。
- 工程分区规则在 [plot_thesis_spatiotemporal_figures.py:37](/Users/dc/Z研究生/TimeFilter/TimeFilter/scripts/plot_thesis_spatiotemporal_figures.py#L37) 和 [plot_thesis_spatiotemporal_figures.py:155](/Users/dc/Z研究生/TimeFilter/TimeFilter/scripts/plot_thesis_spatiotemporal_figures.py#L155)。这部分目前是“基于坐标比例 + 稀有 `area_id` 自动识别背景区”的默认启发式规则，因为你的这份数据里 `area_id` 只有 2 类，不足以直接对应坡顶/坡中/坡脚/平台；如果你后面有真实工程 mask，直接替换这里最合适。
- 各图绘制函数分别在 [plot_thesis_spatiotemporal_figures.py:362](/Users/dc/Z研究生/TimeFilter/TimeFilter/scripts/plot_thesis_spatiotemporal_figures.py#L362)、[plot_thesis_spatiotemporal_figures.py:394](/Users/dc/Z研究生/TimeFilter/TimeFilter/scripts/plot_thesis_spatiotemporal_figures.py#L394)、[plot_thesis_spatiotemporal_figures.py:430](/Users/dc/Z研究生/TimeFilter/TimeFilter/scripts/plot_thesis_spatiotemporal_figures.py#L430)、[plot_thesis_spatiotemporal_figures.py:490](/Users/dc/Z研究生/TimeFilter/TimeFilter/scripts/plot_thesis_spatiotemporal_figures.py#L490)、[plot_thesis_spatiotemporal_figures.py:647](/Users/dc/Z研究生/TimeFilter/TimeFilter/scripts/plot_thesis_spatiotemporal_figures.py#L647)、[plot_thesis_spatiotemporal_figures.py:767](/Users/dc/Z研究生/TimeFilter/TimeFilter/scripts/plot_thesis_spatiotemporal_figures.py#L767)。
- 运行命令如下：
```bash
MPLCONFIGDIR=/tmp/mpl /opt/homebrew/Caskroom/miniforge/base/envs/torch/bin/python \
/Users/dc/Z研究生/TimeFilter/TimeFilter/scripts/plot_thesis_spatiotemporal_figures.py \
--patch-width 24 \
--patch-height 24 \
--time-block-hours 24
```
- 我已经用你的真实数据跑通，图片已导出到 [thesis_spatiotemporal_figures](/Users/dc/Z研究生/TimeFilter/TimeFilter/outputs/thesis_spatiotemporal_figures)。本次自动选中的代表性 patch 为 `x=[120,143], y=[72,95]`，时间块为 `2024-06-16 07:00` 到 `2024-06-17 06:00`。

**3. 输出图说明**
- 图1：蓝色细散点表示所有有效二维投影监测点，横纵坐标分别为 `grid_x` 与 `grid_y`。
- 图2：灰色表示稳定背景区，橙色表示坡脚区，绿色表示坡中区，黄色表示平台区，蓝色表示坡顶区；半透明色块是区域，深色小点是实际监测点。
- 图3：浅色矩形表示非空规则 patch，深红色边框表示自动选中的代表性 patch，灰色点为原始监测点。
- 图4：灰色点为 patch 周边上下文，深蓝框为选定 patch，彩色点按所选时间块内的平均位移着色，右侧特征框依次表示 `位移均值/标准差、速度均值/标准差、加速度均值/标准差、有效点比例`。
- 图5：左下为二维空间 patch 划分，左上为时间块划分，中间框表示“空间 patch × 时间 block”的组合操作，右侧蓝色堆叠块表示最终时空张量。
- 图6：左侧半透明叠片表示原始监测序列 `X∈R^{L×N×C_in}`，中间框表示 patch 统计与投影，右侧蓝色堆叠块表示模型输入张量 `H^{patch}`。

**4. 论文图注**
- 图1：边坡雷达监测点二维投影分布图。
- 图2：基于二维投影坐标的边坡工程分区示意图。
- 图3：边坡监测区域规则网格 patch 划分结果图。
- 图4：代表性 patch 的局部放大及统计特征构造示意图。
- 图5：时空分块机制示意图。
- 图6：模型输入张量结构与维度变换示意图。

