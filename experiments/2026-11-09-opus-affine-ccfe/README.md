# Opus 完整仿射下的 CCFE 消融

用户于2026-11-09明确选择 Opus 完整仿射，并要求以其重新进行消融。

固定 feedback-v2 分组、仿射矩阵、粗动画、prompt、seed=1000、num_iter=1000（实际1001轮）。Full 复用上一轮4个已完成结果；本轮新增四案例 NoMCL（dino=false,timeE=true）和 NoMCL-NoTime（false,false），8卡各一任务。沿用 animate_svg.py、video_svg、原损失与导出，不修改网络代码。

实际源码 55a7108ce65197ed5acce11e22e7b09032586988，已逐文件核对所有Python生成源码和tools/env.sh与b4ba526相同；其余提交仅为记录。原始输入全部哈希复核通过。新输出在 output_videos/2026-11-09-opus-affine-ccfe。conf/保存实际配置与命令，logs/保存PID、进度、退出码。

状态：2026-11-09T05:13:27.880980+08:008个任务已并行启动，均进入真实优化。评分沿用论文仓库 Flash 直接 rubric；复用已有Full评分，不重复调用。相同seed不代表模块关闭后内部随机数消费顺序相同，本批仍为4案例单seed探索。
