# 修复版本与 Opus 自动先验

用户明确授权优化现有代码。原 /data/common/lgt/GroupSketch 的实验继续运行。本工作树位于 /data/common/lgt/GroupSketch-worktrees/llm-affine-prior，分支 liangguotao/feat-llm-affine-prior，不修改原任务引用的源码。

## 已复现并修复的问题

1. get_frame_deltas 的 scale 参数调用 shear 网络，独立 scale 网络没有梯度；改为调用 scale 网络。
2. zero_grad_ 仅清局部优化器，独立全局参数梯度跨 loss/迭代累积；现在两个优化器均清梯度。共享参数仍按原双分支设计更新。
3. SDSVideoLoss 默认 reuse_pipe=False，第一次模型不进入共享缓存，第二个损失又加载一份；默认改为复用同一权重管线。

tests/test_runtime_correctness.py 的三项检查在修复前均失败，修复后通过：实际 scale 梯度、实际全局梯度、模型加载次数与对象身份。日志在 logs/runtime-before*.log 和 runtime-after.log。尚需真实 GPU 完整生成验证；mock加载检查不等于实际显存测量。

## 仿射先验适配

motion_prior.py 负责校验并编译 LLM 先验。tools/prepare_llm_prior.py 是本项目新增适配，不是原仓库官方功能：通过已有 Canvas create_group/generate_animation 导出分组，复用真实 argparse、Painter 初始化、utils 的 SVG/MP4/GIF 导出。输入须为已预处理的256x256单段三次Bezier路径，不重新编号或重画。

生成显式传 --affine_prior 指定 YAML，替代 Canvas 平移；不传保留旧行为。组路径顺序、帧数、首帧恒等、有限非奇异矩阵均核验。矩阵作用于原画布坐标后转回原有中心相对坐标。render_frames_to_tensor_mlp(apply_mlp=False) 通过原渲染链导出初始化；默认仍走原网络预测。

两种自动先验使用相同模型分组和平移，translation只保留平移，affine用全部仿射参数。粗帧SVG与Painter控制点逐帧误差均为0。旧4例Canvas与等价仿射平移控制点最大差7.63e-6，属float32运算顺序误差，见 legacy-translation-validation.json。不改评分规则。

## 环境与状态

生成沿用 video_svg，先 source tools/env.sh。Canvas所需Flask已在用户现有user-site，但全量加载user-site会造成Tokenizers冲突；仅准备进程设置 PYTHONPATH=$PWD/.cache/canvas-python，选择性链接Flask及6个既有依赖，见 canvas-dependencies.json。共享conda环境未修改，未下载新环境。

准备调用 tools/prepare_llm_prior.py，参数 --svg、--plan、--output、--caption、--num-frames、--variant；CUDA_VISIBLE_DEVICES="" 在CPU导出粗先验，不运行SDS。实际优化仍使用 animate_svg.py。

v1预览在 output_videos/llm-prior-preview，只是粗动画；模型原输出保存在各输入 planner-output.json。粗预览暴露语义与连接问题后，正在给Opus回传真实帧做一次反馈修订，v1保留。后续用同一修复版本比较人工先验、LLM平移、LLM仿射，避免混入代码修复收益；原CCFE消融只解释原实现。
