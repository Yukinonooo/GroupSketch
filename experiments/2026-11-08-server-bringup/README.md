# 服务器运行验证

当前入口：`/data/common/lgt/GroupSketch`，使用现成 `/data/anaconda3/envs/video_svg`，启动前 `source tools/env.sh`。不修改共享环境，以 `PYTHONNOUSERSITE=1` 隔离用户目录里不兼容的 Tokenizers。

输入来自只读历史目录 `GroupSketch_bk/afterMove/toulan`，本轮冻结副本位于 `inputs/toulan`；SHA256 见 `conf/input-manifest.json`。原始目录未修改。

两轮真实 SDS 优化（num_iter=1）完成；原 GIF 导出因 ImageIO 不再接受 fps 参数失败。已在 utils.py 改用等价 125ms 帧时长，用已有 SVG 成功重新导出 24 帧 GIF。GIF 格式会将时长量化为 120ms。失败原日志保留于 logs/probe.log。

完整运行：GPU 0 开 CCFE、GPU 1 关 CCFE，均 timeE=true、seed=1000、24 帧、num_iter=1000（实际 1001 轮）；conf 保存实际解析参数和启动命令。输出位于 output_videos/2026-11-08-server-bringup/{ccfe-on,ccfe-off}。

这两组用于跑通及初步对照，单一样例不能证明 CCFE 是否有效。模块开关也会影响初始化随机数消耗；同 seed 不意味着两组所有中间随机数逐一相同。评分沿用用户已选 Gemini 3.7 Flash 直接 rubric，不在运行期间改评分口径。

运行日志和进度在 logs/，退出码写入 *-exit-code.txt；尚未完成的任务不能当作完整结果。临时环境已清除 41,953,847 字节，记录见 conf/cleanup.json。
