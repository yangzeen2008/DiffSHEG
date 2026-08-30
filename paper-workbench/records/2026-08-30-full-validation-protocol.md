# R-2026-08-30-002：全验证集默认参数评估协议

## 目标

在不重训模型的前提下，独立运行 `blend=5` 和 `blend=7` 的顺序推理，根据全验证集统计与去标签盲评冻结最终推理默认值。

## 绑定条件

- 实验：`beat_FM_aa_x0_aligned_v1`
- checkpoint：`pck_best.tar`，Epoch 479，大小 `1,870,617,614` bytes
- 推理：RK4 50 steps，`n_poses=34`，`overlap=24`，seed 1234
- 缓存：Motion cache v2；HuBERT cache v3；按 cache ID 绑定
- 候选：`fm_transition_blend=5/7`
- 预检窗口：`0:20,1000:20,2000:20,3000:20,4000:20`
- 全量阶段：完整 validation dataset，保持原始索引顺序
- `blend` 的输出会反馈到下一顺序窗口，因此两个候选必须独立推理，不能离线复用同一输出。

2026-08-30 启动前，服务器与本地以下 9 个文件 SHA-256 全部一致：`models/flow_matching.py`、`trainers/ddpm_beat_trainer.py`、`runner.py`、`options/base_options.py`、`datasets/beat.py`、`utils/window_stitching.py`、`utils/test_selection.py`、`utils/cache_versions.py`、`utils/motion_metrics.py`。

## 预检通过条件

1. GT、blend5、blend7 均输出 100 个 gesture/expression/audio 窗口。
2. 无 NaN/Inf、文件缺失、音频错位或非预期的重叠误差。
3. 两个候选使用相同索引、checkpoint、seed 和代码哈希。
4. 记录每 100 窗口耗时，据此估算全量完成时间和磁盘占用。

## 最终决策规则

优先级依次为旋转安全、保真度、动态自然度：

1. 统计所有关节以及肩、臂、前臂、手部的边界角速度 P95/P99/P99.9、最大值和异常率。
2. 每个关节的异常阈值定义为 `max(60°/帧, GT P99.9)`，避免用单个任意阈值替代 GT 分布。
3. 比较物理旋转 geodesic MSE/PCK；候选相对另一候选的退化不得超过 1%。
4. 比较全序列速度、加速度、jerk 和动态扩展度相对 GT 的距离。
5. 若 blend5 没有新增系统性边界异常，且动态更接近 GT，则选择 5；否则保留 7。
6. 对最坏边界和随机样例做去标签盲评；若无稳定感知差异，不声称视觉显著提升。

## 当前状态

- 服务器：RTX 4090；启动前数据盘剩余约 149GB。
- 本地/服务器评估代码：哈希一致；本地 `motion_repair_checks.py` 13 项测试通过。
- 100 窗口预检已完成：GT、blend5、blend7 均为 100/100/100，预测与 GT 的窗口重叠 RMS 均为 0，未发现文件缺失或音频错位。
- 每个候选 100 窗口约 201 秒；按 4047 个验证窗口估算约 2 小时 15 分/候选。
- 全量 GT `fullval_gt_v1` 已完成 4047/4047/4047。
- 全量 `fullval_blend5_v1` 已于 2026-08-30 启动并健康运行；30 分钟 heartbeat 已绑定为串行监控，完成并分析 blend5 后才允许启动 blend7。

## 100 窗口预检结果

| 指标 | blend=5 | blend=7 | 解释 |
|:---|---:|---:|:---|
| rotation geodesic MSE (rad²) | 0.117440 | 0.117291 | blend7 低约 0.13%，差异很小 |
| PCK @ 0.5 rad | 0.882048 | 0.881212 | blend5 高约 0.08 个百分点 |
| velocity / GT | 0.9567 | 0.8030 | blend5 更接近 GT 动态 |
| acceleration / GT | 0.7994 | 0.6067 | blend5 更接近 GT 动态 |
| jerk / GT | 0.7239 | 0.5618 | blend5 更接近 GT 动态 |
| motion spread / GT | 0.7555 | 0.7505 | 基本相同 |
| arm boundary P99 (°/frame) | 26.87 | 18.59 | blend7 更稳 |
| arm boundary max (°/frame) | 88.78 | 63.13 | blend7 显著压低最坏旋转 |
| arm abnormal rate | 0.1316% | 0.1316% | 两组各 1 次超过阈值 |

两组唯一超过阈值的事件都位于预检输出窗口 63（对应原验证索引约 3003）、`RArm1`，GT 峰值仅 11.19°/帧。这说明异常主要来自同一模型/样本边界；`blend=7` 能缓解但尚未完全消除。按“旋转安全优先”的协议，预检暂时倾向 `blend=7`，最终结论仍以完整验证集和盲评为准。

指标文件：

- `bvh_output/full_validation/preflight100_blend5_metrics.json`
- `bvh_output/full_validation/preflight100_blend7_metrics.json`

## 表情评估补充

联合模型会同时输出 51 维 BEAT/ARKit Blendshape，不能只用手势指标选择默认推理值。全量分析器现已同步统计：

1. 系数 MSE、MAE、`|error| < 0.05/0.10`、有效通道相关性；
2. 表情速度、加速度、jerk 与 `P90-P10` 动态范围相对 GT；
3. 边界后 7 帧的逐系数最大变化，阈值为 `max(0.10/帧, GT P99.9)`；
4. 原始 `[0,1]` 越界率，以及容差 `[-0.01, 1.01]` 外的显著越界率；
5. 逐 Blendshape 最坏事件和对应输出窗口，供去标签盲评抽样。

100 窗口预检结果：

| 表情指标 | blend=5 | blend=7 | 解释 |
|:---|---:|---:|:---|
| coefficient MSE | 0.011004 | 0.010947 | blend7 略低，差异很小 |
| coefficient MAE | 0.05737 | 0.05749 | blend5 略低，差异很小 |
| correlation | 0.7197 | 0.7208 | 基本相同 |
| velocity / GT | 0.8711 | 0.7637 | blend5 动态更接近 GT |
| acceleration / GT | 0.9503 | 0.7270 | blend5 动态更接近 GT |
| jerk / GT | 0.9040 | 0.6978 | blend5 动态更接近 GT |
| boundary abnormal rate | 0.1651% | 0.0206% | blend7 边界更稳 |
| significant range violation | 0.0070% | 0.0105% | 两者都极低 |

预检盲评包已固定为 4 个最坏边界窗口和 4 个 seed=1234 随机窗口，候选 A/B 按样本独立随机化：`bvh_output/full_validation/expression_blind_preflight/`。其中 `review_manifest.json` 不含答案，`answers.json` 单独保存映射；每个样本同时保留 NPY、原始 face JSON、音频和系数热图。

当前系数空间指标不等于原论文的 FMD/FED，也不等于音素感知的 lip-sync。原作者面部特征提取器和同步评估器未包含在当前本地仓库，补齐前不得在论文中替代或混用这些名称。
