# 主张与证据台账

> 状态建议：`待核验` → `已复现` → `可入正文`。摘要、结论和创新点只使用“可入正文”的主张。

| ID | 论文主张 | 当前数字/表述 | 证据位置 | 状态 | 风险与下一步 |
|---|---|---|---|---|---|
| C-001 | Flow Matching 的关节精度优于同设置 DDPM | PCK `23.05%` vs `20.56%`；MSE `1.0134` vs `1.1810` | [统一实验表](../article/experiment_results.md#1-完整消融与表示对比数据) | 待核验 | 核对 checkpoint、样本数、随机种子和同一评测空间 |
| C-002 | 1-Step 可实现超实时生成 | `533 FPS`，约 `1.6 s / 855 frames` | [1-Step 分析](../article/experiment_results.md#1-step-推理深度分析) | 待核验 | 保存计时代码、硬件、warm-up 和重复测量 |
| C-003 | 1-Step 相比 DDPM 显著加速 | 正文写 `33.3x`，旧结果表另有 `1100x` | [摘要草稿](../article/thesis_draft.md#摘要)；[旧结果表](../article/experiment_results.md#与原版-diffsheg-ddpm-对比) | 冲突 | 两者很可能使用 FPS 与 batch wall-time 两种不同口径；统一后再引用 |
| C-004 | FM_v6 在同一评估器下显著降低 FGD | `36.5` vs `379.3`，约 `10.4x` | [FGD 对比](../article/experiment_results.md#同评估器-fgd-对比-ddpm-vs-fm) | 待核验 | 明确评估器训练集、clip 聚合方式和置信区间 |
| C-005 | 物理约束改善轴角模型精度 | Base/Vel/Vel+Acc 的 PCK 约 `23.01/23.01/23.05%` | [统一消融表](../article/experiment_results.md#1-完整消融与表示对比数据) | 待核验 | 改善幅度很小；需要多次运行或避免使用“显著提升” |
| C-006 | 6D 表示提高多样性但降低精度 | 500 epoch：Diversity `0.6663`，PCK `16.13%` | [统一消融表](../article/experiment_results.md#1-完整消融与表示对比数据) | 待核验 | 将现象与原因解释分开，重投影误差机制需文献或额外实验支持 |
| C-007 | 训练耗时低于 DDPM | 旧表为约 `15h` vs 推估 `63h`，约 `4x` | [主结果表](../article/experiment_results.md#与原版-diffsheg-ddpm-对比) | 待核验 | DDPM 时间标为推估，不宜写成实测结论 |
| C-008 | SO(3) 释放过渡可消除轴角分支导致的小臂长弧旋转，且无需重训 | 典型段左前臂最大角速度 `168.5→20.0°/帧`；最坏路径/端点比 `37.23→1.00`；三段窗口重叠 RMS 仍为 `0` | [修复前后审计](../bvh_output/long_validation_trials/so3_release/forearm_before_after.json)；[三段汇总](../bvh_output/long_validation_trials/so3_release/metrics_summary.json) | 已复现（3 段） | 当前证据是有界长样例；正文若作普遍结论，需补全验证集关节级 SO(3) 统计 |
| C-009 | 固定 7 帧释放并非唯一合理选择；5 帧在当前长样例中更接近 GT 动态幅度，同时压低危险的前臂边界峰值 | 在典型/高强度段，`blend=5` 的速度比分别为 `0.99/0.96`，优于 `blend=7` 的 `0.85/0.82`；jerk 比为 `0.64/0.92`，而不释放时为 `1.46/1.98`；左前臂峰值从不释放的 `52.7/108.5` 降至 `30.3/31.7°/帧` | [验证记录](records/2026-08-30-transition-blend-validation.md)；[固定条件消融](../bvh_output/long_validation_trials/transition_blend_ablation.json)；[视觉材料](../bvh_output/long_validation_trials/transition_blend_ablation/README.md) | 已复现（3 段） | 不能据此声称全数据最优；安静段 `blend=5` 仍有约 `45.5°/帧` 峰值，下一步需全验证集统计、边界自适应或外推限幅，并进行去标签盲评 |
| C-010 | 单向表情—手势联合 FM 构成区别于已有整体 FM 的候选结构创新 | 在同一联合状态中输运表情与手势，并用停止梯度的干净表情估计形成表情到手势的三角速度场 | [创新性边界记录](records/2026-08-30-directed-joint-fm-novelty.md)；[模型实现](../models/transformer.py#L741) | 待核验 | 初步检索未发现完全相同结构，但 GlobalDiff 已采用“先预测表情、再条件化身体 CFM”；需完成系统查新和方向/条件/梯度消融后才能写入创新点 |

## 新主张登记模板

| ID | 论文主张 | 当前数字/表述 | 证据位置 | 状态 | 风险与下一步 |
|---|---|---|---|---|---|
| C-XXX |  |  |  | 待核验 |  |
