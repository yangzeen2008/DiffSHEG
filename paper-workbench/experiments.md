# 实验台账

> 本页是“最终有效实验”的索引，不替代原始日志。早期关节顺序错误的 v1-v4 结果不得作为正文主结论证据。

| ID | 模型 | 关键变量 | 当前结果摘要 | 原始/汇总证据 | 正文用途 | 状态 |
|---|---|---|---|---|---|---|
| E-001 | DDPM_v2 | Axis-Angle；DDIM 25 | PCK 20.56%；MSE 1.1810；Div 0.6495 | [统一结果](../article/experiment_results.md#1-完整消融与表示对比数据) | 同设置基线 | 已汇总，待原始日志复核 |
| E-002 | FM_aa_base | 无 Vel/Acc | PCK 23.01%；MSE 1.0169；Div 0.5240 | [统一结果](../article/experiment_results.md#1-完整消融与表示对比数据) | 物理损失消融基线 | 已汇总，待原始日志复核 |
| E-003 | FM_aa_vel | Vel=100 | PCK 23.01%；MSE 1.0240；Div 0.5266 | [统一结果](../article/experiment_results.md#1-完整消融与表示对比数据) | 一阶约束消融 | 已汇总，待原始日志复核 |
| E-004 | FM_aa_velAcc | Vel=100；Acc=50 | PCK 23.05%；MSE 1.0134；Div 0.5205 | [统一结果](../article/experiment_results.md#1-完整消融与表示对比数据) | 论文主模型 | 已汇总，待原始日志复核 |
| E-005 | FM_6d | 6D；Vel=100；Acc=50；500 epochs | PCK 16.13%；MSE 1.1739；Div 0.6663 | [统一结果](../article/experiment_results.md#1-完整消融与表示对比数据) | 旋转表示消融 | 已汇总，待原始日志复核 |
| E-006 | FM_v6 step ablation | RK4；1-50 steps | 1-Step 533 FPS；质量指标见逐步表 | [步数消融](../article/experiment_results.md#fm-推理步数消融实验) | 速度-质量权衡 | 统计口径待冻结 |
| E-007 | beat_FM_aa_x0_aligned_v1 | Motion cache v2；HuBERT cache v3 且按 cache ID 绑定；FM x0 条件；Axis-Angle；Vel/Acc/Jerk=100/50/10；batch=256；seed=1234；500 epochs | 终态 Epoch 499：MSE 0.09032、PCK 0.91212、SRGR 0.91463、Div 0.53639；`pck_best` 位于 Epoch 479 | [本地训练日志](../logs/train_beat_FM_aa_x0_aligned_v1.log)；[时间轴复核](records/2026-08-30-cross-modal-temporal-alignment-repair.md) | 仅作错误实验历史；不得用于音画同步或主模型结论 | **已废弃**：120/60 FPS 源帧被按 15 FPS 索引，验证集同样错位 |
| E-008 | E-007 pck_best 长序列推理 | RK4 50 steps；overlap=24；7 帧释放；手势按逐关节 SO(3) 最短路径插值，表情保留 Hermite | 3 段各 224 帧：左前臂峰值由 153.1/168.5/73.0 降至 42.3/20.0/22.0°/帧；最坏过渡绕行比均为 1.000；重叠 RMS=0 | [修复前后指标](../bvh_output/long_validation_trials/so3_release/forearm_before_after.json)；`quiet/typical/energetic/before_after_repair.mp4` | 仅保留为窗口边界/SO(3) 算法证据 | 模型质量与音画同步结论随 E-007 废弃；需在新模型复验 |
| E-009 | E-007 pck_best 释放强度消融 | 同一 checkpoint、seed=1234、三段音频与裁剪范围；RK4 50；overlap=24；仅改变 SO(3) 释放帧数 `0/3/5/7` | 典型段速度比 `1.23/1.20/0.99/0.85`、jerk 比 `1.46/1.10/0.64/0.49`；高强度段速度比 `1.23/1.14/0.96/0.82`、jerk 比 `1.98/1.36/0.92/0.69` | [验证记录](records/2026-08-30-transition-blend-validation.md)；[消融汇总](../bvh_output/long_validation_trials/transition_blend_ablation.json) | 后处理候选生成方法 | `blend=5` 不再可冻结；需在新模型全验证集重跑 |
| E-010 | beat_FM_aa_x0_aligned_sync_pilot_v1 | 时间轴 manifest v1；Motion cache v3；HuBERT cache v3；15 FPS/15 FPS/16 kHz；FM x0；Axis-Angle；Vel/Acc/Jerk=100/50/10；batch=256；seed=1234；5 epochs | 755 steps；末段 final loss 362.72；Epoch 4：MSE 0.13425、PCK 0.88329、SRGR 0.87668、Div 0.73205；无 OOM/NaN | [修复与准入记录](records/2026-08-30-cross-modal-temporal-alignment-repair.md)；[训练日志](../logs/train_beat_FM_aa_x0_aligned_sync_pilot_v1.log) | 正式重训前的数据与训练链路准入 | 试跑通过；不是最终模型，等待用户确认 500 epochs |

> 2026-08-30 复核边界：E-001 至 E-006 使用的历史 HuBERT 文件数记录为
> train 316,069 / val 37,881，而动作 LMDB 实际为 train 33,811 / val 4,047；
> 同时历史 FM 使用旧的联合条件逻辑。进一步复核发现 E-007 虽已按 LMDB/HuBERT
> 索引绑定，但其源动作 120 FPS、表情 60 FPS 仍被按 15 FPS 使用，因此 E-001
> 至 E-009 都不能提供当前主模型或音画同步结论；新的有效实验必须基于 E-010
> 所验证的跨模态时间轴与缓存契约。

## 每个实验必须保留

- 运行命令、Git commit、配置文件和随机种子。
- 数据集版本、划分、样本数、预处理与归一化统计量。
- 硬件、软件版本、checkpoint、训练日志和评估原始输出。
- 聚合方式、均值/标准差或置信区间，以及失败运行说明。
- 对应图表、主张 ID 和正文小节。
