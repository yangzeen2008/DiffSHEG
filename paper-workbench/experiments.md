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

## 每个实验必须保留

- 运行命令、Git commit、配置文件和随机种子。
- 数据集版本、划分、样本数、预处理与归一化统计量。
- 硬件、软件版本、checkpoint、训练日志和评估原始输出。
- 聚合方式、均值/标准差或置信区间，以及失败运行说明。
- 对应图表、主张 ID 和正文小节。

