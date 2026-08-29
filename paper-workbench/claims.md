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

## 新主张登记模板

| ID | 论文主张 | 当前数字/表述 | 证据位置 | 状态 | 风险与下一步 |
|---|---|---|---|---|---|
| C-XXX |  |  |  | 待核验 |  |

