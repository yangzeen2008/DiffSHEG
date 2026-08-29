# 图表台账

> `article/figures/` 当前每组图通常同时保留 PNG 和 PDF。论文排版优先使用矢量 PDF，预览与沟通使用 PNG。

| 暂定图号 | 文件主名 | 一句话结论 | 计划位置 | 状态 |
|---|---|---|---|---|
| Fig. 4-1 | `main_comparison_unified` | FM 与 DDPM 的统一指标主对比 | 4.3 主实验 | 已生成，待编号/图注 |
| Fig. 4-2 | `physical_constraints_ablation` | 展示 Vel/Acc 约束的消融变化 | 4.4 物理约束消融 | 已生成，待统计显著性说明 |
| Fig. 4-3 | `rotation_representation_comparison` | 展示 Axis-Angle 与 6D 的精度-多样性权衡 | 4.5 表示消融 | 已生成，待图注 |
| Fig. 4-4 | `step_ablation_fgd` | 展示采样步数与生成质量关系 | 4.6 步数消融 | 已生成，待样本数说明 |
| Fig. 4-5 | `inference_speed` | 展示不同采样设置的推理速度 | 4.7 效率 | 已生成，待计时口径复核 |
| Fig. 4-6 | `training_speed` | 展示 FM 与 DDPM 训练耗时 | 4.7 效率 | 已生成，DDPM 推估值需显式标注 |
| Fig. 4-7 | `one_step_postprocess` | 展示 1-Step 后处理的平滑度权衡 | 4.8 实时管线 | 已生成，待指标定义 |
| Fig. 4-8 | `unified_all_metrics` | 多指标总览 | 4.3 或附录 | 已生成，评估是否信息重复 |
| 备选 | `fgd_comparison` | 旧 FGD 对比 | 附录或删除 | 已生成，需确认是否被统一图替代 |

## 图表交付检查

- 图中字号、单位、颜色和小数位统一。
- 图注能独立说明数据集、模型、采样设置和 `↑/↓` 含义。
- 正文先引用再出现图，且明确写出读图结论。
- 数据可由脚本和固定输入复现，不手工修改图中数值。
- 彩色与灰度、屏幕与打印均可辨识。

