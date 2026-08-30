# DiffSHEG 论文一站式工作台

这是论文工作的统一入口。原项目代码、实验产物和已有文档保持原位；工作台只负责把写作、实验、证据、文献和图表串成一条可检查的流程。

## 现在从这里开始

1. 打开 [动态总览](paper-workbench/DASHBOARD.md)，查看章节、图表、文献和待办状态。
2. 在 [任务看板](paper-workbench/tasks.md) 中只保留当前真正要推进的事项。
3. 写结论前先登记到 [主张与证据台账](paper-workbench/claims.md)，避免数字口径冲突。
4. 参数或口径选择登记到 [决策日志](paper-workbench/decisions.md)，详细验证过程放入 [归档记录](paper-workbench/records/README.md)。
5. 新实验、新文献和章节复核分别使用 [实验记录模板](paper-workbench/templates/experiment-record.md)、[文献笔记模板](paper-workbench/templates/literature-note.md) 和 [章节审校模板](paper-workbench/templates/chapter-review.md)。

## 目录设计

- `paper-workbench/*.md`：少量高频核心台账，保持扁平，便于直接打开。
- `paper-workbench/records/`：带日期和 ID 的详细验证记录；总进展文件只保留摘要和链接。
- `paper-workbench/templates/`：可复用记录与审校模板。
- `paper-workbench/tools/`：自动扫描、断链检查和总览生成脚本。
- `article/`：论文正文、路线和论文图表；不放训练缓存。
- `logs/`、`checkpoints/`、`bvh_output/`：原始运行证据与大体积产物，保持训练代码现有路径，不复制到工作台。

更完整的维护说明见 [工作台目录说明](paper-workbench/README.md)。

## 核心材料

| 模块 | 入口 | 用途 |
|---|---|---|
| 论文正文 | [article/thesis_draft.md](article/thesis_draft.md) | 唯一正文草稿 |
| 写作路线 | [article/thesis_roadmap.md](article/thesis_roadmap.md) | 章节与实验规划 |
| 实验结果 | [article/experiment_results.md](article/experiment_results.md) | 历史结果与最终结果来源 |
| 项目进展 | [PROGRESS.md](PROGRESS.md) | 训练、诊断和产物总览 |
| 图表目录 | [article/figures](article/figures) | 论文 PNG/PDF 图表 |
| 文献台账 | [paper-workbench/literature.md](paper-workbench/literature.md) | 阅读、引用和待补证据 |
| BibTeX | [paper-workbench/references.bib](paper-workbench/references.bib) | 参考文献库 |
| 实验台账 | [paper-workbench/experiments.md](paper-workbench/experiments.md) | 模型口径与证据定位 |
| 决策日志 | [paper-workbench/decisions.md](paper-workbench/decisions.md) | 冻结参数、口径及重新评估条件 |
| 详细记录 | [paper-workbench/records](paper-workbench/records) | 单次训练、验证和评审的完整过程 |
| 图表台账 | [paper-workbench/figures.md](paper-workbench/figures.md) | 图号、结论和正文位置 |

## 常用命令

在仓库根目录执行：

```powershell
# 刷新动态总览
.\.venv\Scripts\python.exe paper-workbench\tools\workbench.py update

# 只在终端查看当前状态
.\.venv\Scripts\python.exe paper-workbench\tools\workbench.py status

# 检查断链、占位符、缺失章节和未完成任务
.\.venv\Scripts\python.exe paper-workbench\tools\workbench.py check

# CI 或提交前使用；发现问题时返回非零退出码
.\.venv\Scripts\python.exe paper-workbench\tools\workbench.py check --strict
```

VS Code 中也可以运行 `Paper: Refresh dashboard` 和 `Paper: Check workspace` 两个任务。

## 工作约定

- `article/thesis_draft.md` 是正文唯一来源；其他文件只提供证据，不复制成第二份正文。
- 数字先进入 `experiments.md`，结论再进入 `claims.md`，核验后才写进摘要和结论。
- 详细过程只写一次到 `records/`；`PROGRESS.md`、台账和正文只保留摘要与链接。
- 参数选择必须写入 `decisions.md`，并注明证据、状态和重新评估条件。
- 每张图都要在 `figures.md` 说明“它证明什么”，不能只登记文件名。
- 文献阅读记录写入 `literature.md`；完整书目信息写入 `references.bib`。
- 每次集中写作前运行 `update`，每次交付或提交前运行 `check --strict`。
