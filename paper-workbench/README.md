# 工作台目录说明

`paper-workbench` 是论文项目的控制层，不复制训练数据、checkpoint 或大体积渲染结果。仓库根目录的 [PAPER_WORKBENCH.md](../PAPER_WORKBENCH.md) 是统一入口。

## 结构

```text
paper-workbench/
├── README.md          # 本说明
├── DASHBOARD.md       # 自动生成的状态总览
├── config.json        # 扫描路径与章节配置
├── tasks.md           # 下一步任务
├── experiments.md     # 实验索引与数字口径
├── claims.md          # 论文主张及证据边界
├── decisions.md       # 已冻结或待冻结的参数/口径决策
├── figures.md         # 图表登记
├── literature.md      # 阅读与引用登记
├── references.bib     # BibTeX 库
├── records/           # 带日期和 ID 的详细验证记录
├── templates/         # 记录与章节审校模板
└── tools/             # 总览生成和一致性检查
```

核心台账数量少且经常一起使用，因此保持扁平；只有会持续增长的详细记录进入 `records/`。这避免了总进展文件无限膨胀，也避免为少量文件制造过深层级。

## 信息流

1. 原始运行输出保留在 `logs/`、`checkpoints/`、`results/` 和 `bvh_output/`。
2. 一次实验或验证结束后，在 `records/` 写完整条件、结果、失败和证据路径。
3. 将可比较数字登记到 `experiments.md`。
4. 将可能写入论文的结论登记到 `claims.md`，并标明证据边界。
5. 参数或统计口径选择登记到 `decisions.md`。
6. 只有达到可入正文状态的内容才进入 `article/thesis_draft.md`。

同一份详细过程只在 `records/` 保存一次；其他文件使用摘要和链接，不复制整段表格。
