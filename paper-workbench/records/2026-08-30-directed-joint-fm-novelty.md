# 单向表情—手势联合 Flow Matching 创新性边界

- 记录 ID：`R-2026-08-30-003`
- 日期：2026-08-30
- 关联主张：`C-010`
- 当前状态：候选创新，待系统检索与消融验证

## 结论

当前方案的创新性不能概括为“首次用 Flow Matching 做语音驱动表情和手势”，因为已有工作已经将 FM/CFM 用于语音手势或包含面部的整体动作生成。更准确的候选创新是：

> 在同一个 Flow Matching 状态空间中联合输运表情与手势，并以停止梯度的干净表情估计构造严格的表情到手势单向条件流。

截至 2026-08-30 的初步公开文献检索，尚未发现与上述完整组合完全相同的方法。这是“未发现完全相同公开工作”，不是穷尽性查新结论；使用“首次”时必须保留“据我们所知”，并在投稿前补充 Google Scholar、Web of Science/Scopus、CNKI 等系统检索。

## 当前实现对应的方法定义

设语音条件为 $A$，时刻 $t$ 的表情和手势状态分别为 $E_t$、$G_t$。当前实现可写为：

$$
v_E=f_E(E_t,A,t),
$$

$$
\hat E_0=E_t-t\,v_E,
$$

$$
v_G=f_G(G_t,A,\operatorname{sg}(\hat E_0),t).
$$

表情分支不依赖手势，手势分支使用停止梯度的表情估计，因此对应条件分解

$$
p(E,G\mid A)=p(E\mid A)\,p(G\mid E,A),
$$

并形成三角式联合速度场。实现证据位于 [表情估计与手势条件分支](../../models/transformer.py#L741) 和 [联合状态 Flow Matching 训练](../../models/flow_matching.py#L72)。

## 与最接近工作的边界

| 工作 | 已有能力 | 与当前方案的关键差异 | 对主张的约束 |
|---|---|---|---|
| DiffSHEG，CVPR 2024 | 扩散模型下的单向表情到手势建模 | 不是 Flow Matching | 不能声称首次提出表情到手势单向结构 |
| Match-TTSG，ICASSP 2024 | 使用 OT-CFM 联合生成语音声学特征与三维手势 | 不生成面部表情 | 不能声称首次将 FM 用于手势生成 |
| GestureLSM，ICCV 2025 | FM/shortcut 框架联合处理身体、手部、下半身与面部区域 | 更接近多区域对称交互，没有严格的表情先行单向分解 | 不能声称首次用 FM 生成包含面部的整体动作 |
| GlobalDiff，AAAI 2026 | 先预测表情，再将表情作为身体动作 CFM 的条件 | 表情在动作 CFM 外部独立预测，不在同一个 FM 联合状态中输运 | 不能声称首次让 FM 手势使用表情条件；这是最接近的直接竞争工作 |
| HolisticSemGes / SemConFlow，2026 | 在包含面部、上肢、下肢和手部的复合潜变量上做 Flow Matching | 采用整体/共享语义潜变量，没有明确表情到手势的三角速度场 | 必须把贡献限定为“单向联合速度场”，而不是“整体联合 FM” |

## 推荐论文表述

中文候选表述：

> 据我们所知，本文首次探索统一的单向表情—手势联合流匹配框架：在同一联合状态空间内同时输运面部表情和身体手势，并利用停止梯度的干净表情估计，对手势速度场施加显式的表情到手势条件依赖。

英文候选表述：

> To our knowledge, we present the first unified directed expression–gesture flow-matching framework that jointly transports facial-expression and gesture states while enforcing an explicit expression-to-gesture dependency through a stop-gradient clean-expression estimate.

方法暂名可使用 `Directed/Triangular Expression–Gesture Flow Matching`，中文为“单向三角表情—手势联合流匹配”。这里不能声称发明了通用 triangular flow matching，只能主张其在语音驱动表情—手势联合生成中的结构设计和实证价值。

## 不应使用的表述

- 首次将 Flow Matching 用于手势或动作生成。
- 首次使用 Flow Matching 生成表情和整体身体动作。
- 首次让 FM 手势生成以表情为条件。
- 首次实现实时或少步语音手势 Flow Matching。
- 在未完成系统检索前使用不带限定语的绝对“首个/首次”。

## 创新成立所需实验

1. 普通拼接式联合 FM，不设置方向依赖。
2. 表情与手势双分支，但手势不使用表情条件。
3. 手势分别使用噪声表情 $E_t$、表情速度 $v_E$、干净估计 $\hat E_0$。
4. `stop-gradient/detach` 与允许梯度从手势分支回传至表情分支的对比。
5. 表情到手势、手势到表情和双向交互三种方向对比。
6. 使用真实表情作为手势条件的性能上界。
7. 同时报告表情质量、手势质量、节奏同步、表情—手势协调性、推理速度/延迟及主观评价。

如果上述实验不能证明三角结构、干净表情估计和停止梯度各自带来稳定收益，则该工作更适合表述为 DiffSHEG 的高效 FM 改造，而不是独立的结构创新。

## 初步检索入口

- DiffSHEG，CVPR 2024：https://openaccess.thecvf.com/content/CVPR2024/html/Chen_DiffSHEG_A_Diffusion-Based_Approach_for_Real-Time_Speech-driven_Holistic_3D_Expression_CVPR_2024_paper.html
- Match-TTSG，ICASSP 2024：https://arxiv.org/abs/2310.05181
- GestureLSM，ICCV 2025：https://arxiv.org/abs/2501.18898
- GlobalDiff，AAAI 2026：https://arxiv.org/abs/2511.10076
- HolisticSemGes / SemConFlow，2026：https://arxiv.org/abs/2603.26553

## 论文定位判断

- 硕士论文：作为候选方法创新基本足够，但必须形成完整的检索、消融、指标和主观评价闭环。
- 会议或期刊：目前可能被评价为增量创新；必须直接比较 GlobalDiff、GestureLSM 和 SemConFlow，并证明单向联合速度场不是简单地把 DiffSHEG 的扩散过程替换为 FM。
