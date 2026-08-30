# DiffSHEG 项目进展

> 更新时间: 2026-08-30

> **当前训练准入版本**：`beat_4english_15_141_sync_v1` 已将约 120 FPS 动作、
> 60 FPS 表情统一重采样到 15 FPS，并与 16 kHz 音频绑定。Motion cache v3、
> HuBERT cache v3 和 5-epoch 全数据试跑均已通过。`beat_FM_aa_x0_aligned_v1`
> 的源时间轴仍然错位，旧 checkpoint 已降级为历史错误实验，不得作为音画同步
> 或当前主模型证据。正式 500-epoch 新实验尚未启动，等待用户明确确认。

---

## 📁 文档索引

| 文件 | 内容 | 状态 |
|:---|:---|:---:|
| **PROGRESS.md** (本文件) | 项目总览、当前状态、训练记录 | ✅ 主文档 |
| **README.md** | 原始 DiffSHEG 官方说明 | 📖 只读 |
| **docs/server_guide.md** | 云端部署完整指南（存储/训练/避坑） | 📖 参考 |
| **docs/TRAINING_HANDOFF_v1_archived.md** | 早期本地训练交接文档（已过时） | 📦 归档 |
| **article/thesis_roadmap.md** | 论文写作路线图 | 📝 论文 |
| **article/experiment_results.md** | 实验结果（待 v6 更新） | ⚠️ 待更新 |
| **article/开题报告_审查报告.md** | 开题报告内容 | 📝 论文 |

---

## 🚀 当前状态

**跨模态时间轴修复与 5-epoch 全数据训练准入已完成；正式重训待确认 ✅**

```
有效数据: beat_4english_15_141_sync_v1
运动缓存: train=38,468 / val=4,609，Motion cache v3，绑定 temporal manifest ID
HuBERT缓存: train=38,468 / val=4,609，HuBERT cache v3，绑定 motion cache ID
准入试跑: beat_FM_aa_x0_aligned_sync_pilot_v1，5 epochs / 755 steps
试跑验证: MSE=0.13425，PCK=0.88329，SRGR=0.87668，Diversity=0.73205
正式训练: beat_FM_aa_x0_aligned_sync_v1，尚未启动
旧模型: beat_FM_aa_x0_aligned_v1 已废弃为同步错误实验
```

### 2026-08-30 跨模态时间轴关键纠正

复核确认，旧预处理把约 120 FPS BVH 和 60 FPS 面部 JSON 原样复制，随后数据集却把两者都按 15 FPS 取窗。这使模型和旧验证集共同学习了错误时间尺度，所以高 PCK/MSE 不能证明表情与语音同步。

现已完成真实时间戳重采样、内容哈希 manifest、运动 LMDB v3、HuBERT v3 全量重建，以及 2-batch 冒烟和 5-epoch 全数据试跑。完整证据、缓存 ID、样本数量和结论边界见 [R-2026-08-30-003](paper-workbench/records/2026-08-30-cross-modal-temporal-alignment-repair.md)。

[15 秒对齐 GT 视频](bvh_output/temporal_alignment_smoke/aligned_ground_truth_audio_smoke.mp4) 只用于证明新数据时间轴正确，不代表 5-epoch 模型质量。新模型完成正式训练后必须重新生成预测对比。

### 2026-08-30 长序列推理与 SO(3) 边界修复

三段长序列已完成 SO(3) 最短路径修复和 `blend=0/3/5/7` 受控消融。`blend=0` 的偶发边界坏帧已在放大慢放中复现；`blend=5` 是当前折中候选，但默认值仍保持 7，等待全验证集统计和去标签盲评后冻结。

完整条件、数字、视觉证据和结论边界已归档到 [R-2026-08-30-001 验证记录](paper-workbench/records/2026-08-30-transition-blend-validation.md)，相关参数决策见 [D-001](paper-workbench/decisions.md)。


### 历史训练配置 (FM_v6)

| 参数 | 值 | vs v5 变化 |
|:---|:---:|:---:|
| 等效 Batch size | 500 × 5 = **2500** | ≈ (batch 降、accum 升) |
| Epochs | 1000 | 不变 |
| 表示 | axis-angle (141维) | 不变 |
| 关节顺序 | **spine_neck_141** | ✅ 不变 |
| vel_loss_weight | **500** | ↑ 5x (v5=100) |
| acc_loss_weight | **200** | ↑ 从 0 恢复 |
| ODE Solver | **RK4** | ↑ 从 Euler 升级 |
| FM sample steps | 50 | 不变 |

### 监控命令

```powershell
f:\study\DiffSHEG\.venv\Scripts\python.exe -c "
import paramiko
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('connect.westd.seetacloud.com', 48360, 'root', '<从安全凭据读取>')
stdin, stdout, stderr = ssh.exec_command('tail -5 /root/autodl-tmp/DiffSHEG/logs/train_fm_v6.log')
print(stdout.read().decode().strip())
ssh.close()
"
```

---

## 📊 问题诊断记录

### 🔴 问题 1: 关节列顺序不匹配 (v1~v4)

> 2026-05-04 发现并修复，v5 验证通过

**现象**：所有 FM/DDPM 模型生成的手势幅度极小（"T-Pose 综合症"）

| 关节 | Original DiffSHEG | FM_v2 (错) | FM_v5 (修复) |
|:---|:---:|:---:|:---:|
| RArm (大臂) | 74.7° | 8.0° | **83.4°** ✅ |
| RForeArm (前臂) | 262.4° | 18.4° | **264.6°** ✅ |
| LArm (大臂) | 76.6° | 12.6° | **76.3°** ✅ |

**根因**：`preprocess_beat.py` 中 `_BVH_COL_141` 的关节排列顺序与 `spine_neck_141` 不匹配。

**修复**：按 `spine_neck_141` 顺序从 raw BVH 重新提取 141 维。**v5 手臂运动范围已恢复至 Original 的 100-112%。**

### 🟡 问题 2: FM 高频抖动 / 动作过快 (v5)

> 2026-05-05 发现，v6 训练中修复

**现象**：FM_v5 手臂运动范围正确，但动作抖动大、速度过快，对每个音素都做反应（缺乏"质感"）

| 指标 | Original (DDPM) | FM_v5 | FM_v5 倍率 |
|:---|:---:|:---:|:---:|
| 速度 std | 10.7 | **18.1** | 1.7x |
| 加速度 std | 15.2 | **29.6** | 1.9x |
| 急动度 std | 25.5 | **53.6** | 2.1x |

**根因**：
- DDPM 的迭代去噪天然是时间低通滤波器，只保留语音大节奏
- FM 的 ODE 积分完整保留模型学到的每个音素级特征 → 过度响应
- vel_loss 仅占总 loss 的 1%，几乎没有平滑约束

**修复方案**（FM_v6 已完成 + 后处理）：
1. vel_loss_weight: 100 → **500** (5x)
2. acc_loss_weight: 0 → **200** (恢复加速度损失)
3. ODE 求解器: Euler → **RK4** (4阶精度)
4. **后处理自适应 Gaussian 滤波** (spine=1.2, arm=1.5, hand=1.5)

| 指标 | Original | FM_v6 原始 | FM_v6 + 后处理 |
|:---|:---:|:---:|:---:|
| Spine vel_std | 0.4 | 0.5 | **0.4** ≈ |
| R_Arm vel_std | 10.7 | 17.7 | **6.0** ✅ |
| L_Arm vel_std | 14.3 | 19.4 | **6.7** ✅ |

### 排查过程中的无效尝试

| 尝试 | 操作 | 结论 |
|:---|:---|:---|
| FM_v3 | 用"正确"stats 重算重训 | ❌ MSE 降 60x 但手臂更弱 |
| FM_v4 | 恢复旧 stats 重训 | ❌ 和 v2 无本质区别 |
| axis_angle_std 放大 | 旧 std 比实际大 7-29x | ⚠️ 是**列错位的症状**，不是根因 |

---

## 🧪 训练历史

| 实验 | 采样 | Batch | vel/acc | 关节顺序 | 状态 | 备注 |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| FM_v1 | FM 50 Euler | 512 | 100/50 | ❌ 错误 | ✅ 完成 | 手势弱 |
| DDPM_v1 | DDIM 25 | 512 | — | ❌ 错误 | ✅ 完成 | 更弱 |
| FM_v2 | FM 50 Euler | 834×3 | 100/0 | ❌ 错误 | ✅ 完成 | 手势弱 |
| FM_v3 | FM 50 Euler | 834×3 | 100/0 | ❌ 错误 | ✅ 完成 | 新 stats, 更弱 |
| FM_v4 | FM 50 Euler | 834×3 | 100/0 | ❌ 错误 | ⏹️ 中止 | |
| **FM_v5** | **FM 50 Euler** | **834×3** | **100/0** | **✅ 修复** | **✅ 完成** | **幅度恢复，抖动大** |
| **FM_v6** | **FM 50 RK4** | **500×5** | **500/200** | **✅ 修复** | **✅ 完成** | **PCK=23.52%, +后处理平滑** |
| **DDPM_v2** | **DDIM 25** | **256** | **—** | **✅ 修复** | **✅ 完成** | **PCK=20.56%, MSE=1.181, SRGR=20.27%, Div=0.649** |
| **FM_aa_base** | **FM 50 RK4** | **512** | **0/0/0** | **✅ 修复** | **✅ 完成** | **PCK=23.01%, MSE=1.0169, SRGR=22.76%, Div=0.5240** |
| **FM_aa_vel** | **FM 50 RK4** | **512** | **100/0/0** | **✅ 修复** | **✅ 完成** | **PCK=23.01%, MSE=1.0240, SRGR=22.86%, Div=0.5266** |
| **FM_aa_velAcc** | **FM 50 RK4** | **512** | **100/50/0** | **✅ 修复** | **✅ 完成** | **PCK=23.05%, MSE=1.0134, SRGR=22.87%, Div=0.5205** |
| **FM_6d** | **FM 50 RK4** | **512** | **100/50/0** | **✅ 修复** | **✅ 完成** | **PCK=16.13%, MSE=1.1739, Div=0.6663** |
| **aligned_v1** | **FM 50 RK4** | **256** | **100/50/10** | **⚠️ 120/60 FPS 源时间轴错位** | **已废弃** | **旧验证指标仅作错误实验历史** |
| **aligned_sync_pilot_v1** | **FM 50 RK4** | **256** | **100/50/10** | **✅ 15 FPS/15 FPS/16 kHz manifest 绑定** | **5-epoch 准入通过** | **PCK=88.33%，不是最终模型** |

---

## 🔧 代码修改记录

### FM_v6 新增改动

1. **models/flow_matching.py**: 
   - `sample()` / `sample_with_inpaint()` 新增 RK4 求解器（默认启用）
   - 4 次模型调用/步 → 推理速度约慢 4x，但 ODE 精度大幅提升
2. **assets/render_comparison.py**: 
   - 并排对比渲染：左(蓝)=Ours, 右(橙)=Original
   - 直接修改 BVH Hips X 坐标实现位置偏移
   - 隐藏面部 mesh（FM 不涉及表情生成）
   - ffmpeg 自动合并音频

### FM_v5 的改动

1. **数据预处理**：按 `spine_neck_141` 顺序重新从 raw BVH 提取 141 维
2. **统计量**：`bvh_mean/std.npy`、`axis_angle_mean/std.npy` 从修正后的数据重新计算
3. **LMDB 缓存**：删除旧缓存，训练时自动重建

### 历史改动 (v1~v2 时期)

详见 `docs/TRAINING_HANDOFF_v1_archived.md`

---

## 🛠️ 工具与流程

### 并排对比渲染

```
1. Blender 打开 assets/beat_visualize.blend
2. Scripting → Open → assets/render_comparison.py → Run Script
   (自动清场、导入两个 BVH、并排渲染、合并音频)
3. 输出: bvh_output/comparison_side_by_side.mp4
```

### 后处理自适应平滑

分区域 Gaussian 滤波（补偿 FM 缺少 DDPM 去噪的低通效果）：

| 区域 | sigma | 效果 |
|:---|:---:|:---|
| Spine/Neck | 1.2 | 减少上半身晃动 |
| Arm/Shoulder | 1.5 | 降低手臂运动速度 |
| Hand/Fingers | 1.5 | 稳定手部 |

脚本: `scratch/smooth_adaptive.py`

### 单独合并音频

```powershell
ffmpeg -y -i input.mp4 -i audio.wav -map 0:v -map 1:a -c:v copy -c:a aac -shortest output.mp4
```

---

## 📦 输出文件

| 文件 | 说明 |
|:---|:---|
| `bvh_output/FM_v6_adaptive.bvh` | FM_v6 + 自适应平滑 (最终版) |
| `bvh_output/FM_v6_rk4.bvh` | FM_v6 原始 (无平滑) |
| `bvh_output/Original_DiffSHEG.bvh` | 原版 DiffSHEG 输出 |
| `bvh_output/comparison_side_by_side_audio.mp4` | 并排对比视频 (含音频) |
| `bvh_output/step1_comparison.mp4` | 1-Step 推理极限对比视频 (含配音) |

---

## ⏭️ 待办

- [ ] 用户确认后启动 `beat_FM_aa_x0_aligned_sync_v1` 的正式 500-epoch 重训
- [ ] 正式训练完成后重做 GT/新模型/旧错误模型的表情—语音与手势视觉对比
- [x] ~~关节顺序修复~~ (v5 已验证)
- [x] ~~手臂运动范围恢复~~ (v5: 100-112% of Original)
- [x] ~~FM_v6 训练完成~~ (PCK=23.52%)
- [x] ~~抖动优化~~ (自适应 Gaussian 后处理)
- [x] ~~并排对比渲染~~ (蓝=Ours, 橙=Original)
- [x] ~~更新 article/experiment_results.md~~ (新增最终结果章节)
- [x] ~~启动 DDPM_v2 和消融实验排队训练~~ (DDPM_v2 PID 4925，队列监控 PID 70283)
- [x] ~~监控云端队列进度，等待 DDPM_v2 及 3 个 FM 消融实验、1 个 6D 实验全部训练完成 (已优化排队任务为每 50 epoch 验证以提速 10 倍)~~
- [x] ~~运行 `eval_metrics.py` 收集各模型指标 (FGD/MSE/PCK/Diversity)~~
- [x] ~~编写统计脚本提取并验证生成的关节标准差 (std) 与表情数据活跃度~~ (已在本地及数据预处理中验证并输出标准差)
- [x] ~~使用 Blender 渲染 Ours (FM/DDPM 联合训练) vs Original 并排对比视频~~ (已生成 comparison_side_by_side_audio.mp4 对比视频)
- [x] ~~填入 `article/experiment_results.md` 并撰写论文正文~~ (数据均已填入第十节，所有最新学术图表均已重新编译生成)
