# DiffSHEG 项目进展

> 更新时间: 2026-05-05 17:52

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

**beat_FM_v6 训练完成 ✅** — 后处理自适应平滑已就绪

```
最终模型: beat_FM_v6 (Epoch 999, PCK=23.52%)
后处理:   自适应 Gaussian 滤波 (spine=1.2, arm=1.5, hand=1.5)
输出:     bvh_output/FM_v6_adaptive.bvh
对比视频: bvh_output/comparison_side_by_side_audio.mp4
```

### 训练配置 (FM_v6)

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
ssh.connect('connect.westd.seetacloud.com', 48360, 'root', 'Yangzeen2008#')
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

---

## ⏭️ 待办

- [x] ~~关节顺序修复~~ (v5 已验证)
- [x] ~~手臂运动范围恢复~~ (v5: 100-112% of Original)
- [x] ~~FM_v6 训练完成~~ (PCK=23.52%)
- [x] ~~抖动优化~~ (自适应 Gaussian 后处理)
- [x] ~~并排对比渲染~~ (蓝=Ours, 橙=Original)
- [x] ~~更新 article/experiment_results.md~~ (新增最终结果章节)
- [ ] 定量评估 (FGD/MSE/PCK) + 论文写作
- [ ] 训练 DDPM_v2 (正确关节顺序, 可选)
