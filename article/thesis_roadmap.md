# DiffSHEG 论文完成路线图（含完整执行参数）

> 最后更新：2026-04-26  
> 项目路径：服务器上 `/root/autodl-tmp/DiffSHEG`  
> 数据缓存：`beat_4english_15_141`  
> GPU：3090 (24GB) 推荐  

---

## 第一阶段：模型训练（服务器，~24h 总计）

> [!IMPORTANT]
> **关于 Epoch 数量**
> 
> 设 `--num_epochs 500` 作为安全上限。FM 通常 100-300 epoch 即收敛（比 DDPM 快）。
> 代码**自动保存最优 checkpoint**（`mse_best.tar` / `pck_best.tar`），无需手动找最佳点。
> 如果日志中 MSE 连续 50+ epoch 不降，可以 `Ctrl+C` 提前停止。

> [!WARNING]
> **警告：请勿在命令中加入 `--gesture_only` 标志**
> 由于默认启用了联合多模态生成（`--unidiffuser True`），模型内部会强制执行手势与表情的分割解包操作。如果设置了 `--gesture_only`，会导致输入维度不足发生 `ValueError: not enough values to unpack` 崩溃。所有的训练和测试均应以表情+手势联合模式（默认）运行。

> [!NOTE]
> **评估指标说明**
> 
> | 指标 | 衡量什么 | 越低/高越好 |
> |------|---------|:-----------:|
> | FGD | 生成动作的分布与真实分布的距离（类似图像的 FID） | ↓ |
> | MSE | 生成动作与 GT 的逐帧逐关节误差 | ↓ |
> | PCK | 关节位置在阈值内的正确率 | ↑ |
> | Diversity | 生成结果的多样性 | ↑ |

### 实验 1.1 — FM + axis-angle（主实验）

| 参数 | 值 | 说明 |
|------|---|------|
| `--name` | `beat_FM_aa` | 实验标识 |
| `--mode` | `train` | |
| `--dataset_name` | `beat` | |
| `--flow_matching` | ✓ | 使用 FM 而非 DDPM |
| `--gesture_only` | ❌ (不加) | 必须为 False (在多模态下避免解包崩溃) |
| `--axis_angle` | `True` (默认) | axis-angle 旋转表示 |
| `--fm_sample_steps` | `50` | ODE 求解步数 |
| `--n_poses` | `34` | 每片段帧数（~2.27s @15fps） |
| `--batch_size` | `512` | 3090 可跑，见 server_guide |
| `--workers` | `32` | 数据加载线程 |
| `--num_epochs` | `500` | 上限，通常 200-300 即收敛 |
| `--lr` | `2e-4` (默认) | Adam 学习率 |
| `--vel_loss_weight` | `100` (默认) | 1阶速度损失权重 |
| `--acc_loss_weight` | `50` (默认) | 2阶加速度损失权重 |
| `--x0_rec_weight` | `100` (默认) | Huber x0 重建权重 |
| `--no_fgd` | ✓ | 训练阶段跳过 FGD（加速） |
| `--gpu_id` | `0` | |
| `--beat_cache_name` | `beat_4english_15_141` | |
| `--log_every` | `50` (默认) | 每 50 iter 打印 loss |
| `--save_every_e` | `5` (默认) | 每 5 epoch 保存 ckpt |
| `--eval_every_e` | `5` (默认) | 每 5 epoch 验证 MSE/PCK |

```bash
nohup python runner.py \
    --dataset_name beat \
    --name beat_FM_aa \
    --mode train \
    --flow_matching \
    --fm_sample_steps 50 \
    --n_poses 34 \
    --batch_size 512 \
    --workers 32 \
    --num_epochs 500 \
    --no_fgd \
    --gpu_id 0 \
    --beat_cache_name beat_4english_15_141 \
    > logs/train_fm_aa.log 2>&1 &
```

| 输出 | 路径 |
|------|------|
| Checkpoint | `checkpoints/beat/beat_FM_aa/model/latest.tar` |
| Best MSE | `checkpoints/beat/beat_FM_aa/model/mse_best.tar` |
| 训练日志 | `logs/train_fm_aa.log` |
| wandb 离线 | `wandb/` |
| 预估时间 | **~8h** (3090, bs=512) |

---

### 实验 1.2 — FM + 6D 旋转（创新点实验）

| 参数 | 值 | 与 1.1 的差异 |
|------|---|------|
| `--name` | `beat_FM_6d` | 不同实验名 |
| `--rot_6d` | ✓ | **新增**：启用 6D 旋转（282 维） |
| `--ortho_loss_weight` | `0.01` (默认) | **新增**：正交化损失 |
| 其余 | 同 1.1 | |

> 网络输入维度自动从 141→282，无需手动改。LMDB 缓存不需要重建。

```bash
nohup python runner.py \
    --dataset_name beat \
    --name beat_FM_6d \
    --mode train \
    --flow_matching \
    --rot_6d \
    --ortho_loss_weight 0.01 \
    --fm_sample_steps 50 \
    --n_poses 34 \
    --batch_size 512 \
    --workers 32 \
    --num_epochs 500 \
    --no_fgd \
    --gpu_id 0 \
    --beat_cache_name beat_4english_15_141 \
    > logs/train_fm_6d.log 2>&1 &
```

| 输出 | 路径 |
|------|------|
| Checkpoint | `checkpoints/beat/beat_FM_6d/model/` |
| 预估时间 | **~8h** |

> [!WARNING]
> 6D 模式下 batch_size 可能需要减小。282 维 > 141 维，显存占用约增大 30%。如果 OOM，改为 `--batch_size 256`。

---

### 实验 1.3 — DDPM Baseline

| 参数 | 值 | 与 1.1 的差异 |
|------|---|------|
| `--name` | `beat_DDPM_baseline` | |
| `--flow_matching` | **不加** | 使用 DDPM |
| `--diffusion_steps` | `1000` (默认) | DDPM 默认 1000 步 |
| `--batch_size` | `256` | DDPM 显存更大，减半 |
| 其余 | 同 1.1 | |

```bash
nohup python runner.py \
    --dataset_name beat \
    --name beat_DDPM_baseline \
    --mode train \
    --n_poses 34 \
    --batch_size 256 \
    --workers 32 \
    --num_epochs 500 \
    --no_fgd \
    --gpu_id 0 \
    --beat_cache_name beat_4english_15_141 \
    > logs/train_ddpm.log 2>&1 &
```

| 输出 | 路径 |
|------|------|
| Checkpoint | `checkpoints/beat/beat_DDPM_baseline/model/` |
| 预估时间 | **~8h** |

---

### 训练顺序（单 GPU）

```
优先级 1: beat_FM_aa        (~8h)  ← 论文主实验，必须跑
优先级 2: beat_DDPM_baseline (~8h)  ← 对比实验，必须跑
优先级 3: beat_FM_6d         (~8h)  ← 创新点，建议跑
```

双 GPU 可并行跑 1+2，然后跑 3。

### 训练监控与恢复

```bash
# 查看实时 loss
tail -f logs/train_fm_aa.log

# 查看当前 epoch
python -c "
import torch
ckpt = torch.load('checkpoints/beat/beat_FM_aa/model/latest.tar', map_location='cpu')
print(f'epoch: {ckpt[\"ep\"]}, total_it: {ckpt[\"total_it\"]}')
"

# 恢复中断的训练（加 --resume）
nohup python runner.py \
    --dataset_name beat --name beat_FM_aa --mode train \
    --flow_matching \
    --fm_sample_steps 50 --n_poses 34 \
    --batch_size 512 --workers 32 --num_epochs 500 \
    --no_fgd --gpu_id 0 \
    --beat_cache_name beat_4english_15_141 \
    --resume \
    > logs/train_fm_aa.log 2>&1 &
```

**自动保存的 Checkpoint 文件：**

| 文件 | 触发条件 | 用途 |
|------|---------|------|
| `latest.tar` | 每 500 iter + 每 5 epoch | 恢复训练 |
| `mse_best.tar` | validation MSE 创新低时 | **最终测试用** |
| `net_best.tar` | validation loss 创新低时 | 备选 |

---

## 第二阶段：推理与评估（~3h）

### 实验 2.1 — FM axis-angle 测试（含 FGD）

| 参数 | 值 | 说明 |
|------|---|------|
| `--name` | `beat_FM_aa` | 对应训好的模型 |
| `--mode` | `test` | |
| `--ckpt` | `mse_best.tar` | 用 MSE 最优的 ckpt |
| `--flow_matching` | ✓ | |
| `--gesture_only` | ❌ (不加) | 必须不加，见第一阶段警告 |
| `--fm_sample_steps` | `50` | |
| `--n_poses` | `34` | |
| `--no_fgd` | **不加**（除非评估器不可用） | 需要 FGD |
| `--gpu_id` | `0` | |

```bash
python runner.py \
    --dataset_name beat \
    --name beat_FM_aa \
    --mode test \
    --flow_matching \
    --fm_sample_steps 50 \
    --n_poses 34 \
    --gpu_id 0 \
    --beat_cache_name beat_4english_15_141 \
    --ckpt mse_best.tar
```

| 输出 | 路径 |
|------|------|
| 生成 npy | `results/beat_34/test/beat_FM_aa/BestMSE_eXXX/` |
| 终端输出 | FGD / MSE / PCK / Diversity 数值 |

### 实验 2.2 — FM 6D 测试

```bash
python runner.py \
    --dataset_name beat \
    --name beat_FM_6d \
    --mode test \
    --flow_matching \
    --rot_6d \
    --fm_sample_steps 50 \
    --n_poses 34 \
    --gpu_id 0 \
    --beat_cache_name beat_4english_15_141 \
    --ckpt mse_best.tar
```

> 6D 模式测试时输出自动转回 141 维 euler，FGD 评估器可直接使用。

### 实验 2.3 — DDPM Baseline 测试

```bash
python runner.py \
    --dataset_name beat \
    --name beat_DDPM_baseline \
    --mode test \
    --n_poses 34 \
    --gpu_id 0 \
    --beat_cache_name beat_4english_15_141 \
    --ckpt mse_best.tar
```

### 实验 2.4 — 采样步数消融（只需 FM_aa 模型）

对训好的 `beat_FM_aa` 模型，**只改 `--fm_sample_steps`**：

```bash
for STEPS in 5 10 25 50 100; do
    echo "=== Testing with $STEPS steps ==="
    python runner.py \
        --dataset_name beat \
        --name beat_FM_aa \
        --mode test \
        --flow_matching \
        --fm_sample_steps $STEPS \
        --n_poses 34 \
        --gpu_id 0 \
        --beat_cache_name beat_4english_15_141 \
        --ckpt mse_best.tar \
        2>&1 | tee logs/test_steps_${STEPS}.log
done
```

**论文 Table 2 — 采样步数消融：**

| 步数 (NFE) | FGD ↓ | MSE ↓ | PCK ↑ | Diversity | 推理时间/batch |
|------|-------|-------|-------|-----------|---------|
| 5    | | | | | |
| 10   | | | | | |
| 25   | | | | | |
| 50   | | | | | |
| 100  | | | | | |
| DDPM (1000) | | | | | |

### 实验 2.5 — 推理时间测量

```bash
python -c "
import time, torch
from runner import *  # 加载模型

# Warm up
for _ in range(3):
    outputs = trainer.generate_batch(audio, pid, dim, add_cond, inpaint)
    torch.cuda.synchronize()

# Measure
times = []
for _ in range(20):
    torch.cuda.synchronize()
    t0 = time.time()
    outputs = trainer.generate_batch(audio, pid, dim, add_cond, inpaint)
    torch.cuda.synchronize()
    times.append(time.time() - t0)

print(f'Mean: {sum(times)/len(times)*1000:.1f}ms')
print(f'Std:  {torch.tensor(times).std().item()*1000:.1f}ms')
"
```

> [!NOTE]
> 需要分别测量 FM (N步) 和 DDPM (1000步) 的推理时间。对比的是**相对加速比**。

---

## 第三阶段：辅助损失消融（~12h 服务器）

### 实验 3.1 — FM 无辅助损失（Base）

| 参数 | 值 | 差异 |
|------|---|------|
| `--name` | `beat_FM_aa_base` | |
| `--vel_loss_weight` | `0` | **关闭** 速度损失 |
| `--acc_loss_weight` | `0` | **关闭** 加速度损失 |
| `--x0_rec_weight` | `0` | **关闭** Huber 重建 |
| 其余 | 同 1.1 | |

```bash
nohup python runner.py \
    --dataset_name beat \
    --name beat_FM_aa_base \
    --mode train \
    --flow_matching \
    --fm_sample_steps 50 \
    --n_poses 34 \
    --batch_size 512 \
    --workers 32 \
    --num_epochs 300 \
    --no_fgd \
    --gpu_id 0 \
    --beat_cache_name beat_4english_15_141 \
    --vel_loss_weight 0 \
    --acc_loss_weight 0 \
    --x0_rec_weight 0 \
    > logs/train_fm_base.log 2>&1 &
```

### 实验 3.2 — FM + Vel only

| 参数 | 值 |
|------|---|
| `--name` | `beat_FM_aa_vel` |
| `--vel_loss_weight` | `100` |
| `--acc_loss_weight` | `0` |
| `--x0_rec_weight` | `0` |

```bash
nohup python runner.py \
    --dataset_name beat \
    --name beat_FM_aa_vel \
    --mode train \
    --flow_matching \
    --fm_sample_steps 50 \
    --n_poses 34 \
    --batch_size 512 \
    --workers 32 \
    --num_epochs 300 \
    --no_fgd \
    --gpu_id 0 \
    --beat_cache_name beat_4english_15_141 \
    --vel_loss_weight 100 \
    --acc_loss_weight 0 \
    --x0_rec_weight 0 \
    > logs/train_fm_vel.log 2>&1 &
```

### 实验 3.3 — FM + Vel + Acc

| 参数 | 值 |
|------|---|
| `--name` | `beat_FM_aa_velAcc` |
| `--vel_loss_weight` | `100` |
| `--acc_loss_weight` | `50` |
| `--x0_rec_weight` | `0` |

```bash
nohup python runner.py \
    --dataset_name beat \
    --name beat_FM_aa_velAcc \
    --mode train \
    --flow_matching \
    --fm_sample_steps 50 \
    --n_poses 34 \
    --batch_size 512 \
    --workers 32 \
    --num_epochs 300 \
    --no_fgd \
    --gpu_id 0 \
    --beat_cache_name beat_4english_15_141 \
    --vel_loss_weight 100 \
    --acc_loss_weight 50 \
    --x0_rec_weight 0 \
    > logs/train_fm_velAcc.log 2>&1 &
```

### 实验 3.4 — FM Full（= 实验 1.1，不需要重训）

> 直接使用 `beat_FM_aa` 的结果即可。

**论文 Table 3 — 辅助损失消融：**

| 配置 | $\mathcal{L}_{vel}$ | $\mathcal{L}_{acc}$ | $\mathcal{L}_{huber}$ | FGD ↓ | MSE ↓ | Jitter ↓ |
|------|:---:|:---:|:---:|---:|---:|---:|
| Base | ✗ | ✗ | ✗ | | | |
| +Vel | ✓ | ✗ | ✗ | | | |
| +Vel+Acc | ✓ | ✓ | ✗ | | | |
| **Full** | ✓ | ✓ | ✓ | | | |

> [!TIP]
> Jitter 指标 = 生成动作的 3 阶有限差分均值，用来量化高频抖动。可以在评估脚本中加一行计算：
> ```python
> jitter = np.mean(np.abs(np.diff(outputs, n=3, axis=1)))
> ```

---

## 第四阶段：可视化（~1-2 天）

### 4.1 测试结果生成

使用 Phase 2 的测试命令，输出 `.npy` 文件。

### 4.2 反标准化 + BVH 转换

```python
# npy → euler degrees → BVH
import numpy as np
from datasets.rotation_converter import *

out = np.load("results/.../00000.npy")  # [T, 141] normalized euler
mean = np.load("data/BEAT/.../train/bvh_rot/bvh_mean.npy")
std = np.load("data/BEAT/.../train/bvh_rot/bvh_std.npy")
euler_deg = out * std + mean  # denormalize
# → 写入 BVH 文件或直接用 Blender 加载
```

### 4.3 Blender 渲染

1. 打开 `beat_visualize.blend`
2. 加载生成的 BVH
3. 录制对比视频：FM vs DDPM vs GT

### 4.4 需要的图表

| 图表 | 数据来源 | 用途 |
|------|---------|------|
| Loss 下降曲线 | wandb offline / 训练日志 | 证明训练收敛 |
| 采样步数 vs FGD | Phase 2.4 | Table 2 对应的折线图 |
| FM vs DDPM 速度对比 | Phase 2.5 | 推理加速比柱状图 |
| 可视化对比帧 | Blender 截图 | 定性展示 |

---

## 第五阶段：论文撰写（~3-5 天）

### 核心表格清单

| 表号 | 内容 | 数据来源 |
|------|------|---------|
| **Table 1** | FM vs DDPM 主结果 | Phase 2.1 + 2.3 |
| **Table 2** | 采样步数消融 (5/10/25/50/100) | Phase 2.4 |
| **Table 3** | Vel/Acc/Huber 损失消融 | Phase 3 |
| **Table 4** | 6D vs axis-angle 旋转表示 | Phase 2.1 + 2.2 |
| **Table 5** | 推理时间对比 | Phase 2.5 |

### 论文章节 → 实验对应

| 章节 | 需要的实验数据 |
|------|--------------|
| 3.1 Flow Matching | 数学推导，不需要实验 |
| 3.2 Velocity + Huber Loss | 公式推导 + Table 3 |
| 3.3 6D 旋转 + Gram-Schmidt | 公式推导 + Table 4 |
| 4.2 主结果 | **Table 1** (FM vs DDPM) |
| 4.3 采样步数消融 | **Table 2** + 折线图 |
| 4.4 辅助损失消融 | **Table 3** |
| 4.5 旋转表示消融 | **Table 4** |
| 4.6 推理时间 | **Table 5** + 柱状图 |
| 4.7 可视化 | Blender 截图/视频 |

---

## 总时间预估

| 阶段 | 任务 | 服务器时间 | 人工时间 | 依赖 |
|------|------|-----------|---------|------|
| Phase 1 | 3 个模型训练 | ~24h | 1h（启动+监控） | 无 |
| Phase 2 | 推理+评估+步数消融 | ~3h | 1h | Phase 1 |
| Phase 3 | 3 个消融训练+测试 | ~15h | 1h | 无（可与 Phase 1 并行） |
| Phase 4 | Blender 可视化+图表 | — | 1-2 天 | Phase 2 |
| Phase 5 | 论文正文撰写 | — | 3-5 天 | Phase 2+3+4 |
| **总计** | | **~42h 服务器** | **~1-2 周** |
| **总费用** | | **~¥75** (3090 ¥1.79/h) | |

---

## 快速检查清单

上服务器前确认：

- [ ] 代码已同步到服务器（含 6D 旋转改动）
- [ ] `axis_angle_mean.npy` / `axis_angle_std.npy` 已上传
- [ ] LMDB 缓存已生成（`bvh_rot_cache_len34_stride10/`）
- [ ] HuBERT 特征已提取（`aud_feat_cache/`）
- [ ] `logs/` 目录已创建
- [ ] conda 环境已激活
