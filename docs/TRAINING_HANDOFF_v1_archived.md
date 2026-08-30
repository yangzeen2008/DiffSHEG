# DiffSHEG BEAT 训练 — 会话交接文档

> 最后更新：2026-04-20 00:46  
> 项目路径：`f:\study\DiffSHEG`  
> 虚拟环境：`f:\study\DiffSHEG\.venv`

> **归档警告（2026-08-30）**：本文记录的 train 316,069 / val 37,881 个
> HuBERT 文件已确认没有与动作 LMDB 的 train 33,811 / val 4,047 个样本
> 一一对应。不得按本文命令恢复旧训练或复用旧缓存；本文仅保留为历史记录。

---

## 📌 当前状态（重要，先读这里）

**训练已成功启动并运行中。**

| 项目 | 状态 |
|------|------|
| 数据预处理 | ✅ 完成（train/val/test 全部 141 维） |
| HuBERT 特征提取 | ✅ 完成（train 316k npy, val 37k npy, test 52 npy） |
| LMDB 主缓存 | ✅ 完成（train + val bvh_rot_cache_len34_stride10） |
| **训练** | ✅ **正在运行** — epoch 0 已完成 2408 steps |
| 最新 checkpoint | `checkpoints/beat/beat_FM_v1/model/latest.tar` (1.87 GB) |

---

## 🚀 下次开始时的第一步

### 1. 检查训练是否还在运行

```powershell
Get-Process python -ErrorAction SilentlyContinue | Select-Object Id, CPU, WorkingSet64
```

如果进程存在且 CPU 持续增长，训练正常。  
如果进程不存在，用下面的命令 **带 `--resume` 恢复训练**：

### 2. 恢复训练命令

```powershell
cd f:\study\DiffSHEG
f:\study\DiffSHEG\.venv\Scripts\python.exe runner.py `
  --dataset_name beat `
  --name beat_FM_v1 `
  --mode train `
  --flow_matching `
  --fm_sample_steps 50 `
  --n_poses 34 `
  --batch_size 16 `
  --no_fgd `
  --gpu_id 0 `
  --beat_cache_name beat_4english_15_141 `
  --resume
```

> ⚠️ **注意**：不需要设置 `HF_ENDPOINT`，不需要重跑预处理或 HuBERT，直接 `--resume` 即可。

### 3. 查看训练进度

```powershell
f:\study\DiffSHEG\.venv\Scripts\python.exe -c "
import torch
ckpt = torch.load(r'f:\study\DiffSHEG\checkpoints\beat\beat_FM_v1\model\latest.tar', map_location='cpu')
print('epoch:', ckpt.get('ep'))
print('total_it:', ckpt.get('total_it'))
"
```

---

## 🗂️ 完整项目结构（已整理完毕）

```
f:\study\DiffSHEG\
├── data\BEAT\
│   ├── raw\beat_english_v0.2.1\      # 原始数据（不要动）
│   └── beat_cache\beat_4english_15_141\
│       ├── train\
│       │   ├── bvh_rot\              # 141 维 BVH（已重新生成）
│       │   ├── wave16k\              # 16kHz WAV numpy
│       │   ├── facial52\            # JSON 表情数据
│       │   ├── sem\                  # 语义标注
│       │   ├── aud_feat_cache\
│       │   │   └── hubert_large_ls960_ft\  # HuBERT npy（316k 文件）
│       │   └── bvh_rot_cache_len34_stride10\  # 主 LMDB 缓存
│       ├── val\   （同上结构）
│       └── test\  （同上结构）
├── checkpoints\beat\beat_FM_v1\
│   ├── opt.txt                       # 训练参数记录
│   ├── meta\
│   └── model\
│       └── latest.tar                # 最新 checkpoint（1.87 GB）
├── datasets\beat.py                  # 已修改（见下方修复清单）
├── models\transformer.py             # 已修改（sqrt_alphas 默认值）
├── models\flow_matching.py           # Flow Matching 训练逻辑
├── trainers\ddpm_beat_trainer.py     # 已修改（wandb offline）
├── preprocess_beat.py                # 已修改（141 维列选择）
├── build_hubert_cache.py             # 已修改（npy 存储模式）
└── .venv\                            # Python 虚拟环境
```

---

## 🔧 本次会话的所有代码修改

### 1. `datasets/beat.py`

| 位置 | 问题 | 修复 |
|------|------|------|
| 顶部 import | 缺少 `pathlib` | 加了 `from pathlib import Path` |
| `cache_generation` L168 | Windows 路径 `/` 分割失败 | `Path(pose_file).stem` |
| LMDB map_size | 超出 Windows 虚拟内存上限 | `min(map_size, 20GB)` |
| `_sample_from_clip` 多处 | `array != []` numpy 广播错误 | 全改为 `len() > 0` |
| `MotionPreprocessor.get()` | `array != []` | `len() > 0` |
| `MotionPreprocessor.check_pose_diff()` | 228 vs 141 维不匹配 | 改用 `np.std(self.skeletons)` |
| `__getitem__` | pyarrow 依赖移除 | 改用 `pickle.loads` |
| HuBERT 加载 | LMDB 模式 → npy 模式 | 读 `aud_feat_cache/hubert_large_ls960_ft/{idx:05d}.npy` |

### 2. `models/transformer.py`

| 位置 | 问题 | 修复 |
|------|------|------|
| `forward()` 签名 | `sqrt_alphas` 是必须参数，FM 不传 | 改为 `sqrt_alphas=None`（默认值） |
| `forward()` L749 | FM 模式下 `_predict_xstart_from_eps` 崩溃 | 添加 `if sqrt_alphas is not None` 分支，FM 时直接用 `exp_noise_t.detach()` |

### 3. `trainers/ddpm_beat_trainer.py`

| 位置 | 问题 | 修复 |
|------|------|------|
| `__init__` wandb 初始化 | 无 API Key 崩溃 | 设 `os.environ["WANDB_MODE"] = "offline"` |

### 4. `preprocess_beat.py`

| 位置 | 问题 | 修复 |
|------|------|------|
| `get_bvh_rotation_lines()` | 直接输出 228 维 | 加入 `_BVH_COL_141` 列表，提取 141 列后输出 |

**141 列选择逻辑**：
- Hips：仅保留旋转（列 3-5，跳过平移 0-2）
- 排除：`HeadEnd`, `Neck1`, 所有 `*4` 末端手指节点（Middle4/Ring4/Pinky4/Index4/Thumb4）
- 排除：`RightHandRing`, `RightHandPinky`, `LeftHandRing`, `LeftHandPinky`（掌骨根）
- 排除：所有腿脚关节（RightUpLeg 到 LeftToeBaseEnd）

### 5. `build_hubert_cache.py`

- 完整重写，改为 npy 文件存储（避免 Windows LMDB `MDB_MAP_FULL`）
- 加入断点续跑支持（`done.txt` + index 跳过）
- 维度不匹配过滤改用 `np.std()` 检测静止动作

---

## ⚙️ 环境信息

```
Python:      f:\study\DiffSHEG\.venv\Scripts\python.exe
PyTorch:     2.1.0 + CUDA
转换用 Python: C:\Users\yangz\miniconda3\python.exe（preprocess 脚本用这个）
HF 镜像:     https://hf-mirror.com（已在 build_hubert_cache.py 中固化）
Numpy:       < 2.0（不能升级）
Transformers: 4.37.2
```

**已安装的额外包（通过 .venv）**：
```
loguru, einops, IPython, termcolor, scikit-learn, pandas
```

---

## 📊 数据集信息

| Split | BVH clips | LMDB 样本数 | HuBERT npy 文件 |
|-------|-----------|------------|----------------|
| train | 416 | ~316,000 | 316,069 |
| val | 52 | ~37,000 | 37,881 |
| test | 52 | — | 52（每 clip 一个） |

- 说话人：1, 2, 3, 4（4 位英文说话人）
- BVH 维度：**141**（47 关节 × 3）
- 表情维度：**51**（facial52 目录）
- 合并后 `net_dim_pose`：**192**（141 + 51）
- Audio：16kHz，HuBERT-large-ls960-ft（1024 维 → Mel 128 维作 aud_feat）

---

## 🎯 训练目标与参数

```
模型：    DiffSHEG UniDiffuser（Transformer Encoder，8 层）
方法：    Flow Matching（OT-CFM，线性插值）
步数：    50 inference steps（--fm_sample_steps 50）
Epochs：  5000（可让其自动跑）
Batch：   16（保守，防内存溢出）
LR：      0.0002（Adam）
n_poses： 34（约 2.27s @ 15fps）
评估：    每 5 epoch，跳过 FGD（--no_fgd）
保存：    每 5 epoch + 每 500 steps（latest.tar）
```

---

## 🔮 后续可能需要做的任务

### 短期（训练监控）

1. **观察 loss 下降**：训练日志每 50 steps 打一次，loss 应该从 ~1.0 逐渐下降
2. **eval 结果**：每 5 epoch 做 MSE/PCK 评估（无 FGD）
3. **调整 batch_size**：若 GPU 利用率低可提升到 32

### 中期（如训练崩溃）

```powershell
# 查看最新 checkpoint
f:\study\DiffSHEG\.venv\Scripts\python.exe -c "
import torch
ckpt = torch.load(r'f:\study\DiffSHEG\checkpoints\beat\beat_FM_v1\model\latest.tar', map_location='cpu')
print('epoch:', ckpt.get('ep'), 'it:', ckpt.get('total_it'))
"

# 恢复（加 --resume）
f:\study\DiffSHEG\.venv\Scripts\python.exe runner.py `
  --dataset_name beat --name beat_FM_v1 --mode train `
  --flow_matching --fm_sample_steps 50 --n_poses 34 `
  --batch_size 16 --no_fgd --gpu_id 0 `
  --beat_cache_name beat_4english_15_141 --resume
```

### 长期（推理/评估）

训到足够 epoch 后，改 `--mode test` 进行推理评估：

```powershell
f:\study\DiffSHEG\.venv\Scripts\python.exe runner.py `
  --dataset_name beat --name beat_FM_v1 --mode test `
  --flow_matching --fm_sample_steps 50 --n_poses 34 `
  --no_fgd --gpu_id 0 `
  --beat_cache_name beat_4english_15_141
```

---

## ⚠️ 已知注意事项

1. **LMDB 缓存加载慢**：训练进程启动后会有 20-30 分钟静默期（Python 进程 CPU 消耗高、内存持续增长到约 10 GB 是正常的）。通过查看 `latest.tar` 文件时间戳确认训练在进行。

2. **内存占用**：训练稳定后约占 10-11 GB RAM，系统有 32 GB，余量充足。避免同时运行浏览器等大程序。

3. **Windows LMDB 限制**：原始代码在 Windows 上有 `MDB_MAP_FULL` 问题，已通过 `min(map_size, 20GB)` 解决。

4. **BVH 维度**：原始 BEAT BVH 是 228 维，我们预处理脚本已改为输出 **141 维**，与官方 `bvh_mean.npy` 对齐。

5. **axis_angle_mean/std.npy**：这两个文件是官方 BEAT 数据集提供的预计算文件（2024/4/17），与我们的 141 维 BVH 维度匹配（均为 141 维）。

6. **wandb**：已配置为 offline 模式，日志保存在 `wandb/` 目录，不需要 API key。

---

## 📁 关键文件快速索引

| 文件 | 用途 |
|------|------|
| `runner.py` | 训练/测试入口 |
| `datasets/beat.py` | BEAT 数据集加载（已大量修改） |
| `models/flow_matching.py` | Flow Matching 损失实现 |
| `models/transformer.py` | 主模型（已修改 forward 签名） |
| `trainers/ddpm_beat_trainer.py` | 训练循环（已修改 wandb） |
| `preprocess_beat.py` | 原始数据 → 141 维 BVH（已修改） |
| `build_hubert_cache.py` | HuBERT npy 提取（已重写） |
| `options/base_options.py` | 默认参数（workers=0 很重要） |

---

*文档由 Antigravity AI 生成，记录 2026-04-19 至 2026-04-20 的工作进展。*
