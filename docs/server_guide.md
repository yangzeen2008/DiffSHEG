# DiffSHEG 云端服务器部署与训练完全指南

> **当前服务器**：SeetaCloud RTX 4090 魔改 48G | 550G 数据盘 | SSH端口 48360

这份指南总结了 DiffSHEG 项目从本地 Windows 迁移到 Linux 云端 GPU 服务器（如 AutoDL / SeetaCloud）的**全部实战经验**，包含核心配置、避坑方案、性能优化策略以及选机建议。

---

## 1. 存储与数据盘管理（防爆盘核心）

云端 GPU 服务器通常分为 **系统盘**（`/`，仅 30GB）和 **数据盘**（`/root/autodl-tmp`，通常 200GB+）。

> ⚠️ **致命警告**：DiffSHEG 的 HuBERT 特征提取阶段会产生约 **35 万个碎片文件，总计 170GB+**。
> 如果放在系统盘，10 分钟内服务器必死无疑。

**正确操作：**
```bash
# 将整个项目上传到数据盘
mv /root/DiffSHEG /root/autodl-tmp/DiffSHEG
# 在系统盘创建软连接，保持路径兼容
ln -s /root/autodl-tmp/DiffSHEG /root/DiffSHEG
```

### 1.1 磁盘空间预算

| 阶段 | 占用空间 | 说明 |
|------|---------|------|
| 原始数据 (raw) | ~25GB | .bvh 动捕 + .wav 音频 |
| 预处理缓存 (beat_cache) | ~170GB | preprocess_beat.py 输出 |
| HuBERT 特征 (aud_feat_cache) | ~60GB | build_hubert_cache.py 输出（含在上面的 170G 中） |
| 预训练模型 (checkpoints) | ~3.5GB | 从作者网盘下载 |
| Conda 环境 + pip 包 | ~10GB | 装在系统盘 |
| **总计** | **约 210GB** | **建议数据盘至少 250GB** |

### 1.2 爆盘急救方案

当数据盘接近满载时，可按以下优先级释放空间：

```bash
# 优先级 1：删除原始数据（预处理完成后不再需要，释放 ~25GB）
rm -rf /root/autodl-tmp/DiffSHEG/data/BEAT/raw

# 优先级 2：删除 test 集缓存（训练时只需 train + val，释放 ~60GB）
rm -rf /root/autodl-tmp/DiffSHEG/data/BEAT/beat_cache/beat_4english_15_141/test
```

---

## 2. 跨平台路径兼容（Windows vs Linux）

本地开发时容易写死绝对路径（如 `f:\study\DiffSHEG\data\...`），导致代码在 Linux 上直接报错。

**正确操作：** 始终使用**相对路径**。
```python
# ❌ 错误：
RAW_DIR = r"f:\study\DiffSHEG\data\BEAT\raw\..."
CACHE_ROOT = r"f:\study\DiffSHEG\data\BEAT\beat_cache\..."

# ✅ 正确：
RAW_DIR = "./data/BEAT/raw/..."
CACHE_ROOT = "./data/BEAT/beat_cache/..."
```

已修改的文件：`preprocess_beat.py`、`build_hubert_cache.py`。

---

## 3. 环境依赖安装（避免 pip 卡死）

### 3.1 PyTorch 版本锁定

在云端安装 PyTorch 极易遇到 **版本依赖无限回溯 (Backtracking)** 的问题。必须**强制绑定**精确版本号：

```bash
conda activate diffsheg

# 强制锁死版本，瞬间完成解析
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --extra-index-url https://download.pytorch.org/whl/cu121
```

### 3.2 必装依赖清单

runner.py 启动时需要以下额外依赖，**必须提前安装**，否则会因缺包而反复崩溃：

```bash
pip install numpy==1.26.4 matplotlib tensorboard pyarrow ipython wandb tensorboardX trimesh smplx imageio
pip install -U openmim && mim install mmcv-full==1.7.2
```

### 3.3 必传代码文件

以下文件/文件夹容易在上传时遗漏，导致 runner.py 报 `ModuleNotFoundError`：

| 文件/文件夹 | 用途 |
|-------------|------|
| `utils/` | 核心工具函数（print_current_loss 等） |
| `datasets/` | 数据集定义（BeatDataset 等） |
| `trainers/` | 训练器定义（DDPMTrainer_beat 等） |
| `models/` | 模型架构定义 |
| `options/` | 参数解析 |

### 3.4 必传数据文件

`preprocess_beat.py` 不会生成 axis-angle 统计文件，需要从本地手动上传：

```
data/BEAT/beat_cache/beat_4english_15_141/train/axis_angle_mean.npy
data/BEAT/beat_cache/beat_4english_15_141/train/axis_angle_std.npy
```

---

## 4. HuggingFace 模型下载（国内网络加速）

`build_hubert_cache.py` 需要从 HuggingFace 下载 `facebook/hubert-large-ls960-ft` 模型（~1.2GB）。国内服务器直连经常超时。

**正确操作：**
```bash
export HF_ENDPOINT=https://hf-mirror.com
python build_hubert_cache.py --split all
```

---

## 5. 训练速度优化（关键参数调优）

### 5.1 性能瓶颈分析

DiffSHEG 是一个**轻量级 1D 时序模型**（骨骼关节旋转角度），不是图像/视频生成模型。其特点：
- 模型参数量小（几千万级别）
- 显存占用低（bs=32 仅需 4GB）
- **瓶颈在 CPU 数据加载，而非 GPU 算力**

### 5.2 参数优化实测数据

| 配置 | batch_size | workers | 显存占用 | GPU利用率 | 每Epoch耗时 | 机型 |
|------|-----------|---------|---------|----------|------------|------|
| 初始版 | 32 | 4 | 4 GB | 7~46% | ~10分钟 | 3090 |
| 优化版 | 128 | 16 | 8 GB | ~58% | ~4分钟 | 3090 |
| 极速版 | 256 | 32 | 31 GB | ~88% | ~1.5分钟 | 3090 48G |
| **当前版** | **512** | **32** | **31 GB** | **~86%** | **~1分钟** | **4090 48G** |

> **推荐配置：batch_size=512, workers=32**
> 相比初始配置，吞吐量提升约 **8 倍**。

### 5.3 最终训练命令

```bash
python runner.py \
    --dataset_name beat \
    --name beat_FM_v1 \
    --mode train \
    --flow_matching \
    --fm_sample_steps 50 \
    --n_poses 34 \
    --batch_size 512 \
    --no_fgd \
    --gpu_id 0 \
    --beat_cache_name beat_4english_15_141 \
    --resume \
    --workers 32
```

### 5.4 训练时长与费用预估

| 目标 Epoch | 4090 48G 耗时 | 3090 48G 耗时 |
|-----------|---------------|---------------|
| 100 | ~1.5 小时 | ~2 小时 |
| 300 | ~4 小时 | ~5 小时 |
| 500 | ~7 小时 | ~8 小时 |

---

## 6. 机型选择指南

### 6.1 核心结论：多卡 3090 > 贵族单卡

DiffSHEG 模型太轻，高端卡（A100/H100/H800）的算力无法被充分利用，**性价比极低**。

| 方案 | 价格 | 预估速度 | 性价比 |
|------|------|---------|--------|
| **1× 3090** | **¥1.79/h** | **1×** | **★★★★★** |
| 2× 3090 | ~¥3.58/h | ~1.8× | ★★★★☆ |
| 4× 3090 | ~¥7.16/h | ~3.5× | ★★★☆☆ |
| 1× H800 | ¥8.88/h | ~1.5× | ★☆☆☆☆ |

> H800 是给训练 GPT/Sora 这种千亿参数怪兽用的。
> 你的模型像一辆灵活的赛车，花 5 倍的钱换 8 车道高速公路，赛车也只能占 1 条道。
> 不如花 2 倍的钱，多开一辆赛车并排跑。

### 6.2 多卡并行启动方式

代码原生支持 `DistributedDataParallel (DDP)`，**零代码改动**，只需加参数：

```bash
# 多卡自动并行（代码会自动检测并使用所有可用 GPU）
python runner.py \
    --dataset_name beat \
    --name beat_FM_v1 \
    --mode train \
    --flow_matching \
    --fm_sample_steps 50 \
    --n_poses 34 \
    --batch_size 1024 \
    --no_fgd \
    --beat_cache_name beat_4english_15_141 \
    --multiprocessing_distributed \
    --workers 64
```

---

## 7. 全自动后台执行流

深度学习训练需要数小时。SSH 断开后进程会被杀死，必须使用 `nohup` 挂载后台。

### 7.1 一键部署脚本 (pipeline.sh)

```bash
#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh || source /opt/conda/etc/profile.d/conda.sh
conda activate diffsheg
cd /root/autodl-tmp/DiffSHEG

# Step 1: 数据预处理（纯 CPU，约 30 分钟）
echo 'Starting Preprocessing...'
python preprocess_beat.py

# Step 2: HuBERT 特征提取（GPU + CPU，约 10 分钟）
echo 'Starting HuBERT...'
export HF_ENDPOINT=https://hf-mirror.com
python build_hubert_cache.py --split all

# Step 3: 正式训练（GPU 全速，约 8 小时跑 500 Epoch）
echo 'Starting Training...'
python runner.py --dataset_name beat --name beat_FM_v1 --mode train \
    --flow_matching --fm_sample_steps 50 --n_poses 34 \
    --batch_size 512 --no_fgd --gpu_id 0 \
    --beat_cache_name beat_4english_15_141 --resume --workers 32
```

### 7.2 启动与监控

```bash
# 启动（后台持久运行）
nohup bash pipeline.sh > logs/pipeline.log 2>&1 &

# 实时查看进度
tail -f /root/autodl-tmp/DiffSHEG/logs/pipeline.log

# 查看 GPU 状态
nvidia-smi

# 查看磁盘使用
df -h /root/autodl-tmp
```

---

## 8. 本地文件备份策略

### 8.1 需要保留的文件（存网盘，约 25GB）

| 文件/文件夹 | 大小 | 说明 |
|-------------|------|------|
| 整个 DiffSHEG 代码目录 | ~50MB | 含修复后的脚本和本指南 |
| data/BEAT/raw/ | ~25GB | 原始数据，可重新生成缓存 |
| axis_angle_mean/std.npy | ~2KB | 需手动上传到服务器 |

### 8.2 可以安全删除的文件

| 文件/文件夹 | 大小 | 原因 |
|-------------|------|------|
| `data/BEAT/beat_cache/` | **~253GB** | 代码自动生成的缓存，可随时重建 |
| `checkpoints/` | ~9GB | 预训练权重，网盘有压缩包兜底 |
| `diffsheg_ckpt.zip` | ~3.4GB | 解压后可删，等服务器训练新模型 |
| `.venv/` | ~2GB | 本地虚拟环境，服务器有独立环境 |

> 删除以上文件可释放本地约 **270GB** 磁盘空间。

---

## 9. 快速排障手册

| 症状 | 原因 | 解决方案 |
|------|------|---------|
| `ValueError: not enough values to unpack` | 启用了联合多模态但加了 `--gesture_only` 导致解包维度不匹配 | **禁用 `--gesture_only` 标志**。默认以表情+手势联合模式训练和评估。 |
| `pip install` 卡死不动 | PyTorch 版本回溯 | 锁死版本号（见第 3.1 节） |
| `OSError: Not enough free space` | 数据盘爆满 | 删 raw 或 test 缓存（见第 1.2 节） |
| `ModuleNotFoundError: utils` | 漏传代码文件 | 上传 utils/ 文件夹（见第 3.3 节） |
| `ModuleNotFoundError: mmcv` | 缺少依赖 | `pip install -U openmim && mim install mmcv-full==1.7.2` |
| `ModuleNotFoundError: wandb` | 缺少依赖 | `pip install wandb matplotlib ipython pyarrow` |
| `FileNotFoundError: axis_angle_mean.npy` | 预处理未生成 | 从本地手动上传（见第 3.4 节） |
| GPU 利用率低 (<50%) | batch_size/workers 太小 | 加大到 bs=512, workers=32（见第 5 节） |
| 显存占用低 (<10GB) | 模型轻量，属正常现象 | 加大 batch_size 压榨显存 |
