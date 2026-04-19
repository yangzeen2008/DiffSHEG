"""
BEAT HuBERT 特征提取脚本（文件夹版）
将每个训练样本窗口的 HuBERT 特征存储为独立的 .npy 文件，
避免 Windows 上 LMDB map_size 限制问题。

文件命名格式：{global_idx:05d}.npy
读取方式：np.load(path / f"{idx:05d}.npy")

用法：
    f:\\study\\DiffSHEG\\.venv\\Scripts\\python.exe build_hubert_cache.py --split train
    f:\\study\\DiffSHEG\\.venv\\Scripts\\python.exe build_hubert_cache.py --split val
    f:\\study\\DiffSHEG\\.venv\\Scripts\\python.exe build_hubert_cache.py --split test
"""

import os, sys, glob, math, argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

# ===== 配置（必须与训练参数完全一致）=====
CACHE_ROOT    = r"f:\study\DiffSHEG\data\BEAT\beat_cache\beat_4english_15_141"
POSE_LEN      = 34        # --n_poses
STRIDE        = 10        # BEAT 数据集 stride（runner.py line 168）
POSE_FPS      = 15        # 姿态帧率
AUDIO_FPS     = 16000     # 音频采样率
MULTI_LEN_TRAINING = [1.0]  # 默认单尺度
DEVICE        = "cuda:0"

# HuBERT CNN 参数（固定，不要改）
HUB_KERNEL = 400
HUB_STRIDE = 320
# ==========================================


def load_hubert_model(device):
    from transformers import Wav2Vec2Processor, HubertModel
    print("Loading HuBERT processor...")
    processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
    print("Loading HuBERT model...")
    model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft").to(device)
    model.eval()
    return processor, model


@torch.no_grad()
def extract_hubert_full(audio_np, processor, model, device):
    """对整段音频提取 HuBERT 特征，返回 [T_hub, 1024]。"""
    if audio_np.ndim == 2:
        audio_np = audio_np[:, 0]
    input_values = processor(
        audio_np, return_tensors="pt", sampling_rate=16000
    ).input_values.to(device)  # [1, N]

    clip_length = HUB_STRIDE * 1000
    num_iter    = input_values.shape[1] // clip_length
    expected_T  = (input_values.shape[1] - (HUB_KERNEL - HUB_STRIDE)) // HUB_STRIDE
    res_lst = []

    for i in range(num_iter):
        if i == 0:
            start, end = 0, clip_length - HUB_STRIDE + HUB_KERNEL
        else:
            start = clip_length * i
            end   = start + (clip_length - HUB_STRIDE + HUB_KERNEL)
        hidden = model(input_values[:, start:end]).last_hidden_state
        res_lst.append(hidden[0].cpu())

    tail = input_values[:, clip_length * num_iter:]
    if tail.shape[1] >= HUB_KERNEL:
        hidden = model(tail).last_hidden_state
        res_lst.append(hidden[0].cpu())

    if not res_lst:
        return None
    ret = torch.cat(res_lst, dim=0)  # [T_hub, 1024]
    if ret.shape[0] < expected_T:
        ret = F.pad(ret.T.unsqueeze(0), (0, expected_T - ret.shape[0])).squeeze(0).T
    else:
        ret = ret[:expected_T]
    return ret  # [T_hub, 1024]


def build_split(split, processor, model):
    split_dir = os.path.join(CACHE_ROOT, split)
    bvh_dir   = os.path.join(split_dir, "bvh_rot")
    wav_dir   = os.path.join(split_dir, "wave16k")
    out_dir   = os.path.join(split_dir, "aud_feat_cache", "hubert_large_ls960_ft")

    # 检查是否已有完整输出
    done_flag = os.path.join(out_dir, "done.txt")
    if os.path.exists(done_flag):
        with open(done_flag) as f:
            n = f.read().strip()
        print(f"[{split}] 已完成，共 {n} 个样本，跳过。")
        return

    os.makedirs(out_dir, exist_ok=True)

    mean_pose_path = os.path.join(CACHE_ROOT, "train", "bvh_rot", "bvh_mean.npy")
    mean_pose = np.load(mean_pose_path)  # 仅用维度做参考，不做逐帧对比

    bvh_files = sorted(glob.glob(os.path.join(bvh_dir, "*.bvh")))
    print(f"[{split}] 处理 {len(bvh_files)} 个 clip → {out_dir}")

    global_idx = 0
    is_test = (split == "test")

    # 支持断点续传：跳过已生成的文件
    existing = set(int(p.stem) for p in Path(out_dir).glob("*.npy"))
    if existing:
        print(f"  断点续传：已有 {len(existing)} 个样本，继续生成...")
        global_idx = max(existing) + 1

    for bvh_path in tqdm(bvh_files, desc=f"[{split}]"):
        stem = Path(bvh_path).stem
        wav_path = os.path.join(wav_dir, stem + ".npy")
        if not os.path.exists(wav_path):
            print(f"  WARN: wav 不存在，跳过 {stem}")
            continue

        # 1. 读 BVH 帧数
        pose_frames = []
        with open(bvh_path, "r") as f:
            for line in f:
                vals = line.strip().split()
                if vals:
                    try:
                        pose_frames.append([float(v) for v in vals])
                    except ValueError:
                        pass
        if not pose_frames:
            continue
        pose_arr = np.array(pose_frames)  # [T, D_raw]

        # 2. 窗口范围（复现 build_cache 逻辑）
        round_seconds = pose_arr.shape[0] // POSE_FPS
        clip_e_f_pose = round_seconds * POSE_FPS
        clip_s_f_pose = 0

        # 3. 提取整段 HuBERT
        audio = np.load(wav_path).astype(np.float32)
        hubert_full = extract_hubert_full(audio, processor, model, DEVICE)
        if hubert_full is None:
            print(f"  WARN: HuBERT 提取失败，跳过 {stem}")
            continue

        # 4. 滑窗分割
        for ratio in MULTI_LEN_TRAINING:
            if is_test:
                cur_pose_len = clip_e_f_pose - clip_s_f_pose
                cur_stride   = cur_pose_len
            else:
                cur_stride   = int(ratio * STRIDE)
                cur_pose_len = int(POSE_LEN * ratio)

            num_subdiv = math.floor(
                (clip_e_f_pose - clip_s_f_pose - cur_pose_len) / cur_stride
            ) + 1

            audio_short_samples = math.floor(cur_pose_len / POSE_FPS * AUDIO_FPS)
            hub_per_window = math.floor(
                (audio_short_samples - (HUB_KERNEL - HUB_STRIDE)) / HUB_STRIDE
            )
            hub_per_window = max(hub_per_window, 1)

            for i in range(num_subdiv):
                start_pose = clip_s_f_pose + i * cur_stride
                sample_pose = pose_arr[start_pose: start_pose + cur_pose_len]

                # 静止过滤
                if not is_test:
                    pose_std = sample_pose.std()
                    if pose_std < 0.5:
                        continue

                # HuBERT 对应窗口
                audio_start_sample = math.floor(
                    i * cur_stride * AUDIO_FPS / POSE_FPS
                )
                hub_start = math.floor(audio_start_sample / HUB_STRIDE)
                hub_end   = hub_start + hub_per_window

                if hub_end <= hubert_full.shape[0]:
                    chunk = hubert_full[hub_start:hub_end].numpy()
                else:
                    chunk = hubert_full[hub_start:].numpy()
                    pad_len = hub_per_window - chunk.shape[0]
                    if pad_len > 0:
                        chunk = np.pad(chunk, ((0, pad_len), (0, 0)))
                    chunk = chunk[:hub_per_window]

                # 跳过断点续传已有的
                if global_idx in existing:
                    global_idx += 1
                    continue

                out_path = os.path.join(out_dir, f"{global_idx:05d}.npy")
                np.save(out_path, chunk.astype(np.float32))
                global_idx += 1

    # 写完成标志
    with open(done_flag, "w") as f:
        f.write(str(global_idx))
    print(f"[{split}] 完成！写入 {global_idx} 个 HuBERT 样本 → {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "val", "test", "all"])
    args = parser.parse_args()

    print(f"CUDA available: {torch.cuda.is_available()}")
    processor, model = load_hubert_model(DEVICE)

    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    for sp in splits:
        build_split(sp, processor, model)

    print("\n=== HuBERT 特征提取完成！===")
    print("注意：beat.py 的 HuBERT 读取部分已同步修改为 npy 文件读取模式")
    print("\n下一步运行训练：")
    print("  f:\\study\\DiffSHEG\\.venv\\Scripts\\python.exe runner.py "
          "--dataset_name beat --name beat_FM_v1 --mode train "
          "--flow_matching --fm_sample_steps 50 --n_poses 34 "
          "--batch_size 32 --no_fgd --gpu_id 0")


if __name__ == "__main__":
    main()
