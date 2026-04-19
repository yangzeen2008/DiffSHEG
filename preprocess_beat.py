"""
BEAT 数据集预处理脚本（一键运行）
将下载好的原始数据整理成 DiffSHEG 训练所需的目录结构。

执行步骤：
  1. 按 train/val/test split 分配文件
  2. 解析 BVH，提取旋转数值行写成新格式
  3. WAV → 16kHz numpy (.npy)
  4. JSON / TXT 直接复制
  5. 计算 mean/std 统计量

用法：
    C:\\Users\\yangz\\miniconda3\\python.exe preprocess_beat.py
"""

import os
import sys
import glob
import json
import shutil
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

# ===== 配置 =====
RAW_DIR   = r"f:\study\DiffSHEG\data\BEAT\raw\beat_english_v0.2.1\beat_english_v0.2.1"
OUT_ROOT  = r"f:\study\DiffSHEG\data\BEAT\beat_cache\beat_4english_15_141"

# 4 位英文说话人的文件夹 ID（对应 BEAT 数据集中的编号）
SPEAKER_IDS = ["1", "2", "3", "4"]

# train/val/test split（按文件名末尾的序号范围划分）
# BEAT 各说话人约有 0_x_x（短片段）和 1_x_x（长片段）两类
# 这里简单按比例划分：80% train / 10% val / 10% test
TRAIN_RATIO = 0.8
VAL_RATIO   = 0.1
# TEST_RATIO = 0.1（剩余）

TARGET_AUDIO_SR = 16000   # 目标采样率
# ====================


# 从 228 维 BVH 中提取 141 维（47 关节 × 3，去掉 Hips 平移和下肢/末端关节）
# 关节列表详见脚本头部注释；Hips 保留旋转列[3:6]，其余各取3列
_BVH_COL_141 = [
    3, 4, 5,   # Hips rot
    6, 7, 8,   # Spine
    9, 10, 11, # Spine1
    12, 13, 14,# Spine2
    15, 16, 17,# Spine3
    18, 19, 20,# Neck
    # Neck1 excluded
    24, 25, 26,# Head
    # HeadEnd excluded
    30, 31, 32,# RightShoulder
    33, 34, 35,# RightArm
    36, 37, 38,# RightForeArm
    39, 40, 41,# RightHand
    42, 43, 44,# RightHandMiddle1
    45, 46, 47,# RightHandMiddle2
    48, 49, 50,# RightHandMiddle3
    # RightHandMiddle4 excluded
    57, 58, 59,# RightHandRing1  (RightHandRing root excluded)
    60, 61, 62,# RightHandRing2
    63, 64, 65,# RightHandRing3
    # RightHandRing4 excluded
    72, 73, 74,# RightHandPinky1 (RightHandPinky root excluded)
    75, 76, 77,# RightHandPinky2
    78, 79, 80,# RightHandPinky3
    # RightHandPinky4 excluded
    84, 85, 86,# RightHandIndex
    87, 88, 89,# RightHandIndex1
    90, 91, 92,# RightHandIndex2
    93, 94, 95,# RightHandIndex3
    # RightHandIndex4 excluded
    99, 100,101,# RightHandThumb1
    102,103,104,# RightHandThumb2
    105,106,107,# RightHandThumb3
    # RightHandThumb4 excluded
    111,112,113,# LeftShoulder
    114,115,116,# LeftArm
    117,118,119,# LeftForeArm
    120,121,122,# LeftHand
    123,124,125,# LeftHandMiddle1
    126,127,128,# LeftHandMiddle2
    129,130,131,# LeftHandMiddle3
    # LeftHandMiddle4 excluded
    138,139,140,# LeftHandRing1  (LeftHandRing root excluded)
    141,142,143,# LeftHandRing2
    144,145,146,# LeftHandRing3
    # LeftHandRing4 excluded
    153,154,155,# LeftHandPinky1 (LeftHandPinky root excluded)
    156,157,158,# LeftHandPinky2
    159,160,161,# LeftHandPinky3
    # LeftHandPinky4 excluded
    165,166,167,# LeftHandIndex
    168,169,170,# LeftHandIndex1
    171,172,173,# LeftHandIndex2
    174,175,176,# LeftHandIndex3
    # LeftHandIndex4 excluded
    180,181,182,# LeftHandThumb1
    183,184,185,# LeftHandThumb2
    186,187,188,# LeftHandThumb3
    # LeftHandThumb4 excluded
    # Legs/feet all excluded
]
assert len(_BVH_COL_141) == 141, f"Expected 141 cols, got {len(_BVH_COL_141)}"
_BVH_COL_141 = np.array(_BVH_COL_141, dtype=np.int32)


def get_bvh_rotation_lines(bvh_path):
    """
    从 BVH 文件提取运动数据部分，裁剪到 141 维（去掉 Hips 平移和下肢/末端/无用指根关节）。
    每行输出 141 个浮点数（以空格分隔）。
    """
    lines_out = []
    in_motion = False
    skipped_header = 0
    with open(bvh_path, "r", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line == "MOTION":
                in_motion = True
                continue
            if in_motion:
                if skipped_header < 2:
                    skipped_header += 1
                    continue
                vals = line.split()
                if len(vals) >= 228:
                    arr = np.array(vals, dtype=np.float32)
                    out = arr[_BVH_COL_141]
                    lines_out.append(" ".join(f"{v:.6f}" for v in out))
                elif len(vals) == 141:
                    # 已经是 141 维，直接保留
                    lines_out.append(line)
    return lines_out



def wav_to_npy(wav_path, target_sr=16000):
    """读取 wav 并重采样为 target_sr，返回 numpy array。"""
    audio, sr = sf.read(wav_path)
    if audio.ndim == 2:
        audio = audio[:, 0]              # 立体声取左声道
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio.astype(np.float32)


def collect_clips(speaker_ids, raw_dir):
    """收集所有说话人的所有 clip 路径（以 .bvh 为基准）。"""
    clips = []
    for sid in speaker_ids:
        spk_dir = os.path.join(raw_dir, sid)
        bvh_files = sorted(glob.glob(os.path.join(spk_dir, "*.bvh")))
        for bvh in bvh_files:
            clips.append(bvh)
    return clips


def split_clips(clips):
    """将 clips 按 train/val/test 比例划分（按说话人内部顺序）。"""
    # 按说话人分组，每个说话人内部分别 split，保证每个人都有验证和测试集
    from collections import defaultdict
    spk_clips = defaultdict(list)
    for c in clips:
        sid = Path(c).parent.name   # "1", "2", "3", "4"
        spk_clips[sid].append(c)

    train, val, test = [], [], []
    for sid, cs in spk_clips.items():
        n = len(cs)
        n_train = int(n * TRAIN_RATIO)
        n_val   = int(n * VAL_RATIO)
        train.extend(cs[:n_train])
        val.extend(cs[n_train:n_train + n_val])
        test.extend(cs[n_train + n_val:])

    return train, val, test


def process_clip(bvh_path, split, out_root):
    """
    处理单个 clip，写入 bvh_rot / wave16k / facial52 / sem 目录。
    返回 True 表示成功，False 表示有文件缺失跳过。
    """
    stem   = Path(bvh_path).stem    # e.g. 1_wayne_0_1_1
    # out dirs
    bvh_dir  = os.path.join(out_root, split, "bvh_rot")
    wav_dir  = os.path.join(out_root, split, "wave16k")
    face_dir = os.path.join(out_root, split, "facial52")
    sem_dir  = os.path.join(out_root, split, "sem")
    for d in [bvh_dir, wav_dir, face_dir, sem_dir]:
        os.makedirs(d, exist_ok=True)

    base_dir = str(Path(bvh_path).parent)

    # --- BVH: 提取旋转行 ---
    out_bvh = os.path.join(bvh_dir, stem + ".bvh")
    if not os.path.exists(out_bvh):
        rot_lines = get_bvh_rotation_lines(bvh_path)
        if not rot_lines:
            print(f"  WARN: no motion lines in {bvh_path}, skip")
            return False
        with open(out_bvh, "w") as f:
            f.write("\n".join(rot_lines) + "\n")

    # --- WAV → NPY ---
    wav_src = os.path.join(base_dir, stem + ".wav")
    out_npy = os.path.join(wav_dir, stem + ".npy")
    if not os.path.exists(wav_src):
        print(f"  WARN: wav not found {wav_src}, skip")
        return False
    if not os.path.exists(out_npy):
        audio = wav_to_npy(wav_src, TARGET_AUDIO_SR)
        np.save(out_npy, audio)

    # --- JSON (facial) ---
    json_src = os.path.join(base_dir, stem + ".json")
    out_json = os.path.join(face_dir, stem + ".json")
    if not os.path.exists(json_src):
        print(f"  WARN: json not found {json_src}, skip")
        return False
    if not os.path.exists(out_json):
        shutil.copy2(json_src, out_json)

    # --- TXT (semantic) ---
    txt_src = os.path.join(base_dir, stem + ".txt")
    out_txt = os.path.join(sem_dir, stem + ".txt")
    if os.path.exists(txt_src) and not os.path.exists(out_txt):
        shutil.copy2(txt_src, out_txt)

    return True


def welford_update(count, mean, M2, new_rows):
    """
    Welford 在线算法：逐行更新均值和 M2（方差的未归一化形式）。
    new_rows: np.ndarray [n, D]
    """
    for row in new_rows:
        count += 1
        delta  = row - mean
        mean  += delta / count
        delta2 = row - mean
        M2    += delta * delta2
    return count, mean, M2


def welford_finalize(count, mean, M2):
    """返回 mean 和 std（样本标准差）。"""
    std = np.sqrt(M2 / max(count - 1, 1))
    std[std < 1e-5] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def compute_stats(split_dir, out_dir):
    """
    用 Welford 在线算法分批计算 pose / facial / axis-angle 的 mean & std。
    每次只把一个文件加载到内存，内存占用从 O(N×D) 降到 O(D)。
    """
    print(f"\n计算 {split_dir} 的统计量（分批 Welford 算法）...")

    # ---- Pose (BVH euler angles) ----
    bvh_files = sorted(glob.glob(os.path.join(split_dir, "bvh_rot", "*.bvh")))
    D_pose = None
    count_p, mean_p, M2_p = 0, None, None
    count_aa, mean_aa, M2_aa = 0, None, None

    use_aa = True
    try:
        import torch
        import datasets.rotation_converter as rot_cvt
    except Exception as e:
        print(f"  WARN: 无法导入 torch/rot_cvt ({e})，跳过 axis-angle 统计。")
        use_aa = False

    for bf in tqdm(bvh_files, desc="pose stats"):
        rows = []
        with open(bf, "r") as f:
            for line in f:
                vals = line.strip().split()
                if vals:
                    try:
                        rows.append([float(v) for v in vals])
                    except ValueError:
                        pass
        if not rows:
            continue
        arr = np.array(rows, dtype=np.float64)   # [T, D]
        D_pose = arr.shape[1]
        if mean_p is None:
            mean_p = np.zeros(D_pose, dtype=np.float64)
            M2_p   = np.zeros(D_pose, dtype=np.float64)
        count_p, mean_p, M2_p = welford_update(count_p, mean_p, M2_p, arr)

        # axis-angle: 逐文件计算，避免大矩阵堆积
        if use_aa:
            if mean_aa is None:
                mean_aa = np.zeros(D_pose, dtype=np.float64)
                M2_aa   = np.zeros(D_pose, dtype=np.float64)
            try:
                t = torch.from_numpy(arr).float() * (np.pi / 180.0)  # [T, D]
                n_joints = D_pose // 3
                aa = rot_cvt.euler_angles_to_axis_angle(
                    t.reshape(1, -1, n_joints, 3), "XYZ"
                ).reshape(-1, D_pose).numpy().astype(np.float64)
                count_aa, mean_aa, M2_aa = welford_update(count_aa, mean_aa, M2_aa, aa)
            except Exception as e:
                print(f"  WARN: axis-angle 转换失败 ({e})，跳过该文件。")

    if mean_p is None:
        print("  ERROR: 未找到任何 BVH 文件，跳过 pose 统计。")
        return

    bvh_mean, bvh_std = welford_finalize(count_p, mean_p, M2_p)
    bvh_stat_dir = os.path.join(split_dir, "bvh_rot")
    np.save(os.path.join(bvh_stat_dir, "bvh_mean.npy"), bvh_mean)
    np.save(os.path.join(bvh_stat_dir, "bvh_std.npy"),  bvh_std)
    print(f"  Pose frames: {count_p}, dim: {D_pose}  → {bvh_stat_dir}")

    # ---- Facial (JSON weights) ----
    json_files = sorted(glob.glob(os.path.join(split_dir, "facial52", "*.json")))
    D_face = None
    count_f, mean_f, M2_f = 0, None, None

    for jf in tqdm(json_files, desc="facial stats"):
        with open(jf, "r") as f:
            data = json.load(f)
        rows = []
        for frame in data.get("frames", []):
            w = frame.get("weights", [])
            if w:
                rows.append(w)
        if not rows:
            continue
        arr = np.array(rows, dtype=np.float64)
        D_face = arr.shape[1]
        if mean_f is None:
            mean_f = np.zeros(D_face, dtype=np.float64)
            M2_f   = np.zeros(D_face, dtype=np.float64)
        count_f, mean_f, M2_f = welford_update(count_f, mean_f, M2_f, arr)

    if mean_f is None:
        print("  ERROR: 未找到任何 JSON 文件，跳过 facial 统计。")
    else:
        face_mean, face_std = welford_finalize(count_f, mean_f, M2_f)
        face_stat_dir = os.path.join(split_dir, "facial52")
        np.save(os.path.join(face_stat_dir, "json_mean.npy"), face_mean)
        np.save(os.path.join(face_stat_dir, "json_std.npy"),  face_std)
        print(f"  Facial frames: {count_f}, dim: {D_face}  → {face_stat_dir}")

    # ---- Axis-angle stats ----
    if use_aa and mean_aa is not None:
        aa_mean, aa_std = welford_finalize(count_aa, mean_aa, M2_aa)
        np.save(os.path.join(split_dir, "axis_angle_mean.npy"), aa_mean)
        np.save(os.path.join(split_dir, "axis_angle_std.npy"),  aa_std)
        print(f"  Axis-angle frames: {count_aa}  → {split_dir}")
    else:
        print("  跳过 axis-angle 统计（需要 torch + datasets.rotation_converter）")



def copy_stats_to_val_test(train_dir, val_dir, test_dir):
    """val/test 复用 train 的统计量（DataLoader 会找这些文件）。"""
    files = [
        "bvh_rot/bvh_mean.npy", "bvh_rot/bvh_std.npy",
        "facial52/json_mean.npy", "facial52/json_std.npy",
        "axis_angle_mean.npy", "axis_angle_std.npy",
    ]
    for target_dir in [val_dir, test_dir]:
        for rel in files:
            src = os.path.join(train_dir, rel)
            dst = os.path.join(target_dir, rel)
            if os.path.exists(src) and not os.path.exists(dst):
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst)
                print(f"  Copied {rel} → {target_dir}")


def main():
    print("=== BEAT 预处理脚本 ===")
    print(f"原始数据: {RAW_DIR}")
    print(f"输出目录: {OUT_ROOT}\n")

    # 1. 收集所有 clip
    all_clips = collect_clips(SPEAKER_IDS, RAW_DIR)
    print(f"共找到 {len(all_clips)} 个 clip")

    # 2. split
    train_clips, val_clips, test_clips = split_clips(all_clips)
    print(f"Train: {len(train_clips)}, Val: {len(val_clips)}, Test: {len(test_clips)}\n")

    # 3. 处理各 split
    for split, clips in [("train", train_clips), ("val", val_clips), ("test", test_clips)]:
        print(f"--- 处理 {split} ---")
        ok = 0
        for bf in tqdm(clips, desc=split):
            if process_clip(bf, split, OUT_ROOT):
                ok += 1
        print(f"  成功处理: {ok}/{len(clips)}")

    # 4. 计算 train 统计量
    train_dir = os.path.join(OUT_ROOT, "train")
    compute_stats(train_dir, train_dir)

    # 5. 将统计量 copy 到 val / test
    val_dir  = os.path.join(OUT_ROOT, "val")
    test_dir = os.path.join(OUT_ROOT, "test")
    copy_stats_to_val_test(train_dir, val_dir, test_dir)

    print("\n=== 预处理完成！===")
    print("下一步：提取 HuBERT 特征（见数据准备指南 Step 5）")
    print("然后运行训练：")
    print("  C:\\Users\\yangz\\miniconda3\\python.exe runner.py --dataset_name beat --name beat_FM_v1 --mode train --flow_matching --fm_sample_steps 50 --n_poses 34 --batch_size 32 --no_fgd --gpu_id 0")


if __name__ == "__main__":
    main()
