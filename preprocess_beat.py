"""
BEAT 数据集预处理脚本（一键运行）
将下载好的原始数据整理成 DiffSHEG 训练所需的目录结构。

执行步骤：
  1. 按 train/val/test split 分配文件
  2. 解析 BVH，提取旋转数值行写成新格式
  3. WAV → 16kHz numpy (.npy)
   4. BVH 120 FPS、表情 60 FPS 按真实时间戳统一到 15 FPS
  5. 计算 mean/std 统计量

用法：
    C:\\Users\\yangz\\miniconda3\\python.exe preprocess_beat.py
"""

import argparse
import glob
import hashlib
import json
import math
import os
import shutil
import sys
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

# ===== 配置 =====
RAW_DIR   = "./data/BEAT/raw/beat_english_v0.2.1/beat_english_v0.2.1"
OUT_ROOT  = "./data/BEAT/beat_cache/beat_4english_15_141_sync_v1"

# 4 位英文说话人的文件夹 ID（对应 BEAT 数据集中的编号）
SPEAKER_IDS = ["1", "2", "3", "4"]

# train/val/test split（按文件名末尾的序号范围划分）
# BEAT 各说话人约有 0_x_x（短片段）和 1_x_x（长片段）两类
# 这里简单按比例划分：80% train / 10% val / 10% test
TRAIN_RATIO = 0.8
VAL_RATIO   = 0.1
# TEST_RATIO = 0.1（剩余）

TARGET_AUDIO_SR = 16000   # 目标采样率
TARGET_POSE_FPS = 15
TEMPORAL_ALIGNMENT_VERSION = 1
TEMPORAL_MANIFEST_NAME = "temporal_alignment_manifest.json"
# ====================


class CrossModalDurationMismatch(RuntimeError):
    pass


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


def _read_bvh_motion(bvh_path):
    """Read selected BVH rotations and the source sampling rate."""
    motion = []
    in_motion = False
    skipped_header = 0
    source_frame_time = None
    with open(bvh_path, "r", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line == "MOTION":
                in_motion = True
                continue
            if not in_motion:
                continue
            if skipped_header < 2:
                skipped_header += 1
                if line.lower().startswith("frame time:"):
                    source_frame_time = float(line.split(":", 1)[1].strip())
                continue
            values = np.fromstring(line, dtype=np.float64, sep=" ")
            if values.size >= 228:
                motion.append(values[_BVH_COL_141])
            elif values.size == 141:
                motion.append(values)
    if source_frame_time is None or not math.isfinite(source_frame_time) or source_frame_time <= 0:
        raise RuntimeError(f"Invalid or missing BVH Frame Time in {bvh_path}")
    if not motion:
        raise RuntimeError(f"No motion frames in {bvh_path}")
    source_fps = 1.0 / source_frame_time
    nearest_integer = round(source_fps)
    if nearest_integer > 0 and abs(source_fps - nearest_integer) / nearest_integer < 1e-3:
        source_fps = float(nearest_integer)
    return np.asarray(motion, dtype=np.float64), source_fps


def _target_frame_count(duration_seconds, target_fps):
    if not math.isfinite(duration_seconds) or duration_seconds <= 0:
        raise ValueError(f"Invalid duration: {duration_seconds}")
    return int(math.floor(duration_seconds * target_fps + 1e-6))


def _target_times(frame_count, target_fps):
    return np.arange(frame_count, dtype=np.float64) / float(target_fps)


def resample_motion_values(motion, source_fps, target_fps, target_frame_count):
    times = _target_times(target_frame_count, target_fps)
    indices = np.rint(times * source_fps).astype(np.int64)
    if indices[-1] >= motion.shape[0]:
        raise RuntimeError(
            "Target BVH time exceeds source: "
            f"index={indices[-1]}, frames={motion.shape[0]}"
        )
    return motion[indices]


def resample_bvh_motion(bvh_path, target_fps, target_frame_count):
    """Decimate BVH rotations on the common target timeline."""
    motion, source_fps = _read_bvh_motion(bvh_path)
    source_duration = motion.shape[0] / source_fps
    return resample_motion_values(motion, source_fps, target_fps, target_frame_count), {
        "source_fps": source_fps,
        "source_frames": int(motion.shape[0]),
        "source_duration_seconds": source_duration,
    }


def read_facial_json(json_path):
    with open(json_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    frames = payload.get("frames", [])
    if not frames:
        raise RuntimeError(f"No facial frames in {json_path}")
    weights = np.asarray([frame.get("weights", []) for frame in frames], dtype=np.float64)
    if weights.ndim != 2 or weights.shape[1] != 51 or not np.isfinite(weights).all():
        raise RuntimeError(f"Invalid facial weights in {json_path}: {weights.shape}")
    times = np.asarray([frame.get("time", np.nan) for frame in frames], dtype=np.float64)
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise RuntimeError(f"Facial timestamps must be finite and strictly increasing: {json_path}")
    steps = np.diff(times)
    source_step = float(np.median(steps))
    if source_step <= 0 or not math.isfinite(source_step):
        raise RuntimeError(f"Invalid facial frame interval in {json_path}")
    source_fps = 1.0 / source_step
    nearest_integer = round(source_fps)
    if nearest_integer > 0 and abs(source_fps - nearest_integer) / nearest_integer < 1e-3:
        source_fps = float(nearest_integer)
        source_step = 1.0 / source_fps
    source_duration = float(times[-1] + source_step)
    return payload, weights, times, {
        "source_fps": source_fps,
        "source_frames": int(weights.shape[0]),
        "source_duration_seconds": source_duration,
        "source_first_timestamp": float(times[0]),
        "source_last_timestamp": float(times[-1]),
    }


def resample_facial_values(payload, weights, source_times, report, target_fps, target_frame_count):
    times = _target_times(target_frame_count, target_fps)
    if times[-1] > source_times[-1] + 1.0 / report["source_fps"]:
        raise RuntimeError("Target facial time exceeds source")
    aligned = np.empty((target_frame_count, weights.shape[1]), dtype=np.float64)
    for coefficient in range(weights.shape[1]):
        aligned[:, coefficient] = np.interp(
            times,
            source_times,
            weights[:, coefficient],
            left=weights[0, coefficient],
            right=weights[-1, coefficient],
        )
    output = {
        "names": payload.get("names", []),
        "frames": [
            {
                "weights": frame.tolist(),
                "time": index / float(target_fps),
                "rotation": [],
            }
            for index, frame in enumerate(aligned)
        ],
    }
    return output


def resample_facial_json(json_path, target_fps, target_frame_count):
    """Linearly resample blendshape coefficients using their real timestamps."""
    payload, weights, source_times, report = read_facial_json(json_path)
    output = resample_facial_values(
        payload, weights, source_times, report, target_fps, target_frame_count
    )
    return output, report


def get_bvh_rotation_lines(bvh_path, target_fps=TARGET_POSE_FPS, target_frame_count=None):
    """
    从 BVH 文件提取运动数据部分，裁剪到 141 维（去掉 Hips 平移和下肢/末端/无用指根关节）。
    每行输出 141 个浮点数（以空格分隔）。
    """
    motion, source_fps = _read_bvh_motion(bvh_path)
    if target_frame_count is None:
        target_frame_count = _target_frame_count(motion.shape[0] / source_fps, target_fps)
    times = _target_times(target_frame_count, target_fps)
    indices = np.rint(times * source_fps).astype(np.int64)
    if indices[-1] >= motion.shape[0]:
        raise RuntimeError(f"Target timeline exceeds BVH source: {bvh_path}")
    return [" ".join(f"{value:.6f}" for value in frame) for frame in motion[indices]]



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


def process_clip(
    bvh_path,
    split,
    out_root,
    *,
    target_fps=TARGET_POSE_FPS,
    force=False,
    fallback_split_root=None,
    allow_motion_tail_trim=False,
):
    """
    处理单个 clip，写入 bvh_rot / wave16k / facial52 / sem 目录。
    返回时间轴审计记录；文件缺失时返回 None。
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

    out_bvh = os.path.join(bvh_dir, stem + ".bvh")
    wav_src = os.path.join(base_dir, stem + ".wav")
    fallback_audio = (
        os.path.join(fallback_split_root, "wave16k", stem + ".npy")
        if fallback_split_root else None
    )
    out_npy = os.path.join(wav_dir, stem + ".npy")

    json_src = os.path.join(base_dir, stem + ".json")
    fallback_facial = (
        os.path.join(fallback_split_root, "facial52", stem + ".json")
        if fallback_split_root else None
    )
    out_json = os.path.join(face_dir, stem + ".json")
    facial_source = json_src if os.path.exists(json_src) else fallback_facial
    if not facial_source or not os.path.exists(facial_source):
        print(f"  WARN: facial JSON not found for {stem}, skip")
        return None

    if os.path.exists(wav_src):
        audio = wav_to_npy(wav_src, TARGET_AUDIO_SR)
        audio_source = wav_src
    elif fallback_audio and os.path.exists(fallback_audio):
        audio = np.load(fallback_audio).astype(np.float32, copy=False)
        if audio.ndim == 2:
            audio = audio[:, 0]
        audio_source = fallback_audio
    else:
        print(f"  WARN: audio not found for {stem}, skip")
        return None
    if audio.ndim != 1 or not np.isfinite(audio).all():
        raise RuntimeError(f"Invalid audio source for {stem}: {audio_source}")
    audio_duration = len(audio) / float(TARGET_AUDIO_SR)
    motion, motion_fps = _read_bvh_motion(bvh_path)
    motion_duration = motion.shape[0] / motion_fps
    facial_payload, facial_weights, facial_times, facial_report = read_facial_json(
        facial_source
    )
    facial_duration = facial_report["source_duration_seconds"]
    durations = {
        "audio": audio_duration,
        "motion": motion_duration,
        "facial": facial_duration,
    }
    duration_delta = max(durations.values()) - min(durations.values())
    tolerance = 2.0 / target_fps
    audio_facial_delta = abs(audio_duration - facial_duration)
    if audio_facial_delta > tolerance:
        raise CrossModalDurationMismatch(
            f"Audio/facial duration mismatch in {stem}: {durations}; "
            f"allowed delta={tolerance:.6f}s"
        )
    if duration_delta > tolerance and not allow_motion_tail_trim:
        raise CrossModalDurationMismatch(
            f"Cross-modal duration mismatch in {stem}: {durations}; "
            f"allowed delta={tolerance:.6f}s"
        )

    aligned_duration = min(durations.values())
    target_frames = _target_frame_count(aligned_duration, target_fps)
    if target_frames < 2:
        raise RuntimeError(f"Aligned clip is too short: {stem}")
    aligned_motion = resample_motion_values(
        motion, motion_fps, target_fps, target_frames
    )
    aligned_facial = resample_facial_values(
        facial_payload,
        facial_weights,
        facial_times,
        facial_report,
        target_fps,
        target_frames,
    )
    if aligned_motion.shape[0] != len(aligned_facial["frames"]):
        raise RuntimeError(f"Aligned motion/facial frame mismatch in {stem}")

    # All three outputs are derived from the same verified timeline. Existing
    # files are retained only when rebuilding the manifest for an interrupted
    # run; --force refreshes them from the raw source.
    if force or not os.path.exists(out_bvh):
        with open(out_bvh, "w", encoding="utf-8") as handle:
            for frame in aligned_motion:
                handle.write(" ".join(f"{value:.6f}" for value in frame) + "\n")
    if force or not os.path.exists(out_npy):
        np.save(out_npy, audio)
    if force or not os.path.exists(out_json):
        with open(out_json, "w", encoding="utf-8") as handle:
            json.dump(aligned_facial, handle, ensure_ascii=False)

    # --- TXT (semantic) ---
    txt_src = os.path.join(base_dir, stem + ".txt")
    fallback_semantic = (
        os.path.join(fallback_split_root, "sem", stem + ".txt")
        if fallback_split_root else None
    )
    semantic_source = txt_src if os.path.exists(txt_src) else fallback_semantic
    out_txt = os.path.join(sem_dir, stem + ".txt")
    if semantic_source and os.path.exists(semantic_source) and (force or not os.path.exists(out_txt)):
        shutil.copy2(semantic_source, out_txt)

    return {
        "clip": stem,
        "target_fps": target_fps,
        "target_frames": target_frames,
        "aligned_duration_seconds": target_frames / float(target_fps),
        "source": {
            "motion_path": str(bvh_path),
            "facial_path": str(facial_source),
            "audio_path": str(audio_source),
            "audio_sample_rate": TARGET_AUDIO_SR,
            "audio_samples": int(len(audio)),
            "audio_duration_seconds": audio_duration,
            "motion_fps": motion_fps,
            "motion_frames": int(motion.shape[0]),
            "motion_duration_seconds": motion_duration,
            "facial_fps": facial_report["source_fps"],
            "facial_frames": facial_report["source_frames"],
            "facial_duration_seconds": facial_duration,
        },
        "maximum_source_duration_delta_seconds": duration_delta,
        "source_tail_trim_seconds": {
            name: duration - aligned_duration for name, duration in durations.items()
        },
    }

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


def write_temporal_manifest(split_dir, split, records, target_fps, skipped=None):
    skipped = list(skipped or [])
    trimmed_clip_count = sum(
        1
        for record in records
        if max(record.get("source_tail_trim_seconds", {}).values(), default=0.0) > 1e-6
    )
    payload = {
        "temporal_alignment_version": TEMPORAL_ALIGNMENT_VERSION,
        "split": split,
        "target_pose_fps": target_fps,
        "target_facial_fps": target_fps,
        "audio_sample_rate": TARGET_AUDIO_SR,
        "clip_count": len(records),
        "trimmed_clip_count": trimmed_clip_count,
        "skipped_clip_count": len(skipped),
        "skipped_clips": skipped,
        "clips": records,
    }
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    payload["manifest_id"] = hashlib.sha256(canonical).hexdigest()
    path = os.path.join(split_dir, TEMPORAL_MANIFEST_NAME)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default=RAW_DIR)
    parser.add_argument("--output-root", default=OUT_ROOT)
    parser.add_argument(
        "--fallback-cache-root",
        help="Optional legacy cache used only as the source of 16 kHz audio and 60 FPS facial JSON",
    )
    parser.add_argument("--target-fps", type=int, default=TARGET_POSE_FPS)
    parser.add_argument("--speakers", nargs="+", default=SPEAKER_IDS)
    parser.add_argument(
        "--split",
        choices=("all", "train_val", "train", "val", "test"),
        default="all",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-stats", action="store_true")
    parser.add_argument(
        "--duration-mismatch-policy",
        choices=("error", "skip", "trim_motion_tail"),
        default="error",
        help="Reject a corrupt clip or explicitly skip it and record the reason",
    )
    parser.add_argument(
        "--max-skip-fraction",
        type=float,
        default=0.01,
        help="Abort if skipped clips exceed this fraction of a split",
    )
    args = parser.parse_args()
    if args.target_fps <= 0:
        raise ValueError("--target-fps must be positive")

    print("=== BEAT 预处理脚本 ===")
    print(f"原始数据: {args.raw_dir}")
    print(f"输出目录: {args.output_root}")
    print(f"统一时间轴: {args.target_fps} FPS\n")

    # 1. 收集所有 clip
    all_clips = collect_clips(args.speakers, args.raw_dir)
    print(f"共找到 {len(all_clips)} 个 clip")

    # 2. split
    train_clips, val_clips, test_clips = split_clips(all_clips)
    print(f"Train: {len(train_clips)}, Val: {len(val_clips)}, Test: {len(test_clips)}\n")

    # 3. 处理各 split
    split_map = {"train": train_clips, "val": val_clips, "test": test_clips}
    if args.split == "all":
        selected_splits = list(split_map)
    elif args.split == "train_val":
        selected_splits = ["train", "val"]
    else:
        selected_splits = [args.split]
    for split in selected_splits:
        clips = split_map[split]
        if args.limit is not None:
            clips = clips[: max(0, args.limit)]
        print(f"--- 处理 {split} ---")
        records = []
        skipped = []
        for bf in tqdm(clips, desc=split):
            try:
                record = process_clip(
                    bf,
                    split,
                    args.output_root,
                    target_fps=args.target_fps,
                    force=args.force,
                    fallback_split_root=(
                        os.path.join(args.fallback_cache_root, split)
                        if args.fallback_cache_root else None
                    ),
                    allow_motion_tail_trim=(
                        args.duration_mismatch_policy == "trim_motion_tail"
                    ),
                )
            except CrossModalDurationMismatch as error:
                if args.duration_mismatch_policy != "skip":
                    raise
                skipped.append({"clip": Path(bf).stem, "reason": str(error)})
                print(f"  WARN: skip corrupt clip {Path(bf).stem}: {error}")
                continue
            if record is not None:
                records.append(record)
            else:
                skipped.append({"clip": Path(bf).stem, "reason": "missing required modality"})
        skip_fraction = len(skipped) / max(len(clips), 1)
        if skip_fraction > args.max_skip_fraction:
            raise RuntimeError(
                f"Skipped {len(skipped)}/{len(clips)} {split} clips "
                f"({skip_fraction:.2%}), above --max-skip-fraction={args.max_skip_fraction:.2%}"
            )
        manifest_path = write_temporal_manifest(
            os.path.join(args.output_root, split),
            split,
            records,
            args.target_fps,
            skipped=skipped,
        )
        print(f"  成功处理: {len(records)}/{len(clips)}")
        if skipped:
            print(f"  显式跳过: {len(skipped)}")
        print(f"  时间轴清单: {manifest_path}")

    # 4. 计算 train 统计量
    train_dir = os.path.join(args.output_root, "train")
    if not args.skip_stats and "train" in selected_splits:
        compute_stats(train_dir, train_dir)

    # 5. 将统计量 copy 到 val / test
    val_dir  = os.path.join(args.output_root, "val")
    test_dir = os.path.join(args.output_root, "test")
    if not args.skip_stats and args.split in ("all", "train_val"):
        copy_stats_to_val_test(train_dir, val_dir, test_dir)

    print("\n=== 预处理完成！===")
    print("下一步：提取 HuBERT 特征（见数据准备指南 Step 5）")
    print("然后运行训练：")
    print(
        "  python runner.py --dataset_name beat "
        f"--beat_cache_name {Path(args.output_root).name} --name beat_FM_v1 "
        "--mode train --flow_matching --fm_sample_steps 50 --n_poses 34 "
        "--batch_size 32 --no_fgd --gpu_id 0"
    )


if __name__ == "__main__":
    main()
