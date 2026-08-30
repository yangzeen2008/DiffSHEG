"""Prepare a continuous GT vs pck_best comparison from eight overlapping windows."""

import json
import os
import sys

import numpy as np
import soundfile as sf
from scipy.spatial.transform import Rotation

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from datasets.data_tools import joints_list


EXPERIMENT = "beat_FM_aa_x0_aligned_v1"
PRED_ROOT = os.path.join(
    ROOT,
    "results",
    "beat_34",
    "test_on_val",
    EXPERIMENT,
    "fixStart24",
    "BestPCK_e479_sequential_overlap24",
)
GT_ROOT = os.path.join(
    ROOT,
    "results",
    "beat_34",
    "test_on_val_GT",
    EXPERIMENT,
    "fixStart0",
    "BestPCK_e479",
)
TEMPLATE = os.path.join(
    ROOT,
    "data",
    "BEAT",
    "raw",
    "beat_english_v0.2.1",
    "beat_english_v0.2.1",
    "1",
    "1_wayne_0_100_100.bvh",
)
OUTPUT = os.path.join(ROOT, "bvh_output", "aligned_v1_visual")
STATS = os.path.join(OUTPUT, "stats")
FPS = 15
WINDOW_STRIDE_FRAMES = 10
TRANSITION_BLEND_FRAMES = 7


def load_numbered(directory):
    files = sorted(name for name in os.listdir(directory) if name.endswith(".npy"))
    if not files:
        raise RuntimeError(f"No .npy samples found in {directory}")
    return np.stack([np.load(os.path.join(directory, name)) for name in files])


def rms(values):
    return float(np.sqrt(np.mean(np.square(values), dtype=np.float64)))


def temporal_metrics(euler_degrees):
    """Measure derivatives in SO(3), avoiding equivalent Euler branches."""
    if euler_degrees.shape[-1] % 3:
        raise ValueError("Euler motion width must be divisible by three")
    frame_count = euler_degrees.shape[0]
    joint_count = euler_degrees.shape[-1] // 3
    rotations = Rotation.from_euler(
        "XYZ", euler_degrees.reshape(-1, 3), degrees=True
    ).as_matrix().reshape(frame_count, joint_count, 3, 3)
    relative = np.matmul(
        np.swapaxes(rotations[:-1], -1, -2), rotations[1:]
    )
    velocity = np.rad2deg(
        Rotation.from_matrix(relative.reshape(-1, 3, 3)).as_rotvec()
    ).reshape(frame_count - 1, joint_count, 3)
    acceleration = np.diff(velocity, axis=0)
    jerk = np.diff(acceleration, axis=0)
    return {
        "velocity_rms": rms(velocity),
        "acceleration_rms": rms(acceleration),
        "jerk_rms": rms(jerk),
    }


def motion_spread(euler_degrees):
    """Return mean P90 geodesic excursion from each joint's mean rotation."""
    joint_count = euler_degrees.shape[-1] // 3
    excursions = []
    for joint_index in range(joint_count):
        start = joint_index * 3
        rotations = Rotation.from_euler(
            "XYZ", euler_degrees[:, start:start + 3], degrees=True
        )
        center = rotations.mean()
        distances = np.rad2deg((center.inv() * rotations).magnitude())
        excursions.append(np.percentile(distances, 90))
    return float(np.mean(excursions))


def overlap_add(windows, stride):
    """Blend overlapping model windows on their actual global frame positions."""
    count, window_length, channels = windows.shape
    total_length = window_length + (count - 1) * stride
    values = np.zeros((total_length, channels), dtype=np.float64)
    weights = np.zeros((total_length, 1), dtype=np.float64)
    taper = np.sin(np.pi * (np.arange(window_length) + 0.5) / window_length) ** 2
    for index, window in enumerate(windows):
        start = index * stride
        end = start + window_length
        values[start:end] += window * taper[:, None]
        weights[start:end] += taper[:, None]
    if np.any(weights <= 0):
        raise RuntimeError("Overlap-add produced an uncovered frame")
    return (values / weights).astype(windows.dtype)


def read_template():
    with open(TEMPLATE, "r", encoding="utf-8") as handle:
        lines = handle.readlines()
    motion_index = next(i for i, line in enumerate(lines) if line.strip() == "MOTION")
    data_start = motion_index + 3
    base_frame = np.fromstring(lines[data_start], dtype=float, sep=" ")
    if base_frame.size < 6:
        raise RuntimeError("Template BVH does not contain a valid motion frame")
    return lines[: motion_index + 1], base_frame


def write_bvh(path, normalized_euler, mean, std):
    hierarchy, base_frame = read_template()
    euler_degrees = normalized_euler * std + mean
    original_joints = joints_list["beat_joints"]
    target_joints = joints_list["spine_neck_141"]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.writelines(hierarchy)
        handle.write(f"Frames: {len(euler_degrees)}\n")
        handle.write("Frame Time: 0.06666667\n")
        for frame in euler_degrees:
            channels = base_frame.copy()
            for index, (joint_name, width) in enumerate(target_joints.items()):
                source_end = original_joints[joint_name][1]
                channels[source_end - width : source_end] = frame[index * 3 : index * 3 + 3]
            handle.write(" ".join(f"{value:.6f}" for value in channels) + "\n")
    return euler_degrees


def stitch_audio():
    chunks = []
    sample_rate = None
    audio_dir = os.path.join(PRED_ROOT, "audio")
    for name in sorted(item for item in os.listdir(audio_dir) if item.endswith(".wav")):
        audio, rate = sf.read(os.path.join(audio_dir, name), dtype="float32")
        if sample_rate is None:
            sample_rate = rate
        elif rate != sample_rate:
            raise RuntimeError(f"Audio sample-rate mismatch: {rate} != {sample_rate}")
        chunks.append(audio)
    stitched = [chunks[0]]
    for index, chunk in enumerate(chunks[1:], start=1):
        previous_start = ((index - 1) * WINDOW_STRIDE_FRAMES * sample_rate) // FPS
        current_start = (index * WINDOW_STRIDE_FRAMES * sample_rate) // FPS
        step_samples = current_start - previous_start
        overlap_samples = len(chunk) - step_samples
        if overlap_samples <= 0:
            raise RuntimeError("Audio windows do not overlap as expected")
        previous_overlap = chunks[index - 1][-overlap_samples:]
        current_overlap = chunk[:overlap_samples]
        if not np.array_equal(previous_overlap, current_overlap):
            raise RuntimeError(f"Audio overlap mismatch between windows {index - 1} and {index}")
        stitched.append(chunk[-step_samples:])
    output_path = os.path.join(OUTPUT, "comparison_audio_sequential_stitched.wav")
    stitched_audio = np.concatenate(stitched)
    sf.write(output_path, stitched_audio, sample_rate)
    return output_path, sample_rate, len(stitched_audio)


def main():
    os.makedirs(OUTPUT, exist_ok=True)
    mean = np.load(os.path.join(STATS, "bvh_mean.npy"))
    std = np.load(os.path.join(STATS, "bvh_std.npy"))
    pred = load_numbered(os.path.join(PRED_ROOT, "gesture"))
    ground_truth = load_numbered(os.path.join(GT_ROOT, "gesture"))
    if pred.shape != ground_truth.shape:
        raise RuntimeError(f"Prediction/GT shape mismatch: {pred.shape} != {ground_truth.shape}")
    if pred.shape[-1] != len(mean):
        raise RuntimeError(f"Gesture/stat shape mismatch: {pred.shape[-1]} != {len(mean)}")
    if not np.isfinite(pred).all() or not np.isfinite(ground_truth).all():
        raise RuntimeError("Prediction or GT contains NaN/Inf")

    pred_joined = overlap_add(pred, WINDOW_STRIDE_FRAMES)
    gt_joined = overlap_add(ground_truth, WINDOW_STRIDE_FRAMES)
    pred_degrees = write_bvh(
        os.path.join(OUTPUT, "pck_best_e479_sequential_stitched.bvh"), pred_joined, mean, std
    )
    gt_degrees = write_bvh(
        os.path.join(OUTPUT, "ground_truth_8windows_stitched.bvh"), gt_joined, mean, std
    )
    audio_path, sample_rate, audio_samples = stitch_audio()

    pred_metrics = temporal_metrics(pred_degrees)
    gt_metrics = temporal_metrics(gt_degrees)
    report = {
        "experiment": EXPERIMENT,
        "checkpoint": "pck_best.tar",
        "checkpoint_epoch": 479,
        "inference_mode": "sequential_test_windows",
        "transition_blend_frames": TRANSITION_BLEND_FRAMES,
        "window_count": int(pred.shape[0]),
        "frames_per_sample": int(pred.shape[1]),
        "window_stride_frames": WINDOW_STRIDE_FRAMES,
        "overlap_frames": int(pred.shape[1] - WINDOW_STRIDE_FRAMES),
        "total_frames": int(pred_joined.shape[0]),
        "fps": FPS,
        "audio_sample_rate": int(sample_rate),
        "audio_samples": int(audio_samples),
        "audio_path": audio_path,
        "prediction": pred_metrics,
        "ground_truth": gt_metrics,
        "ratios": {
            key.replace("_rms", ""): pred_metrics[key] / gt_metrics[key]
            for key in pred_metrics
        },
        "motion_spread_degrees": {
            "prediction": motion_spread(pred_degrees),
            "ground_truth": motion_spread(gt_degrees),
            "ratio": motion_spread(pred_degrees) / motion_spread(gt_degrees),
        },
    }
    report_path = os.path.join(OUTPUT, "comparison_metrics.json")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
