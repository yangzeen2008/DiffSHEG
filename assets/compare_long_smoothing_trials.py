"""Compare rotation-aware smoothing strengths on the blend-10 long outputs."""

from __future__ import annotations

import json
import os
import sys

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from assets.prepare_long_validation_visuals import (  # noqa: E402
    GT_ROOT,
    SELECTION_PATH,
    STATS,
    load_numbered,
    motion_spread,
    stitch_windows,
    temporal_metrics,
)
from smooth_adaptive import smooth_euler_rotation  # noqa: E402


PRED_ROOT = os.path.join(
    ROOT, "results", "beat_34", "test_on_val", "beat_FM_aa_x0_aligned_v1",
    "fixStart24", "BestPCK_e479_sequential_overlap24_long3_blend10",
)
OUTPUT = os.path.join(
    ROOT, "bvh_output", "long_validation_trials", "smoothing_comparison.json"
)
TRIALS = (
    (0.0, 0.0),
    (0.40, 0.60),
    (0.55, 0.80),
    (0.70, 1.00),
    (0.85, 1.20),
    (1.00, 1.40),
)


def smooth_motion(values, spine_sigma, arm_sigma):
    output = values.copy()
    for joint_index in range(values.shape[1] // 3):
        sigma = spine_sigma if joint_index < 3 else arm_sigma
        if sigma > 0:
            start = joint_index * 3
            output[:, start:start + 3] = smooth_euler_rotation(
                values[:, start:start + 3], "XYZ", sigma, mode="quaternion"
            )
    return output


def main():
    with open(SELECTION_PATH, "r", encoding="utf-8") as handle:
        selection = json.load(handle)
    pred_windows = load_numbered(os.path.join(PRED_ROOT, "gesture"))
    gt_windows = load_numbered(os.path.join(GT_ROOT, "gesture"))
    mean = np.load(os.path.join(STATS, "bvh_mean.npy"))
    std = np.load(os.path.join(STATS, "bvh_std.npy"))
    reports = []
    offset = 0
    for segment in selection["segments"]:
        count = segment["count"]
        selected = slice(offset, offset + count)
        pred, _ = stitch_windows(pred_windows[selected])
        gt, _ = stitch_windows(gt_windows[selected])
        pred_degrees = pred * std + mean
        gt_degrees = gt * std + mean
        gt_metrics = temporal_metrics(gt_degrees)
        gt_spread = motion_spread(gt_degrees)
        trials = []
        for spine_sigma, arm_sigma in TRIALS:
            smoothed = smooth_motion(pred_degrees, spine_sigma, arm_sigma)
            metrics = temporal_metrics(smoothed)
            spread = motion_spread(smoothed)
            trials.append({
                "spine_sigma": spine_sigma,
                "arm_sigma": arm_sigma,
                "velocity_ratio": metrics["velocity_rms"] / gt_metrics["velocity_rms"],
                "acceleration_ratio": metrics["acceleration_rms"] / gt_metrics["acceleration_rms"],
                "jerk_ratio": metrics["jerk_rms"] / gt_metrics["jerk_rms"],
                "motion_spread_ratio": spread / gt_spread,
            })
        reports.append({"label": segment["label"], "trials": trials})
        offset += count
    payload = {"prediction": "blend10", "segments": reports}
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
