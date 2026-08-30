"""Plot per-frame left-forearm angular velocity for blend ablations."""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from assets.analyze_forearm_rotations import load_joint_rotations  # noqa: E402


TRIAL_ROOT = os.path.join(ROOT, "bvh_output", "long_validation_trials")
OUTPUT = os.path.join(
    TRIAL_ROOT, "transition_blend_ablation", "forearm_velocity_comparison.png"
)
SERIES = (
    ("GT", "so3_release", "ground_truth.bvh", "#222222"),
    ("Blend 0", "so3_release_blend0", "model_pck_best_e479.bvh", "#d62728"),
    ("Blend 3", "so3_release_blend3", "model_pck_best_e479.bvh", "#ff7f0e"),
    ("Blend 5", "so3_release_blend5", "model_pck_best_e479.bvh", "#1f77b4"),
    ("Blend 7", "so3_release", "model_pck_best_e479.bvh", "#2ca02c"),
)


def angular_velocity(path: str) -> np.ndarray:
    rotations = load_joint_rotations(path)["LeftForeArm"]["rotation"]
    matrices = rotations.as_matrix()
    relative = np.matmul(np.swapaxes(matrices[:-1], -1, -2), matrices[1:])
    return np.rad2deg(Rotation.from_matrix(relative).magnitude())


fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True, constrained_layout=True)
for axis, segment in zip(axes, ("typical", "energetic")):
    for label, folder, filename, color in SERIES:
        values = angular_velocity(os.path.join(TRIAL_ROOT, folder, segment, filename))
        frames = np.arange(1, len(values) + 1)
        axis.plot(frames, values, label=label, color=color, linewidth=1.8)
        if label in {"Blend 0", "Blend 5"}:
            maximum = int(np.argmax(values))
            axis.scatter(frames[maximum], values[maximum], color=color, s=35, zorder=5)
            axis.annotate(
                f"{label}: {values[maximum]:.1f}°/frame",
                (frames[maximum], values[maximum]),
                xytext=(6, 8), textcoords="offset points", fontsize=9, color=color,
            )
    for boundary in range(34, 224, 10):
        axis.axvline(boundary, color="#999999", linewidth=0.6, alpha=0.25)
    axis.set_title(f"{segment.capitalize()} — LeftForeArm SO(3) angular velocity")
    axis.set_ylabel("degrees / frame")
    axis.grid(axis="y", alpha=0.2)
axes[-1].set_xlabel("BVH frame (vertical lines are window boundaries)")
axes[0].legend(ncol=5, loc="upper right")
fig.savefig(OUTPUT, dpi=180)
print(OUTPUT)
