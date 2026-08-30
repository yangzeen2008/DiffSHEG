"""Audit physically meaningful per-joint rotations in long validation BVHs."""

from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from smooth_adaptive import parse_bvh_header  # noqa: E402


VISUAL_ROOT = os.environ.get(
    "DIFFSHEG_LONG_OUTPUT_ROOT",
    os.path.join(ROOT, "bvh_output", "long_validation_visuals"),
)
SEGMENTS = ("quiet", "typical", "energetic")
TRANSITION_BLEND = int(os.environ.get("DIFFSHEG_TRANSITION_BLEND", "7"))
BOUNDARY_WINDOW = 7
JOINTS = (
    "RightShoulder", "RightArm", "RightForeArm", "RightHand",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
)


def load_joint_rotations(path):
    with open(path, "r", encoding="utf-8") as handle:
        lines = handle.readlines()
    _, joints, data_start = parse_bvh_header(lines)
    motion = np.stack([
        np.fromstring(line, dtype=np.float64, sep=" ")
        for line in lines[data_start:] if line.strip()
    ])
    result = {}
    for joint in joints:
        channels = joint["channels"]
        offsets = [
            index for index, channel in enumerate(channels)
            if channel.lower().endswith("rotation")
        ]
        if len(offsets) != 3:
            continue
        indices = [joint["start_idx"] + offset for offset in offsets]
        order = "".join(channels[offset][0].upper() for offset in offsets)
        result[joint["name"]] = {
            "rotation": Rotation.from_euler(order, motion[:, indices], degrees=True),
            "euler": motion[:, indices],
            "order": order,
        }
    return result


def summarize(entry):
    rotations = entry["rotation"]
    matrices = rotations.as_matrix()
    relative = np.matmul(np.swapaxes(matrices[:-1], -1, -2), matrices[1:])
    velocity = np.rad2deg(Rotation.from_matrix(relative).magnitude())
    center = rotations.mean()
    excursion = np.rad2deg((center.inv() * rotations).magnitude())
    euler_step = np.diff(entry["euler"], axis=0)
    euler_step = (euler_step + 180.0) % 360.0 - 180.0
    top_indices = np.argsort(velocity)[-12:][::-1]
    top_transitions = []
    for index in top_indices:
        to_frame = int(index + 1)
        block_offset = None if to_frame < 34 else int((to_frame - 34) % 10)
        top_transitions.append({
            "from_frame": int(index),
            "to_frame": to_frame,
            "velocity_deg": float(velocity[index]),
            "new_block_offset": block_offset,
            "inside_release_blend": bool(
                block_offset is not None and block_offset < TRANSITION_BLEND
            ),
        })
    release_blocks = []
    for start in range(34, len(rotations) - 6, 10):
        endpoint_distance = float(np.rad2deg(
            (rotations[start].inv() * rotations[start + 6]).magnitude()
        ))
        path_distance = float(np.sum(velocity[start:start + 6]))
        release_blocks.append({
            "start_frame": start,
            "endpoint_distance_deg": endpoint_distance,
            "path_distance_deg": path_distance,
            "path_to_endpoint_ratio": (
                path_distance / endpoint_distance
                if endpoint_distance > 1e-6 else None
            ),
            "peak_velocity_deg": float(np.max(velocity[start:start + 6])),
        })
    release_blocks.sort(key=lambda value: value["path_distance_deg"], reverse=True)
    boundary_blocks = []
    for start in range(34, len(rotations), 10):
        end = min(start + BOUNDARY_WINDOW - 1, len(rotations) - 1)
        transition_slice = velocity[start - 1:end]
        if transition_slice.size == 0:
            continue
        endpoint_distance = float(np.rad2deg(
            (rotations[start - 1].inv() * rotations[end]).magnitude()
        ))
        path_distance = float(np.sum(transition_slice))
        boundary_blocks.append({
            "start_frame": start,
            "end_frame": end,
            "entry_jump_deg": float(velocity[start - 1]),
            "path_distance_deg": path_distance,
            "endpoint_distance_deg": endpoint_distance,
            "path_to_endpoint_ratio": (
                path_distance / endpoint_distance
                if endpoint_distance > 1e-6 else None
            ),
            "peak_velocity_deg": float(np.max(transition_slice)),
        })
    boundary_blocks.sort(key=lambda value: value["peak_velocity_deg"], reverse=True)
    return {
        "order": entry["order"],
        "excursion_p90_deg": float(np.percentile(excursion, 90)),
        "excursion_max_deg": float(np.max(excursion)),
        "velocity_rms_deg_per_frame": float(np.sqrt(np.mean(velocity ** 2))),
        "velocity_p95_deg_per_frame": float(np.percentile(velocity, 95)),
        "velocity_max_deg_per_frame": float(np.max(velocity)),
        "euler_axis_step_p95_deg": [
            float(value) for value in np.percentile(np.abs(euler_step), 95, axis=0)
        ],
        "top_velocity_transitions": top_transitions,
        "release_blend_blocks_by_path": release_blocks,
        "boundary_blocks_by_peak": boundary_blocks,
    }


def main():
    report = {
        "transition_blend": TRANSITION_BLEND,
        "boundary_window": BOUNDARY_WINDOW,
        "segments": {},
    }
    for segment in SEGMENTS:
        paths = {
            "prediction": os.path.join(
                VISUAL_ROOT, segment, "model_pck_best_e479.bvh"
            ),
            "ground_truth": os.path.join(VISUAL_ROOT, segment, "ground_truth.bvh"),
        }
        segment_report = {}
        loaded = {key: load_joint_rotations(path) for key, path in paths.items()}
        for joint_name in JOINTS:
            if any(joint_name not in value for value in loaded.values()):
                raise RuntimeError(f"Joint {joint_name} is missing from {segment}")
            prediction = summarize(loaded["prediction"][joint_name])
            ground_truth = summarize(loaded["ground_truth"][joint_name])
            segment_report[joint_name] = {
                "prediction": prediction,
                "ground_truth": ground_truth,
                "ratios": {
                    "excursion_p90": (
                        prediction["excursion_p90_deg"]
                        / ground_truth["excursion_p90_deg"]
                    ),
                    "velocity_rms": (
                        prediction["velocity_rms_deg_per_frame"]
                        / ground_truth["velocity_rms_deg_per_frame"]
                    ),
                    "velocity_p95": (
                        prediction["velocity_p95_deg_per_frame"]
                        / ground_truth["velocity_p95_deg_per_frame"]
                    ),
                },
            }
        report["segments"][segment] = segment_report
    output = os.path.join(VISUAL_ROOT, "forearm_rotation_audit.json")
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
