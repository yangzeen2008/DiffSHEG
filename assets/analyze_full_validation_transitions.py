"""Analyze full/preview sequential validation outputs in physical SO(3)."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import numpy as np
import soundfile as sf
from scipy.spatial.transform import Rotation

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.data_tools import joints_list  # noqa: E402


STATS_ROOT = ROOT / "bvh_output" / "aligned_v1_visual" / "stats"
WINDOW_LENGTH = 34
OVERLAP = 24
STRIDE = WINDOW_LENGTH - OVERLAP
POSE_FPS = 15
AUDIO_FPS = 16000
ARM_JOINTS = (
    "RShoulder", "RArm", "RArm1", "RHand",
    "LShoulder", "LArm", "LArm1", "LHand",
)
FACE_COEFFICIENTS = (
    "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft",
    "browOuterUpRight", "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight",
    "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight",
    "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft", "jawOpen",
    "jawRight", "mouthClose", "mouthDimpleLeft", "mouthDimpleRight",
    "mouthFrownLeft", "mouthFrownRight", "mouthFunnel", "mouthLeft",
    "mouthLowerDownLeft", "mouthLowerDownRight", "mouthPressLeft",
    "mouthPressRight", "mouthPucker", "mouthRight", "mouthRollLower",
    "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft",
    "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight",
    "mouthUpperUpLeft", "mouthUpperUpRight", "noseSneerLeft", "noseSneerRight",
)
EXPRESSION_BOUNDARY_FLOOR = 0.10
EXPRESSION_RANGE_TOLERANCE = 0.01


def numbered_files(directory: pathlib.Path, suffix: str):
    files = sorted(directory.glob(f"*{suffix}"))
    if not files:
        raise RuntimeError(f"No {suffix} files in {directory}")
    return files


def load_windows(directory: pathlib.Path):
    files = numbered_files(directory, ".npy")
    windows = [np.load(path) for path in files]
    shape = windows[0].shape
    if any(window.shape != shape for window in windows):
        raise RuntimeError(f"Inconsistent window shapes in {directory}")
    values = np.stack(windows)
    if not np.isfinite(values).all():
        raise RuntimeError(f"NaN/Inf in {directory}")
    return values, [path.name for path in files]


def audio_is_consecutive(previous, current):
    if previous is None or current is None or previous.shape != current.shape:
        return False
    exact_step = STRIDE * AUDIO_FPS / POSE_FPS
    for step in {math.floor(exact_step), math.ceil(exact_step)}:
        overlap = len(current) - step
        if overlap > 0 and np.allclose(
            previous[-overlap:], current[:overlap], rtol=0.0, atol=1e-7
        ):
            return True
    return False


def chain_slices(pred_root: pathlib.Path, gt_root: pathlib.Path, expected: int):
    pred_files = numbered_files(pred_root / "audio", ".wav")
    gt_files = numbered_files(gt_root / "audio", ".wav")
    if len(pred_files) != expected or len(gt_files) != expected:
        raise RuntimeError(
            f"Audio count mismatch: pred={len(pred_files)} gt={len(gt_files)} expected={expected}"
        )
    chains = []
    chain_start = 0
    previous = None
    for index, (pred_path, gt_path) in enumerate(zip(pred_files, gt_files)):
        pred_audio, pred_rate = sf.read(pred_path, dtype="float32")
        gt_audio, gt_rate = sf.read(gt_path, dtype="float32")
        if pred_rate != AUDIO_FPS or gt_rate != AUDIO_FPS:
            raise RuntimeError(f"Unexpected audio rate at window {index}")
        if pred_audio.shape != gt_audio.shape or not np.array_equal(pred_audio, gt_audio):
            raise RuntimeError(f"Prediction/GT audio mismatch at window {index}")
        if index and not audio_is_consecutive(previous, pred_audio):
            chains.append((chain_start, index))
            chain_start = index
        previous = pred_audio
    chains.append((chain_start, expected))
    return chains


def overlap_rms(previous, current):
    difference = previous[-OVERLAP:] - current[:OVERLAP]
    return float(np.sqrt(np.mean(np.square(difference), dtype=np.float64)))


def stitch(values, chains):
    stitched = []
    boundaries = []
    overlap_errors = []
    for start, end in chains:
        chain = [values[start]]
        frame_count = WINDOW_LENGTH
        for index in range(start + 1, end):
            error = overlap_rms(values[index - 1], values[index])
            overlap_errors.append(error)
            boundaries.append({
                "chain_index": len(stitched),
                "frame": frame_count,
                "output_window_index": index,
            })
            chain.append(values[index, OVERLAP:])
            frame_count += STRIDE
        stitched.append(np.concatenate(chain, axis=0))
    return stitched, boundaries, overlap_errors


def rotation_matrices(normalized, mean, std):
    degrees = normalized * std + mean
    joint_count = degrees.shape[-1] // 3
    return Rotation.from_euler(
        "XYZ", degrees.reshape(-1, 3), degrees=True
    ).as_matrix().reshape(len(degrees), joint_count, 3, 3)


def velocity_vectors(matrices):
    relative = np.matmul(np.swapaxes(matrices[:-1], -1, -2), matrices[1:])
    return np.rad2deg(
        Rotation.from_matrix(relative.reshape(-1, 3, 3)).as_rotvec()
    ).reshape(len(matrices) - 1, matrices.shape[1], 3)


def rms(chunks):
    total = sum(float(np.square(chunk, dtype=np.float64).sum()) for chunk in chunks)
    count = sum(chunk.size for chunk in chunks)
    return math.sqrt(total / count) if count else None


def percentile(values, value):
    return float(np.percentile(values, value)) if values.size else None


def safe_ratio(prediction, target):
    if target == 0:
        return None if prediction != 0 else 1.0
    return prediction / target


def correlation(prediction, target):
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    if prediction.size < 2 or np.std(prediction) < 1e-12 or np.std(target) < 1e-12:
        return None
    return float(np.corrcoef(prediction, target)[0, 1])


def summarize_expressions(
    pred_chains,
    gt_chains,
    pred_boundaries,
    gt_boundaries,
):
    """Summarize raw BEAT blendshape coefficients without learned evaluators.

    These deterministic coefficient-space metrics complement, but do not
    replace, the paper's learned FMD/FED or a phoneme-aware lip-sync metric.
    """
    if len(FACE_COEFFICIENTS) != pred_chains[0].shape[-1]:
        raise RuntimeError(
            f"Unexpected expression dimension {pred_chains[0].shape[-1]}"
        )

    pred_all = np.concatenate(pred_chains, axis=0).astype(np.float64)
    gt_all = np.concatenate(gt_chains, axis=0).astype(np.float64)
    errors = pred_all - gt_all

    pred_velocity = [np.diff(values.astype(np.float64), axis=0) for values in pred_chains]
    gt_velocity = [np.diff(values.astype(np.float64), axis=0) for values in gt_chains]
    pred_acceleration = [np.diff(values, axis=0) for values in pred_velocity]
    gt_acceleration = [np.diff(values, axis=0) for values in gt_velocity]
    pred_jerk = [np.diff(values, axis=0) for values in pred_acceleration]
    gt_jerk = [np.diff(values, axis=0) for values in gt_acceleration]

    pred_boundary_peaks = []
    gt_boundary_peaks = []
    for pred_event, gt_event in zip(pred_boundaries, gt_boundaries):
        chain_index = pred_event["chain_index"]
        boundary = pred_event["frame"]
        if (
            chain_index != gt_event["chain_index"]
            or boundary != gt_event["frame"]
        ):
            raise RuntimeError("Prediction/GT expression boundary indexing mismatch")
        pred_slice = np.abs(pred_velocity[chain_index][boundary - 1:boundary + 6])
        gt_slice = np.abs(gt_velocity[chain_index][boundary - 1:boundary + 6])
        pred_boundary_peaks.append(pred_slice.max(axis=0))
        gt_boundary_peaks.append(gt_slice.max(axis=0))
    pred_boundary_peaks = np.stack(pred_boundary_peaks)
    gt_boundary_peaks = np.stack(gt_boundary_peaks)
    boundary_thresholds = np.maximum(
        EXPRESSION_BOUNDARY_FLOOR,
        np.percentile(gt_boundary_peaks, 99.9, axis=0),
    )
    boundary_abnormal = pred_boundary_peaks > boundary_thresholds.reshape(1, -1)

    top_events = []
    flat_order = np.argsort(pred_boundary_peaks.reshape(-1))[::-1][:20]
    for flat_index in flat_order:
        event_index, coefficient_index = np.unravel_index(
            flat_index, pred_boundary_peaks.shape
        )
        event = pred_boundaries[event_index]
        top_events.append({
            "output_window_index": int(event["output_window_index"]),
            "chain_index": int(event["chain_index"]),
            "boundary_frame_in_chain": int(event["frame"]),
            "coefficient": FACE_COEFFICIENTS[coefficient_index],
            "prediction_peak_per_frame": float(
                pred_boundary_peaks[event_index, coefficient_index]
            ),
            "ground_truth_peak_per_frame": float(
                gt_boundary_peaks[event_index, coefficient_index]
            ),
            "abnormal_threshold_per_frame": float(
                boundary_thresholds[coefficient_index]
            ),
        })

    active = np.std(gt_all, axis=0) > 1e-5
    per_coefficient = {}
    for coefficient_index, coefficient_name in enumerate(FACE_COEFFICIENTS):
        pred_values = pred_all[:, coefficient_index]
        gt_values = gt_all[:, coefficient_index]
        coefficient_errors = errors[:, coefficient_index]
        pred_boundary = pred_boundary_peaks[:, coefficient_index]
        gt_boundary = gt_boundary_peaks[:, coefficient_index]
        threshold = boundary_thresholds[coefficient_index]
        per_coefficient[coefficient_name] = {
            "fidelity": {
                "mse": float(np.mean(np.square(coefficient_errors))),
                "mae": float(np.mean(np.abs(coefficient_errors))),
                "correlation": correlation(pred_values, gt_values),
            },
            "range": {
                "prediction_min": float(pred_values.min()),
                "prediction_max": float(pred_values.max()),
                "ground_truth_min": float(gt_values.min()),
                "ground_truth_max": float(gt_values.max()),
                "significant_out_of_range_rate": float(np.mean(
                    (pred_values < -EXPRESSION_RANGE_TOLERANCE)
                    | (pred_values > 1.0 + EXPRESSION_RANGE_TOLERANCE)
                )),
            },
            "boundary": {
                "prediction_p99": percentile(pred_boundary, 99),
                "prediction_max": float(pred_boundary.max()),
                "ground_truth_p99": percentile(gt_boundary, 99),
                "ground_truth_max": float(gt_boundary.max()),
                "abnormal_threshold": float(threshold),
                "abnormal_rate": float(np.mean(pred_boundary > threshold)),
            },
        }

    pred_velocity_rms = rms(pred_velocity)
    gt_velocity_rms = rms(gt_velocity)
    pred_acceleration_rms = rms(pred_acceleration)
    gt_acceleration_rms = rms(gt_acceleration)
    pred_jerk_rms = rms(pred_jerk)
    gt_jerk_rms = rms(gt_jerk)
    pred_spread = float(np.mean(
        np.percentile(pred_all, 90, axis=0) - np.percentile(pred_all, 10, axis=0)
    ))
    gt_spread = float(np.mean(
        np.percentile(gt_all, 90, axis=0) - np.percentile(gt_all, 10, axis=0)
    ))

    return {
        "frames": int(len(pred_all)),
        "boundaries": int(len(pred_boundary_peaks)),
        "fidelity": {
            "coefficient_mse": float(np.mean(np.square(errors))),
            "coefficient_mae": float(np.mean(np.abs(errors))),
            "coefficient_pck_at_0_05": float(np.mean(np.abs(errors) < 0.05)),
            "coefficient_pck_at_0_10": float(np.mean(np.abs(errors) < 0.10)),
            "correlation_active_coefficients": correlation(
                pred_all[:, active], gt_all[:, active]
            ),
            "active_coefficient_count": int(active.sum()),
        },
        "range": {
            "prediction_min": float(pred_all.min()),
            "prediction_max": float(pred_all.max()),
            "ground_truth_min": float(gt_all.min()),
            "ground_truth_max": float(gt_all.max()),
            "raw_out_of_0_1_rate": float(np.mean((pred_all < 0.0) | (pred_all > 1.0))),
            "significant_out_of_0_1_rate": float(np.mean(
                (pred_all < -EXPRESSION_RANGE_TOLERANCE)
                | (pred_all > 1.0 + EXPRESSION_RANGE_TOLERANCE)
            )),
            "tolerance": EXPRESSION_RANGE_TOLERANCE,
        },
        "temporal": {
            "prediction": {
                "velocity_rms": pred_velocity_rms,
                "acceleration_rms": pred_acceleration_rms,
                "jerk_rms": pred_jerk_rms,
                "spread_p90_minus_p10_mean": pred_spread,
            },
            "ground_truth": {
                "velocity_rms": gt_velocity_rms,
                "acceleration_rms": gt_acceleration_rms,
                "jerk_rms": gt_jerk_rms,
                "spread_p90_minus_p10_mean": gt_spread,
            },
            "ratios_to_gt": {
                "velocity": safe_ratio(pred_velocity_rms, gt_velocity_rms),
                "acceleration": safe_ratio(pred_acceleration_rms, gt_acceleration_rms),
                "jerk": safe_ratio(pred_jerk_rms, gt_jerk_rms),
                "spread": safe_ratio(pred_spread, gt_spread),
            },
        },
        "boundary": {
            "threshold_floor_per_frame": EXPRESSION_BOUNDARY_FLOOR,
            "prediction_p95": percentile(pred_boundary_peaks.reshape(-1), 95),
            "prediction_p99": percentile(pred_boundary_peaks.reshape(-1), 99),
            "prediction_p99_9": percentile(pred_boundary_peaks.reshape(-1), 99.9),
            "prediction_max": float(pred_boundary_peaks.max()),
            "ground_truth_p99": percentile(gt_boundary_peaks.reshape(-1), 99),
            "ground_truth_max": float(gt_boundary_peaks.max()),
            "abnormal_rate": float(np.mean(boundary_abnormal)),
            "top_events": top_events,
        },
        "coefficients": per_coefficient,
        "limitations": {
            "learned_fmd_fed": "not_evaluated_missing_original_face_feature_extractor",
            "phoneme_aware_lip_sync": "not_evaluated_missing_sync_evaluator",
        },
    }


def summarize(
    pred_chains,
    gt_chains,
    pred_boundaries,
    gt_boundaries,
    joint_names,
):
    pred_velocity_vectors = [velocity_vectors(chain) for chain in pred_chains]
    gt_velocity_vectors = [velocity_vectors(chain) for chain in gt_chains]
    pred_acceleration = [np.diff(values, axis=0) for values in pred_velocity_vectors]
    gt_acceleration = [np.diff(values, axis=0) for values in gt_velocity_vectors]
    pred_jerk = [np.diff(values, n=2, axis=0) for values in pred_velocity_vectors]
    gt_jerk = [np.diff(values, n=2, axis=0) for values in gt_velocity_vectors]

    pred_velocity = [np.linalg.norm(values, axis=-1) for values in pred_velocity_vectors]
    gt_velocity = [np.linalg.norm(values, axis=-1) for values in gt_velocity_vectors]
    pred_boundary_peaks = []
    gt_boundary_peaks = []
    for pred_event, gt_event in zip(
        pred_boundaries, gt_boundaries
    ):
        chain_index = pred_event["chain_index"]
        boundary = pred_event["frame"]
        gt_chain_index = gt_event["chain_index"]
        gt_boundary = gt_event["frame"]
        if chain_index != gt_chain_index or boundary != gt_boundary:
            raise RuntimeError("Prediction/GT boundary indexing mismatch")
        pred_slice = pred_velocity[chain_index][boundary - 1:boundary + 6]
        gt_slice = gt_velocity[chain_index][boundary - 1:boundary + 6]
        pred_boundary_peaks.append(pred_slice.max(axis=0))
        gt_boundary_peaks.append(gt_slice.max(axis=0))
    pred_boundary_peaks = np.stack(pred_boundary_peaks)
    gt_boundary_peaks = np.stack(gt_boundary_peaks)

    fidelity_errors = []
    for prediction, target in zip(pred_chains, gt_chains):
        relative = np.matmul(np.swapaxes(prediction, -1, -2), target)
        errors = Rotation.from_matrix(relative.reshape(-1, 3, 3)).magnitude()
        fidelity_errors.append(errors.reshape(prediction.shape[:2]))
    fidelity_errors = np.concatenate(fidelity_errors, axis=0)

    all_pred_rotations = np.concatenate(pred_chains, axis=0)
    all_gt_rotations = np.concatenate(gt_chains, axis=0)
    pred_spread = []
    gt_spread = []
    for joint_index in range(len(joint_names)):
        pred_rotation = Rotation.from_matrix(all_pred_rotations[:, joint_index])
        gt_rotation = Rotation.from_matrix(all_gt_rotations[:, joint_index])
        pred_center = pred_rotation.mean()
        gt_center = gt_rotation.mean()
        pred_spread.append(
            np.percentile(np.rad2deg((pred_center.inv() * pred_rotation).magnitude()), 90)
        )
        gt_spread.append(
            np.percentile(np.rad2deg((gt_center.inv() * gt_rotation).magnitude()), 90)
        )

    gt_thresholds = np.maximum(60.0, np.percentile(gt_boundary_peaks, 99.9, axis=0))
    joint_report = {}
    for joint_index, joint_name in enumerate(joint_names):
        pred_values = pred_boundary_peaks[:, joint_index]
        gt_values = gt_boundary_peaks[:, joint_index]
        threshold = gt_thresholds[joint_index]
        joint_report[joint_name] = {
            "prediction": {
                "p95_deg_per_frame": percentile(pred_values, 95),
                "p99_deg_per_frame": percentile(pred_values, 99),
                "p99_9_deg_per_frame": percentile(pred_values, 99.9),
                "max_deg_per_frame": float(pred_values.max()),
                "abnormal_rate": float(np.mean(pred_values > threshold)),
            },
            "ground_truth": {
                "p95_deg_per_frame": percentile(gt_values, 95),
                "p99_deg_per_frame": percentile(gt_values, 99),
                "p99_9_deg_per_frame": percentile(gt_values, 99.9),
                "max_deg_per_frame": float(gt_values.max()),
                "abnormal_rate": float(np.mean(gt_values > threshold)),
            },
            "abnormal_threshold_deg_per_frame": float(threshold),
        }

    arm_indices = [joint_names.index(name) for name in ARM_JOINTS]
    arm_prediction = pred_boundary_peaks[:, arm_indices]
    arm_ground_truth = gt_boundary_peaks[:, arm_indices]
    arm_thresholds = gt_thresholds[arm_indices]
    arm_abnormal = arm_prediction > arm_thresholds.reshape(1, -1)
    top_arm_events = []
    flat_order = np.argsort(arm_prediction.reshape(-1))[::-1][:20]
    for flat_index in flat_order:
        event_index, arm_offset = np.unravel_index(
            flat_index, arm_prediction.shape
        )
        joint_index = arm_indices[arm_offset]
        event = pred_boundaries[event_index]
        top_arm_events.append({
            "output_window_index": int(event["output_window_index"]),
            "chain_index": int(event["chain_index"]),
            "boundary_frame_in_chain": int(event["frame"]),
            "joint": joint_names[joint_index],
            "prediction_peak_deg_per_frame": float(
                pred_boundary_peaks[event_index, joint_index]
            ),
            "ground_truth_peak_deg_per_frame": float(
                gt_boundary_peaks[event_index, joint_index]
            ),
            "abnormal_threshold_deg_per_frame": float(gt_thresholds[joint_index]),
        })

    pred_velocity_rms = rms(pred_velocity_vectors)
    gt_velocity_rms = rms(gt_velocity_vectors)
    pred_acceleration_rms = rms(pred_acceleration)
    gt_acceleration_rms = rms(gt_acceleration)
    pred_jerk_rms = rms(pred_jerk)
    gt_jerk_rms = rms(gt_jerk)
    pred_spread_mean = float(np.mean(pred_spread))
    gt_spread_mean = float(np.mean(gt_spread))
    return {
        "frames": int(sum(len(chain) for chain in pred_chains)),
        "boundaries": int(len(pred_boundary_peaks)),
        "fidelity": {
            "rotation_geodesic_mse_rad2": float(np.mean(fidelity_errors ** 2)),
            "rotation_pck_at_0_5_rad": float(np.mean(fidelity_errors < 0.5)),
            "rotation_error_p95_deg": float(np.percentile(np.rad2deg(fidelity_errors), 95)),
        },
        "temporal": {
            "prediction": {
                "velocity_rms": pred_velocity_rms,
                "acceleration_rms": pred_acceleration_rms,
                "jerk_rms": pred_jerk_rms,
                "motion_spread_p90_mean_deg": pred_spread_mean,
            },
            "ground_truth": {
                "velocity_rms": gt_velocity_rms,
                "acceleration_rms": gt_acceleration_rms,
                "jerk_rms": gt_jerk_rms,
                "motion_spread_p90_mean_deg": gt_spread_mean,
            },
            "ratios_to_gt": {
                "velocity": pred_velocity_rms / gt_velocity_rms,
                "acceleration": pred_acceleration_rms / gt_acceleration_rms,
                "jerk": pred_jerk_rms / gt_jerk_rms,
                "motion_spread": pred_spread_mean / gt_spread_mean,
            },
        },
        "arm_boundary": {
            "p95_deg_per_frame": percentile(arm_prediction.reshape(-1), 95),
            "p99_deg_per_frame": percentile(arm_prediction.reshape(-1), 99),
            "p99_9_deg_per_frame": percentile(arm_prediction.reshape(-1), 99.9),
            "max_deg_per_frame": float(arm_prediction.max()),
            "abnormal_rate": float(np.mean(arm_abnormal)),
            "ground_truth_p99_deg_per_frame": percentile(
                arm_ground_truth.reshape(-1), 99
            ),
            "top_events": top_arm_events,
        },
        "joints": joint_report,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-root", type=pathlib.Path, required=True)
    parser.add_argument("--ground-truth-root", type=pathlib.Path, required=True)
    parser.add_argument("--blend", type=int, required=True)
    parser.add_argument("--scope", choices=("preflight", "full"), required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()

    prediction, prediction_names = load_windows(args.prediction_root / "gesture")
    ground_truth, ground_truth_names = load_windows(args.ground_truth_root / "gesture")
    if prediction_names != ground_truth_names or prediction.shape != ground_truth.shape:
        raise RuntimeError(
            f"Prediction/GT mismatch: {prediction.shape} vs {ground_truth.shape}"
        )
    if prediction.shape[1:] != (WINDOW_LENGTH, 141):
        raise RuntimeError(f"Unexpected gesture shape {prediction.shape}")

    pred_expression, pred_expression_names = load_windows(
        args.prediction_root / "expression"
    )
    gt_expression, gt_expression_names = load_windows(
        args.ground_truth_root / "expression"
    )
    if (
        pred_expression_names != gt_expression_names
        or pred_expression_names != prediction_names
        or pred_expression.shape != gt_expression.shape
    ):
        raise RuntimeError(
            "Prediction/GT expression filenames or shapes do not match gesture windows"
        )
    if pred_expression.shape[1:] != (WINDOW_LENGTH, len(FACE_COEFFICIENTS)):
        raise RuntimeError(f"Unexpected expression shape {pred_expression.shape}")

    chains = chain_slices(args.prediction_root, args.ground_truth_root, len(prediction))
    pred_stitched, pred_boundaries, pred_overlap = stitch(prediction, chains)
    gt_stitched, gt_boundaries, gt_overlap = stitch(ground_truth, chains)
    pred_expression_stitched, pred_expression_boundaries, pred_expression_overlap = stitch(
        pred_expression, chains
    )
    gt_expression_stitched, gt_expression_boundaries, gt_expression_overlap = stitch(
        gt_expression, chains
    )
    mean = np.load(STATS_ROOT / "bvh_mean.npy")
    std = np.load(STATS_ROOT / "bvh_std.npy")
    if mean.shape != (141,) or std.shape != (141,):
        raise RuntimeError("Unexpected BVH normalization statistics")

    pred_rotations = [rotation_matrices(chain, mean, std) for chain in pred_stitched]
    gt_rotations = [rotation_matrices(chain, mean, std) for chain in gt_stitched]
    joint_names = list(joints_list["spine_neck_141"].keys())
    if len(joint_names) != 47:
        raise RuntimeError(f"Unexpected joint count {len(joint_names)}")
    report = {
        "scope": args.scope,
        "blend": args.blend,
        "window_count": int(len(prediction)),
        "chain_count": int(len(chains)),
        "chain_lengths_windows": [int(end - start) for start, end in chains],
        "overlap_rms": {
            "prediction_max": max(pred_overlap, default=0.0),
            "ground_truth_max": max(gt_overlap, default=0.0),
        },
        "expression_overlap_rms": {
            "prediction_max": max(pred_expression_overlap, default=0.0),
            "ground_truth_max": max(gt_expression_overlap, default=0.0),
        },
        "metrics": summarize(
            pred_rotations,
            gt_rotations,
            pred_boundaries,
            gt_boundaries,
            joint_names,
        ),
        "expression_metrics": summarize_expressions(
            pred_expression_stitched,
            gt_expression_stitched,
            pred_expression_boundaries,
            gt_expression_boundaries,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.summary_only:
        metrics = report["metrics"]
        expression_metrics = report["expression_metrics"]
        print(json.dumps({
            "scope": report["scope"],
            "blend": report["blend"],
            "window_count": report["window_count"],
            "chain_count": report["chain_count"],
            "overlap_rms": report["overlap_rms"],
            "expression_overlap_rms": report["expression_overlap_rms"],
            "fidelity": metrics["fidelity"],
            "temporal_ratios_to_gt": metrics["temporal"]["ratios_to_gt"],
            "arm_boundary": metrics["arm_boundary"],
            "expression_fidelity": expression_metrics["fidelity"],
            "expression_range": expression_metrics["range"],
            "expression_temporal_ratios_to_gt": expression_metrics["temporal"]["ratios_to_gt"],
            "expression_boundary": expression_metrics["boundary"],
        }, indent=2))
    else:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
