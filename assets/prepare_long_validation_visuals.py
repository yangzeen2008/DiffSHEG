"""Prepare three continuity-safe long validation comparisons for rendering."""

from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import soundfile as sf


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from assets.prepare_aligned_v1_visual import (  # noqa: E402
    motion_spread,
    temporal_metrics,
    write_bvh,
)


EXPERIMENT = "beat_FM_aa_x0_aligned_v1"
RESULT_NAME = os.environ.get(
    "DIFFSHEG_PRED_RESULT_NAME",
    "BestPCK_e479_sequential_overlap24_long3_diverse20",
)
GT_RESULT_NAME = os.environ.get(
    "DIFFSHEG_GT_RESULT_NAME",
    "BestPCK_e479_sequential_overlap24_long3_diverse20",
)
PRED_ROOT = os.path.join(
    ROOT, "results", "beat_34", "test_on_val", EXPERIMENT,
    "fixStart24", RESULT_NAME,
)
GT_ROOT = os.path.join(
    ROOT, "results", "beat_34", "test_on_val_GT", EXPERIMENT,
    "fixStart24", GT_RESULT_NAME,
)
DEFAULT_OUTPUT_ROOT = os.path.join(ROOT, "bvh_output", "long_validation_visuals")
OUTPUT_ROOT = os.environ.get("DIFFSHEG_LONG_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT)
SELECTION_PATH = os.environ.get(
    "DIFFSHEG_SELECTION_PATH", os.path.join(DEFAULT_OUTPUT_ROOT, "selection.json")
)
STATS = os.path.join(ROOT, "bvh_output", "aligned_v1_visual", "stats")
WINDOW_LENGTH = 34
STRIDE = 10
OVERLAP = WINDOW_LENGTH - STRIDE
FPS = 15
AUDIO_FPS = 16000
TRANSITION_BLEND = int(os.environ.get("DIFFSHEG_TRANSITION_BLEND", "7"))


def load_numbered(directory):
    files = sorted(name for name in os.listdir(directory) if name.endswith(".npy"))
    return [np.load(os.path.join(directory, name)) for name in files]


def load_audio(directory):
    chunks = []
    for name in sorted(item for item in os.listdir(directory) if item.endswith(".wav")):
        audio, sample_rate = sf.read(os.path.join(directory, name), dtype="float32")
        if sample_rate != AUDIO_FPS:
            raise RuntimeError(f"Unexpected audio rate {sample_rate} in {name}")
        chunks.append(audio)
    return chunks


def stitch_windows(windows, *, require_exact=True):
    if not windows:
        raise RuntimeError("No windows to stitch")
    joined = [windows[0]]
    errors = []
    for previous, current in zip(windows, windows[1:]):
        difference = previous[-OVERLAP:] - current[:OVERLAP]
        error = float(np.sqrt(np.mean(np.square(difference), dtype=np.float64)))
        errors.append(error)
        if require_exact and error != 0.0:
            raise RuntimeError(f"Non-zero motion overlap RMS: {error}")
        joined.append(current[-STRIDE:])
    return np.concatenate(joined), errors


def stitch_audio(chunks):
    if not chunks:
        raise RuntimeError("No audio windows to stitch")
    joined = [chunks[0]]
    steps = []
    exact_step = STRIDE * AUDIO_FPS / FPS
    for previous, current in zip(chunks, chunks[1:]):
        for step in {math.floor(exact_step), math.ceil(exact_step)}:
            overlap = len(current) - step
            if overlap > 0 and np.array_equal(previous[-overlap:], current[:overlap]):
                steps.append(step)
                joined.append(current[-step:])
                break
        else:
            raise RuntimeError("Audio windows are not consecutive")
    return np.concatenate(joined), steps


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def main():
    with open(SELECTION_PATH, "r", encoding="utf-8") as handle:
        selection = json.load(handle)
    pred_gesture = load_numbered(os.path.join(PRED_ROOT, "gesture"))
    pred_expression = load_numbered(os.path.join(PRED_ROOT, "expression"))
    pred_audio = load_audio(os.path.join(PRED_ROOT, "audio"))
    gt_gesture = load_numbered(os.path.join(GT_ROOT, "gesture"))
    gt_audio = load_audio(os.path.join(GT_ROOT, "audio"))
    expected = sum(segment["count"] for segment in selection["segments"])
    collections = [pred_gesture, pred_expression, pred_audio, gt_gesture, gt_audio]
    if any(len(collection) != expected for collection in collections):
        raise RuntimeError(
            f"Expected {expected} windows, got {[len(collection) for collection in collections]}"
        )

    mean = np.load(os.path.join(STATS, "bvh_mean.npy"))
    std = np.load(os.path.join(STATS, "bvh_std.npy"))
    reports = []
    offset = 0
    for segment in selection["segments"]:
        label = segment["label"]
        count = segment["count"]
        segment_slice = slice(offset, offset + count)
        output_dir = os.path.join(OUTPUT_ROOT, label)
        os.makedirs(output_dir, exist_ok=True)

        pred_joined, pred_overlap = stitch_windows(pred_gesture[segment_slice])
        _, expression_overlap = stitch_windows(pred_expression[segment_slice])
        gt_joined, gt_overlap = stitch_windows(gt_gesture[segment_slice])
        audio, audio_steps = stitch_audio(pred_audio[segment_slice])
        gt_audio_joined, _ = stitch_audio(gt_audio[segment_slice])
        if not np.array_equal(audio, gt_audio_joined):
            raise RuntimeError(f"Prediction and GT audio differ for {label}")

        pred_degrees = write_bvh(
            os.path.join(output_dir, "model_pck_best_e479.bvh"), pred_joined, mean, std
        )
        gt_degrees = write_bvh(
            os.path.join(output_dir, "ground_truth.bvh"), gt_joined, mean, std
        )
        audio_path = os.path.join(output_dir, "audio.wav")
        sf.write(audio_path, audio, AUDIO_FPS)
        pred_metrics = temporal_metrics(pred_degrees)
        gt_metrics = temporal_metrics(gt_degrees)
        report = {
            "label": label,
            "source_window_start": segment["start"],
            "window_count": count,
            "frames": int(pred_joined.shape[0]),
            "duration_seconds": pred_joined.shape[0] / FPS,
            "audio_rms": float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))),
            "overlap_rms": {
                "gesture_max": max(pred_overlap, default=0.0),
                "expression_max": max(expression_overlap, default=0.0),
                "ground_truth_max": max(gt_overlap, default=0.0),
            },
            "audio_step_samples": sorted(set(audio_steps)),
            "prediction": pred_metrics,
            "ground_truth": gt_metrics,
            "ratios": {
                key.replace("_rms", ""): ratio(pred_metrics[key], gt_metrics[key])
                for key in pred_metrics
            },
            "motion_spread_degrees": {
                "prediction": motion_spread(pred_degrees),
                "ground_truth": motion_spread(gt_degrees),
                "ratio": ratio(motion_spread(pred_degrees), motion_spread(gt_degrees)),
            },
            "audio_path": audio_path,
        }
        with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
        reports.append(report)
        offset += count

    summary = {
        "experiment": EXPERIMENT,
        "checkpoint": "pck_best.tar",
        "checkpoint_epoch": 479,
        "prediction_result": RESULT_NAME,
        "ground_truth_result": GT_RESULT_NAME,
        "inference": (
            "RK4, 50 steps, overlap 24, "
            f"transition blend {TRANSITION_BLEND}"
        ),
        "segments": reports,
    }
    summary_path = os.path.join(OUTPUT_ROOT, "metrics_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
