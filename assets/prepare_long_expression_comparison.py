"""Prepare a continuity-safe long expression stream for Blender rendering."""

from __future__ import annotations

import argparse
import json
import pathlib
import wave

import numpy as np


WINDOW_LENGTH = 34
STRIDE = 10
OVERLAP = WINDOW_LENGTH - STRIDE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=pathlib.Path, required=True)
    parser.add_argument("--mapping", type=pathlib.Path, required=True)
    parser.add_argument("--ground-truth-root", type=pathlib.Path, required=True)
    parser.add_argument("--prediction-root", type=pathlib.Path, required=True)
    parser.add_argument("--audio", type=pathlib.Path, required=True)
    parser.add_argument("--segment", default="energetic")
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--fps", type=int, default=15)
    return parser.parse_args()


def load_windows(root: pathlib.Path, coefficient_count: int) -> list[np.ndarray]:
    paths = sorted((root / "expression").glob("*.npy"))
    windows = [np.load(path) for path in paths]
    expected = (WINDOW_LENGTH, coefficient_count)
    invalid = [str(path) for path, value in zip(paths, windows) if value.shape != expected]
    if invalid:
        raise RuntimeError(f"Unexpected expression shapes: {invalid}")
    return windows


def stitch(windows: list[np.ndarray]) -> tuple[np.ndarray, float]:
    if not windows:
        raise RuntimeError("No expression windows selected")
    maximum_overlap_error = 0.0
    joined = [windows[0]]
    for previous, current in zip(windows, windows[1:]):
        error = float(np.max(np.abs(previous[-OVERLAP:] - current[:OVERLAP])))
        maximum_overlap_error = max(maximum_overlap_error, error)
        if error != 0.0:
            raise RuntimeError(f"Expression windows are not continuous: max error {error}")
        joined.append(current[-STRIDE:])
    return np.concatenate(joined), maximum_overlap_error


def stream_metrics(values: np.ndarray) -> dict[str, float | int]:
    velocity = np.diff(values, axis=0)
    return {
        "frames": int(values.shape[0]),
        "raw_min": float(values.min()),
        "raw_max": float(values.max()),
        "dynamic_range_mean_p90_p10": float(
            np.mean(np.percentile(values, 90, axis=0) - np.percentile(values, 10, axis=0))
        ),
        "velocity_rms": float(np.sqrt(np.mean(velocity.astype(np.float64) ** 2))),
        "max_frame_delta": float(np.max(np.abs(velocity))),
        "values_below_zero": int(np.count_nonzero(values < 0.0)),
        "values_above_one": int(np.count_nonzero(values > 1.0)),
    }


def write_stream(
    path: pathlib.Path,
    values: np.ndarray,
    fps: int,
    coefficient_names: list[str],
) -> None:
    payload = {
        "names": coefficient_names,
        "frames": [
            {
                "weights": frame.astype(float).tolist(),
                "time": index / fps,
                "rotation": [],
            }
            for index, frame in enumerate(values)
        ],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    selection = json.loads(args.selection.resolve().read_text(encoding="utf-8"))
    mapping = json.loads(args.mapping.resolve().read_text(encoding="utf-8"))
    coefficient_names = [entry["source"] for entry in mapping["channels"]]
    if len(coefficient_names) != 51 or len(set(coefficient_names)) != len(coefficient_names):
        raise RuntimeError("Expected 51 unique source coefficients in mapping")
    segments = selection["segments"]
    segment = next((item for item in segments if item["label"] == args.segment), None)
    if segment is None:
        raise RuntimeError(f"Unknown segment {args.segment!r}")
    offset = sum(item["count"] for item in segments[: segments.index(segment)])
    count = int(segment["count"])
    selection_slice = slice(offset, offset + count)

    ground_truth_windows = load_windows(
        args.ground_truth_root.resolve(), len(coefficient_names)
    )
    prediction_windows = load_windows(
        args.prediction_root.resolve(), len(coefficient_names)
    )
    expected_windows = sum(int(item["count"]) for item in segments)
    if len(ground_truth_windows) != expected_windows or len(prediction_windows) != expected_windows:
        raise RuntimeError(
            f"Expected {expected_windows} windows; got GT={len(ground_truth_windows)}, "
            f"prediction={len(prediction_windows)}"
        )

    ground_truth, ground_truth_overlap = stitch(ground_truth_windows[selection_slice])
    prediction, prediction_overlap = stitch(prediction_windows[selection_slice])
    if ground_truth.shape != prediction.shape:
        raise RuntimeError(f"Stream shape mismatch: {ground_truth.shape} != {prediction.shape}")

    audio_path = args.audio.resolve()
    with wave.open(str(audio_path), "rb") as audio:
        audio_rate = audio.getframerate()
        audio_samples = audio.getnframes()
        audio_channels = audio.getnchannels()
    video_duration = ground_truth.shape[0] / args.fps
    audio_duration = audio_samples / audio_rate
    if abs(video_duration - audio_duration) > 1.0 / args.fps:
        raise RuntimeError(
            f"Audio/video duration mismatch: video={video_duration}, audio={audio_duration}"
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ground_truth_path = output_dir / "ground_truth.json"
    prediction_path = output_dir / "prediction_blend7.json"
    write_stream(ground_truth_path, ground_truth, args.fps, coefficient_names)
    write_stream(prediction_path, prediction, args.fps, coefficient_names)

    report = {
        "segment": args.segment,
        "source_window_start": segment["start"],
        "source_window_count": count,
        "window_length": WINDOW_LENGTH,
        "stride": STRIDE,
        "overlap": OVERLAP,
        "fps": args.fps,
        "frames": int(ground_truth.shape[0]),
        "duration_seconds": video_duration,
        "audio": {
            "path": str(audio_path),
            "sample_rate": audio_rate,
            "samples": audio_samples,
            "channels": audio_channels,
            "duration_seconds": audio_duration,
        },
        "overlap_max_abs_error": {
            "ground_truth": ground_truth_overlap,
            "prediction_blend7": prediction_overlap,
        },
        "ground_truth": stream_metrics(ground_truth),
        "prediction_blend7": stream_metrics(prediction),
        "render_inputs": {
            "ground_truth": str(ground_truth_path),
            "prediction_blend7": str(prediction_path),
        },
    }
    report_path = output_dir / "preparation_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("DIFFSHEG_LONG_EXPRESSION=" + json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
