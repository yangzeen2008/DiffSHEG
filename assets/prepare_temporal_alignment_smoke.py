"""Choose an active aligned GT segment for audio/face synchronization review."""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import soundfile as sf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-root", type=pathlib.Path, required=True)
    parser.add_argument("--clip", required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--audio-rate", type=int, default=16000)
    parser.add_argument(
        "--motion",
        type=pathlib.Path,
        help="Optional aligned 141D BVH rotation file. Defaults to split-root/bvh_rot/<clip>.bvh.",
    )
    return parser.parse_args()


def normalized(values: np.ndarray) -> np.ndarray:
    minimum = float(values.min())
    maximum = float(values.max())
    if maximum <= minimum:
        return np.zeros_like(values, dtype=np.float64)
    return (values - minimum) / (maximum - minimum)


def euler_xyz_matrix(euler_degrees: np.ndarray) -> np.ndarray:
    """Match the project's XYZ Euler convention (Rx @ Ry @ Rz)."""
    x, y, z = np.radians(euler_degrees)
    cx, cy, cz = np.cos((x, y, z))
    sx, sy, sz = np.sin((x, y, z))
    rx = np.array(((1.0, 0.0, 0.0), (0.0, cx, -sx), (0.0, sx, cx)))
    ry = np.array(((cy, 0.0, sy), (0.0, 1.0, 0.0), (-sy, 0.0, cy)))
    rz = np.array(((cz, -sz, 0.0), (sz, cz, 0.0), (0.0, 0.0, 1.0)))
    return rx @ ry @ rz


def relative_head_matrices(motion: np.ndarray, start: int, end: int) -> np.ndarray:
    """Compose Hips-through-Head rotations, then express the segment relative to frame zero."""
    if motion.ndim != 2 or motion.shape[1] < 21:
        raise RuntimeError(f"Expected aligned motion shaped [frames, >=21], got {motion.shape}")
    world = []
    for frame in motion[start:end, :21]:
        rotation = np.eye(3, dtype=np.float64)
        for joint_euler in frame.reshape(7, 3):
            rotation = rotation @ euler_xyz_matrix(joint_euler)
        world.append(rotation)
    world = np.asarray(world)
    reference_inverse = world[0].T
    return world @ reference_inverse


def main() -> None:
    args = parse_args()
    split_root = args.split_root.resolve()
    face_path = split_root / "facial52" / f"{args.clip}.json"
    audio_path = split_root / "wave16k" / f"{args.clip}.npy"
    motion_path = (
        args.motion.resolve()
        if args.motion
        else (split_root / "bvh_rot" / f"{args.clip}.bvh").resolve()
    )
    face = json.loads(face_path.read_text(encoding="utf-8"))
    weights = np.asarray([frame["weights"] for frame in face["frames"]], dtype=np.float64)
    audio = np.load(audio_path).astype(np.float32, copy=False)
    if audio.ndim == 2:
        audio = audio[:, 0]
    expected_audio = round(len(weights) / args.fps * args.audio_rate)
    if abs(len(audio) - expected_audio) > args.audio_rate / args.fps:
        raise RuntimeError(
            f"Aligned source duration mismatch: face={len(weights)}, audio={len(audio)}"
        )

    segment_frames = round(args.duration * args.fps)
    if segment_frames > len(weights):
        raise RuntimeError("Requested segment is longer than the aligned clip")
    starts = np.arange(0, len(weights) - segment_frames + 1, args.fps, dtype=int)
    audio_rms = []
    expression_range = []
    expression_velocity = []
    for start in starts:
        end = start + segment_frames
        audio_start = round(start / args.fps * args.audio_rate)
        audio_end = round(end / args.fps * args.audio_rate)
        audio_window = audio[audio_start:audio_end].astype(np.float64)
        face_window = weights[start:end]
        audio_rms.append(float(np.sqrt(np.mean(audio_window ** 2))))
        expression_range.append(
            float(np.mean(np.percentile(face_window, 90, axis=0) - np.percentile(face_window, 10, axis=0)))
        )
        expression_velocity.append(float(np.sqrt(np.mean(np.diff(face_window, axis=0) ** 2))))
    audio_rms = np.asarray(audio_rms)
    expression_range = np.asarray(expression_range)
    expression_velocity = np.asarray(expression_velocity)
    score = normalized(audio_rms) + normalized(expression_range) + normalized(expression_velocity)
    chosen = int(np.argmax(score))
    start = int(starts[chosen])
    end = start + segment_frames
    audio_start = round(start / args.fps * args.audio_rate)
    audio_end = round(end / args.fps * args.audio_rate)
    head_matrices = None
    if motion_path.exists():
        motion = np.loadtxt(motion_path, dtype=np.float64)
        if len(motion) != len(weights):
            raise RuntimeError(
                f"Aligned motion/face frame mismatch: motion={len(motion)}, face={len(weights)}"
            )
        head_matrices = relative_head_matrices(motion, start, end)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_face = {
        "names": face.get("names", []),
        "frames": [
            {
                "weights": frame["weights"],
                "time": index / args.fps,
                "rotation": [],
                **(
                    {"head_rotation_matrix_bvh": head_matrices[index].reshape(-1).tolist()}
                    if head_matrices is not None
                    else {}
                ),
            }
            for index, frame in enumerate(face["frames"][start:end])
        ],
    }
    output_face_path = output_dir / "ground_truth_aligned.json"
    output_audio_path = output_dir / "audio_aligned.wav"
    output_face_path.write_text(
        json.dumps(output_face, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    sf.write(output_audio_path, audio[audio_start:audio_end], args.audio_rate)
    report = {
        "source_clip": args.clip,
        "source_face": str(face_path),
        "source_audio": str(audio_path),
        "source_motion": str(motion_path) if head_matrices is not None else None,
        "fps": args.fps,
        "audio_rate": args.audio_rate,
        "start_frame": start,
        "end_frame_exclusive": end,
        "start_seconds": start / args.fps,
        "duration_seconds": segment_frames / args.fps,
        "frames": segment_frames,
        "audio_samples": audio_end - audio_start,
        "head_motion": {
            "included": head_matrices is not None,
            "source_chain": ["Hips", "Spine", "Spine1", "Spine2", "Spine3", "Neck", "Head"],
            "representation": "relative 3x3 BVH rotation matrix, XYZ local rotations composed root-to-head",
            "reference_frame": start if head_matrices is not None else None,
        },
        "selection_metrics": {
            "audio_rms": float(audio_rms[chosen]),
            "expression_range_p90_p10": float(expression_range[chosen]),
            "expression_velocity_rms": float(expression_velocity[chosen]),
            "combined_normalized_score": float(score[chosen]),
        },
        "face_output": str(output_face_path),
        "audio_output": str(output_audio_path),
    }
    report_path = output_dir / "smoke_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("DIFFSHEG_TEMPORAL_SMOKE=" + json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
