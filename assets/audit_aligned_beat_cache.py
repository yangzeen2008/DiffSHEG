"""Audit every file in a temporally aligned BEAT source cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib

import numpy as np


TEMPORAL_MANIFEST_NAME = "temporal_alignment_manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=pathlib.Path, required=True)
    parser.add_argument("--splits", nargs="+", default=("train", "val"))
    parser.add_argument("--output", type=pathlib.Path)
    return parser.parse_args()


def manifest_identity(payload: dict) -> str:
    content = dict(payload)
    content.pop("manifest_id", None)
    canonical = json.dumps(
        content, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def numbered_source_stems(directory: pathlib.Path, suffix: str) -> set[str]:
    return {
        path.stem
        for path in directory.glob(f"*{suffix}")
        if path.name not in {"bvh_mean.npy", "bvh_std.npy", "json_mean.npy", "json_std.npy"}
    }


def audit_split(root: pathlib.Path, split: str) -> dict:
    split_root = root / split
    manifest_path = split_root / TEMPORAL_MANIFEST_NAME
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("manifest_id") != manifest_identity(payload):
        raise RuntimeError(f"Manifest identity mismatch: {manifest_path}")
    if payload.get("target_pose_fps") != 15 or payload.get("target_facial_fps") != 15:
        raise RuntimeError(f"Unexpected target frame rate: {manifest_path}")
    records = payload.get("clips", [])
    expected_stems = {record["clip"] for record in records}
    if len(expected_stems) != len(records):
        raise RuntimeError(f"Duplicate clip in {manifest_path}")
    actual = {
        "motion": numbered_source_stems(split_root / "bvh_rot", ".bvh"),
        "facial": numbered_source_stems(split_root / "facial52", ".json"),
        "audio": numbered_source_stems(split_root / "wave16k", ".npy"),
    }
    for modality, stems in actual.items():
        if stems != expected_stems:
            raise RuntimeError(
                f"{split} {modality} file set mismatch: "
                f"missing={sorted(expected_stems - stems)[:5]} extra={sorted(stems - expected_stems)[:5]}"
            )

    total_frames = 0
    maximum_audio_overhang = 0.0
    maximum_recorded_trim = 0.0
    for record in records:
        stem = record["clip"]
        expected_frames = int(record["target_frames"])
        with (split_root / "bvh_rot" / f"{stem}.bvh").open(
            "r", encoding="utf-8"
        ) as handle:
            motion_frames = sum(1 for line in handle if line.strip())
        face = json.loads(
            (split_root / "facial52" / f"{stem}.json").read_text(encoding="utf-8")
        )
        face_frames = face.get("frames", [])
        audio = np.load(split_root / "wave16k" / f"{stem}.npy", mmap_mode="r")
        if motion_frames != expected_frames or len(face_frames) != expected_frames:
            raise RuntimeError(
                f"{split}/{stem} length mismatch: expected={expected_frames}, "
                f"motion={motion_frames}, face={len(face_frames)}"
            )
        if expected_frames:
            if abs(float(face_frames[0]["time"])) > 1e-9:
                raise RuntimeError(f"{split}/{stem} facial time does not start at zero")
            expected_last = (expected_frames - 1) / 15.0
            if abs(float(face_frames[-1]["time"]) - expected_last) > 1e-8:
                raise RuntimeError(f"{split}/{stem} facial timestamp mismatch")
        audio_duration = int(audio.shape[0]) / 16000.0
        aligned_duration = expected_frames / 15.0
        if audio_duration + 1.0 / 16000.0 < aligned_duration:
            raise RuntimeError(f"{split}/{stem} audio is shorter than aligned frames")
        maximum_audio_overhang = max(
            maximum_audio_overhang, audio_duration - aligned_duration
        )
        maximum_recorded_trim = max(
            maximum_recorded_trim,
            max(record.get("source_tail_trim_seconds", {}).values(), default=0.0),
        )
        total_frames += expected_frames

    return {
        "split": split,
        "manifest_id": payload["manifest_id"],
        "clips": len(records),
        "skipped_clips": int(payload.get("skipped_clip_count", 0)),
        "trimmed_clips": int(payload.get("trimmed_clip_count", 0)),
        "total_frames": total_frames,
        "maximum_audio_overhang_seconds": maximum_audio_overhang,
        "maximum_recorded_source_trim_seconds": maximum_recorded_trim,
    }


def main() -> None:
    args = parse_args()
    cache_root = args.cache_root.resolve()
    reports = [audit_split(cache_root, split) for split in args.splits]
    stats = {}
    for relative in (
        "train/bvh_rot/bvh_mean.npy",
        "train/bvh_rot/bvh_std.npy",
        "train/facial52/json_mean.npy",
        "train/facial52/json_std.npy",
        "train/axis_angle_mean.npy",
        "train/axis_angle_std.npy",
    ):
        values = np.load(cache_root / relative)
        if not np.isfinite(values).all():
            raise RuntimeError(f"Non-finite statistics: {relative}")
        stats[relative] = {"shape": list(values.shape), "min": float(values.min()), "max": float(values.max())}
    result = {"cache_root": str(cache_root), "splits": reports, "statistics": stats}
    if args.output:
        args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
        args.output.resolve().write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
