"""Select diverse, continuity-safe long validation ranges from the BEAT cache."""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle

import lmdb
import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CACHE = os.path.join(
    ROOT,
    "data",
    "BEAT",
    "beat_cache",
    "beat_4english_15_141",
    "val",
    "bvh_rot_cache_len34_stride10",
)
DEFAULT_OUTPUT = os.path.join(
    ROOT, "bvh_output", "long_validation_visuals", "selection.json"
)
POSE_FPS = 15
AUDIO_FPS = 16000
POSE_STRIDE = 10


def is_consecutive(previous, current):
    if previous is None or len(previous) != len(current):
        return False
    exact_step = POSE_STRIDE * AUDIO_FPS / POSE_FPS
    for step in {math.floor(exact_step), math.ceil(exact_step)}:
        overlap = len(previous) - step
        if overlap > 0 and np.array_equal(previous[-overlap:], current[:overlap]):
            return True
    return False


def read_samples(cache_path):
    environment = lmdb.open(cache_path, readonly=True, lock=False, readahead=False)
    samples = []
    try:
        with environment.begin(write=False) as transaction:
            count = transaction.stat()["entries"]
            for index in range(count):
                value = transaction.get(f"{index:05d}".encode("ascii"))
                pose, _, audio, _, _, _, _, _, speaker = pickle.loads(value)
                samples.append({
                    "index": index,
                    "pose": np.asarray(pose),
                    "audio": np.asarray(audio),
                    "speaker": tuple(np.asarray(speaker).reshape(-1).tolist()),
                })
    finally:
        environment.close()
    return samples


def group_samples(samples):
    groups = []
    current = []
    previous = None
    for sample in samples:
        chained = (
            previous is not None
            and previous["speaker"] == sample["speaker"]
            and is_consecutive(previous["audio"], sample["audio"])
        )
        if current and not chained:
            groups.append(current)
            current = []
        current.append(sample)
        previous = sample
    if current:
        groups.append(current)
    return groups


def unique_audio(group):
    output = [group[0]["audio"]]
    for previous, current in zip(group, group[1:]):
        for step in (math.floor(POSE_STRIDE * AUDIO_FPS / POSE_FPS),
                     math.ceil(POSE_STRIDE * AUDIO_FPS / POSE_FPS)):
            overlap = len(current["audio"]) - step
            if overlap > 0 and np.array_equal(
                previous["audio"][-overlap:], current["audio"][:overlap]
            ):
                output.append(current["audio"][-step:])
                break
        else:
            raise RuntimeError("A grouped sequence contains a discontinuous audio pair")
    return np.concatenate(output)


def describe(group, window_count):
    selected = group[:window_count]
    audio = unique_audio(selected).astype(np.float64)
    poses = np.stack([sample["pose"] for sample in selected])
    step = POSE_STRIDE
    joined_pose = np.concatenate([poses[0], *[pose[-step:] for pose in poses[1:]]])
    block = max(1, AUDIO_FPS // 20)
    usable = len(audio) // block * block
    block_rms = np.sqrt(np.mean(audio[:usable].reshape(-1, block) ** 2, axis=1))
    pose_velocity = np.diff(joined_pose, axis=0)
    total_frames = joined_pose.shape[0]
    return {
        "start": int(selected[0]["index"]),
        "count": int(len(selected)),
        "end_exclusive": int(selected[-1]["index"] + 1),
        "speaker": list(selected[0]["speaker"]),
        "duration_seconds": total_frames / POSE_FPS,
        "audio_rms": float(np.sqrt(np.mean(audio ** 2))),
        "audio_rms_variation": float(np.std(block_rms)),
        "audio_peak": float(np.max(np.abs(audio))),
        "normalized_pose_velocity_rms": float(np.sqrt(np.mean(pose_velocity ** 2))),
    }


def choose_diverse(candidates):
    """Pick quiet, typical, and energetic ranges without reusing a sequence."""
    ordered = sorted(candidates, key=lambda item: item["audio_rms"])
    picks = [
        ("quiet", ordered[0]),
        ("typical", ordered[len(ordered) // 2]),
        ("energetic", ordered[-1]),
    ]
    result = []
    used = set()
    for label, candidate in picks:
        if candidate["start"] in used:
            continue
        item = dict(candidate)
        item["label"] = label
        result.append(item)
        used.add(candidate["start"])
    if len(result) != 3:
        raise RuntimeError("Could not select three distinct validation sequences")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default=DEFAULT_CACHE)
    parser.add_argument("--window-count", type=int, default=20)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.window_count < 2:
        raise ValueError("window-count must be at least 2")

    samples = read_samples(args.cache)
    groups = group_samples(samples)
    eligible = [group for group in groups if len(group) >= args.window_count]
    if len(eligible) < 3:
        raise RuntimeError(
            f"Only {len(eligible)} continuous groups contain at least "
            f"{args.window_count} windows"
        )
    candidates = [describe(group, args.window_count) for group in eligible]
    selected = choose_diverse(candidates)
    payload = {
        "cache": args.cache,
        "sample_count": len(samples),
        "continuous_group_count": len(groups),
        "eligible_group_count": len(eligible),
        "window_count_per_segment": args.window_count,
        "pose_fps": POSE_FPS,
        "pose_stride_frames": POSE_STRIDE,
        "segments": selected,
        "test_window_ranges": ",".join(
            f"{item['start']}:{item['count']}" for item in selected
        ),
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
