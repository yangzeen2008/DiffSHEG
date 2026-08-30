"""Prepare deterministic blind-review bundles for expression candidates.

The bundle contains coefficient heatmaps and the original face JSON files so a
real character renderer can be attached later without rerunning inference.
It intentionally keeps candidate identities in a separate answers file.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil

import matplotlib.pyplot as plt
import numpy as np

from analyze_full_validation_transitions import FACE_COEFFICIENTS, numbered_files


def load_metrics(path: pathlib.Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_collection(root: pathlib.Path, expected_names=None):
    files = numbered_files(root / "expression", ".npy")
    names = [path.name for path in files]
    if expected_names is not None and names != expected_names:
        raise RuntimeError(f"Expression filename mismatch in {root}")
    values = [np.load(path) for path in files]
    if any(value.shape != (34, len(FACE_COEFFICIENTS)) for value in values):
        raise RuntimeError(f"Unexpected expression shape in {root}")
    return files, values


def event_ranking(*reports):
    scores = {}
    details = {}
    for report in reports:
        events = report["expression_metrics"]["boundary"]["top_events"]
        for event in events:
            index = int(event["output_window_index"])
            score = float(event["prediction_peak_per_frame"])
            if score > scores.get(index, -1.0):
                scores[index] = score
                details[index] = event
    return sorted(scores, key=scores.get, reverse=True), details


def face_json_path(root: pathlib.Path, npy_name: str):
    stem = pathlib.Path(npy_name).stem
    directory = root / "expression" / "face_json"
    candidates = (directory / f"{stem}.json", directory / f"{stem}_rank0.json")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise RuntimeError(f"Missing face JSON for {npy_name} in {directory}")


def plot_review(path, audio_path, ground_truth, candidate_a, candidate_b, title):
    import soundfile as sf

    audio, sample_rate = sf.read(audio_path, dtype="float32")
    figure, axes = plt.subplots(
        4, 1, figsize=(14, 10), constrained_layout=True,
        gridspec_kw={"height_ratios": [1, 2, 2, 2]},
    )
    seconds = np.arange(len(audio)) / sample_rate
    axes[0].plot(seconds, audio, color="#555555", linewidth=0.6)
    axes[0].set_title(title)
    axes[0].set_ylabel("Audio")
    axes[0].set_xlim(0, seconds[-1])
    axes[0].set_xticks([])

    images = []
    for axis, values, label in zip(
        axes[1:],
        (ground_truth, candidate_a, candidate_b),
        ("GROUND TRUTH", "CANDIDATE A", "CANDIDATE B"),
    ):
        image = axis.imshow(
            values.T, aspect="auto", origin="lower", vmin=0.0, vmax=1.0,
            interpolation="nearest", cmap="viridis",
        )
        images.append(image)
        axis.set_ylabel(label)
        axis.set_yticks([2, 8, 24, 31, 43, 50])
        axis.set_yticklabels([FACE_COEFFICIENTS[i] for i in [2, 8, 24, 31, 43, 50]])
    axes[-1].set_xlabel("Frame (15 FPS)")
    figure.colorbar(images[-1], ax=axes[1:], label="Blendshape coefficient")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--blend5-root", type=pathlib.Path, required=True)
    parser.add_argument("--blend7-root", type=pathlib.Path, required=True)
    parser.add_argument("--ground-truth-root", type=pathlib.Path, required=True)
    parser.add_argument("--blend5-metrics", type=pathlib.Path, required=True)
    parser.add_argument("--blend7-metrics", type=pathlib.Path, required=True)
    parser.add_argument("--output-root", type=pathlib.Path, required=True)
    parser.add_argument("--worst-count", type=int, default=4)
    parser.add_argument("--random-count", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    # Keep plotting and JSON output reproducible across machines.
    rng = np.random.default_rng(args.seed)
    blend5_report = load_metrics(args.blend5_metrics)
    blend7_report = load_metrics(args.blend7_metrics)
    gt_files, gt_values = load_collection(args.ground_truth_root)
    names = [path.name for path in gt_files]
    blend5_files, blend5_values = load_collection(args.blend5_root, names)
    blend7_files, blend7_values = load_collection(args.blend7_root, names)

    ranked, event_details = event_ranking(blend5_report, blend7_report)
    worst = ranked[: min(args.worst_count, len(ranked))]
    remaining = np.array([index for index in range(len(names)) if index not in worst])
    random_count = min(args.random_count, len(remaining))
    random_indices = sorted(
        rng.choice(remaining, size=random_count, replace=False).tolist()
    )
    selected = [(index, "worst_boundary") for index in worst]
    selected.extend((index, "fixed_random") for index in random_indices)

    args.output_root.mkdir(parents=True, exist_ok=True)
    review_manifest = {
        "seed": args.seed,
        "blind": True,
        "candidate_labels_are_randomized_per_sample": True,
        "samples": [],
    }
    answer_manifest = {"seed": args.seed, "answers": []}
    for order, (index, reason) in enumerate(selected):
        sample_id = f"sample_{order + 1:02d}"
        sample_root = args.output_root / sample_id
        sample_root.mkdir(parents=True, exist_ok=True)
        swap = bool(rng.integers(0, 2))
        candidate_a_label = "blend7" if swap else "blend5"
        candidate_b_label = "blend5" if swap else "blend7"
        candidate_a_root = args.blend7_root if swap else args.blend5_root
        candidate_b_root = args.blend5_root if swap else args.blend7_root
        candidate_a = blend7_values[index] if swap else blend5_values[index]
        candidate_b = blend5_values[index] if swap else blend7_values[index]

        np.save(sample_root / "ground_truth.npy", gt_values[index])
        np.save(sample_root / "candidate_A.npy", candidate_a)
        np.save(sample_root / "candidate_B.npy", candidate_b)
        shutil.copy2(
            args.ground_truth_root / "audio" / names[index].replace(".npy", ".wav"),
            sample_root / "audio.wav",
        )
        shutil.copy2(
            face_json_path(args.ground_truth_root, names[index]),
            sample_root / "ground_truth.json",
        )
        shutil.copy2(
            face_json_path(candidate_a_root, names[index]),
            sample_root / "candidate_A.json",
        )
        shutil.copy2(
            face_json_path(candidate_b_root, names[index]),
            sample_root / "candidate_B.json",
        )
        plot_review(
            sample_root / "coefficient_review.png",
            sample_root / "audio.wav",
            gt_values[index], candidate_a, candidate_b,
            f"{sample_id} | {reason} | source window {index}",
        )
        review_manifest["samples"].append({
            "sample_id": sample_id,
            "reason": reason,
            "source_output_window_index": index,
            "source_window_file": names[index],
            "boundary_event": event_details.get(index),
            "review_image": str(sample_root / "coefficient_review.png"),
            "audio": str(sample_root / "audio.wav"),
            "render_inputs": {
                "ground_truth": str(sample_root / "ground_truth.json"),
                "candidate_A": str(sample_root / "candidate_A.json"),
                "candidate_B": str(sample_root / "candidate_B.json"),
            },
        })
        answer_manifest["answers"].append({
            "sample_id": sample_id,
            "candidate_A": candidate_a_label,
            "candidate_B": candidate_b_label,
        })

    (args.output_root / "review_manifest.json").write_text(
        json.dumps(review_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.output_root / "answers.json").write_text(
        json.dumps(answer_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({
        "output_root": str(args.output_root),
        "sample_count": len(selected),
        "worst_indices": worst,
        "random_indices": random_indices,
    }, indent=2))


if __name__ == "__main__":
    main()
