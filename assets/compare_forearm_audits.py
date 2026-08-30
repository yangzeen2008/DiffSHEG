"""Summarize left-forearm SO(3) metrics before and after the release repair."""

from __future__ import annotations

import json
import os
import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[1]
BEFORE = ROOT / "bvh_output" / "long_validation_visuals" / "forearm_rotation_audit.json"
AFTER_ROOT = pathlib.Path(os.environ.get(
    "DIFFSHEG_LONG_OUTPUT_ROOT",
    ROOT / "bvh_output" / "long_validation_trials" / "so3_release",
))
AFTER = AFTER_ROOT / "forearm_rotation_audit.json"


def load(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def extract(report, segment):
    joint = report["segments"][segment]["LeftForeArm"]
    prediction = joint["prediction"]
    worst_release = prediction["release_blend_blocks_by_path"][0]
    return {
        "excursion_p90_deg": prediction["excursion_p90_deg"],
        "velocity_rms_deg_per_frame": prediction["velocity_rms_deg_per_frame"],
        "velocity_p95_deg_per_frame": prediction["velocity_p95_deg_per_frame"],
        "velocity_max_deg_per_frame": prediction["velocity_max_deg_per_frame"],
        "worst_release_path_deg": worst_release["path_distance_deg"],
        "worst_release_endpoint_deg": worst_release["endpoint_distance_deg"],
        "worst_release_detour_ratio": worst_release["path_to_endpoint_ratio"],
    }


def main():
    before = load(BEFORE)
    after = load(AFTER)
    comparison = {"segments": {}}
    for segment in ("quiet", "typical", "energetic"):
        old = extract(before, segment)
        new = extract(after, segment)
        comparison["segments"][segment] = {
            "before": old,
            "after": new,
            "after_over_before": {
                key: new[key] / old[key] if old[key] else None for key in old
            },
        }
    output = AFTER_ROOT / "forearm_before_after.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(comparison, handle, indent=2)
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
