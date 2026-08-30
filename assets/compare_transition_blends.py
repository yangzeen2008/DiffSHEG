"""Compare controlled long-form transition-blend ablations."""

from __future__ import annotations

import json
import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[1]
TRIAL_ROOT = ROOT / "bvh_output" / "long_validation_trials"
VARIANTS = {
    "blend0": TRIAL_ROOT / "so3_release_blend0",
    "blend3": TRIAL_ROOT / "so3_release_blend3",
    "blend5": TRIAL_ROOT / "so3_release_blend5",
    "blend7": TRIAL_ROOT / "so3_release",
}


def load(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main():
    payload = {"controlled_variables": {
        "checkpoint": "pck_best.tar e479",
        "solver": "RK4 50 steps",
        "overlap": 24,
        "ranges": "3571:20,3799:20,944:20",
        "changed_only": "fm_transition_blend",
    }, "segments": {}}
    loaded = {}
    for label, root in VARIANTS.items():
        loaded[label] = {
            "summary": load(root / "metrics_summary.json"),
            "rotation": load(root / "forearm_rotation_audit.json"),
        }

    for segment_index, segment in enumerate(("quiet", "typical", "energetic")):
        payload["segments"][segment] = {}
        for label, reports in loaded.items():
            summary = reports["summary"]["segments"][segment_index]
            forearm = reports["rotation"]["segments"][segment]["LeftForeArm"]["prediction"]
            worst_boundary = forearm["boundary_blocks_by_peak"][0]
            payload["segments"][segment][label] = {
                "velocity_ratio_to_gt": summary["ratios"]["velocity"],
                "acceleration_ratio_to_gt": summary["ratios"]["acceleration"],
                "jerk_ratio_to_gt": summary["ratios"]["jerk"],
                "motion_spread_ratio_to_gt": summary["motion_spread_degrees"]["ratio"],
                "left_forearm_velocity_rms_deg_per_frame": forearm["velocity_rms_deg_per_frame"],
                "left_forearm_velocity_p95_deg_per_frame": forearm["velocity_p95_deg_per_frame"],
                "left_forearm_velocity_max_deg_per_frame": forearm["velocity_max_deg_per_frame"],
                "worst_boundary_entry_jump_deg": worst_boundary["entry_jump_deg"],
                "worst_boundary_peak_deg_per_frame": worst_boundary["peak_velocity_deg"],
                "worst_boundary_path_deg": worst_boundary["path_distance_deg"],
                "worst_boundary_detour_ratio": worst_boundary["path_to_endpoint_ratio"],
            }

    output = TRIAL_ROOT / "transition_blend_ablation.json"
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
