"""Render DiffSHEG expression JSON streams through the Rocketbox clay scene."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import bpy
from mathutils import Matrix, Quaternion


EYE_SIDES = ("Left", "Right")


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping", type=pathlib.Path, required=True)
    parser.add_argument("--stream", action="append", required=True, help="stream_id=expression.json")
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--report", type=pathlib.Path, required=True)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--resolution", type=int, default=640)
    parser.add_argument("--head-motion-scale", type=float, default=0.7)
    parser.add_argument("--head-motion-max-degrees", type=float, default=22.0)
    parser.add_argument(
        "--eye-safety-profile",
        choices=("raw", "rocketbox_safe"),
        default="rocketbox_safe",
        help=(
            "Render-only guard for the Rocketbox eyeWide Shape Keys. "
            "The source JSON is never modified."
        ),
    )
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument(
        "--frame-count",
        type=int,
        default=None,
        help="Render only this many frames after --frame-start (diagnostic use).",
    )
    parser.add_argument(
        "--zero-channel",
        action="append",
        default=[],
        help="Set one source coefficient to zero for render diagnostics; may be repeated.",
    )
    parser.add_argument(
        "--eye-wide-max",
        type=float,
        default=None,
        help="Optional explicit cap for eyeWideLeft/Right after the selected safety profile.",
    )
    return parser.parse_args(argv)


def parse_streams(values: list[str]) -> list[tuple[str, pathlib.Path]]:
    streams = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Stream must use stream_id=path syntax: {value!r}")
        stream_id, path = value.split("=", 1)
        if not stream_id or not stream_id.replace("_", "").isalnum():
            raise ValueError(f"Invalid stream id: {stream_id!r}")
        streams.append((stream_id, pathlib.Path(path).resolve()))
    if len({stream_id for stream_id, _ in streams}) != len(streams):
        raise ValueError("Duplicate stream ids")
    return streams


def find_key(key_blocks: bpy.types.KeyBlocks, target: str) -> bpy.types.ShapeKey | None:
    for key in key_blocks:
        if key.name == target or key.name.endswith("." + target):
            return key
    return None


def reset_arkit(key_blocks: bpy.types.KeyBlocks) -> None:
    for key in key_blocks:
        if key.name.startswith("AK_") or ".AK_" in key.name:
            key.value = 0.0


def apply_eye_safety(
    source_names: list[str],
    weights: list[float],
    profile: str,
    explicit_eye_wide_max: float | None = None,
) -> tuple[list[float], list[dict]]:
    """Keep Rocketbox's eyeWide morph below its mesh-folding threshold."""
    adjusted = [float(value) for value in weights]
    if explicit_eye_wide_max is not None:
        wide_scale = 1.0
        wide_max = explicit_eye_wide_max
    elif profile == "rocketbox_safe":
        # A/B renders at 0.1/0.2/0.3/0.4 show the detached dark upper-lid
        # crease beginning above 0.2 on the verified Rocketbox mesh.  Scaling
        # preserves the source eye-wide dynamics instead of saturating nearly
        # every captured frame at the safety limit.
        wide_scale = 0.25
        wide_max = 0.20
    elif profile == "raw":
        return adjusted, []
    else:
        raise RuntimeError(f"Unsupported eye-safety profile: {profile}")

    values = dict(zip(source_names, adjusted))
    original = dict(values)
    for side in EYE_SIDES:
        wide_name = f"eyeWide{side}"
        values[wide_name] = min(max(values[wide_name], 0.0) * wide_scale, wide_max)

    changes = []
    for index, name in enumerate(source_names):
        adjusted[index] = values[name]
        if abs(values[name] - original[name]) > 1e-9:
            changes.append({
                "channel": name,
                "raw": original[name],
                "applied": values[name],
            })
    return adjusted, changes


def rotation_only(matrix: Matrix) -> Matrix:
    return matrix.to_quaternion().to_matrix()


def scaled_shortest_quaternion(matrix: Matrix, scale: float, max_degrees: float) -> Quaternion:
    quaternion = matrix.to_quaternion()
    if quaternion.w < 0.0:
        quaternion.negate()
    angle = quaternion.angle
    if angle < 1e-8:
        return Quaternion((1.0, 0.0, 0.0, 0.0))
    applied_angle = min(angle * scale, math.radians(max_degrees))
    return Quaternion(quaternion.axis, applied_angle)


def head_driver() -> dict | None:
    armatures = [obj for obj in bpy.data.objects if obj.type == "ARMATURE"]
    if len(armatures) != 1:
        return None
    armature = armatures[0]
    pose_bone = armature.pose.bones.get("Bip01 Head")
    if pose_bone is None:
        return None
    pose_bone.rotation_mode = "QUATERNION"
    armature_world_rotation = rotation_only(armature.matrix_world.to_3x3())
    rest_rotation = rotation_only(pose_bone.bone.matrix_local.to_3x3())
    return {
        "armature": armature,
        "bone": pose_bone,
        "armature_world_rotation": armature_world_rotation,
        "rest_rotation": rest_rotation,
    }


def reset_head(driver: dict | None) -> None:
    if driver is not None:
        driver["bone"].rotation_quaternion = Quaternion((1.0, 0.0, 0.0, 0.0))


def apply_head_rotation(
    driver: dict | None,
    flattened_bvh_matrix: list[float] | None,
    scale: float,
    max_degrees: float,
) -> float:
    reset_head(driver)
    if driver is None or flattened_bvh_matrix is None:
        return 0.0
    if len(flattened_bvh_matrix) != 9:
        raise RuntimeError(f"Expected 9 head-rotation matrix values, got {len(flattened_bvh_matrix)}")
    bvh_rotation = Matrix((
        flattened_bvh_matrix[0:3],
        flattened_bvh_matrix[3:6],
        flattened_bvh_matrix[6:9],
    ))
    # BVH: X right, Y up, Z forward. Blender scene: X right, Z up, -Y forward.
    bvh_to_blender = Matrix(((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)))
    world_delta = bvh_to_blender @ bvh_rotation @ bvh_to_blender.transposed()
    world_quaternion = scaled_shortest_quaternion(world_delta, scale, max_degrees)
    world_delta = world_quaternion.to_matrix()
    armature_world_rotation = driver["armature_world_rotation"]
    armature_delta = (
        armature_world_rotation.inverted() @ world_delta @ armature_world_rotation
    )
    rest_rotation = driver["rest_rotation"]
    bone_local_delta = rest_rotation.inverted() @ armature_delta @ rest_rotation
    driver["bone"].rotation_quaternion = bone_local_delta.to_quaternion()
    return math.degrees(world_quaternion.angle)


def main() -> None:
    args = parse_args()
    streams = parse_streams(args.stream)
    mapping = json.loads(args.mapping.resolve().read_text(encoding="utf-8"))
    source_names = [entry["source"] for entry in mapping["channels"]]
    target_names = [entry["target"] for entry in mapping["channels"]]
    if args.frame_start < 0:
        raise ValueError("--frame-start must be non-negative")
    if args.frame_count is not None and args.frame_count <= 0:
        raise ValueError("--frame-count must be positive")
    unknown_zero_channels = sorted(set(args.zero_channel) - set(source_names))
    if unknown_zero_channels:
        raise ValueError(f"Unknown --zero-channel values: {unknown_zero_channels}")
    zero_channel_indices = {source_names.index(name) for name in args.zero_channel}
    if args.eye_wide_max is not None and not 0.0 <= args.eye_wide_max <= 1.0:
        raise ValueError("--eye-wide-max must be in [0, 1]")
    if args.eye_wide_max is not None:
        eye_wide_transfer = {"scale": 1.0, "maximum": args.eye_wide_max}
    elif args.eye_safety_profile == "rocketbox_safe":
        eye_wide_transfer = {"scale": 0.25, "maximum": 0.20}
    else:
        eye_wide_transfer = None

    meshes = [obj for obj in bpy.data.objects if obj.type == "MESH" and obj.data.shape_keys]
    if len(meshes) != 1:
        raise RuntimeError(f"Expected one shape-key mesh, found {[obj.name for obj in meshes]}")
    mesh = meshes[0]
    key_blocks = mesh.data.shape_keys.key_blocks
    target_keys = [find_key(key_blocks, target) for target in target_names]
    missing = [target for target, key in zip(target_names, target_keys) if key is None]
    if missing:
        raise RuntimeError(f"Missing mapped shape keys: {missing}")
    driver = head_driver()

    scene = bpy.context.scene
    supported_variants = {
        "texture_free_clay",
        "clean_clay_eye_cards",
        "clean_clay_textured_eyes",
        "clean_clay_complete_eyes",
    }
    if scene.get("diffsheg_render_variant") not in supported_variants:
        raise RuntimeError("Input .blend is not a verified Rocketbox clay variant")
    scene.render.resolution_x = args.resolution
    scene.render.resolution_y = args.resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.fps = args.fps

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    expected_frame_count = None
    for stream_id, json_path in streams:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        if payload.get("names") != source_names:
            raise RuntimeError(f"Coefficient order mismatch in {json_path}")
        frames = payload.get("frames", [])
        if not frames:
            raise RuntimeError(f"No frames in {json_path}")
        source_frame_count = len(frames)
        stop = source_frame_count if args.frame_count is None else args.frame_start + args.frame_count
        selected_frames = frames[args.frame_start : min(stop, source_frame_count)]
        if not selected_frames:
            raise RuntimeError(
                f"Selected frame range is empty for {json_path}: "
                f"start={args.frame_start}, count={args.frame_count}, source={source_frame_count}"
            )
        if expected_frame_count is None:
            expected_frame_count = len(selected_frames)
        elif len(selected_frames) != expected_frame_count:
            raise RuntimeError(
                f"Selected frame-count mismatch in {json_path}: "
                f"{len(selected_frames)} != {expected_frame_count}"
            )

        frame_dir = output_dir / stream_id
        frame_dir.mkdir(parents=True, exist_ok=True)
        clipped_low = 0
        clipped_high = 0
        raw_min = float("inf")
        raw_max = float("-inf")
        head_pose_frames = 0
        applied_head_angles = []
        eye_adjusted_frames = 0
        eye_adjusted_values = 0
        eye_max_absolute_adjustment = 0.0
        eye_channel_adjustments: dict[str, int] = {}
        for output_index, frame in enumerate(selected_frames):
            source_frame_index = args.frame_start + output_index
            weights = frame.get("weights", [])
            if len(weights) != len(target_keys):
                raise RuntimeError(
                    f"Expected {len(target_keys)} weights in {json_path} "
                    f"frame {source_frame_index}, got {len(weights)}"
                )
            applied_weights, eye_changes = apply_eye_safety(
                source_names,
                weights,
                args.eye_safety_profile,
                args.eye_wide_max,
            )
            for index in zero_channel_indices:
                applied_weights[index] = 0.0
            eye_adjusted_frames += int(bool(eye_changes))
            eye_adjusted_values += len(eye_changes)
            for change in eye_changes:
                eye_max_absolute_adjustment = max(
                    eye_max_absolute_adjustment,
                    abs(change["raw"] - change["applied"]),
                )
                channel = change["channel"]
                eye_channel_adjustments[channel] = eye_channel_adjustments.get(channel, 0) + 1

            scene.frame_set(source_frame_index + 1)
            reset_arkit(key_blocks)
            for key, raw_value, applied_value in zip(target_keys, weights, applied_weights):
                value = float(raw_value)
                raw_min = min(raw_min, value)
                raw_max = max(raw_max, value)
                clipped_low += int(value < 0.0)
                clipped_high += int(value > 1.0)
                key.value = min(max(float(applied_value), 0.0), 1.0)
            head_matrix = frame.get("head_rotation_matrix_bvh")
            head_pose_frames += int(head_matrix is not None)
            applied_head_angles.append(
                apply_head_rotation(
                    driver,
                    head_matrix,
                    args.head_motion_scale,
                    args.head_motion_max_degrees,
                )
            )
            scene.render.filepath = str(frame_dir / f"frame_{source_frame_index:05d}.png")
            bpy.ops.render.render(write_still=True)

        reports.append({
            "stream_id": stream_id,
            "source": str(json_path),
            "source_frame_count": source_frame_count,
            "frame_start": args.frame_start,
            "frame_count": len(selected_frames),
            "raw_min": raw_min,
            "raw_max": raw_max,
            "clipped_low_values": clipped_low,
            "clipped_high_values": clipped_high,
            "head_pose_frames": head_pose_frames,
            "eye_safety": {
                "profile": args.eye_safety_profile,
                "adjusted_frames": eye_adjusted_frames,
                "adjusted_values": eye_adjusted_values,
                "max_absolute_adjustment": eye_max_absolute_adjustment,
                "channel_adjustments": eye_channel_adjustments,
            },
            "applied_head_angle_degrees": {
                "minimum": min(applied_head_angles),
                "maximum": max(applied_head_angles),
                "mean": sum(applied_head_angles) / len(applied_head_angles),
            },
            "frame_dir": str(frame_dir),
        })

    reset_arkit(key_blocks)
    reset_head(driver)
    report = {
        "blender_version": bpy.app.version_string,
        "input_blend": bpy.data.filepath,
        "render_variant": scene.get("diffsheg_render_variant"),
        "fps": args.fps,
        "resolution": [args.resolution, args.resolution],
        "frame_count": expected_frame_count,
        "duration_seconds": expected_frame_count / args.fps,
        "mapped_coefficients": len(target_keys),
        "head_driver": driver["bone"].name if driver else None,
        "head_motion_scale": args.head_motion_scale,
        "head_motion_max_degrees": args.head_motion_max_degrees,
        "eye_safety_profile": args.eye_safety_profile,
        "eye_wide_max": args.eye_wide_max,
        "eye_wide_transfer": eye_wide_transfer,
        "frame_start": args.frame_start,
        "zero_channels": args.zero_channel,
        "streams": reports,
    }
    args.report.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.report.resolve().write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("DIFFSHEG_EXPRESSION_RENDER=" + json.dumps({
        "streams": len(reports),
        "frames_per_stream": expected_frame_count,
        "duration_seconds": report["duration_seconds"],
        "report": str(args.report.resolve()),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
