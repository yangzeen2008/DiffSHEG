"""Prepare and verify the Rocketbox ARKit avatar with Blender in background mode.

Run with Blender, not the system Python::

    blender --background --factory-startup \
      --python assets/prepare_rocketbox_blender_scene.py -- \
      --asset-dir assets/face_models/microsoft_rocketbox_female_adult_01_arkit \
      --output-blend assets/face_models/microsoft_rocketbox_female_adult_01_arkit/rocketbox_arkit_ready.blend \
      --report bvh_output/face_model_validation/rocketbox_import_report.json \
      --preview-dir bvh_output/face_model_validation/orientation
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import re
import sys

import bpy
from mathutils import Vector


ARKIT_PATTERN = re.compile(r"(?:^|\.)AK_(\d{2})_([A-Za-z]+)$")


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-dir", type=pathlib.Path, required=True)
    parser.add_argument("--output-blend", type=pathlib.Path, required=True)
    parser.add_argument("--report", type=pathlib.Path, required=True)
    parser.add_argument("--preview-dir", type=pathlib.Path, required=True)
    return parser.parse_args(argv)


def clear_factory_scene() -> None:
    # This script always runs with --factory-startup in an isolated process.
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection in list(bpy.data.collections):
        if collection.users == 0:
            bpy.data.collections.remove(collection)


def scene_bounds(mesh_objects: list[bpy.types.Object]) -> tuple[Vector, Vector]:
    corners = [obj.matrix_world @ Vector(corner) for obj in mesh_objects for corner in obj.bound_box]
    if not corners:
        raise RuntimeError("Imported FBX contains no mesh bounds")
    low = Vector((min(p.x for p in corners), min(p.y for p in corners), min(p.z for p in corners)))
    high = Vector((max(p.x for p in corners), max(p.y for p in corners), max(p.z for p in corners)))
    return low, high


def point_at(obj: bpy.types.Object, target: Vector) -> None:
    obj.rotation_euler = (target - obj.location).to_track_quat("-Z", "Y").to_euler()


def add_area_light(name: str, location: Vector, target: Vector, energy: float, size: float) -> None:
    data = bpy.data.lights.new(name=name, type="AREA")
    data.energy = energy
    data.shape = "DISK"
    data.size = size
    obj = bpy.data.objects.new(name=name, object_data=data)
    bpy.context.scene.collection.objects.link(obj)
    obj.location = location
    point_at(obj, target)


def prepare_rendering(low: Vector, high: Vector) -> bpy.types.Object:
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = 640
    scene.render.resolution_y = 640
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    scene.world.color = (0.025, 0.025, 0.025)

    if scene.world.use_nodes:
        background = scene.world.node_tree.nodes.get("Background")
        if background:
            background.inputs["Color"].default_value = (0.025, 0.025, 0.025, 1.0)
            background.inputs["Strength"].default_value = 0.15

    scene.view_settings.exposure = -1.25

    size = high - low
    height = max(size.z, 1e-3)
    center = (low + high) * 0.5
    face_target = Vector((center.x, center.y, high.z - 0.12 * height))
    light_energy = 90.0 * max(height * height, 1.0)
    add_area_light(
        "Key_Light",
        face_target + Vector((0.45 * height, -0.50 * height, 0.25 * height)),
        face_target,
        light_energy,
        0.35 * height,
    )
    add_area_light(
        "Fill_Light",
        face_target + Vector((-0.40 * height, -0.30 * height, 0.10 * height)),
        face_target,
        light_energy * 0.45,
        0.45 * height,
    )
    add_area_light(
        "Rim_Light",
        face_target + Vector((0.0, 0.45 * height, 0.30 * height)),
        face_target,
        light_energy * 0.70,
        0.30 * height,
    )

    camera_data = bpy.data.cameras.new("Face_Camera")
    camera = bpy.data.objects.new("Face_Camera", camera_data)
    bpy.context.scene.collection.objects.link(camera)
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = 0.34 * height
    camera_data.lens = 70
    camera_data.clip_start = max(height * 0.001, 0.001)
    camera_data.clip_end = max(height * 10.0, 100.0)
    scene.camera = camera
    return camera


def render_orientations(
    camera: bpy.types.Object,
    low: Vector,
    high: Vector,
    preview_dir: pathlib.Path,
) -> list[str]:
    preview_dir.mkdir(parents=True, exist_ok=True)
    size = high - low
    height = max(size.z, 1e-3)
    center = (low + high) * 0.5
    target = Vector((center.x, center.y, high.z - 0.12 * height))
    radius = 0.58 * height
    directions = {
        "camera_neg_y": Vector((0.0, -1.0, 0.0)),
        "camera_pos_y": Vector((0.0, 1.0, 0.0)),
        "camera_pos_x": Vector((1.0, 0.0, 0.0)),
        "camera_neg_x": Vector((-1.0, 0.0, 0.0)),
    }
    outputs = []
    for name, direction in directions.items():
        camera.location = target + radius * direction
        point_at(camera, target)
        output = (preview_dir / f"{name}.png").resolve()
        bpy.context.scene.render.filepath = str(output)
        bpy.ops.render.render(write_still=True)
        outputs.append(str(output))
    # Rocketbox faces the -Y camera. Leave the saved scene at this canonical
    # front view instead of at the final orientation-diagnostic position.
    camera.location = target + radius * directions["camera_neg_y"]
    point_at(camera, target)
    front_output = (preview_dir.parent / "neutral_front.png").resolve()
    bpy.context.scene.render.filepath = str(front_output)
    bpy.ops.render.render(write_still=True)
    outputs.append(str(front_output))
    return outputs


def main() -> None:
    args = parse_args()
    asset_dir = args.asset_dir.resolve()
    fbx_path = asset_dir / "Female_Adult_01_facial.fbx"
    mapping_path = asset_dir / "arkit_channel_map.json"
    if not fbx_path.is_file():
        raise FileNotFoundError(fbx_path)
    if not mapping_path.is_file():
        raise FileNotFoundError(mapping_path)

    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    expected_targets = [entry["target"] for entry in mapping["channels"]]
    clear_factory_scene()

    before_objects = set(bpy.data.objects)
    result = bpy.ops.import_scene.fbx(
        filepath=str(fbx_path),
        use_anim=False,
        use_image_search=True,
    )
    imported_objects = [obj for obj in bpy.data.objects if obj not in before_objects]
    mesh_objects = [obj for obj in imported_objects if obj.type == "MESH"]
    armatures = [obj for obj in imported_objects if obj.type == "ARMATURE"]
    if not mesh_objects:
        raise RuntimeError("FBX import completed without mesh objects")

    shape_key_objects = []
    all_shape_keys = []
    for obj in mesh_objects:
        key_data = getattr(obj.data, "shape_keys", None)
        key_names = [key.name for key in key_data.key_blocks] if key_data else []
        if key_names:
            shape_key_objects.append({"object": obj.name, "count": len(key_names), "keys": key_names})
            all_shape_keys.extend(key_names)

    arkit_keys = sorted(
        {name for name in all_shape_keys if ARKIT_PATTERN.search(name)},
        key=lambda name: int(ARKIT_PATTERN.search(name).group(1)),
    )
    matched_targets = {
        target: next((name for name in all_shape_keys if name == target or name.endswith("." + target)), None)
        for target in expected_targets
    }
    missing_targets = [target for target, actual in matched_targets.items() if actual is None]

    low, high = scene_bounds(mesh_objects)
    camera = prepare_rendering(low, high)
    previews = render_orientations(camera, low, high, args.preview_dir.resolve())

    args.output_blend.resolve().parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(args.output_blend.resolve()), check_existing=False)

    image_files = []
    for image in bpy.data.images:
        if image.source == "FILE":
            image_files.append({
                "name": image.name,
                "filepath": bpy.path.abspath(image.filepath),
                "packed": image.packed_file is not None,
            })

    report = {
        "blender_version": bpy.app.version_string,
        "fbx": str(fbx_path),
        "import_result": sorted(result),
        "output_blend": str(args.output_blend.resolve()),
        "imported_object_count": len(imported_objects),
        "mesh_count": len(mesh_objects),
        "armature_count": len(armatures),
        "material_count": len(bpy.data.materials),
        "image_count": len(image_files),
        "images": image_files,
        "shape_key_objects": shape_key_objects,
        "arkit_shape_key_count": len(arkit_keys),
        "arkit_shape_keys": arkit_keys,
        "diffsheg_target_count": len(expected_targets),
        "matched_target_count": len(expected_targets) - len(missing_targets),
        "missing_targets": missing_targets,
        "target_mapping": matched_targets,
        "bounds": {"low": list(low), "high": list(high), "size": list(high - low)},
        "orientation_previews": previews,
    }
    args.report.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.report.resolve().write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("DIFFSHEG_ROCKETBOX_REPORT=" + json.dumps({
        "arkit": len(arkit_keys),
        "matched": report["matched_target_count"],
        "missing": missing_targets,
        "images": len(image_files),
        "blend": report["output_blend"],
        "report": str(args.report.resolve()),
    }, ensure_ascii=False))
    if missing_targets:
        raise RuntimeError(f"Missing DiffSHEG targets: {missing_targets}")


if __name__ == "__main__":
    main()
