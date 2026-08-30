"""Create a texture-free clay variant from the verified Rocketbox scene."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import defaultdict, deque

import bpy
from mathutils import Vector


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping", type=pathlib.Path, required=True)
    parser.add_argument("--output-blend", type=pathlib.Path, required=True)
    parser.add_argument("--preview-dir", type=pathlib.Path, required=True)
    parser.add_argument("--report", type=pathlib.Path, required=True)
    parser.add_argument(
        "--keep-eye-cards",
        action="store_true",
        help="Keep only the original opacity-card components around the eyes/eyelashes.",
    )
    parser.add_argument(
        "--keep-eye-texture",
        action="store_true",
        help="Keep the original textured material only on the two disconnected eyeball meshes.",
    )
    return parser.parse_args(argv)


def make_principled_material(name: str, color: tuple[float, float, float, float]) -> bpy.types.Material:
    material = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    shader = nodes.new("ShaderNodeBsdfPrincipled")
    shader.inputs["Base Color"].default_value = color
    shader.inputs["Metallic"].default_value = 0.0
    shader.inputs["Roughness"].default_value = 0.58
    shader.inputs["IOR"].default_value = 1.45
    material.node_tree.links.new(shader.outputs["BSDF"], output.inputs["Surface"])
    return material


def make_hidden_material(name: str) -> bpy.types.Material:
    """Hide non-face geometry without retaining any image texture dependency."""
    material = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    transparent = nodes.new("ShaderNodeBsdfTransparent")
    material.node_tree.links.new(transparent.outputs["BSDF"], output.inputs["Surface"])
    if hasattr(material, "surface_render_method"):
        material.surface_render_method = "DITHERED"
    return material


def connected_material_components(mesh_object: bpy.types.Object, material_index: int) -> list[dict]:
    """Return connected polygon islands for one material slot in mesh-local space."""
    mesh = mesh_object.data
    polygons = [polygon for polygon in mesh.polygons if polygon.material_index == material_index]
    vertex_to_polygons: dict[int, list[int]] = defaultdict(list)
    polygon_vertices: dict[int, set[int]] = {}
    for polygon in polygons:
        vertices = set(polygon.vertices)
        polygon_vertices[polygon.index] = vertices
        for vertex in vertices:
            vertex_to_polygons[vertex].append(polygon.index)

    remaining = set(polygon_vertices)
    components = []
    while remaining:
        seed = remaining.pop()
        queue = deque([seed])
        component_polygons = {seed}
        component_vertices = set(polygon_vertices[seed])
        while queue:
            polygon_index = queue.popleft()
            for vertex in polygon_vertices[polygon_index]:
                for neighbour in vertex_to_polygons[vertex]:
                    if neighbour in remaining:
                        remaining.remove(neighbour)
                        component_polygons.add(neighbour)
                        component_vertices.update(polygon_vertices[neighbour])
                        queue.append(neighbour)
        coordinates = [mesh.vertices[index].co for index in component_vertices]
        low = Vector((min(value[axis] for value in coordinates) for axis in range(3)))
        high = Vector((max(value[axis] for value in coordinates) for axis in range(3)))
        components.append(
            {
                "polygons": sorted(component_polygons),
                "center": (low + high) * 0.5,
                "size": high - low,
            }
        )
    return components


def keep_only_eye_cards(mesh: bpy.types.Object, opacity_index: int) -> dict:
    """Split eye/eyelash cards from the combined hair and opacity material."""
    original = mesh.material_slots[opacity_index].material
    if original is None:
        raise RuntimeError("Rocketbox opacity slot has no material")
    components = connected_material_components(mesh, opacity_index)
    eye_components = [
        component
        for component in components
        if component["center"].y < 0.0 and component["center"].z > 158.0
    ]
    if len(eye_components) != 8:
        raise RuntimeError(
            f"Expected 8 Rocketbox eye-card islands, found {len(eye_components)}; refusing ambiguous split"
        )
    eye_material = original.copy()
    eye_material.name = "Clay_Eye_Eyelash_Cards"
    mesh.data.materials.append(eye_material)
    eye_material_index = len(mesh.material_slots) - 1
    eye_polygon_indices = {
        polygon_index
        for component in eye_components
        for polygon_index in component["polygons"]
    }
    for polygon_index in eye_polygon_indices:
        mesh.data.polygons[polygon_index].material_index = eye_material_index
    return {
        "material": eye_material,
        "component_count": len(eye_components),
        "polygon_count": len(eye_polygon_indices),
        "material_index": eye_material_index,
        "components": [
            {
                "polygons": len(component["polygons"]),
                "center": list(component["center"]),
                "size": list(component["size"]),
            }
            for component in eye_components
        ],
    }


def keep_only_eye_geometry(mesh: bpy.types.Object, head_index: int) -> dict:
    """Split the two eyeballs from the shared head material before making the face clay."""
    original = mesh.material_slots[head_index].material
    if original is None:
        raise RuntimeError("Rocketbox head slot has no material")
    components = connected_material_components(mesh, head_index)
    eye_components = [
        component
        for component in components
        if component["center"].y < 0.0
        and component["center"].z > 159.0
        and component["size"].x < 4.0
        and component["size"].y < 3.0
    ]
    if len(eye_components) != 2:
        raise RuntimeError(
            f"Expected 2 Rocketbox eyeball islands, found {len(eye_components)}; refusing ambiguous split"
        )
    eye_material = original.copy()
    eye_material.name = "Clay_Textured_Eyeballs"
    mesh.data.materials.append(eye_material)
    eye_material_index = len(mesh.material_slots) - 1
    eye_polygon_indices = {
        polygon_index
        for component in eye_components
        for polygon_index in component["polygons"]
    }
    for polygon_index in eye_polygon_indices:
        mesh.data.polygons[polygon_index].material_index = eye_material_index
    return {
        "material": eye_material,
        "component_count": len(eye_components),
        "polygon_count": len(eye_polygon_indices),
        "material_index": eye_material_index,
        "components": [
            {
                "polygons": len(component["polygons"]),
                "center": list(component["center"]),
                "size": list(component["size"]),
            }
            for component in eye_components
        ],
    }


def find_key(key_blocks: bpy.types.KeyBlocks, target: str) -> bpy.types.ShapeKey | None:
    for key in key_blocks:
        if key.name == target or key.name.endswith("." + target):
            return key
    return None


def point_at(obj: bpy.types.Object, target: Vector) -> None:
    obj.rotation_euler = (target - obj.location).to_track_quat("-Z", "Y").to_euler()


def configure_depth_lighting(scene: bpy.types.Scene, mesh: bpy.types.Object) -> None:
    corners = [mesh.matrix_world @ Vector(corner) for corner in mesh.bound_box]
    low = Vector((min(p.x for p in corners), min(p.y for p in corners), min(p.z for p in corners)))
    high = Vector((max(p.x for p in corners), max(p.y for p in corners), max(p.z for p in corners)))
    height = max(high.z - low.z, 1e-3)
    center = (low + high) * 0.5
    target = Vector((center.x, center.y, high.z - 0.12 * height))

    if scene.world and scene.world.use_nodes:
        background = scene.world.node_tree.nodes.get("Background")
        if background:
            background.inputs["Strength"].default_value = 0.008

    lighting = {
        "Key_Light": {
            "location": target + Vector((-0.36 * height, -0.44 * height, 0.28 * height)),
            "energy": 70.0 * height * height,
            "size": 0.16 * height,
        },
        "Fill_Light": {
            "location": target + Vector((0.32 * height, -0.26 * height, 0.06 * height)),
            "energy": 5.0 * height * height,
            "size": 0.30 * height,
        },
        "Rim_Light": {
            "location": target + Vector((0.20 * height, 0.36 * height, 0.22 * height)),
            "energy": 18.0 * height * height,
            "size": 0.18 * height,
        },
    }
    for name, settings in lighting.items():
        light = bpy.data.objects.get(name)
        if light is None or light.type != "LIGHT":
            raise RuntimeError(f"Verified scene is missing light {name!r}")
        light.location = settings["location"]
        light.data.energy = settings["energy"]
        light.data.size = settings["size"]
        point_at(light, target)

    camera = scene.camera
    if camera is None:
        raise RuntimeError("Verified Rocketbox scene has no face camera")
    camera.data.type = "PERSP"
    camera.data.lens = 85.0
    camera.location = target + Vector((0.0, -0.58 * height, 0.01 * height))
    point_at(camera, target)


def reset_arkit(key_blocks: bpy.types.KeyBlocks) -> None:
    for key in key_blocks:
        if ".AK_" in key.name or key.name.startswith("AK_"):
            key.value = 0.0


def main() -> None:
    args = parse_args()
    mapping = json.loads(args.mapping.resolve().read_text(encoding="utf-8"))
    target_names = [entry["target"] for entry in mapping["channels"]]

    meshes = [obj for obj in bpy.data.objects if obj.type == "MESH" and obj.data.shape_keys]
    if len(meshes) != 1:
        raise RuntimeError(f"Expected exactly one shape-key mesh, found {[obj.name for obj in meshes]}")
    mesh = meshes[0]
    key_blocks = mesh.data.shape_keys.key_blocks
    missing = [name for name in target_names if find_key(key_blocks, name) is None]
    if missing:
        raise RuntimeError(f"Missing mapped shape keys: {missing}")

    original_material_names = [slot.material.name if slot.material else "" for slot in mesh.material_slots]
    if original_material_names != ["f001_body", "f001_head", "f001_opacity"]:
        raise RuntimeError(f"Unexpected Rocketbox material slots: {original_material_names}")
    eye_geometry_split = keep_only_eye_geometry(mesh, 1) if args.keep_eye_texture else None
    eye_card_split = keep_only_eye_cards(mesh, 2) if args.keep_eye_cards else None

    hidden_body = make_hidden_material("Clay_Hidden_Body")
    head = make_principled_material("Clay_Head", (0.12, 0.12, 0.12, 1.0))
    hidden_cards = make_hidden_material("Clay_Hidden_Cards")
    replacements = {
        "f001_body": hidden_body,
        "f001_head": head,
        "f001_opacity": hidden_cards,
    }
    replaced_slots = {}
    for slot in list(mesh.material_slots)[:3]:
        original = slot.material.name if slot.material else ""
        if original not in replacements:
            raise RuntimeError(f"Unexpected Rocketbox material slot: {original!r}")
        slot.material = replacements[original]
        replaced_slots[original] = slot.material.name

    active_materials = {hidden_body, head, hidden_cards}
    if eye_card_split:
        active_materials.add(eye_card_split["material"])
    if eye_geometry_split:
        active_materials.add(eye_geometry_split["material"])
    for material in list(bpy.data.materials):
        if material not in active_materials and material.users == 0:
            bpy.data.materials.remove(material)
    for image in list(bpy.data.images):
        if image.users == 0:
            bpy.data.images.remove(image)

    scene = bpy.context.scene
    if args.keep_eye_cards and args.keep_eye_texture:
        scene["diffsheg_render_variant"] = "clean_clay_complete_eyes"
    elif args.keep_eye_cards:
        scene["diffsheg_render_variant"] = "clean_clay_eye_cards"
    elif args.keep_eye_texture:
        scene["diffsheg_render_variant"] = "clean_clay_textured_eyes"
    else:
        scene["diffsheg_render_variant"] = "texture_free_clay"
    scene["diffsheg_shape_key_count"] = 52
    scene["diffsheg_source_dimension"] = 51
    scene.render.resolution_x = 768
    scene.render.resolution_y = 768
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.view_settings.exposure = -1.35
    configure_depth_lighting(scene, mesh)

    presets = {
        "neutral": {},
        "smile": {
            "AK_07_CheekSquintLeft": 0.22,
            "AK_08_CheekSquintRight": 0.22,
            "AK_44_MouthSmileLeft": 0.78,
            "AK_45_MouthSmileRight": 0.78,
        },
        "jaw_open": {
            "AK_25_JawOpen": 0.68,
            "AK_32_MouthFunnel": 0.18,
        },
        "blink_brow": {
            "AK_03_BrowInnerUp": 0.35,
            "AK_09_EyeBlinkLeft": 0.88,
            "AK_10_EyeBlinkRight": 0.88,
        },
    }
    args.preview_dir.resolve().mkdir(parents=True, exist_ok=True)
    renders = []
    for preset_name, values in presets.items():
        reset_arkit(key_blocks)
        for target, value in values.items():
            find_key(key_blocks, target).value = value
        output = (args.preview_dir.resolve() / f"{preset_name}.png").resolve()
        scene.render.filepath = str(output)
        bpy.ops.render.render(write_still=True)
        renders.append({"preset": preset_name, "values": values, "output": str(output)})

    reset_arkit(key_blocks)
    source_blend = bpy.data.filepath
    args.output_blend.resolve().parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(args.output_blend.resolve()), check_existing=False)

    texture_nodes = []
    for material in active_materials:
        for node in material.node_tree.nodes:
            if node.type == "TEX_IMAGE":
                texture_nodes.append({"material": material.name, "node": node.name})
    report = {
        "blender_version": bpy.app.version_string,
        "source_blend": source_blend,
        "output_blend": str(args.output_blend.resolve()),
        "mesh": mesh.name,
        "shape_key_count": len(key_blocks),
        "diffsheg_mapped_count": len(target_names),
        "missing_targets": missing,
        "material_replacements": replaced_slots,
        "eye_card_split": {
            key: value
            for key, value in (eye_card_split or {}).items()
            if key != "material"
        } if eye_card_split else None,
        "eye_geometry_split": {
            key: value
            for key, value in (eye_geometry_split or {}).items()
            if key != "material"
        } if eye_geometry_split else None,
        "image_texture_node_count": len(texture_nodes),
        "image_texture_nodes": texture_nodes,
        "image_datablock_count": len(bpy.data.images),
        "file_image_datablock_count": sum(image.source == "FILE" for image in bpy.data.images),
        "viewer_image_datablock_count": sum(image.source == "VIEWER" for image in bpy.data.images),
        "visibility_policy": {
            "face": "neutral clay",
            "eyeballs": "original textured material" if eye_geometry_split else "neutral clay",
            "eye_eyelash_cards": "original alpha material" if eye_card_split else "hidden",
            "hair_body_cards": "hidden",
            "clothing_body": "hidden",
        },
        "render_setup": {
            "engine": scene.render.engine,
            "camera_type": scene.camera.data.type,
            "camera_lens_mm": scene.camera.data.lens,
            "camera_location": list(scene.camera.location),
            "exposure_ev": scene.view_settings.exposure,
            "head_base_color": list(head.node_tree.nodes.get("Principled BSDF").inputs["Base Color"].default_value),
            "head_roughness": head.node_tree.nodes.get("Principled BSDF").inputs["Roughness"].default_value,
            "lights": {
                name: {
                    "energy": bpy.data.objects[name].data.energy,
                    "size": bpy.data.objects[name].data.size,
                    "location": list(bpy.data.objects[name].location),
                }
                for name in ("Key_Light", "Fill_Light", "Rim_Light")
            },
        },
        "renders": renders,
    }
    args.report.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.report.resolve().write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("DIFFSHEG_CLAY_REPORT=" + json.dumps({
        "mapped": len(target_names),
        "missing": missing,
        "texture_nodes": len(texture_nodes),
        "blend": str(args.output_blend.resolve()),
        "report": str(args.report.resolve()),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
