"""Inspect Rocketbox rig, material regions, and opacity-card components."""

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
    parser.add_argument("--output", type=pathlib.Path, required=True)
    return parser.parse_args(argv)


def vector(values) -> list[float]:
    return [float(value) for value in values]


def material_report(material: bpy.types.Material | None) -> dict | None:
    if material is None:
        return None
    entry = {
        "name": material.name,
        "use_nodes": material.use_nodes,
        "surface_render_method": getattr(material, "surface_render_method", None),
        "nodes": [],
        "links": [],
    }
    if not material.use_nodes or material.node_tree is None:
        return entry
    for node in material.node_tree.nodes:
        node_entry = {
            "name": node.name,
            "label": node.label,
            "type": node.type,
        }
        if node.type == "TEX_IMAGE":
            node_entry["image"] = node.image.name if node.image else None
            node_entry["image_filepath"] = (
                bpy.path.abspath(node.image.filepath) if node.image else None
            )
            node_entry["interpolation"] = node.interpolation
            node_entry["extension"] = node.extension
        entry["nodes"].append(node_entry)
    entry["links"] = [
        {
            "from_node": link.from_node.name,
            "from_socket": link.from_socket.name,
            "to_node": link.to_node.name,
            "to_socket": link.to_socket.name,
        }
        for link in material.node_tree.links
    ]
    return entry


def opacity_components(mesh_object: bpy.types.Object, material_index: int) -> list[dict]:
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
                "vertices": len(component_vertices),
                "polygons": len(component_polygons),
                "bounds_low": vector(low),
                "bounds_high": vector(high),
                "center": vector((low + high) * 0.5),
                "size": vector(high - low),
            }
        )
    components.sort(key=lambda item: (-item["polygons"], item["center"][2]))
    return components


def main() -> None:
    args = parse_args()
    meshes = [obj for obj in bpy.data.objects if obj.type == "MESH"]
    armatures = [obj for obj in bpy.data.objects if obj.type == "ARMATURE"]
    report = {
        "blend": bpy.data.filepath,
        "blender_version": bpy.app.version_string,
        "objects": [],
        "armatures": [],
    }
    for obj in meshes:
        material_slots = [slot.material.name if slot.material else None for slot in obj.material_slots]
        polygon_counts = {
            str(index): sum(1 for polygon in obj.data.polygons if polygon.material_index == index)
            for index in range(len(material_slots))
        }
        entry = {
            "name": obj.name,
            "parent": obj.parent.name if obj.parent else None,
            "parent_type": obj.parent_type,
            "location": vector(obj.location),
            "rotation_mode": obj.rotation_mode,
            "rotation_euler": vector(obj.rotation_euler),
            "scale": vector(obj.scale),
            "matrix_world": [vector(row) for row in obj.matrix_world],
            "vertex_count": len(obj.data.vertices),
            "polygon_count": len(obj.data.polygons),
            "materials": material_slots,
            "material_details": [material_report(slot.material) for slot in obj.material_slots],
            "material_polygon_counts": polygon_counts,
            "modifiers": [
                {
                    "name": modifier.name,
                    "type": modifier.type,
                    "object": getattr(modifier, "object", None).name
                    if getattr(modifier, "object", None)
                    else None,
                }
                for modifier in obj.modifiers
            ],
        }
        opacity_indices = [
            index for index, name in enumerate(material_slots) if name and "opacity" in name.lower()
        ]
        entry["opacity_material_indices"] = opacity_indices
        entry["opacity_components"] = [
            {"material_index": index, "components": opacity_components(obj, index)}
            for index in opacity_indices
        ]
        entry["material_components"] = [
            {
                "material_index": index,
                "material_name": name,
                "components": opacity_components(obj, index),
            }
            for index, name in enumerate(material_slots)
        ]
        report["objects"].append(entry)

    for armature in armatures:
        world_matrix = armature.matrix_world
        report["armatures"].append(
            {
                "name": armature.name,
                "location": vector(armature.location),
                "rotation_mode": armature.rotation_mode,
                "rotation_euler": vector(armature.rotation_euler),
                "scale": vector(armature.scale),
                "matrix_world": [vector(row) for row in world_matrix],
                "bones": [
                    {
                        "name": bone.name,
                        "parent": bone.parent.name if bone.parent else None,
                        "head_local": vector(bone.head_local),
                        "tail_local": vector(bone.tail_local),
                        "head_world": vector(world_matrix @ bone.head_local),
                        "tail_world": vector(world_matrix @ bone.tail_local),
                    }
                    for bone in armature.data.bones
                ],
            }
        )

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("DIFFSHEG_ROCKETBOX_INSPECTION=" + json.dumps({
        "output": str(output),
        "meshes": len(meshes),
        "armatures": len(armatures),
    }))


if __name__ == "__main__":
    main()
