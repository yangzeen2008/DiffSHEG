"""Render the quiet, typical, and energetic long validation comparisons."""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import tempfile

import bpy


ROOT = r"F:\study\DiffSHEG"
OUTPUT_ROOT = os.environ.get(
    "DIFFSHEG_LONG_OUTPUT_ROOT",
    os.path.join(ROOT, "bvh_output", "long_validation_visuals"),
)
MODEL_LABEL = os.environ.get("DIFFSHEG_MODEL_LABEL", "FM pck_best e479")
FFMPEG = r"C:\Users\yangz\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1-full_build\bin\ffmpeg.exe"
SEGMENTS = tuple(
    name.strip()
    for name in os.environ.get(
        "DIFFSHEG_RENDER_SEGMENTS", "quiet,typical,energetic"
    ).split(",")
    if name.strip()
)
SPACING = 150.0


def frame_count(path):
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip().startswith("Frames:"):
                return int(line.split(":", 1)[1].strip())
    raise RuntimeError(f"No frame count in {path}")


def offset_bvh(source, x_offset, destination):
    with open(source, "r", encoding="utf-8") as handle:
        lines = handle.readlines()
    motion_index = next(i for i, line in enumerate(lines) if line.strip() == "MOTION")
    data_start = motion_index + 3
    output = lines[:data_start]
    for line in lines[data_start:]:
        values = line.strip().split()
        if len(values) >= 6:
            values[0] = f"{float(values[0]) + x_offset:.6f}"
            output.append(" ".join(values) + "\n")
    with open(destination, "w", encoding="utf-8", newline="\n") as handle:
        handle.writelines(output)


def clear_scene_assets():
    for obj in list(bpy.data.objects):
        if obj.type in {"ARMATURE", "MESH", "FONT"}:
            bpy.data.objects.remove(obj, do_unlink=True)
    for action in list(bpy.data.actions):
        bpy.data.actions.remove(action)


def add_label(text, x_position, color):
    bpy.ops.object.text_add(location=(x_position, 0, -20))
    label = bpy.context.object
    label.data.body = text
    label.data.size = 11
    label.data.align_x = "CENTER"
    label.rotation_euler = (math.radians(90), 0, 0)
    label.color = color


def import_colored(path, name, palette, color):
    bpy.ops.import_anim.bvh(filepath=path)
    armature = bpy.context.object
    armature.name = name
    armature.show_in_front = True
    armature.color = color
    armature.data.show_bone_colors = True
    for bone in armature.pose.bones:
        bone.color.palette = palette
    return armature


def frame_view_on_armatures(armatures):
    for obj in bpy.context.selected_objects:
        obj.select_set(False)
    for armature in armatures:
        armature.select_set(True)
    bpy.context.view_layer.objects.active = armatures[0]
    view_areas = [area for area in bpy.context.screen.areas if area.type == "VIEW_3D"]
    if not view_areas:
        raise RuntimeError("No Blender 3D viewport is available for OpenGL rendering")
    area = view_areas[0]
    region = next(region for region in area.regions if region.type == "WINDOW")
    with bpy.context.temp_override(area=area, region=region):
        bpy.ops.view3d.view_axis(type="FRONT", align_active=False)
        bpy.ops.view3d.view_selected(use_all_regions=False)
    area.spaces[0].region_3d.view_distance *= 1.20
    return area


def label_filter(segment):
    label = segment.upper()
    return (
        "drawbox=x=0:y=0:w=iw/2:h=82:color=black@0.55:t=fill,"
        "drawbox=x=iw/2:y=0:w=iw/2:h=82:color=black@0.55:t=fill,"
        "drawbox=x=iw/2-2:y=0:w=4:h=ih:color=white@0.30:t=fill,"
        f"drawtext=fontfile='C\\:/Windows/Fonts/arialbd.ttf':text='{label} | GROUND TRUTH':"
        "fontcolor=0xFF6B1A:fontsize=32:x=(w/4-text_w/2):y=23,"
        f"drawtext=fontfile='C\\:/Windows/Fonts/arialbd.ttf':text='{label} | {MODEL_LABEL.upper()}':"
        "fontcolor=0x268DFF:fontsize=32:x=(3*w/4-text_w/2):y=23"
    )


def render_segment(segment):
    output_dir = os.path.join(OUTPUT_ROOT, segment)
    gt_bvh = os.path.join(output_dir, "ground_truth.bvh")
    pred_bvh = os.path.join(output_dir, "model_pck_best_e479.bvh")
    audio = os.path.join(output_dir, "audio.wav")
    output_video = os.path.join(output_dir, "comparison.mp4")
    for required in (gt_bvh, pred_bvh, audio, FFMPEG):
        if not os.path.exists(required):
            raise FileNotFoundError(required)

    scene = bpy.context.scene
    scene.render.fps = 15
    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1080
    scene.render.resolution_percentage = 100
    clear_scene_assets()
    temporary = tempfile.mkdtemp(prefix=f"diffsheg_{segment}_")
    try:
        gt_offset = os.path.join(temporary, "gt.bvh")
        pred_offset = os.path.join(temporary, "pred.bvh")
        offset_bvh(gt_bvh, -SPACING / 2, gt_offset)
        offset_bvh(pred_bvh, SPACING / 2, pred_offset)
        gt_color = (1.0, 0.32, 0.06, 1.0)
        pred_color = (0.08, 0.48, 1.0, 1.0)
        gt_armature = import_colored(gt_offset, "Ground Truth", "THEME09", gt_color)
        pred_armature = import_colored(pred_offset, MODEL_LABEL, "THEME01", pred_color)
        add_label("Ground Truth", -SPACING / 2, gt_color)
        add_label(MODEL_LABEL, SPACING / 2, pred_color)

        frame_dir = os.path.join(temporary, "frames")
        os.makedirs(frame_dir, exist_ok=True)
        scene.render.image_settings.file_format = "PNG"
        scene.render.filepath = os.path.join(frame_dir, "frame_")
        scene.frame_start = 1
        scene.frame_end = max(frame_count(gt_bvh), frame_count(pred_bvh))
        view_area = frame_view_on_armatures((gt_armature, pred_armature))
        shading = view_area.spaces[0].shading
        shading.type = "SOLID"
        shading.color_type = "OBJECT"
        shading.light = "STUDIO"
        bpy.ops.render.opengl(animation=True)
        subprocess.run(
            [
                FFMPEG, "-y", "-framerate", "15", "-start_number", "1",
                "-i", os.path.join(frame_dir, "frame_%04d.png"), "-i", audio,
                "-map", "0:v:0", "-map", "1:a:0", "-vf", label_filter(segment),
                "-c:v", "libx264", "-crf", "18", "-preset", "fast",
                "-c:a", "aac", "-shortest", output_video,
            ],
            check=True,
        )
        print(f"COMPARISON_VIDEO={output_video}")
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


for name in SEGMENTS:
    render_segment(name)

if os.environ.get("QUIT_AFTER_RENDER") == "1":
    bpy.ops.wm.quit_blender()
