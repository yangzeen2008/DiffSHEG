"""Render GT and transition-blend ablations in one synchronized view."""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import tempfile

import bpy


ROOT = r"F:\study\DiffSHEG"
TRIAL_ROOT = os.path.join(ROOT, "bvh_output", "long_validation_trials")
OUTPUT_ROOT = os.path.join(TRIAL_ROOT, "transition_blend_ablation")
FFMPEG = r"C:\Users\yangz\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1-full_build\bin\ffmpeg.exe"
SEGMENTS = tuple(
    value.strip()
    for value in os.environ.get(
        "DIFFSHEG_RENDER_SEGMENTS", "quiet,typical,energetic"
    ).split(",")
    if value.strip()
)
SPACING = 120.0
VARIANTS = (
    ("GROUND TRUTH", "so3_release", "ground_truth.bvh", "THEME09", (1.0, 0.32, 0.06, 1.0)),
    ("BLEND 0", "so3_release_blend0", "model_pck_best_e479.bvh", "THEME03", (0.95, 0.75, 0.10, 1.0)),
    ("BLEND 3", "so3_release_blend3", "model_pck_best_e479.bvh", "THEME04", (0.20, 0.85, 0.35, 1.0)),
    ("BLEND 5", "so3_release_blend5", "model_pck_best_e479.bvh", "THEME01", (0.08, 0.48, 1.0, 1.0)),
    ("BLEND 7", "so3_release", "model_pck_best_e479.bvh", "THEME06", (0.72, 0.30, 0.95, 1.0)),
)


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
    bpy.ops.object.text_add(location=(x_position, 0, -22))
    label = bpy.context.object
    label.data.body = text
    label.data.size = 9
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


def frame_view(armatures):
    for obj in bpy.context.selected_objects:
        obj.select_set(False)
    for armature in armatures:
        armature.select_set(True)
    bpy.context.view_layer.objects.active = armatures[0]
    area = next(area for area in bpy.context.screen.areas if area.type == "VIEW_3D")
    region = next(region for region in area.regions if region.type == "WINDOW")
    with bpy.context.temp_override(area=area, region=region):
        bpy.ops.view3d.view_axis(type="FRONT", align_active=False)
        bpy.ops.view3d.view_selected(use_all_regions=False)
    area.spaces[0].region_3d.view_distance *= 1.12
    return area


def render(segment):
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    audio = os.path.join(TRIAL_ROOT, "so3_release", segment, "audio.wav")
    output_video = os.path.join(OUTPUT_ROOT, f"{segment}_all.mp4")
    paths = [
        os.path.join(TRIAL_ROOT, folder, segment, filename)
        for _, folder, filename, _, _ in VARIANTS
    ]
    for required in (*paths, audio, FFMPEG):
        if not os.path.exists(required):
            raise FileNotFoundError(required)

    scene = bpy.context.scene
    scene.render.fps = 15
    scene.render.resolution_x = 2560
    scene.render.resolution_y = 900
    scene.render.resolution_percentage = 100
    clear_scene_assets()
    temporary = tempfile.mkdtemp(prefix=f"diffsheg_ablation_{segment}_")
    try:
        armatures = []
        center = (len(VARIANTS) - 1) / 2.0
        for index, ((label, _, _, palette, color), source) in enumerate(zip(VARIANTS, paths)):
            x_position = (index - center) * SPACING
            shifted = os.path.join(temporary, f"variant_{index}.bvh")
            offset_bvh(source, x_position, shifted)
            armatures.append(import_colored(shifted, label, palette, color))
            add_label(label, x_position, color)

        frame_dir = os.path.join(temporary, "frames")
        os.makedirs(frame_dir, exist_ok=True)
        scene.render.image_settings.file_format = "PNG"
        scene.render.filepath = os.path.join(frame_dir, "frame_")
        scene.frame_start = 1
        scene.frame_end = min(frame_count(path) for path in paths)
        area = frame_view(armatures)
        shading = area.spaces[0].shading
        shading.type = "SOLID"
        shading.color_type = "OBJECT"
        shading.light = "STUDIO"
        bpy.ops.render.opengl(animation=True)
        subprocess.run(
            [
                FFMPEG, "-y", "-framerate", "15", "-start_number", "1",
                "-i", os.path.join(frame_dir, "frame_%04d.png"), "-i", audio,
                "-map", "0:v:0", "-map", "1:a:0",
                "-vf", (
                    "drawbox=x=0:y=0:w=iw:h=62:color=black@0.55:t=fill,"
                    f"drawtext=fontfile='C\\:/Windows/Fonts/arialbd.ttf':"
                    f"text='{segment.upper()} | SAME CHECKPOINT-SEED-AUDIO | ONLY BLEND CHANGES':"
                    "fontcolor=white:fontsize=28:x=(w-text_w)/2:y=17"
                ),
                "-c:v", "libx264", "-crf", "18", "-preset", "fast",
                "-c:a", "aac", "-shortest", output_video,
            ],
            check=True,
        )
        print(f"ABLATION_VIDEO={output_video}")
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


for segment_name in SEGMENTS:
    render(segment_name)

if os.environ.get("QUIT_AFTER_RENDER") == "1":
    bpy.ops.wm.quit_blender()
