"""Render enlarged slow-motion boundary windows for transition blending."""

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
WINDOWS = {
    "typical": {"start": 106, "end": 126, "spike": 114},
    "energetic": {"start": 66, "end": 86, "spike": 74},
}
VARIANTS = (
    ("NO RELEASE (BLEND 0)", "so3_release_blend0", "THEME03", (0.95, 0.72, 0.08, 1.0)),
    ("CANDIDATE (BLEND 5)", "so3_release_blend5", "THEME01", (0.08, 0.48, 1.0, 1.0)),
)


def offset_bvh(source: str, x_offset: float, destination: str) -> None:
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


def clear_scene_assets() -> None:
    for obj in list(bpy.data.objects):
        if obj.type in {"ARMATURE", "MESH", "FONT"}:
            bpy.data.objects.remove(obj, do_unlink=True)
    for action in list(bpy.data.actions):
        bpy.data.actions.remove(action)


def add_label(text: str, x_position: float, color: tuple[float, ...]) -> None:
    bpy.ops.object.text_add(location=(x_position, 0, -22))
    label = bpy.context.object
    label.data.body = text
    label.data.size = 7
    label.data.align_x = "CENTER"
    label.rotation_euler = (math.radians(90), 0, 0)
    label.color = color


def import_colored(path: str, name: str, palette: str, color: tuple[float, ...]):
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
    area.spaces[0].region_3d.view_distance *= 1.05
    return area


def render(segment: str, window: dict[str, int]) -> None:
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    sources = [
        os.path.join(TRIAL_ROOT, folder, segment, "model_pck_best_e479.bvh")
        for _, folder, _, _ in VARIANTS
    ]
    for required in (*sources, FFMPEG):
        if not os.path.exists(required):
            raise FileNotFoundError(required)

    scene = bpy.context.scene
    scene.render.fps = 15
    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1080
    scene.render.resolution_percentage = 100
    clear_scene_assets()
    temporary = tempfile.mkdtemp(prefix=f"diffsheg_boundary_{segment}_")
    try:
        armatures = []
        for index, ((label, _, palette, color), source) in enumerate(zip(VARIANTS, sources)):
            x_position = -55.0 if index == 0 else 55.0
            shifted = os.path.join(temporary, f"variant_{index}.bvh")
            offset_bvh(source, x_position, shifted)
            armatures.append(import_colored(shifted, label, palette, color))
            add_label(label, x_position, color)

        frame_dir = os.path.join(temporary, "frames")
        os.makedirs(frame_dir, exist_ok=True)
        scene.render.image_settings.file_format = "PNG"
        scene.render.filepath = os.path.join(frame_dir, "frame_")
        # Blender's imported frame 1 corresponds to BVH frame 0.
        scene.frame_start = window["start"] + 1
        scene.frame_end = window["end"] + 1
        area = frame_view(armatures)
        shading = area.spaces[0].shading
        shading.type = "SOLID"
        shading.color_type = "OBJECT"
        shading.light = "STUDIO"
        bpy.ops.render.opengl(animation=True)

        spike_time = (window["spike"] - window["start"]) * 4.0 / 15.0
        output_video = os.path.join(
            OUTPUT_ROOT, f"{segment}_blend0_vs_blend5_boundary_slow4x.mp4"
        )
        title = (
            f"{segment.upper()} | 4X SLOW | BVH {window['start']}-{window['end']} | "
            f"BLEND0 SPIKE AT {window['spike']}"
        )
        video_filter = (
            "drawbox=x=0:y=0:w=iw:h=72:color=black@0.62:t=fill,"
            f"drawtext=fontfile='C\\:/Windows/Fonts/arialbd.ttf':text='{title}':"
            "fontcolor=white:fontsize=30:x=(w-text_w)/2:y=20,"
            f"drawtext=fontfile='C\\:/Windows/Fonts/arialbd.ttf':text='BOUNDARY SPIKE':"
            "fontcolor=red:fontsize=38:x=(w-text_w)/2:y=82:"
            f"enable='between(t,{spike_time:.3f},{spike_time + 0.267:.3f})'"
        )
        subprocess.run(
            [
                FFMPEG, "-y", "-framerate", "3.75",
                "-start_number", str(scene.frame_start),
                "-i", os.path.join(frame_dir, "frame_%04d.png"),
                "-vf", video_filter,
                "-r", "30", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-crf", "18", "-preset", "fast", output_video,
            ],
            check=True,
        )
        print(f"BOUNDARY_VIDEO={output_video}")
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


try:
    for segment_name, segment_window in WINDOWS.items():
        render(segment_name, segment_window)
finally:
    if os.environ.get("QUIT_AFTER_RENDER") == "1":
        bpy.ops.wm.quit_blender()
