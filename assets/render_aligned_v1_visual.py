"""Render the prepared pck_best vs GT BVHs in the Blender viewport."""

import math
import os
import shutil
import subprocess
import tempfile

import bpy


ROOT = r"F:\study\DiffSHEG"
OUTPUT_DIR = os.path.join(ROOT, "bvh_output", "aligned_v1_visual")
GT_BVH = os.path.join(OUTPUT_DIR, "ground_truth_8windows_stitched.bvh")
PRED_BVH = os.path.join(OUTPUT_DIR, "pck_best_e479_sequential_stitched.bvh")
AUDIO = os.path.join(OUTPUT_DIR, "comparison_audio_sequential_stitched.wav")
RAW_VIDEO = os.path.join(OUTPUT_DIR, "pck_best_vs_gt_continuity_fixed_silent.mp4")
OUTPUT_VIDEO = os.path.join(OUTPUT_DIR, "pck_best_vs_gt_continuity_fixed.mp4")
FFMPEG = r"C:\Users\yangz\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1-full_build\bin\ffmpeg.exe"
SPACING = 150.0
LABEL_FILTER = (
    "drawbox=x=0:y=0:w=iw/2:h=82:color=black@0.55:t=fill,"
    "drawbox=x=iw/2:y=0:w=iw/2:h=82:color=black@0.55:t=fill,"
    "drawbox=x=iw/2-2:y=0:w=4:h=ih:color=white@0.30:t=fill,"
    "drawtext=fontfile='C\\:/Windows/Fonts/arialbd.ttf':text='GROUND TRUTH':"
    "fontcolor=0xFF6B1A:fontsize=34:x=(w/4-text_w/2):y=23,"
    "drawtext=fontfile='C\\:/Windows/Fonts/arialbd.ttf':text='MODEL  PCK BEST  E479  CONTINUITY FIX':"
    "fontcolor=0x268DFF:fontsize=34:x=(3*w/4-text_w/2):y=23"
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


def render():
    for required in (GT_BVH, PRED_BVH, AUDIO, FFMPEG):
        if not os.path.exists(required):
            raise FileNotFoundError(required)
    scene = bpy.context.scene
    scene.render.fps = 15
    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1080
    scene.render.resolution_percentage = 100
    clear_scene_assets()

    temporary = tempfile.mkdtemp(prefix="diffsheg_aligned_v1_")
    try:
        gt_offset = os.path.join(temporary, "gt.bvh")
        pred_offset = os.path.join(temporary, "pred.bvh")
        offset_bvh(GT_BVH, -SPACING / 2, gt_offset)
        offset_bvh(PRED_BVH, SPACING / 2, pred_offset)
        gt_color = (1.0, 0.32, 0.06, 1.0)
        pred_color = (0.08, 0.48, 1.0, 1.0)
        import_colored(gt_offset, "Ground Truth", "THEME09", gt_color)
        import_colored(pred_offset, "FM x0 pck_best e479", "THEME01", pred_color)
        add_label("Ground Truth", -SPACING / 2, gt_color)
        add_label("FM x0 - pck_best e479", SPACING / 2, pred_color)

        scene.render.image_settings.file_format = "FFMPEG"
        scene.render.ffmpeg.format = "MPEG4"
        scene.render.ffmpeg.codec = "H264"
        scene.render.filepath = RAW_VIDEO
        scene.frame_start = 1
        scene.frame_end = max(frame_count(GT_BVH), frame_count(PRED_BVH))
        view_areas = [area for area in bpy.context.screen.areas if area.type == "VIEW_3D"]
        if not view_areas:
            raise RuntimeError("No Blender 3D viewport is available for OpenGL rendering")
        shading = view_areas[0].spaces[0].shading
        shading.type = "SOLID"
        shading.color_type = "OBJECT"
        shading.light = "STUDIO"
        view_areas[0].spaces[0].region_3d.view_perspective = "CAMERA"
        bpy.ops.render.opengl(animation=True)
        subprocess.run(
            [
                FFMPEG,
                "-y",
                "-i",
                RAW_VIDEO,
                "-i",
                AUDIO,
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-vf",
                LABEL_FILTER,
                "-c:v",
                "libx264",
                "-crf",
                "18",
                "-preset",
                "fast",
                "-c:a",
                "aac",
                "-shortest",
                OUTPUT_VIDEO,
            ],
            check=True,
        )
        print(f"COMPARISON_VIDEO={OUTPUT_VIDEO}")
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


render()
if os.environ.get("QUIT_AFTER_RENDER") == "1":
    bpy.ops.wm.quit_blender()
