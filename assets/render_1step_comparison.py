"""
DiffSHEG 1-Step 模型对比渲染 (Blender 5.x)
==========================================
并排展示不同流匹配模型在 1-Step 推理下的动作生成效果，直观呈现积分误差与表示流形退化。

1. 用 Blender 打开 assets/beat_visualize.blend
2. Scripting -> Open -> 此脚本 -> Run Script
"""

import bpy
import os
import shutil
import tempfile
import math


# ============================================================
# 配置区域
# ============================================================

FFMPEG = r"C:\Users\yangz\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1-full_build\bin\ffmpeg.exe"

SOUND_NAME = r"F:\study\DiffSHEG\data\BEAT\raw\beat_english_v0.2.1\beat_english_v0.2.1\1\1_wayne_0_100_100.wav"

BVH_DIR = r"F:\study\DiffSHEG\bvh_output"

# 对比模型: (标签, BVH文件名, 骨骼颜色主题)
MODELS = [
    ("FM_aa_base (1-Step)",   "FM_aa_base_step1.bvh",   'THEME04'),  # 绿色
    ("FM_aa_vel (1-Step)",    "FM_aa_vel_step1.bvh",    'THEME07'),  # 青色
    ("FM_aa_velAcc (1-Step)", "FM_aa_velAcc_step1.bvh", 'THEME06'),  # 粉色
    ("FM_6d (1-Step)",        "FM_6d_step1.bvh",        'THEME03'),  # 黄色
    ("Original (DDPM)",       "Original_DiffSHEG.bvh",   'THEME09'),  # 橙色
]

OUTPUT_VIDEO = os.path.join(BVH_DIR, "step1_comparison.mp4")

SPACING = 100.0        # 骨骼间距 (BVH 单位)
CAMERA_DISTANCE = 500  # 相机距离中心
CAMERA_HEIGHT = 120    # 相机高度

# ============================================================
# 以下无需修改
# ============================================================


def get_bvh_frame_count(bvh_path):
    with open(bvh_path, 'r') as f:
        for line in f:
            if line.strip().startswith('Frames:'):
                return int(line.strip().split(':')[1])
    return 0


def create_offset_bvh(src_bvh, x_offset, out_path):
    """复制 BVH, 给 Hips X 位置加偏移"""
    with open(src_bvh, 'r') as f:
        lines = f.readlines()

    data_start = 0
    for i, line in enumerate(lines):
        if line.strip() == 'MOTION':
            data_start = i + 3
            break

    out_lines = lines[:data_start]
    for line in lines[data_start:]:
        vals = line.strip().split()
        if len(vals) >= 6:
            vals[0] = str(float(vals[0]) + x_offset)
            out_lines.append(' '.join(vals) + '\n')
        else:
            out_lines.append(line)

    with open(out_path, 'w') as f:
        f.writelines(out_lines)
    return out_path


def clean_all():
    """删除场景中所有骨骼、Mesh 和文字"""
    for obj in list(bpy.data.objects):
        if obj.type in ('ARMATURE', 'MESH', 'FONT'):
            bpy.data.objects.remove(obj, do_unlink=True)
    for action in list(bpy.data.actions):
        bpy.data.actions.remove(action)
    for arm in list(bpy.data.armatures):
        bpy.data.armatures.remove(arm)
    for mesh in list(bpy.data.meshes):
        bpy.data.meshes.remove(mesh)
    print("Scene cleaned")


def add_text_label(text, x_pos, z_pos=-20):
    """在骨骼下方添加文字标签"""
    bpy.ops.object.text_add(location=(x_pos, 0, z_pos))
    txt = bpy.context.object
    txt.data.body = text
    txt.data.size = 12
    txt.data.align_x = 'CENTER'
    txt.rotation_euler = (math.radians(90), 0, 0)  # 面向 +Y
    # 白色材质
    mat = bpy.data.materials.new(name=f"mat_{text}")
    mat.diffuse_color = (1, 1, 1, 1)
    txt.data.materials.append(mat)
    return txt


def setup_camera(n_models):
    """自动调整相机位置以容纳所有模型"""
    cam = None
    for obj in bpy.data.objects:
        if obj.type == 'CAMERA':
            cam = obj
            break

    if not cam:
        bpy.ops.object.camera_add()
        cam = bpy.context.object
        bpy.context.scene.camera = cam

    # 相机放在 +Y 方向看向原点
    total_width = (n_models - 1) * SPACING
    distance = max(CAMERA_DISTANCE, total_width * 1.2)

    cam.location = (0, -distance, CAMERA_HEIGHT)
    cam.rotation_euler = (math.radians(80), 0, 0)  # 略微俯视

    # 设置镜头焦距, 越宽越能容纳
    cam.data.lens = 35  # mm
    cam.data.sensor_width = 36

    print(f"Camera: distance={distance:.0f}, height={CAMERA_HEIGHT}, lens=35mm")
    return cam


def render_model_comparison():
    scene = bpy.context.scene
    scene.render.fps = 15
    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1080

    # 1. 清场
    clean_all()

    n = len(MODELS)
    total_width = (n - 1) * SPACING
    start_x = -total_width / 2

    tmp_dir = tempfile.mkdtemp(prefix="diffsheg_1step_")
    max_frames = 0

    # 2. 导入每个模型
    for i, (label, bvh_name, color_theme) in enumerate(MODELS):
        bvh_path = os.path.join(BVH_DIR, bvh_name)
        if not os.path.exists(bvh_path):
            print(f"WARNING: {bvh_path} not found, skipping")
            continue

        x_offset = start_x + i * SPACING
        tmp_bvh = create_offset_bvh(bvh_path, x_offset,
                                     os.path.join(tmp_dir, f"{label}.bvh"))

        bpy.ops.import_anim.bvh(filepath=tmp_bvh)
        obj = bpy.context.object
        obj.name = label
        obj.data.name = label
        obj.show_in_front = True

        # 骨骼着色
        obj.data.show_bone_colors = True
        for bone in obj.pose.bones:
            bone.color.palette = color_theme

        # 文字标签
        add_text_label(label, x_offset)

        frames = get_bvh_frame_count(bvh_path)
        max_frames = max(max_frames, frames)
        print(f"[{i+1}/{n}] {label}: {frames} frames, x={x_offset:.0f}")

    # 3. 设置相机
    setup_camera(n)

    # 4. 渲染设置
    scene.render.image_settings.file_format = 'FFMPEG'
    scene.render.ffmpeg.format = 'MPEG4'
    scene.render.ffmpeg.codec = 'H264'
    scene.render.ffmpeg.audio_codec = 'AAC'
    scene.render.filepath = OUTPUT_VIDEO
    scene.frame_start = 1
    scene.frame_end = int(max_frames + 1)

    # 切到相机视角
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            area.spaces[0].region_3d.view_perspective = 'CAMERA'
            break

    # 5. 渲染
    print(f"\nRendering {max_frames} frames, {n} models side-by-side...")
    bpy.ops.render.opengl(animation=True)
    print(f"Video saved: {OUTPUT_VIDEO}")

    # 6. 合并音频
    if SOUND_NAME and os.path.exists(SOUND_NAME):
        out_audio = OUTPUT_VIDEO.replace('.mp4', '_audio.mp4')
        cmd = (
            f'"{FFMPEG}" -y -i "{OUTPUT_VIDEO}" -i "{SOUND_NAME}" '
            f'-map 0:v -map 1:a -c:v copy -c:a aac -shortest "{out_audio}"'
        )
        ret = os.system(cmd)
        if ret == 0:
            os.replace(out_audio, OUTPUT_VIDEO)
            print("Audio merged!")

    shutil.rmtree(tmp_dir, ignore_errors=True)


# ============================================================
render_model_comparison()
print(f"\n✅ Done! -> {OUTPUT_VIDEO}")
print(f"Layout: {' | '.join(m[0] for m in MODELS)}")

if os.environ.get("QUIT_AFTER_RENDER") == "1":
    print("Quitting Blender as QUIT_AFTER_RENDER is set...")
    bpy.ops.wm.quit_blender()
