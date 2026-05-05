"""
DiffSHEG 并排对比渲染 (Blender 5.x)
=====================================
1. 用 Blender 打开 assets/beat_visualize.blend
2. Scripting -> Open -> 此脚本 -> Run Script
"""

import bpy
import os
import shutil
import tempfile


# ============================================================
# 配置区域
# ============================================================

FFMPEG = r"C:\Users\yangz\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1-full_build\bin\ffmpeg.exe"

SOUND_NAME = r"F:\study\DiffSHEG\data\BEAT\raw\beat_english_v0.2.1\beat_english_v0.2.1\1\1_wayne_0_100_100.wav"

LEFT_BVH  = r"F:\study\DiffSHEG\bvh_output\FM_v6_adaptive.bvh"
RIGHT_BVH = r"F:\study\DiffSHEG\bvh_output\Original_DiffSHEG.bvh"

OUTPUT_VIDEO = r"F:\study\DiffSHEG\bvh_output\comparison_side_by_side.mp4"

SPACING = 150.0  # BVH 单位 (厘米级), 需要较大的值

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
    """
    复制 BVH 文件, 给每一帧的 Hips X 位置(第0列) 加偏移。
    """
    with open(src_bvh, 'r') as f:
        lines = f.readlines()

    # 找到 MOTION 数据起始行
    data_start = 0
    for i, line in enumerate(lines):
        if line.strip() == 'MOTION':
            data_start = i + 3  # MOTION -> Frames: -> Frame Time: -> 第一帧
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
    """删除场景中所有骨骼和 Mesh"""
    for obj in bpy.data.objects:
        if obj.type in ('ARMATURE', 'MESH'):
            bpy.data.objects.remove(obj, do_unlink=True)
    for action in bpy.data.actions:
        bpy.data.actions.remove(action)
    for arm in bpy.data.armatures:
        bpy.data.armatures.remove(arm)
    for mesh in bpy.data.meshes:
        bpy.data.meshes.remove(mesh)
    print("Scene cleaned")


def render_side_by_side():
    scene = bpy.context.scene
    scene.render.fps = 15

    # 1. 清场
    clean_all()

    # 2. 创建带偏移的临时 BVH
    tmp_dir = tempfile.mkdtemp(prefix="diffsheg_")
    left_tmp = create_offset_bvh(LEFT_BVH, -SPACING / 2, os.path.join(tmp_dir, "left.bvh"))
    right_tmp = create_offset_bvh(RIGHT_BVH, SPACING / 2, os.path.join(tmp_dir, "right.bvh"))
    print(f"Temp BVH files: {tmp_dir}")

    # 3. 导入左边 (蓝色 = Ours)
    bpy.ops.import_anim.bvh(filepath=left_tmp)
    left = bpy.context.object
    left.name = "FM_v6_Ours"
    left.data.name = "FM_v6_Ours"
    left.show_in_front = True
    # 骨骼着色
    left.data.show_bone_colors = True
    for bone in left.pose.bones:
        bone.color.palette = 'THEME01'  # 蓝色
    print(f"Left imported: {left.name}")

    # 4. 导入右边 (橙色 = Original)
    bpy.ops.import_anim.bvh(filepath=right_tmp)
    right = bpy.context.object
    right.name = "Original"
    right.data.name = "Original"
    right.show_in_front = True
    right.data.show_bone_colors = True
    for bone in right.pose.bones:
        bone.color.palette = 'THEME09'  # 橙色
    print(f"Right imported: {right.name}")

    # 5. 帧数
    num_frames = max(get_bvh_frame_count(LEFT_BVH), get_bvh_frame_count(RIGHT_BVH))
    print(f"Frames: {num_frames}")

    # 6. 渲染设置
    scene.render.image_settings.file_format = 'FFMPEG'
    scene.render.ffmpeg.format = 'MPEG4'
    scene.render.ffmpeg.codec = 'H264'
    scene.render.ffmpeg.audio_codec = 'AAC'
    scene.render.filepath = OUTPUT_VIDEO
    scene.frame_start = 1
    scene.frame_end = int(num_frames + 1)

    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            area.spaces[0].region_3d.view_perspective = 'CAMERA'
            break

    # 7. 渲染
    print(f"\nRendering {num_frames} frames ...")
    bpy.ops.render.opengl(animation=True)
    print(f"Video saved: {OUTPUT_VIDEO}")

    # 8. 合并音频
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

    # 清理临时文件
    shutil.rmtree(tmp_dir, ignore_errors=True)


# ============================================================
render_side_by_side()
print(f"\n✅ Done! -> {OUTPUT_VIDEO}")
print("Left (Blue) = FM_v6 (Ours)  |  Right (Orange) = Original")
