"""
渲染后音频合成脚本
用法: python merge_audio.py
"""
import subprocess, os

FFMPEG = r"C:\Users\yangz\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1-full_build\bin\ffmpeg.exe"
VIDEO = r"F:\study\DiffSHEG\bvh_output\step_comparison.mp4"
AUDIO = r"F:\study\DiffSHEG\data\BEAT\raw\beat_english_v0.2.1\beat_english_v0.2.1\1\1_wayne_0_100_100.wav"
OUTPUT = r"F:\study\DiffSHEG\bvh_output\step_comparison_audio.mp4"

assert os.path.exists(FFMPEG), f"ffmpeg not found: {FFMPEG}"
assert os.path.exists(VIDEO), f"video not found: {VIDEO}"
assert os.path.exists(AUDIO), f"audio not found: {AUDIO}"

cmd = [
    FFMPEG, "-y",
    "-i", VIDEO,
    "-i", AUDIO,
    "-map", "0:v",
    "-map", "1:a",
    "-c:v", "copy",
    "-c:a", "aac",
    "-shortest",
    OUTPUT
]

print(f"Merging:\n  Video: {VIDEO}\n  Audio: {AUDIO}\n  Output: {OUTPUT}\n")
result = subprocess.run(cmd, capture_output=True, text=True)

if result.returncode == 0:
    size = os.path.getsize(OUTPUT) / 1024 / 1024
    print(f"Done! {OUTPUT} ({size:.1f} MB)")
else:
    print(f"Error:\n{result.stderr[-500:]}")
