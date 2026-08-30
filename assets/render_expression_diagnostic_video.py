"""Render a clearly labelled 2D diagnostic face for blind expression review."""

from __future__ import annotations

import argparse
import pathlib
import shutil
import subprocess
import tempfile

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import soundfile as sf

from analyze_full_validation_transitions import FACE_COEFFICIENTS, POSE_FPS


INDEX = {name: index for index, name in enumerate(FACE_COEFFICIENTS)}


def value(frame, name):
    return float(np.clip(frame[INDEX[name]], 0.0, 1.0))


def draw_face(axis, frame, label, color):
    axis.set_xlim(-1.05, 1.05)
    axis.set_ylim(-1.2, 1.15)
    axis.set_aspect("equal")
    axis.axis("off")
    axis.set_title(label, fontsize=15, color=color, fontweight="bold")
    axis.add_patch(Ellipse((0, 0), 1.6, 2.0, fill=False, linewidth=2.2, color="#333333"))

    blink_left = value(frame, "eyeBlinkLeft")
    blink_right = value(frame, "eyeBlinkRight")
    wide_left = value(frame, "eyeWideLeft")
    wide_right = value(frame, "eyeWideRight")
    eye_heights = (
        max(0.015, 0.16 * (1.0 - blink_left) + 0.08 * wide_left),
        max(0.015, 0.16 * (1.0 - blink_right) + 0.08 * wide_right),
    )
    look_x = 0.07 * (
        value(frame, "eyeLookOutRight") - value(frame, "eyeLookInRight")
        + value(frame, "eyeLookInLeft") - value(frame, "eyeLookOutLeft")
    )
    look_y = 0.06 * (
        value(frame, "eyeLookUpLeft") + value(frame, "eyeLookUpRight")
        - value(frame, "eyeLookDownLeft") - value(frame, "eyeLookDownRight")
    )
    for center_x, height in zip((-0.36, 0.36), eye_heights):
        axis.add_patch(Ellipse((center_x, 0.35), 0.42, height, fill=False, linewidth=2, color="#333333"))
        if height > 0.035:
            axis.add_patch(Ellipse(
                (center_x + look_x, 0.35 + look_y), 0.075, 0.075,
                color="#222222",
            ))

    inner_up = value(frame, "browInnerUp")
    brow_left = value(frame, "browOuterUpLeft") - value(frame, "browDownLeft")
    brow_right = value(frame, "browOuterUpRight") - value(frame, "browDownRight")
    axis.plot(
        [-0.62, -0.16],
        [0.64 + 0.15 * brow_left, 0.61 + 0.18 * inner_up],
        color="#333333", linewidth=3,
    )
    axis.plot(
        [0.16, 0.62],
        [0.61 + 0.18 * inner_up, 0.64 + 0.15 * brow_right],
        color="#333333", linewidth=3,
    )

    axis.plot([0, -0.05, 0.05], [0.28, -0.02, -0.06], color="#777777", linewidth=1.5)

    smile_left = value(frame, "mouthSmileLeft") - value(frame, "mouthFrownLeft")
    smile_right = value(frame, "mouthSmileRight") - value(frame, "mouthFrownRight")
    pucker = 0.5 * (value(frame, "mouthPucker") + value(frame, "mouthFunnel"))
    stretch = 0.5 * (
        value(frame, "mouthStretchLeft") + value(frame, "mouthStretchRight")
    )
    width = np.clip(0.58 + 0.22 * stretch - 0.30 * pucker, 0.25, 0.82)
    opening = np.clip(
        0.025 + 0.34 * value(frame, "jawOpen") + 0.12 * value(frame, "mouthFunnel"),
        0.025, 0.42,
    )
    center_y = -0.43 - 0.06 * value(frame, "jawOpen")
    xs = np.linspace(-width, width, 80)
    normalized = xs / max(width, 1e-6)
    corner_curve = np.where(
        normalized < 0,
        -0.15 * smile_left * np.abs(normalized),
        -0.15 * smile_right * np.abs(normalized),
    )
    arch = 0.06 * (1.0 - normalized ** 2)
    upper = center_y + corner_curve + arch + opening / 2.0
    lower = center_y + corner_curve - arch - opening / 2.0
    axis.fill_between(xs, lower, upper, color="#ad3a4a", alpha=0.30)
    axis.plot(xs, upper, color="#7c2230", linewidth=2)
    axis.plot(xs, lower, color="#7c2230", linewidth=2)

    axis.text(
        0, -1.08,
        (
            f"jaw {value(frame, 'jawOpen'):.2f}   "
            f"blink {(blink_left + blink_right) / 2:.2f}   "
            f"smile {(max(smile_left, 0) + max(smile_right, 0)) / 2:.2f}"
        ),
        ha="center", va="center", fontsize=10, color="#555555",
    )


def find_ffmpeg(explicit):
    if explicit:
        path = pathlib.Path(explicit)
        if path.is_file():
            return str(path)
        raise FileNotFoundError(path)
    discovered = shutil.which("ffmpeg")
    if discovered:
        return discovered
    fallback = pathlib.Path(
        r"C:\Users\yangz\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1-full_build\bin\ffmpeg.exe"
    )
    if fallback.is_file():
        return str(fallback)
    raise RuntimeError("ffmpeg not found")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-root", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--ffmpeg")
    args = parser.parse_args()

    sample_root = args.sample_root.resolve()
    output = (args.output or sample_root / "expression_diagnostic.mp4").resolve()
    ground_truth = np.load(sample_root / "ground_truth.npy")
    candidate_a = np.load(sample_root / "candidate_A.npy")
    candidate_b = np.load(sample_root / "candidate_B.npy")
    if not (
        ground_truth.shape == candidate_a.shape == candidate_b.shape
        and ground_truth.shape[1] == len(FACE_COEFFICIENTS)
    ):
        raise RuntimeError("Expression arrays have incompatible shapes")
    audio_path = sample_root / "audio.wav"
    audio, sample_rate = sf.read(audio_path, dtype="float32")
    audio_time = np.arange(len(audio)) / sample_rate

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = pathlib.Path(tempfile.mkdtemp(prefix="diffsheg_expression_video_"))
    try:
        for frame_index in range(len(ground_truth)):
            figure = plt.figure(figsize=(16, 7.5), constrained_layout=True)
            grid = figure.add_gridspec(2, 3, height_ratios=[4, 1])
            face_axes = [figure.add_subplot(grid[0, index]) for index in range(3)]
            audio_axis = figure.add_subplot(grid[1, :])
            draw_face(face_axes[0], ground_truth[frame_index], "GROUND TRUTH", "#d95f02")
            draw_face(face_axes[1], candidate_a[frame_index], "CANDIDATE A", "#1b78c5")
            draw_face(face_axes[2], candidate_b[frame_index], "CANDIDATE B", "#7b3294")
            figure.suptitle(
                "SCHEMATIC BLENDSHAPE DIAGNOSTIC — NOT FINAL CHARACTER RENDER",
                fontsize=17, fontweight="bold",
            )
            audio_axis.plot(audio_time, audio, color="#777777", linewidth=0.6)
            current_time = frame_index / POSE_FPS
            audio_axis.axvline(current_time, color="#d62728", linewidth=2)
            audio_axis.set_xlim(0, min(audio_time[-1], len(ground_truth) / POSE_FPS))
            audio_axis.set_ylim(-1.05, 1.05)
            audio_axis.set_xlabel(f"Audio time — {current_time:.2f}s")
            audio_axis.set_ylabel("waveform")
            figure.savefig(temporary / f"frame_{frame_index:04d}.png", dpi=120)
            plt.close(figure)

        subprocess.run([
            find_ffmpeg(args.ffmpeg), "-y",
            "-framerate", str(POSE_FPS), "-start_number", "0",
            "-i", str(temporary / "frame_%04d.png"),
            "-i", str(audio_path),
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest", str(output),
        ], check=True)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    print(f"EXPRESSION_DIAGNOSTIC_VIDEO={output}")


if __name__ == "__main__":
    main()
