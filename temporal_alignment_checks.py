"""Regression checks for BEAT cross-modal temporal alignment."""

from __future__ import annotations

import hashlib
import json
import pathlib
import tempfile
import unittest

import numpy as np
import soundfile as sf

import preprocess_beat


ROOT = pathlib.Path(__file__).resolve().parent
REAL_CLIP = (
    ROOT
    / "data"
    / "BEAT"
    / "raw"
    / "beat_english_v0.2.1"
    / "beat_english_v0.2.1"
    / "1"
    / "1_wayne_0_1_1.bvh"
)
REAL_FACE = REAL_CLIP.with_suffix(".json")
REAL_AUDIO = REAL_CLIP.with_suffix(".wav")


def write_bvh(path: pathlib.Path, frames: int, fps: int) -> None:
    lines = [
        "HIERARCHY",
        "ROOT Hips",
        "{",
        "}",
        "MOTION",
        f"Frames: {frames}",
        f"Frame Time: {1.0 / fps:.12f}",
    ]
    base = np.arange(228, dtype=np.float64)
    for index in range(frames):
        values = base + index
        lines.append(" ".join(f"{value:.6f}" for value in values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_face(path: pathlib.Path, frames: int, fps: int) -> None:
    payload = {
        "names": [f"coefficient_{index:02d}" for index in range(51)],
        "frames": [
            {
                "weights": [frame / fps + coefficient / 1000.0 for coefficient in range(51)],
                "time": frame / fps,
                "rotation": [],
            }
            for frame in range(frames)
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


class TemporalAlignmentChecks(unittest.TestCase):
    def make_synthetic_clip(self, directory: pathlib.Path, *, face_frames: int = 60):
        stem = "1_test_0_1_1"
        bvh = directory / f"{stem}.bvh"
        write_bvh(bvh, frames=120, fps=120)
        write_face(directory / f"{stem}.json", frames=face_frames, fps=60)
        sf.write(directory / f"{stem}.wav", np.zeros(16000, dtype=np.float32), 16000)
        return bvh

    def test_common_timeline_resamples_120_and_60_fps_to_15(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary = pathlib.Path(temporary)
            source = temporary / "source"
            output = temporary / "output"
            source.mkdir()
            bvh = self.make_synthetic_clip(source)

            report = preprocess_beat.process_clip(
                bvh, "train", output, target_fps=15, force=True
            )
            self.assertEqual(report["target_frames"], 15)
            self.assertAlmostEqual(report["source"]["motion_fps"], 120.0, places=5)
            self.assertAlmostEqual(report["source"]["facial_fps"], 60.0, places=5)

            pose_lines = (output / "train" / "bvh_rot" / f"{bvh.stem}.bvh").read_text().splitlines()
            self.assertEqual(len(pose_lines), 15)
            first_values = [float(line.split()[0]) for line in pose_lines]
            self.assertEqual(first_values[0], 3.0)
            self.assertEqual(first_values[1], 11.0)
            self.assertEqual(first_values[-1], 115.0)

            face = json.loads(
                (output / "train" / "facial52" / f"{bvh.stem}.json").read_text()
            )
            self.assertEqual(len(face["frames"]), 15)
            self.assertAlmostEqual(face["frames"][1]["time"], 1.0 / 15.0)
            self.assertAlmostEqual(face["frames"][1]["weights"][0], 1.0 / 15.0)

    def test_duration_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary = pathlib.Path(temporary)
            source = temporary / "source"
            source.mkdir()
            bvh = self.make_synthetic_clip(source, face_frames=30)
            with self.assertRaisesRegex(RuntimeError, "duration mismatch"):
                preprocess_beat.process_clip(
                    bvh, "train", temporary / "output", target_fps=15, force=True
                )

    def test_motion_tail_mismatch_can_use_verified_common_prefix(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary = pathlib.Path(temporary)
            source = temporary / "source"
            source.mkdir()
            stem = "1_test_0_1_1"
            bvh = source / f"{stem}.bvh"
            write_bvh(bvh, frames=60, fps=120)
            write_face(source / f"{stem}.json", frames=60, fps=60)
            sf.write(source / f"{stem}.wav", np.zeros(16000, dtype=np.float32), 16000)

            report = preprocess_beat.process_clip(
                bvh,
                "train",
                temporary / "output",
                target_fps=15,
                force=True,
                allow_motion_tail_trim=True,
            )
            self.assertEqual(report["target_frames"], 7)
            self.assertAlmostEqual(report["source_tail_trim_seconds"]["audio"], 0.5)
            self.assertAlmostEqual(report["source_tail_trim_seconds"]["facial"], 0.5)
            self.assertAlmostEqual(report["source_tail_trim_seconds"]["motion"], 0.0)

    def test_legacy_cache_can_supply_audio_and_face_without_reusing_motion(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary = pathlib.Path(temporary)
            source = temporary / "source"
            fallback = temporary / "legacy" / "train"
            source.mkdir()
            (fallback / "facial52").mkdir(parents=True)
            (fallback / "wave16k").mkdir(parents=True)
            stem = "1_test_0_1_1"
            bvh = source / f"{stem}.bvh"
            write_bvh(bvh, frames=120, fps=120)
            write_face(fallback / "facial52" / f"{stem}.json", frames=60, fps=60)
            np.save(fallback / "wave16k" / f"{stem}.npy", np.zeros(16000, dtype=np.float32))

            report = preprocess_beat.process_clip(
                bvh,
                "train",
                temporary / "output",
                target_fps=15,
                force=True,
                fallback_split_root=fallback,
            )
            self.assertEqual(report["target_frames"], 15)
            self.assertEqual(report["source"]["motion_path"], str(bvh))
            self.assertIn("legacy", report["source"]["facial_path"])
            self.assertIn("legacy", report["source"]["audio_path"])

    def test_temporal_manifest_is_content_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            split_dir = pathlib.Path(temporary) / "train"
            split_dir.mkdir()
            path = preprocess_beat.write_temporal_manifest(
                split_dir,
                "train",
                [{"clip": "synthetic", "target_frames": 15}],
                15,
            )
            payload = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
            manifest_id = payload.pop("manifest_id")
            canonical = json.dumps(
                payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ).encode("utf-8")
            self.assertEqual(manifest_id, hashlib.sha256(canonical).hexdigest())

    def test_motion_cache_contract_requires_temporal_manifest_identity(self):
        from datasets.beat import BeatDataset
        from utils.cache_versions import MOTION_CACHE_VERSION

        with tempfile.TemporaryDirectory() as temporary:
            split_dir = pathlib.Path(temporary) / "train"
            split_dir.mkdir()
            path = preprocess_beat.write_temporal_manifest(
                split_dir,
                "train",
                [{"clip": "synthetic", "target_frames": 15}],
                15,
            )
            manifest = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
            dataset = BeatDataset.__new__(BeatDataset)
            dataset.audio_fps = 16000
            dataset.pose_fps = 15
            dataset.loader_type = "train"
            dataset.pose_length = 34
            dataset.ori_stride = 10
            dataset.pose_rep = "bvh_rot"
            dataset.temporal_manifest = manifest
            self.assertTrue(dataset._temporal_manifest_is_valid(manifest))
            payload = dataset._motion_manifest_payload(sample_count=123)
            self.assertEqual(payload["cache_version"], MOTION_CACHE_VERSION)
            self.assertEqual(payload["temporal_manifest_id"], manifest["manifest_id"])

            tampered = dict(manifest)
            tampered["target_facial_fps"] = 60
            self.assertFalse(dataset._temporal_manifest_is_valid(tampered))

    @unittest.skipUnless(
        REAL_CLIP.exists() and REAL_FACE.exists() and REAL_AUDIO.exists(),
        "Complete real BEAT sample is not available",
    )
    def test_real_beat_clip_has_expected_source_rates_and_aligned_length(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = preprocess_beat.process_clip(
                REAL_CLIP,
                "val",
                pathlib.Path(temporary),
                target_fps=15,
                force=True,
            )
            self.assertAlmostEqual(report["source"]["motion_fps"], 120.0, places=2)
            self.assertAlmostEqual(report["source"]["facial_fps"], 60.0, places=2)
            self.assertEqual(report["target_frames"], 1035)
            self.assertLess(report["maximum_source_duration_delta_seconds"], 1.0 / 15.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
