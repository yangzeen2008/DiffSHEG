"""
BEAT HuBERT 特征提取脚本（文件夹版）
将每个训练样本窗口的 HuBERT 特征存储为独立的 .npy 文件，
避免 Windows 上 LMDB map_size 限制问题。

文件命名格式：{global_idx:05d}.npy
读取方式：np.load(path / f"{idx:05d}.npy")

用法：
    f:\\study\\DiffSHEG\\.venv\\Scripts\\python.exe build_hubert_cache.py --split train --force
    f:\\study\\DiffSHEG\\.venv\\Scripts\\python.exe build_hubert_cache.py --split val --force
    f:\\study\\DiffSHEG\\.venv\\Scripts\\python.exe build_hubert_cache.py --split test --force
"""

import os, sys, argparse, json, pickle
import numpy as np
import torch
import lmdb
from pathlib import Path
from tqdm import tqdm
from utils.hubert import HUBERT_CACHE_VERSION
from utils.cache_versions import MOTION_CACHE_VERSION, MOTION_MANIFEST_NAME

sys.path.insert(0, str(Path(__file__).parent))

# ===== 配置（必须与训练参数完全一致）=====
CACHE_ROOT    = "./data/BEAT/beat_cache/beat_4english_15_141_sync_v1"
POSE_LEN      = 34        # --n_poses
STRIDE        = 10        # BEAT 数据集 stride（runner.py line 168）
POSE_FPS      = 15        # 姿态帧率
AUDIO_FPS     = 16000     # 音频采样率
DEVICE        = "cuda:0"
CACHE_VERSION = HUBERT_CACHE_VERSION

# ==========================================


def load_hubert_model(device):
    from utils.hubert import load_hubert_components
    print("Loading HuBERT processor...")
    print("Loading HuBERT model...")
    return load_hubert_components(device=device)


def _lmdb_path(split):
    cache_name = "bvh_rot_cache" if split == "test" else f"bvh_rot_cache_len{POSE_LEN}_stride{STRIDE}"
    return os.path.join(CACHE_ROOT, split, cache_name)


def _output_path(split):
    return os.path.join(
        CACHE_ROOT, split, "aud_feat_cache", "hubert_large_ls960_ft"
    )


def _read_manifest(out_dir):
    path = os.path.join(out_dir, "manifest.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _cache_is_current(out_dir, sample_count, motion_cache_id):
    manifest = _read_manifest(out_dir)
    if not manifest:
        return False
    npy_count = sum(1 for _ in Path(out_dir).glob("*.npy"))
    return (
        manifest.get("cache_version") == CACHE_VERSION
        and manifest.get("sample_count") == sample_count
        and manifest.get("source_motion_cache_version") == MOTION_CACHE_VERSION
        and manifest.get("source_motion_cache_id") == motion_cache_id
        and npy_count == sample_count
    )


def _load_audio_batch(environment, indices):
    audio_batch = []
    with environment.begin(write=False) as transaction:
        for index in indices:
            value = transaction.get(f"{index:05d}".encode("ascii"))
            if value is None:
                raise RuntimeError(f"LMDB sample {index:05d} is missing")
            sample = pickle.loads(value)
            audio = np.asarray(sample[2], dtype=np.float32)
            if audio.ndim == 2:
                audio = audio[:, 0]
            audio_batch.append(audio)
    return audio_batch


@torch.no_grad()
def _extract_batch(audio_batch, processor, model, device):
    processed = processor(
        audio_batch,
        return_tensors="pt",
        sampling_rate=AUDIO_FPS,
        padding=True,
    )
    model_inputs = {
        key: value.to(device)
        for key, value in processed.items()
        if key in ("input_values", "attention_mask")
    }
    hidden = model(**model_inputs).last_hidden_state.cpu().numpy()
    features = []
    for row, audio in zip(hidden, audio_batch):
        feature_length = int(
            model._get_feat_extract_output_lengths(torch.tensor(len(audio))).item()
        )
        features.append(row[:feature_length].astype(np.float32, copy=False))
    return features


def build_split(split, processor, model, batch_size=8, force=False):
    """Build a cache whose numeric keys exactly match the dataset LMDB."""
    split_dir = os.path.join(CACHE_ROOT, split)
    lmdb_dir = _lmdb_path(split)
    out_dir = _output_path(split)
    if not os.path.isdir(lmdb_dir):
        raise FileNotFoundError(
            f"Dataset LMDB does not exist: {lmdb_dir}. Build BeatDataset first."
        )
    motion_manifest_path = os.path.join(lmdb_dir, MOTION_MANIFEST_NAME)
    if not os.path.exists(motion_manifest_path):
        raise RuntimeError(
            f"Motion cache is unversioned: {lmdb_dir}. Rebuild it with "
            "runner.py --mode prepare_cache --rebuild_motion_cache first."
        )
    with open(motion_manifest_path, "r", encoding="utf-8") as manifest_file:
        motion_manifest = json.load(manifest_file)
    if motion_manifest.get("cache_version") != MOTION_CACHE_VERSION:
        raise RuntimeError(
            f"Motion cache version mismatch in {motion_manifest_path}: "
            f"expected {MOTION_CACHE_VERSION}"
        )
    motion_cache_id = motion_manifest.get("cache_id")
    if not isinstance(motion_cache_id, str) or not motion_cache_id:
        raise RuntimeError(
            f"Motion cache has no cache_id: {motion_manifest_path}. "
            "Re-run prepare_cache adoption/rebuild with the current code."
        )

    environment = lmdb.open(lmdb_dir, readonly=True, lock=False, readahead=False)
    with environment.begin(write=False) as transaction:
        sample_count = transaction.stat()["entries"]

    if _cache_is_current(out_dir, sample_count, motion_cache_id) and not force:
        print(f"[{split}] aligned cache already complete: {sample_count} samples")
        environment.close()
        return

    old_files = list(Path(out_dir).glob("*.npy")) if os.path.isdir(out_dir) else []
    progress_path = os.path.join(out_dir, ".aligned_cache_progress")
    if old_files and not force and not os.path.exists(progress_path):
        environment.close()
        raise RuntimeError(
            f"[{split}] found an unversioned or mismatched HuBERT cache "
            f"({len(old_files)} files vs {sample_count} LMDB samples). "
            "Re-run with --force to rebuild it by exact LMDB index."
        )

    os.makedirs(out_dir, exist_ok=True)
    start_index = 0
    if os.path.exists(progress_path):
        with open(progress_path, "r", encoding="utf-8") as file:
            start_index = int(file.read().strip() or 0)
        print(f"[{split}] resuming aligned rebuild at sample {start_index}")
    else:
        for marker_name in ("done.txt", "manifest.json"):
            marker_path = os.path.join(out_dir, marker_name)
            if os.path.exists(marker_path):
                os.remove(marker_path)
        with open(progress_path, "w", encoding="utf-8") as file:
            file.write("0")

    print(f"[{split}] LMDB-aligned HuBERT cache: {sample_count} samples → {out_dir}")
    indices = range(start_index, sample_count, batch_size)
    for batch_start in tqdm(indices, desc=f"[{split}]", total=(sample_count - start_index + batch_size - 1) // batch_size):
        batch_indices = list(range(batch_start, min(batch_start + batch_size, sample_count)))
        audio_batch = _load_audio_batch(environment, batch_indices)
        feature_batch = _extract_batch(audio_batch, processor, model, DEVICE)
        for index, features in zip(batch_indices, feature_batch):
            np.save(os.path.join(out_dir, f"{index:05d}.npy"), features)
        with open(progress_path, "w", encoding="utf-8") as file:
            file.write(str(batch_indices[-1] + 1))

    # Remove stale tail files only after every aligned sample was written.
    for path in Path(out_dir).glob("*.npy"):
        if int(path.stem) >= sample_count:
            path.unlink()

    manifest = {
        "cache_version": CACHE_VERSION,
        "sample_count": sample_count,
        "source_lmdb": os.path.relpath(lmdb_dir, split_dir),
        "source_motion_cache_version": MOTION_CACHE_VERSION,
        "source_motion_cache_id": motion_cache_id,
        "model": "facebook/hubert-large-ls960-ft",
        "weight_norm_repaired": True,
        "sample_rate": AUDIO_FPS,
        "pose_length": POSE_LEN,
        "stride": STRIDE,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
    with open(os.path.join(out_dir, "done.txt"), "w", encoding="utf-8") as file:
        file.write(str(sample_count))
    os.remove(progress_path)
    environment.close()
    print(f"[{split}] complete: {sample_count} aligned HuBERT samples")


def main():
    global CACHE_ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "val", "test", "all"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--cache-root",
        default=CACHE_ROOT,
        help="Aligned BEAT cache root created by preprocess_beat.py",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Rebuild a legacy/mismatched cache in place; supports resume",
    )
    args = parser.parse_args()
    CACHE_ROOT = args.cache_root

    print(f"CUDA available: {torch.cuda.is_available()}")
    processor, model = load_hubert_model(DEVICE)

    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    for sp in splits:
        build_split(sp, processor, model, batch_size=max(1, args.batch_size), force=args.force)

    print("\n=== HuBERT 特征提取完成！===")
    print("注意：beat.py 的 HuBERT 读取部分已同步修改为 npy 文件读取模式")
    print("\n下一步运行训练：")
    print("  f:\\study\\DiffSHEG\\.venv\\Scripts\\python.exe runner.py "
          "--dataset_name beat --name beat_FM_v1 --mode train "
          "--flow_matching --fm_sample_steps 50 --n_poses 34 "
          "--batch_size 32 --no_fgd --gpu_id 0")


if __name__ == "__main__":
    main()
