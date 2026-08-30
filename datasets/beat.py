import os
from pathlib import Path

import pickle
import math
import shutil
import numpy as np
import lmdb as lmdb
import pandas as pd
import torch
import glob
import hashlib
import json
import uuid
from termcolor import colored
from loguru import logger
from collections import defaultdict
from torch.utils.data import Dataset
import torch.distributed as dist
import pickle
import pyarrow  # kept for compatibility checks only
from sklearn.preprocessing import normalize
import librosa 
import datasets.rotation_converter as rot_cvt
import torch.nn.functional as F
import pickle
from utils.hubert import HUBERT_CACHE_VERSION
from utils.cache_versions import (
    ALIGNED_FACIAL_FPS,
    ALIGNED_POSE_FPS,
    MOTION_CACHE_VERSION,
    MOTION_MANIFEST_NAME,
    MOTION_STD_FLOOR,
    TEMPORAL_ALIGNMENT_VERSION,
    TEMPORAL_MANIFEST_NAME,
)


class BeatDataset(Dataset):
    def __init__(
        self,
        args,
        loader_type,
        augmentation=None,
        kwargs=None,
        build_cache=True,
        validate_hubert=True,
    ):
        self.opt = args
        self.loader_type = loader_type
        # self.rank = dist.get_rank()
        self.new_cache = args.new_cache
        self.pose_length = args.n_poses #150
        self.stride = args.stride #50
        self.pose_fps = args.pose_fps #15
        self.pose_dims = args.dim_pose # 141

        self.speaker_dims = args.speaker_dim
        self.motion_cache_legacy = False
        self.motion_cache_id = None
        self.hubert_manifest = None
        self.loader_type = loader_type
        self.audio_rep = 'wave16k'
        self.pose_rep = 'bvh_rot'
        self.facial_rep = 'facial52'
        self.word_rep = 'text'
        self.emo_rep = 'emo'
        self.sem_rep = 'sem'
        self.audio_fps = 16000
        self.id_rep = 'id'
        
        self.disable_filtering = False
        self.clean_first_seconds = 0
        self.clean_final_seconds = 0
        
        self.ori_stride = self.stride
        self.ori_length = self.pose_length
        self.alignment = [0,0] # for beat
        
        if loader_type == "train":
            self.data_dir = args.train_data_path
            self.multi_length_training = args.multi_length_training
        elif loader_type == "val":
            self.data_dir = args.val_data_path
            self.multi_length_training = args.multi_length_training 
        else:
            self.data_dir = args.test_data_path
            self.multi_length_training = [1.0]
        self.temporal_manifest = self._load_temporal_manifest()
        self.temporal_alignment_valid = self._temporal_manifest_is_valid(
            self.temporal_manifest
        )
      
        self.max_length = int(self.pose_length * self.multi_length_training[-1])
    

        if self.loader_type == 'test':
            cache_dir_name = f"{self.pose_rep}_cache"
            # preloaded_dir = self.data_dir + f"{self.pose_rep}_cache"
        else:
            cache_dir_name = f"{self.pose_rep}_cache_len{self.pose_length}_stride{self.stride}"
            # preloaded_dir = self.data_dir + f"{self.pose_rep}_cache_len{self.pose_length}_stride{self.stride}"
        
        preloaded_dir = self.data_dir + cache_dir_name

        if args.use_aud_feat or self.opt.expAddHubert or self.opt.addHubert:
            self.aud_feat_path = preloaded_dir.replace(cache_dir_name, "aud_feat_cache/hubert_large_ls960_ft")
            
        
        self.mean_pose = np.load(args.mean_pose_path+f"{args.pose_rep}/bvh_mean.npy")
        self.std_pose = np.load(args.mean_pose_path+f"{args.pose_rep}/bvh_std.npy")
        self.mean_pose_axis_angle = np.load(args.mean_pose_path+"axis_angle_mean.npy")
        self.std_pose_axis_angle = np.load(args.mean_pose_path+"axis_angle_std.npy")
        self.audio_norm = args.audio_norm
        self.facial_norm = args.facial_norm
        if self.audio_norm:
            self.mean_audio = np.load(args.mean_pose_path+f"{args.audio_rep}/npy_mean.npy")
            self.std_audio = np.load(args.mean_pose_path+f"{args.audio_rep}/npy_std.npy")
        self.mean_facial = np.load(args.mean_pose_path+f"{args.facial_rep}/json_mean.npy")
        self.std_facial = np.load(args.mean_pose_path+f"{args.facial_rep}/json_std.npy")
        
        if self.opt.expression_only or self.opt.gesCondition_expression_only:
                self.motion_std = self.std_facial 
                self.motion_mean = self.mean_facial
        elif self.opt.gesture_only or self.opt.expCondition_gesture_only:
            if getattr(self.opt, 'rot_6d', False):
                # 6D mode: no normalize (values are in [-1,1] from rotation matrix)
                # Use identity mean/std so inv_standardize is a no-op
                n_joints = len(self.mean_pose_axis_angle) // 3
                self.motion_std = np.ones(n_joints * 6, dtype=np.float32)
                self.motion_mean = np.zeros(n_joints * 6, dtype=np.float32)
            elif self.opt.axis_angle:
                self.motion_std = self.std_pose_axis_angle
                self.motion_mean = self.mean_pose_axis_angle
            else:
                self.motion_std = self.std_pose
                self.motion_mean = self.mean_pose
        else:
            if getattr(self.opt, 'rot_6d', False):
                n_joints = len(self.mean_pose_axis_angle) // 3
                ges_std = np.ones(n_joints * 6, dtype=np.float32)
                ges_mean = np.zeros(n_joints * 6, dtype=np.float32)
                self.motion_std = np.concatenate((ges_std, self.std_facial), axis=-1)
                self.motion_mean = np.concatenate((ges_mean, self.mean_facial), axis=-1)
            elif self.opt.axis_angle:
                self.motion_std = np.concatenate((self.std_pose_axis_angle, self.std_facial), axis=-1)
                self.motion_mean = np.concatenate((self.mean_pose_axis_angle, self.mean_facial), axis=-1)
            else:
                self.motion_std = np.concatenate((self.std_pose, self.std_facial), axis=-1)
                self.motion_mean = np.concatenate((self.mean_pose, self.mean_facial), axis=-1)
        
        if self.opt.mode != "test_custom_audio":
            if build_cache:
                self.build_cache(preloaded_dir)
            self.lmdb_env = lmdb.open(preloaded_dir, readonly=True, lock=False)

            # HuBERT npy 文件夹路径（通过 aud_feat_path 直接访问 .npy 文件，不用 LMDB）
            pass


            with self.lmdb_env.begin() as txn:
                self.n_samples = txn.stat()["entries"]

            self._validate_motion_manifest(preloaded_dir)

            if (
                validate_hubert
                and
                (args.use_aud_feat or self.opt.expAddHubert or self.opt.addHubert)
                and self.opt.mode != "test_custom_audio"
            ):
                manifest_path = os.path.join(self.aud_feat_path, "manifest.json")
                manifest = None
                if os.path.exists(manifest_path):
                    with open(manifest_path, "r", encoding="utf-8") as manifest_file:
                        manifest = json.load(manifest_file)
                if (
                    not manifest
                    or manifest.get("cache_version") != HUBERT_CACHE_VERSION
                    or manifest.get("sample_count") != self.n_samples
                    or (
                        manifest.get("source_motion_cache_version") != MOTION_CACHE_VERSION
                        and not self.motion_cache_legacy
                    )
                    or (
                        manifest.get("source_motion_cache_id") != self.motion_cache_id
                        and not self.motion_cache_legacy
                    )
                ):
                    raise RuntimeError(
                        "HuBERT cache is legacy or misaligned with the dataset LMDB. "
                        f"Rebuild it with: python build_hubert_cache.py --split {loader_type} --force"
                    )
                self.hubert_manifest = manifest

        self._configure_motion_statistics()
        self._record_cache_contract()


    def _load_temporal_manifest(self):
        path = os.path.join(self.data_dir, TEMPORAL_MANIFEST_NAME)
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as manifest_file:
            return json.load(manifest_file)


    def _temporal_manifest_is_valid(self, manifest):
        if not manifest or not isinstance(manifest, dict):
            return False
        manifest_id = manifest.get("manifest_id")
        payload = dict(manifest)
        payload.pop("manifest_id", None)
        canonical = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        expected_id = hashlib.sha256(canonical).hexdigest()
        return (
            manifest.get("temporal_alignment_version") == TEMPORAL_ALIGNMENT_VERSION
            and manifest.get("target_pose_fps") == ALIGNED_POSE_FPS
            and manifest.get("target_facial_fps") == ALIGNED_FACIAL_FPS
            and manifest.get("audio_sample_rate") == self.audio_fps
            and manifest.get("target_pose_fps") == self.pose_fps
            and isinstance(manifest.get("clip_count"), int)
            and manifest.get("clip_count") > 0
            and isinstance(manifest_id, str)
            and manifest_id == expected_id
        )


    def _require_temporal_alignment(self):
        if self.temporal_alignment_valid:
            return
        raise RuntimeError(
            "BEAT source data has no valid 15 FPS cross-modal timeline manifest. "
            "The legacy cache paired 120 FPS BVH and 60 FPS facial frames with "
            "15 FPS audio windows. Rebuild source data with preprocess_beat.py "
            "into a new beat_4english_15_141_sync_v1 cache before building LMDB."
        )


    def _record_cache_contract(self):
        if self.motion_cache_id:
            motion_ids = dict(getattr(self.opt, "motion_cache_ids", {}))
            motion_ids[self.loader_type] = self.motion_cache_id
            self.opt.motion_cache_ids = motion_ids
        if self.hubert_manifest:
            hubert_bindings = dict(getattr(self.opt, "hubert_cache_bindings", {}))
            hubert_bindings[self.loader_type] = {
                "cache_version": self.hubert_manifest.get("cache_version"),
                "sample_count": self.hubert_manifest.get("sample_count"),
                "source_motion_cache_id": self.hubert_manifest.get("source_motion_cache_id"),
            }
            self.opt.hubert_cache_bindings = hubert_bindings


    def _motion_manifest_payload(self, sample_count):
        temporal = self.temporal_manifest or {}
        return {
            "cache_version": MOTION_CACHE_VERSION,
            "sample_count": int(sample_count),
            "split": self.loader_type,
            "pose_length": "full" if self.loader_type == "test" else int(self.pose_length),
            "stride": "full" if self.loader_type == "test" else int(self.ori_stride),
            "pose_rep": self.pose_rep,
            "axis_angle_std_floor": MOTION_STD_FLOOR,
            "temporal_alignment_version": temporal.get("temporal_alignment_version"),
            "temporal_manifest_id": temporal.get("manifest_id"),
            "target_pose_fps": temporal.get("target_pose_fps"),
            "target_facial_fps": temporal.get("target_facial_fps"),
            "audio_sample_rate": temporal.get("audio_sample_rate"),
        }


    def _write_motion_manifest(self, preloaded_dir, audit=None):
        self._require_temporal_alignment()
        if hasattr(self, "n_samples"):
            sample_count = self.n_samples
        else:
            environment = lmdb.open(preloaded_dir, readonly=True, lock=False)
            with environment.begin(write=False) as transaction:
                sample_count = transaction.stat()["entries"]
            environment.close()
        payload = self._motion_manifest_payload(sample_count)
        payload["cache_id"] = uuid.uuid4().hex
        if audit is not None:
            payload["adoption_audit"] = audit
        manifest_path = os.path.join(preloaded_dir, MOTION_MANIFEST_NAME)
        with open(manifest_path, "w", encoding="utf-8") as manifest_file:
            json.dump(payload, manifest_file, indent=2)
        return payload


    def _audit_motion_cache_for_adoption(self, preloaded_dir):
        logger.warning("Fully auditing legacy motion cache before adoption: {}", preloaded_dir)
        environment = getattr(self, "lmdb_env", None)
        owns_environment = environment is None
        if owns_environment:
            environment = lmdb.open(preloaded_dir, readonly=True, lock=False, readahead=False)
        total_squares = 0.0
        total_values = 0
        max_abs = 0.0
        try:
            with environment.begin(write=False) as transaction:
                sample_count = transaction.stat()["entries"]
                for index in range(sample_count):
                    value = transaction.get(f"{index:05d}".encode("ascii"))
                    if value is None:
                        raise RuntimeError(f"Motion cache key {index:05d} is missing")
                    sample = pickle.loads(value)
                    if not isinstance(sample, (tuple, list)) or len(sample) != 9:
                        raise RuntimeError(f"Motion cache sample {index:05d} has invalid schema")
                    pose = np.asarray(sample[1], dtype=np.float32)
                    facial = np.asarray(sample[4], dtype=np.float32)
                    if self.loader_type != "test" and pose.shape != (self.pose_length, self.pose_dims):
                        raise RuntimeError(
                            f"Motion cache sample {index:05d} pose shape {pose.shape}; "
                            f"expected {(self.pose_length, self.pose_dims)}"
                        )
                    if facial.shape[0] != pose.shape[0]:
                        raise RuntimeError(
                            f"Motion cache sample {index:05d} facial length does not match pose"
                        )
                    if not np.isfinite(pose).all() or not np.isfinite(facial).all():
                        raise RuntimeError(f"Motion cache sample {index:05d} contains NaN or Inf")
                    max_abs = max(max_abs, float(np.max(np.abs(pose))))
                    pose64 = pose.astype(np.float64, copy=False)
                    total_squares += float(np.sum(pose64 * pose64))
                    total_values += pose.size
        finally:
            if owns_environment:
                environment.close()
        rms = (total_squares / max(total_values, 1)) ** 0.5
        if rms > 10.0 or max_abs > 100.0:
            raise RuntimeError(
                f"Legacy motion cache failed adoption audit: RMS={rms:.4f}, "
                f"max_abs={max_abs:.4f}. Rebuild it instead."
            )
        return {
            "scope": "all_samples",
            "sample_count": sample_count,
            "motion_rms": rms,
            "motion_max_abs": max_abs,
        }


    def _validate_motion_manifest(self, preloaded_dir):
        manifest_path = os.path.join(preloaded_dir, MOTION_MANIFEST_NAME)
        manifest = None
        if os.path.exists(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as manifest_file:
                manifest = json.load(manifest_file)
        expected = self._motion_manifest_payload(self.n_samples)
        fields = (
            "cache_version",
            "sample_count",
            "split",
            "pose_length",
            "stride",
            "pose_rep",
            "axis_angle_std_floor",
            "temporal_alignment_version",
            "temporal_manifest_id",
            "target_pose_fps",
            "target_facial_fps",
            "audio_sample_rate",
        )
        is_current = (
            self.temporal_alignment_valid
            and
            bool(manifest)
            and isinstance(manifest.get("cache_id"), str)
            and bool(manifest.get("cache_id"))
            and all(manifest.get(field) == expected[field] for field in fields)
        )
        if is_current:
            self.motion_cache_id = manifest["cache_id"]
            return
        if getattr(self.opt, "adopt_motion_cache", False):
            self._require_temporal_alignment()
            audit = self._audit_motion_cache_for_adoption(preloaded_dir)
            payload = self._write_motion_manifest(preloaded_dir, audit=audit)
            self.motion_cache_id = payload["cache_id"]
            logger.info("Adopted versioned BEAT motion cache: {}", preloaded_dir)
            return
        if getattr(self.opt, "allow_legacy_motion_cache", False):
            self.motion_cache_legacy = True
            logger.warning(
                "Using an unversioned/legacy BEAT motion cache for diagnostics only: {}",
                preloaded_dir,
            )
            return
        raise RuntimeError(
            "BEAT motion cache is legacy or uses incompatible normalization/time alignment. "
            "First rebuild aligned source data with preprocess_beat.py, then rebuild LMDB "
            "with: python runner.py --dataset_name beat "
            "--beat_cache_name beat_4english_15_141_sync_v1 --n_poses 34 "
            "--mode prepare_cache --cache_splits "
            f"{self.loader_type} --rebuild_motion_cache"
        )


    def _configure_motion_statistics(self):
        axis_std = (
            self.std_pose_axis_angle
            if self.motion_cache_legacy
            else np.maximum(self.std_pose_axis_angle, MOTION_STD_FLOOR)
        )
        self.axis_angle_std_for_cache = axis_std
        if self.opt.expression_only or self.opt.gesCondition_expression_only:
            self.motion_std = self.std_facial
            self.motion_mean = self.mean_facial
        elif self.opt.gesture_only or self.opt.expCondition_gesture_only:
            if getattr(self.opt, 'rot_6d', False):
                n_joints = len(self.mean_pose_axis_angle) // 3
                self.motion_std = np.ones(n_joints * 6, dtype=np.float32)
                self.motion_mean = np.zeros(n_joints * 6, dtype=np.float32)
            elif self.opt.axis_angle:
                self.motion_std = axis_std
                self.motion_mean = self.mean_pose_axis_angle
            else:
                self.motion_std = self.std_pose
                self.motion_mean = self.mean_pose
        else:
            if getattr(self.opt, 'rot_6d', False):
                n_joints = len(self.mean_pose_axis_angle) // 3
                ges_std = np.ones(n_joints * 6, dtype=np.float32)
                ges_mean = np.zeros(n_joints * 6, dtype=np.float32)
                self.motion_std = np.concatenate((ges_std, self.std_facial), axis=-1)
                self.motion_mean = np.concatenate((ges_mean, self.mean_facial), axis=-1)
            elif self.opt.axis_angle:
                self.motion_std = np.concatenate((axis_std, self.std_facial), axis=-1)
                self.motion_mean = np.concatenate((self.mean_pose_axis_angle, self.mean_facial), axis=-1)
            else:
                self.motion_std = np.concatenate((self.std_pose, self.std_facial), axis=-1)
                self.motion_mean = np.concatenate((self.mean_pose, self.mean_facial), axis=-1)
        
            
    def build_cache(self, preloaded_dir):
        logger.info(f"Audio bit rate: {self.audio_fps}")
        logger.info("Reading data '{}'...".format(self.data_dir))
        
        # pose_length_extended = int(round(self.pose_length))
        logger.info("Creating the dataset cache...")
        if self.new_cache:
            if os.path.exists(preloaded_dir):
                shutil.rmtree(preloaded_dir)

        cache_generated = False
        if os.path.exists(preloaded_dir):
            logger.info("Found the cache {}".format(preloaded_dir))
        elif self.loader_type == "test":
            self.cache_generation(
                preloaded_dir, True, 
                0, 0,
                is_test=True)
            cache_generated = True
        else: 
            self.cache_generation(
                preloaded_dir, self.disable_filtering, 
                self.clean_first_seconds, self.clean_final_seconds,
                is_test=False)
            cache_generated = True
        if cache_generated:
            self._write_motion_manifest(preloaded_dir)
        
    
    def __len__(self):
        return self.n_samples

    def cache_generation(self, out_lmdb_dir, disable_filtering, clean_first_seconds,  clean_final_seconds, is_test=False):
        self.n_out_samples = 0
        pose_files = sorted(glob.glob(os.path.join(self.data_dir, f"{self.pose_rep}") + "/*.bvh"), key=str,)  
        if not pose_files:
            raise FileNotFoundError(
                f"No source BVH files found under {os.path.join(self.data_dir, self.pose_rep)}"
            )
        # create db for samples
        map_size = int(1024 * 1024 * 2048 * (self.audio_fps/16000)**3 * 4) * (len(pose_files)/30*(self.pose_fps/15)) * len(self.multi_length_training) * self.multi_length_training[-1] * 2 # in 1024 MB
        map_size = min(map_size, 1024 * 1024 * 1024 * 20)  # 最大 20 GB，兼容 Windows
        dst_lmdb_env = lmdb.open(out_lmdb_dir, map_size=int(map_size))

        n_filtered_out = defaultdict(int)
    
        for pose_file in pose_files:
            pose_each_file = []
            audio_each_file = []
            facial_each_file = []
            word_each_file = []
            emo_each_file = []
            sem_each_file = []
            vid_each_file = []
            
            id_pose = Path(pose_file).stem  # e.g. 1_wayne_0_1_1  (works on Windows & Linux)

            logger.info(colored(f"# ---- Building cache for Pose   {id_pose} ---- #", "blue"))
            with open(pose_file, "r") as pose_data:
                for j, line in enumerate(pose_data.readlines()):
                    data = np.fromstring(line, dtype=float, sep=" ") # 1*27 e.g., 27 rotation 
                    pose_each_file.append(data)
            pose_each_file = np.array(pose_each_file) # n frames * 27

            if self.audio_rep is not None:
                logger.info(f"# ---- Building cache for Audio  {id_pose} and Pose {id_pose} ---- #")
                audio_file = pose_file.replace(self.pose_rep, self.audio_rep).replace("bvh", "npy")
                try:
                    # the librosa cannot use on the cloud sever
#                     audio_data, _ = librosa.load(audio_file, sr=None)
#                     if self.audio_rep == "melspec":
#                         audio_each_file = np.load(f"{audio_file[:-4]}_melspec_128_64.npy").transpose(1,0)
#                         self.audio_fps = 32
#                     elif self.audio_rep == "disentangled":
#                         audio_each_file = np.load(f"{audio_file[:-4]}_disentangled_v1.npy").transpose(1,0)
#                     else:
#                         sr, audio_each_file = scipy.io.wavfile.read(audio_file) # np array
#                     audio_each_file = audio_each_file[::sr//16000]
                    audio_each_file = np.load(audio_file)
                except:
                    logger.warning(f"# ---- file not found for Audio {id_pose}, skip all files with the same id ---- #")
                    continue
                if self.audio_norm: 
                    audio_each_file = (audio_each_file - self.mean_audio) / self.std_audio
                    
            if self.facial_rep is not None:
                logger.info(f"# ---- Building cache for Facial {id_pose} and Pose {id_pose} ---- #")
                facial_file = pose_file.replace(self.pose_rep, self.facial_rep).replace("bvh", "json")
                try:
                    with open(facial_file, 'r') as facial_data_file:
                        facial_data = json.load(facial_data_file)
                        for j, frame_data in enumerate(facial_data['frames']):
                            if self.facial_norm:
                                facial_each_file.append((frame_data['weights']-self.mean_facial) / self.std_facial)
                            else:
                                facial_each_file.append(frame_data['weights'])
                    facial_each_file = np.array(facial_each_file)
                except:
                    logger.warning(f"# ---- file not found for Facial {id_pose}, skip all files with the same id ---- #")
                    continue
                    
            if id_pose.split("_")[-1] == "b":
                time_offset = 30 if int(id_pose.split("_")[-3]) % 2 == 0 else 300
                logger.warning(time_offset)
            else:
                time_offset = 0
                
                    
            if self.sem_rep is not None:
                logger.info(f"# ---- Building cache for Sem    {id_pose} and Pose {id_pose} ---- #")
                sem_file = pose_file.replace(self.pose_rep, self.sem_rep).replace("bvh", "txt")
                try:
                    sem_all = pd.read_csv(sem_file, 
                        sep='\t', 
                        names=["name", "start_time", "end_time", "duration", "score", "keywords"])
                except:
                    logger.warning(f"# ---- file not found for Sem {id_pose}, skip all files with the same id ---- #")
                    continue
                # we adopt motion-level semantic score here. 
                for i in range(pose_each_file.shape[0]):
                    found_flag = False
                    for j, (start, end, score) in enumerate(zip(sem_all['start_time'],sem_all['end_time'], sem_all['score'])):
                        current_time = i/self.pose_fps + time_offset
                        if start<=current_time and current_time<=end: 
                            sem_each_file.append(score)
                            found_flag=True
                            break
                        else: continue 
                    if not found_flag: sem_each_file.append(0.)
                sem_each_file = np.array(sem_each_file)
                #print(sem_each_file)
                
            if self.id_rep is not None:
                vid_each_file.append(int(id_pose.split("_")[0])-1)
            
            filtered_result = self._sample_from_clip(
                dst_lmdb_env,
                audio_each_file, pose_each_file, facial_each_file, word_each_file,
                vid_each_file, emo_each_file, sem_each_file,
                disable_filtering, clean_first_seconds, clean_final_seconds, is_test,
                ) 

            for type in filtered_result.keys():
                n_filtered_out[type] += filtered_result[type]
                                
        with dst_lmdb_env.begin() as txn:
            logger.info(colored(f"no. of samples: {txn.stat()['entries']}", "cyan"))
            n_total_filtered = 0
            for type, n_filtered in n_filtered_out.items():
                logger.info("{}: {}".format(type, n_filtered))
                n_total_filtered += n_filtered
            logger.info(colored("no. of excluded samples: {} ({:.1f}%)".format(
                n_total_filtered, 100 * n_total_filtered / (txn.stat()["entries"] + n_total_filtered)), "cyan"))
        dst_lmdb_env.sync()
        dst_lmdb_env.close()
    
    def _sample_from_clip(
        self, dst_lmdb_env, audio_each_file, pose_each_file, facial_each_file, word_each_file,
        vid_each_file, emo_each_file, sem_each_file,
        disable_filtering, clean_first_seconds, clean_final_seconds, is_test,
        ):
        """
        for data cleaning, we ignore the data for first and final n s
        for test, we return all data 
        """
        audio_start = int(self.alignment[0] * self.audio_fps)
        pose_start = int(self.alignment[1] * self.pose_fps)
        audio_each_file = audio_each_file[audio_start:]
        pose_each_file = pose_each_file[pose_start:]
        
        round_seconds_skeleton = pose_each_file.shape[0] // self.pose_fps  # assume 1500 frames / 15 fps = 100 s
        if len(audio_each_file) > 0:
            round_seconds_audio = len(audio_each_file) // self.audio_fps # assume 16,000,00 / 16,000 = 100 s
            if len(facial_each_file) > 0:
                round_seconds_facial = facial_each_file.shape[0] // self.pose_fps
                logger.info(f"audio: {round_seconds_skeleton}s, pose: {round_seconds_audio}s, facial: {round_seconds_facial}s")
                round_seconds_skeleton = min(round_seconds_audio, round_seconds_skeleton, round_seconds_facial)
                max_round = max(round_seconds_audio, round_seconds_skeleton, round_seconds_facial)
                if round_seconds_skeleton != max_round: 
                    logger.warning(f"reduce to {round_seconds_skeleton}s, ignore {max_round-round_seconds_skeleton}s")  
            else:
                logger.info(f"audio: {round_seconds_skeleton}s, pose: {round_seconds_audio}s")
                round_seconds_skeleton = min(round_seconds_audio, round_seconds_skeleton)
                max_round = max(round_seconds_audio, round_seconds_skeleton)
                if round_seconds_skeleton != max_round: 
                    logger.warning(f"reduce to {round_seconds_skeleton}s, ignore {max_round-round_seconds_skeleton}s")
        
        clip_s_t, clip_e_t = clean_first_seconds, round_seconds_skeleton - clean_final_seconds # assume [10, 90]s
        clip_s_f_audio, clip_e_f_audio = self.audio_fps * clip_s_t, clip_e_t * self.audio_fps # [160,000,90*160,000]
        clip_s_f_pose, clip_e_f_pose = clip_s_t * self.pose_fps, clip_e_t * self.pose_fps # [150,90*15]

        for ratio in self.multi_length_training:
            if is_test:# stride = length for test
                self.pose_length = clip_e_f_pose - clip_s_f_pose
                self.stride = self.pose_length
                self.max_length = self.pose_length
            else:
                self.stride = int(ratio*self.ori_stride)
                self.pose_length = int(self.ori_length*ratio)
                
            num_subdivision = math.floor((clip_e_f_pose - clip_s_f_pose - self.pose_length) / self.stride) + 1
            logger.info(f"pose from frame {clip_s_f_pose} to {clip_e_f_pose}, length {self.pose_length}")
            logger.info(f"{num_subdivision} clips is expected with stride {self.stride}")
            
            if len(audio_each_file) > 0:
                audio_short_length = math.floor(self.pose_length / self.pose_fps * self.audio_fps)
                """
                for audio sr = 16000, fps = 15, pose_length = 34, 
                audio short length = 36266.7 -> 36266
                this error is fine.
                """
                logger.info(f"audio from frame {clip_s_f_audio} to {clip_e_f_audio}, length {audio_short_length}")
             
            n_filtered_out = defaultdict(int)
            sample_pose_list = []
            sample_audio_list = []
            sample_facial_list = []
            sample_word_list = []
            sample_emo_list = []
            sample_sem_list = []
            sample_vid_list = []
           
            for i in range(num_subdivision): # cut into around 2s chip, (self npose)
                start_idx = clip_s_f_pose + i * self.stride
                fin_idx = start_idx + self.pose_length # 34
                sample_pose = pose_each_file[start_idx:fin_idx]
                # print(sample_pose.shape)
                if len(audio_each_file) > 0:
                    audio_start = clip_s_f_audio + math.floor(i * self.stride * self.audio_fps / self.pose_fps)
                    audio_end = audio_start + audio_short_length
                    sample_audio = audio_each_file[audio_start:audio_end]
                elif self.audio_rep is not None:
                    logger.warning("audio file is incorrect")
                    continue
                else:
                    sample_audio = np.array([-1])
                
                sample_facial = facial_each_file[start_idx:fin_idx] if len(facial_each_file) > 0 else np.array([-1])
                sample_word = word_each_file[start_idx:fin_idx] if len(word_each_file) > 0 else np.array([-1])
                sample_emo = emo_each_file[start_idx:fin_idx] if len(emo_each_file) > 0 else np.array([-1])
                sample_sem = sem_each_file[start_idx:fin_idx] if len(sem_each_file) > 0 else np.array([-1])
                sample_vid = np.array(vid_each_file) if len(vid_each_file) > 0 else np.array([-1])

                if sample_pose.any() != None:
                    # filtering motion skeleton data
                    sample_pose, filtering_message = MotionPreprocessor(sample_pose, self.mean_pose).get()
                    is_correct_motion = len(sample_pose) > 0
                    if is_correct_motion or disable_filtering:
                        sample_pose_list.append(sample_pose)
                        sample_audio_list.append(sample_audio)
                        sample_facial_list.append(sample_facial)
                        sample_word_list.append(sample_word)
                        sample_vid_list.append(sample_vid)
                        sample_emo_list.append(sample_emo)
                        sample_sem_list.append(sample_sem)
                    else:
                        n_filtered_out[filtering_message] += 1
            
            sample_mel_list = []
            for aud, pose in zip(sample_audio_list, sample_pose_list):
                aud = librosa.resample(aud, orig_sr=16000, target_sr=18000)
                mel = librosa.feature.melspectrogram(y=aud, sr=18000, hop_length=1200, n_mels=128)
                mel = mel[..., :pose.shape[0]]
                mel = np.swapaxes(mel, -1, -2)
                sample_mel_list.append(mel)
            
            sample_pose_axis_angle_list = []
            euler_tensor = torch.from_numpy(np.array(sample_pose_list))
            if len(euler_tensor.shape) == 2:
                euler_tensor.unsqueeze(0)
            euler_tensor = euler_tensor * np.pi / 180.0 # degree to radian
            B, T, C = euler_tensor.shape
            euler_tensor = euler_tensor.reshape(B, T, (C//3), 3)
            axis_angle_tensor = rot_cvt.euler_angles_to_axis_angle(euler_tensor, "XYZ")
            axis_angle_tensor = axis_angle_tensor.reshape(B, T, C)
            sample_pose_axis_angle_list = [pp.numpy() for pp in axis_angle_tensor]

            if len(sample_pose_list) > 0:
                with dst_lmdb_env.begin(write=True) as txn:
                    for pose, pose_axis_angle, audio, mel, facial, word, vid, emo, sem in zip(
                                                        sample_pose_list,
                                                        sample_pose_axis_angle_list,
                                                        sample_audio_list,
                                                        sample_mel_list,
                                                        sample_facial_list,
                                                        sample_word_list,
                                                        sample_vid_list,
                                                        sample_emo_list,
                                                        sample_sem_list,
                                                        ):
                        normalized_pose = self.normalize_pose(pose, self.mean_pose, self.std_pose)
                        normalized_pose_axis_angle = self.normalize_pose(pose_axis_angle, self.mean_pose_axis_angle, self.std_pose_axis_angle)
                        k = "{:005}".format(self.n_out_samples).encode("ascii")
                        v = pickle.dumps([normalized_pose, normalized_pose_axis_angle, audio, mel, facial, word, emo, sem, vid],
                                         protocol=pickle.HIGHEST_PROTOCOL)
                        txn.put(k, v)
                        self.n_out_samples += 1
        return n_filtered_out

    @staticmethod
    def normalize_pose(dir_vec, mean_pose, std_pose=None):
        # Clamp near-zero std to avoid exploding values for near-static joints.
        # 20 axis-angle dims have std ~1e-8 (static joints); 1e-2 is safely
        # below the 21st-smallest std (0.0088) so only truly static dims are affected.
        std_safe = np.maximum(std_pose, MOTION_STD_FLOOR)
        return (dir_vec - mean_pose) / std_safe 
    
    def __getitem__(self, idx):
        with self.lmdb_env.begin(write=False) as txn:
            key = "{:005}".format(idx).encode("ascii")
            sample = txn.get(key)
            sample = pickle.loads(sample)
            tar_pose, tar_pose_axis_angle, in_audio, in_mel, in_facial, in_word, emo, sem, vid = sample
            vid = torch.from_numpy(vid.copy()).int()
            emo = torch.from_numpy(emo.copy()).int()
            sem = torch.from_numpy(sem.copy()).float() 
            in_audio = torch.from_numpy(in_audio.copy()).float() 
            in_mel = torch.from_numpy(in_mel.copy()).float() 
            in_word = torch.from_numpy(in_word.copy()).int()  
            if self.loader_type == "test":
                tar_pose_axis_angle = torch.from_numpy(tar_pose_axis_angle.copy()).float()
                tar_pose = torch.from_numpy(tar_pose.copy()).float()
                in_facial = torch.from_numpy(in_facial.copy()).float()
                if self.opt.usePredExpr:
                    pred_face = np.load(self.face_list[idx]).squeeze()
                    in_facial = torch.from_numpy(pred_face).float()
            else:
                tar_pose_axis_angle = torch.from_numpy(tar_pose_axis_angle.copy()).reshape((tar_pose_axis_angle.shape[0], -1)).float()
                tar_pose = torch.from_numpy(tar_pose.copy()).reshape((tar_pose.shape[0], -1)).float()
                in_facial = torch.from_numpy(in_facial.copy()).reshape((in_facial.shape[0], -1)).float()

        # --- 6D rotation on-the-fly conversion ---
        # Denorm axis_angle, convert to 6D (no normalize needed for 6D).
        if getattr(self.opt, 'rot_6d', False):
            # Denormalize axis_angle back to raw values
            aa_raw = tar_pose_axis_angle * torch.from_numpy(self.axis_angle_std_for_cache).float() \
                   + torch.from_numpy(self.mean_pose_axis_angle).float()
            T = aa_raw.shape[0]
            n_joints = aa_raw.shape[-1] // 3
            aa_reshaped = aa_raw.reshape(T, n_joints, 3)
            pose_6d = rot_cvt.axis_angle_to_rotation_6d(aa_reshaped)  # [T, J, 6]
            tar_pose_6d = pose_6d.reshape(T, n_joints * 6)            # [T, 282]
        else:
            tar_pose_6d = None

        # Build base output dict
        out = {
            "pose": tar_pose,
            "pose_axis_angle": tar_pose_axis_angle,
            "audio": in_audio,
            "facial": in_facial,
            "word": in_word,
            "id": vid,
            "emo": emo,
            "sem": sem,
        }
        if tar_pose_6d is not None:
            out["pose_6d"] = tar_pose_6d

        if self.opt.use_aud_feat or self.opt.expAddHubert or self.opt.addHubert:
            npy_path = os.path.join(self.aud_feat_path, f"{idx:05d}.npy")
            aud_feat = np.load(npy_path)
            aud_feat = torch.from_numpy(aud_feat).float()

                
            if self.opt.use_aud_feat == "interpolate" or self.opt.expAddHubert or self.opt.addHubert:
                aud_feat = F.interpolate(aud_feat.swapaxes(-1,-2).unsqueeze(0), size=tar_pose.shape[0], mode='linear', align_corners=True).swapaxes(-1,-2).squeeze()
            
            if self.opt.use_aud_feat:
                out["aud_feat"] = aud_feat
            elif self.opt.expAddHubert or self.opt.addHubert:
                out["aud_feat"] = in_mel
                out["pretrain_aud_feat"] = aud_feat
        else:
            out["aud_feat"] = in_mel

        return out


class MotionPreprocessor:
    def __init__(self, skeletons, mean_pose):
        self.skeletons = skeletons
        self.mean_pose = mean_pose
        self.filtering_message = "PASS"

    def get(self):
        assert (self.skeletons is not None)

        # filtering
        if len(self.skeletons) > 0:
            if self.check_pose_diff():
                self.skeletons = []
                self.filtering_message = "pose"

        return self.skeletons, self.filtering_message

    def check_static_motion(self, verbose=True):
        def get_variance(skeleton, joint_idx):
            wrist_pos = skeleton[:, joint_idx]
            variance = np.sum(np.var(wrist_pos, axis=0))
            return variance

        left_arm_var = get_variance(self.skeletons, 6)
        right_arm_var = get_variance(self.skeletons, 9)

        th = 0.0014  # exclude 13110
        # th = 0.002  # exclude 16905
        if left_arm_var < th and right_arm_var < th:
            if verbose:
                print("skip - check_static_motion left var {}, right var {}".format(left_arm_var, right_arm_var))
            return True
        else:
            if verbose:
                print("pass - check_static_motion left var {}, right var {}".format(left_arm_var, right_arm_var))
            return False


    def check_pose_diff(self, verbose=False):
        # 用全局标准差替代逐维差值，兼容 raw BVH (228 dims) 和裁剪后 (141 dims) 的 mean_pose
        diff = np.std(self.skeletons)

        # th = 0.017
        th = 0.02 #0.02  # exclude 3594
        if diff < th:
            if verbose:
                print("skip - check_pose_diff {:.5f}".format(diff))
            return True
        else:
            if verbose:
                print("pass - check_pose_diff {:.5f}".format(diff))
            return False


    def check_spine_angle(self, verbose=True):
        def angle_between(v1, v2):
            v1_u = v1 / np.linalg.norm(v1)
            v2_u = v2 / np.linalg.norm(v2)
            return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

        angles = []
        for i in range(self.skeletons.shape[0]):
            spine_vec = self.skeletons[i, 1] - self.skeletons[i, 0]
            angle = angle_between(spine_vec, [0, -1, 0])
            angles.append(angle)

        if np.rad2deg(max(angles)) > 30 or np.rad2deg(np.mean(angles)) > 20:  # exclude 4495
        # if np.rad2deg(max(angles)) > 20:  # exclude 8270
            if verbose:
                print("skip - check_spine_angle {:.5f}, {:.5f}".format(max(angles), np.mean(angles)))
            return True
        else:
            if verbose:
                print("pass - check_spine_angle {:.5f}".format(max(angles)))
            return False
