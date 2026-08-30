import numpy
import os
import random
from os.path import join as pjoin

import utils.paramUtil as paramUtil
from options.train_options import TrainCompOptions
# from utils.plot_script import *

from models import MotionTransformer, UniDiffuser
from trainers import DDPMTrainer_beat, DDPMTrainer_show, DDPMTrainer
from datasets import ShowDataset
from utils.test_selection import IndexedDatasetView, parse_test_window_ranges

from mmcv.runner import get_dist_info, init_dist
from mmcv.parallel import MMDistributedDataParallel, MMDataParallel
import warnings

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

import sys
sys.path.append(os.path.join(sys.path[2], "A_TalkSHOW_ori"))


def validate_beat_training_data(dataset, opt, sample_count=8):
    """Fail fast on schema/finite-value errors before the long train loop."""
    if len(dataset) <= 0:
        raise RuntimeError("BEAT training cache contains no samples")

    indices = numpy.linspace(
        0, len(dataset) - 1, num=min(sample_count, len(dataset)), dtype=int
    )
    total_squares = 0.0
    total_values = 0
    max_abs = 0.0
    for index in numpy.unique(indices):
        sample = dataset[int(index)]
        gesture_key = 'pose_6d' if getattr(opt, 'rot_6d', False) else (
            'pose_axis_angle' if opt.axis_angle else 'pose'
        )
        required = [gesture_key, 'facial', 'aud_feat', 'id']
        if opt.addHubert or opt.expAddHubert:
            required.append('pretrain_aud_feat')
        missing = [key for key in required if key not in sample]
        if missing:
            raise RuntimeError(f"BEAT cache sample {index} is missing fields: {missing}")

        gesture = sample[gesture_key]
        facial = sample['facial']
        if opt.expression_only or opt.gesCondition_expression_only:
            motion = facial
        elif opt.gesture_only or opt.expCondition_gesture_only is not None or opt.textExpEmoCondition_gesture_only:
            motion = gesture
        else:
            motion = torch.cat((gesture, facial), dim=-1)
        if motion.shape != (opt.n_poses, opt.net_dim_pose):
            raise RuntimeError(
                f"BEAT cache sample {index} motion shape {tuple(motion.shape)}; "
                f"expected {(opt.n_poses, opt.net_dim_pose)}"
            )
        if sample['aud_feat'].shape[:1] != (opt.n_poses,):
            raise RuntimeError(
                f"BEAT cache sample {index} audio length {sample['aud_feat'].shape[0]}; "
                f"expected {opt.n_poses}"
            )
        if opt.addHubert or opt.expAddHubert:
            hubert = sample['pretrain_aud_feat']
            if hubert.shape != (opt.n_poses, 1024):
                raise RuntimeError(
                    f"BEAT cache sample {index} HuBERT shape {tuple(hubert.shape)}; "
                    f"expected {(opt.n_poses, 1024)}"
                )

        tensors = [sample[key] for key in required if torch.is_tensor(sample[key])]
        if any(not torch.isfinite(tensor.float()).all() for tensor in tensors):
            raise RuntimeError(f"BEAT cache sample {index} contains NaN or Inf")
        motion_float = motion.float()
        max_abs = max(max_abs, motion_float.abs().max().item())
        total_squares += motion_float.square().sum().item()
        total_values += motion_float.numel()

    rms = (total_squares / max(total_values, 1)) ** 0.5
    print(
        f"BEAT preflight passed: {len(numpy.unique(indices))} samples, "
        f"motion RMS={rms:.4f}, max_abs={max_abs:.4f}"
    )
    if rms > 10.0 or max_abs > 100.0:
        raise RuntimeError(
            "BEAT normalized motion magnitude is implausibly large. "
            "Rebuild the versioned motion cache before training."
        )



def build_models(opt, dim_pose, audio_dim=128, audio_latent_dim=256, style_dim=4):
    if opt.unidiffuser:
        encoder = UniDiffuser(
            opt=opt,
            input_feats=dim_pose,
            audio_dim=audio_dim,
            aud_latent_dim=audio_latent_dim,
            style_dim=style_dim,
            num_frames=opt.n_poses,
            num_layers=opt.num_layers,
            latent_dim=opt.latent_dim,
            no_clip=opt.no_clip,
            no_eff=opt.no_eff,
            pe_type=opt.PE)
    else:
        encoder = MotionTransformer(
            opt=opt,
            input_feats=dim_pose,
            audio_dim=audio_dim,
            style_dim=style_dim,
            num_frames=opt.n_poses,
            num_layers=opt.num_layers,
            latent_dim=opt.latent_dim,
            no_clip=opt.no_clip,
            no_eff=opt.no_eff,
            pe_type=opt.PE)
    return encoder

def build_fgd_val_model(opt):
    eval_model_module = __import__(f"models.motion_autoencoder", fromlist=["something"])
    import copy
    eval_opt = copy.deepcopy(opt)
    if opt.expression_only or opt.gesCondition_expression_only:
        eval_opt.net_dim_pose = opt.expression_dim
    else:
        # The BEAT gesture FGD encoder is trained on normalized axis-angle,
        # including when the generator itself uses 6D rotations.
        eval_opt.net_dim_pose = opt.split_pos // 2 if getattr(opt, 'rot_6d', False) else opt.split_pos
    eval_model = getattr(eval_model_module, 'HalfEmbeddingNet')(eval_opt)

    print(f"init 'HalfEmbeddingNet' success with net_dim_pose={eval_opt.net_dim_pose}")
    return eval_model

def main():
    parser = TrainCompOptions()
    opt = parser.parse()

    if opt.mode == 'train' and opt.allow_legacy_motion_cache:
        raise ValueError("--allow_legacy_motion_cache is diagnostic-only and cannot be used for training")
    if opt.mode != 'prepare_cache' and (opt.rebuild_motion_cache or opt.adopt_motion_cache):
        raise ValueError(
            "--rebuild_motion_cache/--adopt_motion_cache require --mode prepare_cache"
        )

    if opt.dist_url == "env://" and opt.world_size == -1:
        opt.world_size = int(os.environ["WORLD_SIZE"])

    opt.distributed = opt.world_size > 1 or opt.multiprocessing_distributed

    if torch.cuda.is_available():
        ngpus_per_node = torch.cuda.device_count()
    else:
        ngpus_per_node = 1
    if opt.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        opt.world_size = ngpus_per_node * opt.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, opt))
    else:
        # Simply call main_worker function
        main_worker(opt.gpu_id, ngpus_per_node, opt)



def main_worker(gpu_id, ngpus_per_node, opt):
    # rank, world_size = get_dist_info()
    opt.gpu_id = gpu_id

    if opt.gpu_id is not None:
        print("Use GPU: {}".format(opt.gpu_id))

    if opt.distributed:
        if opt.dist_url == "env://" and opt.rank == -1:
            opt.rank = int(os.environ["RANK"])
        if opt.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            opt.rank = opt.rank * ngpus_per_node + gpu_id
        dist.init_process_group(backend=opt.dist_backend, init_method=opt.dist_url,
                                world_size=opt.world_size, rank=opt.rank)

    seed = int(getattr(opt, 'seed', 1234)) + int(getattr(opt, 'rank', 0))
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if getattr(opt, 'deterministic', False):
        cudnn.deterministic = True
        cudnn.benchmark = False


    torch.autograd.set_detect_anomaly(bool(opt.debug))

    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    opt.meta_dir = pjoin(opt.save_root, 'meta')

    # if opt.rank == 0:
    os.makedirs(opt.model_dir, exist_ok=True)
    os.makedirs(opt.meta_dir, exist_ok=True)
    if opt.world_size > 1:
        dist.barrier()
        
    if opt.dataset_name.lower() == 'beat':
        opt.data_root = 'data/BEAT'
        opt.fps = 15
        opt.net_dim_pose = 192 # body: [16, 34, 141], expression: [16, 34, 51], in_audio: [16, 36266]
        opt.dim_pose = 141
        if opt.remove_hand:
            opt.dim_pose = 33
        # 6D rotation: 47 joints × 6 = 282 dims for gesture
        if opt.rot_6d:
            opt.dim_pose = 282
        opt.expression_dim = 51 

        if opt.expression_only or opt.gesCondition_expression_only:
            opt.net_dim_pose = opt.expression_dim # expression
            opt.e_path = f'data/BEAT/beat_cache/{opt.beat_cache_name}/weights/face_300.bin'
        elif opt.gesture_only or opt.expCondition_gesture_only != None or \
                opt.textExpEmoCondition_gesture_only:
            opt.net_dim_pose = opt.dim_pose # gesture (141 or 282)
            if opt.rot_6d:
                # 6D eval converts to euler before FGD, so use gesture-only axis-angle weights
                opt.e_path = f'data/BEAT/beat_cache/{opt.beat_cache_name}/weights/ges_axis_angle_300.bin'
            elif opt.axis_angle:
                opt.e_path = f'data/BEAT/beat_cache/{opt.beat_cache_name}/weights/ges_axis_angle_300.bin'
            else:
                opt.e_path = f'data/BEAT/beat_cache/{opt.beat_cache_name}/weights/ae_300.bin'
        else:
            opt.net_dim_pose = opt.dim_pose + opt.expression_dim # gesture + expression
            if opt.rot_6d:
                # 6D eval converts to euler before FGD, so use gesture-only axis-angle weights
                opt.e_path = f'data/BEAT/beat_cache/{opt.beat_cache_name}/weights/ges_axis_angle_300.bin'
            elif opt.axis_angle:
                # Use gesture-only FGD weights; eval code slices to split_pos before feeding
                opt.e_path = f'data/BEAT/beat_cache/{opt.beat_cache_name}/weights/ges_axis_angle_300.bin'
            else:
                raise NotImplementedError
        
        # split_pos separates gesture dims from expression dims
        opt.split_pos = opt.dim_pose
        
        opt.audio_dim = 128
        if opt.use_aud_feat:
            opt.audio_dim = 1024
        opt.style_dim = 30 # totally 30 subjects
        opt.speaker_dim = 30 
        opt.word_index_num = 5793
        opt.word_dims = 300
        opt.word_f = 128
        opt.emotion_f = 8
        opt.emotion_dims = 8
        opt.freeze_wordembed = False
        opt.hidden_size = 256
        opt.n_layer = 4
        # Allow CLI override for model capacity experiments
        if getattr(opt, 'hidden_size_override', 0) > 0:
            opt.hidden_size = opt.hidden_size_override
        if getattr(opt, 'n_layer_override', 0) > 0:
            opt.n_layer = opt.n_layer_override

        if opt.n_poses == 150:
            opt.stride = 50
        elif opt.n_poses == 34: 
            opt.stride = 10
        from utils.window_stitching import resolve_overlap_len
        sequential_inference = (
            opt.mode in ("test_arbitrary_len", "test_custom_audio")
            or getattr(opt, "sequential_test_windows", False)
        )
        resolved_overlap = resolve_overlap_len(
            opt.n_poses,
            opt.stride,
            opt.overlap_len,
            auto_overlap=getattr(opt, "auto_overlap", True),
            sequential=sequential_inference,
        )
        if resolved_overlap != opt.overlap_len:
            print(
                f"Auto overlap: {opt.overlap_len} -> {resolved_overlap} "
                f"(n_poses={opt.n_poses}, stride={opt.stride})"
            )
        opt.overlap_len = resolved_overlap
        opt.pose_fps = 15
        opt.vae_length = 300
        if opt.n_poses not in (34, 150):
            raise ValueError("BEAT --n_poses must be 34 or 150")
        opt.new_cache = bool(getattr(opt, 'rebuild_motion_cache', False))
        opt.audio_norm = False
        opt.facial_norm = True
        opt.pose_norm = True
        opt.train_data_path = f'data/BEAT/beat_cache/{opt.beat_cache_name}/train/'
        opt.val_data_path = f'data/BEAT/beat_cache/{opt.beat_cache_name}/val/'
        opt.test_data_path = f'data/BEAT/beat_cache/{opt.beat_cache_name}/test/'
        opt.mean_pose_path = f'data/BEAT/beat_cache/{opt.beat_cache_name}/train/'
        opt.std_pose_path = f'data/BEAT/beat_cache/{opt.beat_cache_name}/train/'
        opt.multi_length_training = [1.0]
        opt.audio_rep = 'wave16k'
        opt.facial_rep = 'facial52'
        opt.speaker_id = 'id'
        opt.pose_rep = 'bvh_rot'
        opt.word_rep = 'text'
        opt.sem_rep = 'sem'
        opt.emo_rep = 'emo'

    elif opt.dataset_name.lower() == 'talkshow':
        opt.talkshow_config = 'options/talkshow_configs/body_pixel.json'
        opt.speaker_dim = 4
        opt.fps = 30
        opt.dim_pose = 129
        opt.split_pos = 129
        if opt.remove_hand:
            opt.dim_pose = 39
        opt.expression_dim = 103
        if opt.ablation == "reverse_ges2exp":
            opt.expression_dim, opt.dim_pose = opt.dim_pose, opt.expression_dim
        if opt.expression_only or opt.gesCondition_expression_only:
            opt.net_dim_pose = opt.expression_dim # expression
            opt.e_path = f'data/SHOW/ae_weights/expression.pth.tar'
        elif opt.gesture_only or opt.expCondition_gesture_only != None:
            opt.net_dim_pose = opt.dim_pose # gesture
            opt.e_path = f'data/SHOW/ae_weights/gesture.pth.tar'
        else:
            opt.net_dim_pose = opt.dim_pose + opt.expression_dim # gesture + expression
            opt.e_path = f'data/SHOW/ae_weights/gesture_expression.pth.tar'
        
        if opt.audio_feat == 'mfcc':
            opt.audio_dim = 64
        elif opt.audio_feat == 'mel':
            opt.audio_dim = 128
        elif opt.audio_feat == 'raw':
            opt.audio_dim = 1
        elif opt.audio_feat == 'hubert':
            opt.audio_dim = 1024
        opt.style_dim = 4
        opt.speaker_dim = 4 
        opt.n_poses = 88
        opt.pose_fps = 30
        opt.vae_length = 300
    
    else:
        raise KeyError('Dataset Does Not Exist')



    if opt.mode == 'prepare_cache':
        if opt.dataset_name.lower() != 'beat':
            raise ValueError("--mode prepare_cache currently supports BEAT only")
        if opt.rebuild_motion_cache and opt.adopt_motion_cache:
            raise ValueError("Choose only one of --rebuild_motion_cache or --adopt_motion_cache")
        dataset_module = __import__(f"datasets.{opt.dataset_name}", fromlist=["something"])
        if opt.cache_splits == 'all':
            splits = ['train', 'val', 'test']
        elif opt.cache_splits == 'train_val':
            splits = ['train', 'val']
        else:
            splits = [opt.cache_splits]
        for split in splits:
            dataset = dataset_module.BeatDataset(
                opt, split, build_cache=True, validate_hubert=False
            )
            print(f"Prepared BEAT {split} motion cache: {len(dataset)} samples")
        print("Motion cache preparation complete. Build the aligned HuBERT cache next.")
        return

    print("=> creating model '{}'".format(opt.model_base))
    model = build_models(opt, opt.net_dim_pose, opt.audio_dim, opt.audio_latent_dim, opt.style_dim)

    if opt.no_fgd == False:
        eval_model = build_fgd_val_model(opt)
    else:
        eval_model = None

    if not torch.cuda.is_available() and not torch.backends.mps.is_available():
        print('using CPU, this will be slow')
    elif opt.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if torch.cuda.is_available():
            if opt.gpu_id is not None:
                torch.cuda.set_device(opt.gpu_id)
                model.cuda(opt.gpu_id)
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs of the current node.
                opt.batch_size = int(opt.batch_size / ngpus_per_node)
                opt.workers = int((opt.workers + ngpus_per_node - 1) / ngpus_per_node)
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[opt.gpu_id], find_unused_parameters=False)

                if not opt.no_fgd:
                    eval_model.cuda(opt.gpu_id)
                    eval_model = torch.nn.parallel.DistributedDataParallel(eval_model, device_ids=[opt.gpu_id], find_unused_parameters=False)
            else:
                # DistributedDataParallel will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                # model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
                model = torch.nn.parallel.DistributedDataParallel(model)
                if not opt.no_fgd:
                    # eval_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(eval_model)  
                    eval_model = torch.nn.parallel.DistributedDataParallel(eval_model, device_ids=[opt.rank], broadcast_buffers=True, find_unused_parameters=False).to(opt.rank)
    elif opt.gpu_id is not None and opt.gpu_id < 0:
        model = model.to('cpu')
        if not opt.no_fgd:
            eval_model = eval_model.to('cpu')
    elif opt.gpu_id is not None and torch.cuda.is_available():
        torch.cuda.set_device(opt.gpu_id)
        model = model.cuda(opt.gpu_id)
        if not opt.no_fgd:
            eval_model = eval_model.cuda(opt.gpu_id)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        model = model.to(device)
        if not opt.no_fgd:
            eval_model = eval_model.to(device)
    else:
        if torch.cuda.is_available():
            # Use single gpu
            model = model.cuda()
            if not opt.no_fgd:
                eval_model = eval_model.cuda()
        else:
            model = model.to('cpu')
            if not opt.no_fgd:
                eval_model = eval_model.to('cpu')

    if torch.cuda.is_available() and (opt.gpu_id is None or opt.gpu_id >= 0):
        if opt.gpu_id:
            device = torch.device('cuda:{}'.format(opt.gpu_id))
        else:
            device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    opt.device = device

    if opt.dataset_name == 'beat':
        runner = DDPMTrainer_beat(opt, model, eval_model=eval_model)
    elif opt.dataset_name == 'talkshow':
        runner = DDPMTrainer_show(opt, model, eval_model=eval_model)
    else:
        runner = DDPMTrainer(opt, model)

    if opt.mode == "train":
        if opt.dataset_name.lower() == 'beat':
            train_dataset = __import__(f"datasets.{opt.dataset_name}", fromlist=["something"]).BeatDataset(opt, "train")  
            val_dataset = __import__(f"datasets.{opt.dataset_name}", fromlist=["something"]).BeatDataset(opt, "val")
            validate_beat_training_data(train_dataset, opt)
        
        elif opt.dataset_name.lower() == 'talkshow':
            train_dataset = ShowDataset(opt, 'data/SHOW/cached_data/talkshow_train_cache')
            val_dataset = ShowDataset(opt, 'data/SHOW/cached_data/talkshow_val_cache')


        runner.train(train_dataset, val_dataset)
    elif opt.mode == "eval":
        # Standalone evaluation: loads val set, runs one validation epoch, prints MSE/PCK/Diversity
        if opt.dataset_name.lower() == 'beat':
            val_dataset = __import__(f"datasets.{opt.dataset_name}", fromlist=["something"]).BeatDataset(opt, "val")
        elif opt.dataset_name.lower() == 'talkshow':
            val_dataset = ShowDataset(opt, 'data/SHOW/cached_data/talkshow_val_cache')
        
        # Copy specified ckpt to latest.tar so resume loads it
        import shutil
        src = pjoin(opt.model_dir, opt.ckpt)
        dst = pjoin(opt.model_dir, 'latest.tar')
        if opt.ckpt != 'latest.tar':
            shutil.copy2(src, dst)
            print(f"Copied {opt.ckpt} -> latest.tar for eval")
        
        # Use the trainer's built-in load mechanism
        opt.resume = True
        opt.eval_every_e = 1
        opt.num_epochs = 99999
        opt.save_every_e = 9999
        opt.debug = True  # 1 train batch + 1 eval batch per epoch, then break
        
        train_dataset = val_dataset
        
        print(f"Running eval with fm_sample_steps={getattr(opt, 'fm_sample_steps', 'N/A')}")
        runner.train(train_dataset, val_dataset)

    elif "test" in opt.mode:
        split = "val" if opt.test_on_val else "test"
        if opt.dataset_name.lower() == 'beat':
            test_dataset = __import__(f"datasets.{opt.dataset_name}", fromlist=["something"]).BeatDataset(opt, split)

        elif opt.dataset_name.lower() == 'talkshow':
            test_dataset = ShowDataset(opt, 'data/SHOW/cached_data/talkshow_test_cache')

        selected_indices = parse_test_window_ranges(opt.test_window_ranges)
        if selected_indices:
            if max(selected_indices) >= len(test_dataset):
                raise IndexError(
                    f"test_window_ranges selects index {max(selected_indices)}, "
                    f"but the {split} dataset contains only {len(test_dataset)} windows"
                )
            test_dataset = IndexedDatasetView(test_dataset, selected_indices)
            print(
                f"Selected {len(selected_indices)} cached test windows from "
                f"{opt.test_window_ranges}"
            )
                
        if opt.mode == "test":
            results_dir = runner.test(test_dataset)
        elif opt.mode == "test_arbitrary_len":
            opt.batch_size = 1
            results_dir = runner.test_arbitrary_len(test_dataset)
        elif opt.mode == "test_custom_audio":
            results_dir = runner.test_custom_aud(opt.test_audio_path, test_dataset)
        print(results_dir)




if __name__ == '__main__':
    main()
    
