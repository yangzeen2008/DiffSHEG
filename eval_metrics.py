"""
Standalone evaluation: loads checkpoint, runs val set, prints FGD/MSE/PCK/Diversity.
Upload to server, then run:
    python eval_metrics.py --name beat_FM_v1 --flow_matching --fm_sample_steps 50 --ckpt pck_best.tar
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from tqdm import tqdm
from options.train_options import TrainCompOptions

class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = self.sum = self.count = 0
    def update(self, val, n=1):
        self.val = val; self.sum += val * n; self.count += n
    @property
    def avg(self):
        return self.sum / self.count if self.count else 0

def main():
    opt = TrainCompOptions().parse()
    opt.is_train = False
    opt.save_root = os.path.join(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = os.path.join(opt.save_root, 'model')
    opt.meta_dir = os.path.join(opt.save_root, 'meta')
    
    # Initialize BEAT option dimensions
    if opt.dataset_name.lower() == 'beat':
        opt.data_root = 'data/BEAT'
        opt.fps = 15
        opt.net_dim_pose = 192  # body: 141, expression: 51
        opt.dim_pose = 141
        if opt.remove_hand:
            opt.dim_pose = 33
        if opt.rot_6d:
            opt.dim_pose = 282
        opt.expression_dim = 51

        if opt.expression_only or opt.gesCondition_expression_only:
            opt.net_dim_pose = opt.expression_dim
            opt.e_path = f'data/BEAT/beat_cache/{opt.beat_cache_name}/weights/face_300.bin'
        elif opt.gesture_only or opt.expCondition_gesture_only != None or opt.textExpEmoCondition_gesture_only:
            opt.net_dim_pose = opt.dim_pose
            if opt.rot_6d:
                opt.e_path = f'data/BEAT/beat_cache/{opt.beat_cache_name}/weights/ges_axis_angle_300.bin'
            elif opt.axis_angle:
                opt.e_path = f'data/BEAT/beat_cache/{opt.beat_cache_name}/weights/ges_axis_angle_300.bin'
            else:
                opt.e_path = f'data/BEAT/beat_cache/{opt.beat_cache_name}/weights/ae_300.bin'
        else:
            opt.net_dim_pose = opt.dim_pose + opt.expression_dim
            if opt.rot_6d:
                opt.e_path = f'data/BEAT/beat_cache/{opt.beat_cache_name}/weights/ges_axis_angle_300.bin'
            elif opt.axis_angle:
                opt.e_path = f'data/BEAT/beat_cache/{opt.beat_cache_name}/weights/ges_axis_angle_300.bin'
            else:
                raise NotImplementedError
        
        opt.split_pos = opt.dim_pose
        opt.audio_dim = 128
        if opt.use_aud_feat:
            opt.audio_dim = 1024
        opt.style_dim = 30
        opt.speaker_dim = 30
        opt.word_index_num = 5793
        opt.word_dims = 300
        opt.word_f = 128
        opt.emotion_f = 8
        opt.emotion_dims = 8
        opt.freeze_wordembed = False
        opt.hidden_size = 256
        opt.n_layer = 4
        if getattr(opt, 'hidden_size_override', 0) > 0:
            opt.hidden_size = opt.hidden_size_override
        if getattr(opt, 'n_layer_override', 0) > 0:
            opt.n_layer = opt.n_layer_override

        if opt.n_poses == 150:
            opt.stride = 50
        elif opt.n_poses == 34: 
            opt.stride = 10
        opt.pose_fps = 15
        opt.vae_length = 300
        opt.new_cache = False
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

    # Build model
    from runner import build_models
    encoder = build_models(opt, opt.net_dim_pose, opt.audio_dim, opt.audio_latent_dim, opt.style_dim)
    
    # Build FGD eval model
    eval_model = None
    if not opt.no_fgd:
        from runner import build_fgd_val_model
        eval_model = build_fgd_val_model(opt)
    
    if not torch.cuda.is_available() or (opt.gpu_id is not None and opt.gpu_id < 0):
        device = torch.device('cpu')
    else:
        gpu_id = opt.gpu_id if opt.gpu_id is not None else 0
        device = torch.device(f'cuda:{gpu_id}')
    opt.device = device
    
    encoder = encoder.to(device)
    if eval_model is not None:
        eval_model = eval_model.to(device)
    
    # Load checkpoint
    ckpt_path = os.path.join(opt.model_dir, opt.ckpt)
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu')
    encoder.load_state_dict(ckpt['encoder'], strict=False)
    epoch = ckpt.get('ep', 0)
    print(f"Loaded epoch {epoch}")
    
    # Load val dataset
    print("Loading validation dataset...")
    val_dataset = __import__("datasets.beat", fromlist=["something"]).BeatDataset(opt, "val")
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=opt.batch_size, shuffle=False,
        num_workers=0, drop_last=True, pin_memory=True)
    
    encoder.eval()
    
    # Build trainer to encapsulate generation/sampling logic (DDPM vs FM)
    from trainers import DDPMTrainer_beat
    trainer = DDPMTrainer_beat(opt, encoder, eval_model=eval_model)
    
    # Metrics
    mse_meter = AverageMeter()
    pck_meter = AverageMeter()
    diversity_meter = AverageMeter()
    
    latent_out_all = None
    latent_ori_all = None
    
    print(f"\n{'='*60}")
    print(f"Evaluating: FM {opt.fm_sample_steps} steps | Epoch {epoch}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    with torch.no_grad():
        for i, batch_data in tqdm(enumerate(val_loader), total=len(val_loader)):
            # Get target pose (matching ddpm_beat_trainer.py)
            if getattr(opt, 'rot_6d', False):
                tar_pose = batch_data["pose_6d"]
            elif opt.axis_angle:
                tar_pose = batch_data["pose_axis_angle"]
            else:
                tar_pose = batch_data["pose"]
            tar_pose = tar_pose.to(device)
            
            if not opt.gesture_only:
                tar_facial = batch_data["facial"].to(device)
                motions = torch.cat([tar_pose, tar_facial], dim=-1)
            else:
                motions = tar_pose
            
            # Audio & Person ID & add_cond setup (matching ddpm_beat_trainer.py val loop)
            audio_emb = batch_data["aud_feat"].to(device) if opt.audio_rep is not None else None
            
            if opt.expCondition_gesture_only:
                in_facial = batch_data["facial"].to(device) if opt.facial_rep is not None else None
                audio_emb = torch.cat((audio_emb, in_facial), dim=-1)
            elif opt.gesCondition_expression_only:
                audio_emb = torch.cat((audio_emb, tar_pose), dim=-1)
            
            add_cond = {}
            if opt.addTextCond:
                in_word = batch_data["word"].to(device) if opt.word_rep is not None else None
                add_cond['text'] = in_word
            if opt.addEmoCond:
                in_emo = batch_data["emo"].to(device) if opt.emo_rep is not None else None
                add_cond['emo'] = in_emo
            if opt.expAddHubert or opt.addHubert:
                add_cond["pretrain_aud_feat"] = batch_data["pretrain_aud_feat"].to(device)
            
            p_id = batch_data["id"] if opt.speaker_id else None
            if p_id is not None:
                p_id = trainer.one_hot(p_id, opt.speaker_dim).to(device)
            
            if opt.remove_audio and audio_emb is not None:
                audio_emb = torch.zeros_like(audio_emb).to(audio_emb.device)
            if opt.remove_style and p_id is not None:
                p_id = torch.zeros_like(p_id).to(p_id.device)
                
            B = motions.shape[0]
            
            # Inpainting dict
            overlap_len = getattr(opt, 'overlap_len', 4)
            inpaint_dict = {}
            if overlap_len > 0:
                inpaint_dict['gt'] = motions
                inpaint_dict['outpainting_mask'] = torch.zeros_like(motions, dtype=torch.bool, device=device)
                inpaint_dict['outpainting_mask'][:, :overlap_len] = True
            
            # Generate
            outputs = trainer.generate_batch(
                audio_emb, p_id, opt.net_dim_pose, add_cond, inpaint_dict
            )
            
            # FGD
            if eval_model is not None and not opt.no_fgd:
                latent_out = eval_model(outputs[:, :34, :opt.split_pos].float())
                latent_ori = eval_model(motions[:, :34, :opt.split_pos].float())
                if latent_out_all is None:
                    latent_out_all = latent_out.cpu().numpy()
                    latent_ori_all = latent_ori.cpu().numpy()
                else:
                    latent_out_all = np.concatenate([latent_out_all, latent_out.cpu().numpy()])
                    latent_ori_all = np.concatenate([latent_ori_all, latent_ori.cpu().numpy()])
            
            # MSE & PCK
            seq = outputs.shape[1]
            if opt.rot_6d:
                import datasets.rotation_converter as rot_cvt
                n_j = opt.split_pos // 6
                
                # Outputs: convert 6D to normalized axis-angle
                out_ges_6d = outputs[..., :opt.split_pos]
                mat_o = rot_cvt.rotation_6d_to_matrix(out_ges_6d.reshape(B*seq, n_j, 6))
                aa_o = rot_cvt.matrix_to_axis_angle(mat_o).reshape(B, seq, n_j*3)
                
                std_safe = np.maximum(val_dataset.std_pose_axis_angle, 1e-2)
                mean_t = torch.from_numpy(val_dataset.mean_pose_axis_angle).to(device)
                std_t = torch.from_numpy(std_safe).to(device)
                aa_norm_o = (aa_o - mean_t) / std_t
                
                # Motions: retrieve normalized axis-angle ground truth
                aa_norm_g = batch_data["pose_axis_angle"].to(device)
                
                # Append facial expression if present
                if outputs.shape[-1] > opt.split_pos:
                    outputs_for_metric = torch.cat([aa_norm_o, outputs[..., opt.split_pos:]], dim=-1)
                    motions_for_metric = torch.cat([aa_norm_g, motions[..., opt.split_pos:]], dim=-1)
                else:
                    outputs_for_metric = aa_norm_o
                    motions_for_metric = aa_norm_g
                    
                outputs_np = outputs_for_metric.cpu()
                motions_np = motions_for_metric.cpu()
            else:
                outputs_np = outputs.cpu()
                motions_np = motions.cpu()
            
            C = outputs_np.shape[-1]
            seq = outputs_np.shape[1]
            outputs_r = outputs_np.reshape(B, seq, C // 3, 3).numpy()
            motions_r = motions_np.reshape(B, seq, C // 3, 3).numpy()
            
            diff = outputs_r - motions_r
            diff_sq = diff ** 2
            correct = np.sum(diff_sq, axis=3)
            correct = np.sqrt(correct) < 0.5
            pck_val = np.mean(correct)
            mse_val = np.mean(diff_sq)
            
            pck_meter.update(pck_val, B)
            mse_meter.update(mse_val, B)
            
            # Diversity (pairs within batch)
            B_div = min(B, 32)
            out_split = np.split(outputs_r, np.arange(B_div, B, B_div), axis=0)
            if len(out_split) >= 1 and out_split[0].shape[0] >= B_div:
                div_val = 0.0
                count = 0
                for ii in range(B_div):
                    for jj in range(ii + 1, B_div):
                        div_val += np.mean(np.abs(out_split[0][ii] - out_split[0][jj]))
                        count += 1
                if count > 0:
                    diversity_meter.update(div_val / count, 1)
    
    elapsed = time.time() - start_time
    
    # Compute FGD
    fgd_val = -1
    if latent_out_all is not None:
        from utils.metrics import FIDCalculator
        try:
            fid_calc = FIDCalculator()
            fgd_val = fid_calc.frechet_distance(latent_out_all, latent_ori_all)
        except:
            # Manual FGD calculation
            mu1 = np.mean(latent_out_all, axis=0)
            mu2 = np.mean(latent_ori_all, axis=0)
            sigma1 = np.cov(latent_out_all, rowvar=False)
            sigma2 = np.cov(latent_ori_all, rowvar=False)
            from scipy import linalg
            diff_mu = mu1 - mu2
            covmean = linalg.sqrtm(sigma1 @ sigma2)
            if np.iscomplexobj(covmean):
                covmean = covmean.real
            fgd_val = float(diff_mu @ diff_mu + np.trace(sigma1 + sigma2 - 2 * covmean))
    
    # Print results
    print(f"\n{'='*60}")
    print(f"RESULTS: FM {opt.fm_sample_steps} steps | Epoch {epoch}")
    print(f"{'='*60}")
    print(f"  FGD:       {fgd_val:.4f}")
    print(f"  MSE:       {mse_meter.avg:.6f}")
    print(f"  PCK:       {pck_meter.avg:.6f}")
    print(f"  Diversity: {diversity_meter.avg:.4f}")
    print(f"  Time:      {elapsed:.1f}s")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
