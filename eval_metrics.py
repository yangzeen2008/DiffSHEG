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
    
    # Build model
    from models.transformer import TransformerModel
    if opt.flow_matching:
        from models.flow_matching import FlowMatchingWrapper
        model = FlowMatchingWrapper(TransformerModel(opt), opt)
    else:
        from models.gaussian_diffusion import GaussianDiffusion
        model = GaussianDiffusion(opt, TransformerModel(opt))
    
    # Build FGD eval model
    eval_model = None
    if not opt.no_fgd:
        from train_eval_net import build_fgd_val_model
        eval_model = build_fgd_val_model(opt)
    
    model = model.cuda()
    if eval_model is not None:
        eval_model = eval_model.cuda()
    
    # Load checkpoint
    ckpt_path = os.path.join(opt.save_dir, "model", opt.ckpt)
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt['model'], strict=False)
    epoch = ckpt.get('ep', 0)
    print(f"Loaded epoch {epoch}")
    
    # Load val dataset
    print("Loading validation dataset...")
    val_dataset = __import__("datasets.beat", fromlist=["something"]).BeatDataset(opt, "val")
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=opt.batch_size, shuffle=False,
        num_workers=0, drop_last=True, pin_memory=True)
    
    device = torch.device(f'cuda:{opt.gpu_id}')
    model.eval()
    
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
            # Get target pose
            if opt.axis_angle:
                tar_pose = batch_data["pose_axis_angle"]
            else:
                tar_pose = batch_data["pose"]
            tar_pose = tar_pose.to(device)
            
            if not opt.gesture_only:
                tar_facial = batch_data["facial"].to(device)
                motions = torch.cat([tar_pose, tar_facial], dim=-1)
            else:
                motions = tar_pose
            
            # Audio
            aud_feat = batch_data.get("aud_feat")
            if aud_feat is not None:
                aud_feat = aud_feat.to(device)
            audio_emb = batch_data["audio_emb"].to(device) if "audio_emb" in batch_data else None
            
            # Person ID
            p_id = batch_data.get("p_id")
            if p_id is not None:
                p_id = p_id.to(device)
            
            B = motions.shape[0]
            
            # Inpainting dict
            overlap_len = getattr(opt, 'overlap_len', 4)
            inpaint_dict = {
                'gt': motions[:, :overlap_len],
                'outpainting_mask': torch.zeros(B, motions.shape[1], dtype=torch.bool, device=device)
            }
            inpaint_dict['outpainting_mask'][:, :overlap_len] = True
            
            add_cond = {}
            if aud_feat is not None:
                add_cond['aud_feat'] = aud_feat
            
            # Generate
            net_dim = opt.net_dim_pose if opt.gesture_only else opt.net_dim_pose + opt.net_dim_facial
            outputs = model.sample(
                audio_emb, p_id, net_dim, add_cond, inpaint_dict,
                fm_steps=opt.fm_sample_steps if opt.flow_matching else None
            )
            
            # FGD
            if eval_model is not None and not opt.no_fgd:
                latent_out = eval_model(outputs[:, :34, :].float())
                latent_ori = eval_model(motions[:, :34, :].float())
                if latent_out_all is None:
                    latent_out_all = latent_out.cpu().numpy()
                    latent_ori_all = latent_ori.cpu().numpy()
                else:
                    latent_out_all = np.concatenate([latent_out_all, latent_out.cpu().numpy()])
                    latent_ori_all = np.concatenate([latent_ori_all, latent_ori.cpu().numpy()])
            
            # MSE & PCK
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
