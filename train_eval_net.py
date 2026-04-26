"""
train_eval_net.py — Train the FGD evaluator (motion autoencoder) for BEAT dataset.

This script trains a PoseEncoderConv + PoseDecoderConv autoencoder on the
BEAT training set. The trained encoder is then used as the FGD evaluator
(HalfEmbeddingNet) for computing Fréchet Gesture Distance.

Usage:
    python train_eval_net.py \
        --beat_cache_name beat_4english_15_141 \
        --n_poses 34 \
        --num_epochs 300 \
        --batch_size 256 \
        --gesture_only

Output:
    data/BEAT/beat_cache/{beat_cache_name}/weights/ges_axis_angle_300.bin
    (or ae_300.bin if not gesture_only)
"""

import os
import sys
import argparse
import pickle
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import lmdb

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.motion_autoencoder import PoseEncoderConv, PoseDecoderConv


class SimpleMotionDataset(Dataset):
    """Minimal LMDB loader — only extracts axis-angle pose data."""

    def __init__(self, lmdb_dir, n_poses, pose_dim):
        self.n_poses = n_poses
        self.pose_dim = pose_dim
        self.lmdb_env = lmdb.open(lmdb_dir, readonly=True, lock=False,
                                   map_size=int(20e9))
        with self.lmdb_env.begin() as txn:
            self.n_samples = txn.stat()['entries']
        print(f"  Loaded {self.n_samples} samples from {lmdb_dir}")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        key = "{:010}".format(idx).encode("ascii")
        with self.lmdb_env.begin(write=False) as txn:
            sample = txn.get(key)
            if sample is None:
                # fallback: try next index
                return self.__getitem__((idx + 1) % self.n_samples)
            sample = pickle.loads(sample)

        # sample layout: (tar_pose, tar_pose_axis_angle, in_audio, in_mel,
        #                  in_facial, in_word, emo, sem, vid)
        tar_pose_axis_angle = sample[1]  # [T, dim] or [T, J, 3]
        pose = torch.from_numpy(tar_pose_axis_angle.copy()).float()
        pose = pose.reshape(pose.shape[0], -1)  # [T, pose_dim]
        return pose


class AutoEncoder(nn.Module):
    """Simple autoencoder wrapping PoseEncoderConv + PoseDecoderConv."""

    def __init__(self, n_frames, pose_dim, feature_length=300):
        super().__init__()
        self.pose_encoder = PoseEncoderConv(n_frames, pose_dim,
                                             feature_length=feature_length)
        self.decoder = PoseDecoderConv(n_frames, pose_dim,
                                       feature_length=feature_length)

    def forward(self, poses, variational_encoding=False):
        feat, mu, logvar = self.pose_encoder(poses, variational_encoding)
        out = self.decoder(feat)
        return out, mu, logvar


def train(args):
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available()
                          else 'cpu')
    print(f"Device: {device}")

    # --- Paths ---
    cache_root = f'data/BEAT/beat_cache/{args.beat_cache_name}'
    lmdb_dir = os.path.join(cache_root, 'train',
                            f'bvh_rot_cache_len{args.n_poses}_stride10')
    weight_dir = os.path.join(cache_root, 'weights')
    os.makedirs(weight_dir, exist_ok=True)

    if args.gesture_only:
        save_name = f'ges_axis_angle_{args.vae_length}.bin'
    else:
        save_name = f'ae_{args.vae_length}.bin'
    save_path = os.path.join(weight_dir, save_name)

    # --- Dataset ---
    print("Loading dataset...")
    dataset = SimpleMotionDataset(lmdb_dir, args.n_poses, args.pose_dim)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, drop_last=True,
                        pin_memory=True)

    # --- Model ---
    model = AutoEncoder(args.n_poses, args.pose_dim,
                        feature_length=args.vae_length).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    print(f"Pose dim: {args.pose_dim}, Frames: {args.n_poses}, "
          f"VAE length: {args.vae_length}")
    print(f"Save to: {save_path}")
    print(f"{'='*60}")

    # --- Training ---
    best_loss = float('inf')
    for epoch in range(1, args.num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_rec = 0.0
        epoch_kl = 0.0
        n_batches = 0
        t0 = time.time()

        for batch_idx, poses in enumerate(loader):
            poses = poses.to(device)  # [B, T, D]

            # VAE training with KL annealing
            use_vae = (epoch > 50)  # warm up with pure AE first
            recon, mu, logvar = model(poses, variational_encoding=use_vae)

            # Reconstruction loss
            rec_loss = nn.functional.mse_loss(recon, poses)

            # KL divergence
            if use_vae:
                kl_loss = -0.5 * torch.mean(
                    1 + logvar - mu.pow(2) - logvar.exp())
                kl_weight = min(1.0, (epoch - 50) / 100)  # linear anneal
                loss = rec_loss + kl_weight * 0.01 * kl_loss
            else:
                kl_loss = torch.tensor(0.0)
                loss = rec_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_rec += rec_loss.item()
            epoch_kl += kl_loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / n_batches
        avg_rec = epoch_rec / n_batches
        avg_kl = epoch_kl / n_batches
        elapsed = time.time() - t0

        # Logging
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d}/{args.num_epochs} | "
                  f"Loss: {avg_loss:.6f} | "
                  f"Rec: {avg_rec:.6f} | "
                  f"KL: {avg_kl:.6f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f} | "
                  f"{elapsed:.1f}s")

        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'model_state': model.state_dict(),
                'epoch': epoch,
                'loss': avg_loss,
                'pose_dim': args.pose_dim,
                'n_poses': args.n_poses,
                'vae_length': args.vae_length,
            }, save_path)

        # Also save periodic checkpoints
        if epoch % 100 == 0:
            ckpt_path = save_path.replace('.bin', f'_e{epoch}.bin')
            torch.save({
                'model_state': model.state_dict(),
                'epoch': epoch,
                'loss': avg_loss,
                'pose_dim': args.pose_dim,
                'n_poses': args.n_poses,
                'vae_length': args.vae_length,
            }, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

    print(f"\n{'='*60}")
    print(f"Training complete! Best loss: {best_loss:.6f}")
    print(f"Saved to: {save_path}")
    print(f"\nTo use this for FGD evaluation, the trainer will")
    print(f"automatically load it from: {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train FGD evaluator (motion autoencoder) for BEAT')
    parser.add_argument('--beat_cache_name', type=str,
                        default='beat_4english_15_141')
    parser.add_argument('--n_poses', type=int, default=34)
    parser.add_argument('--pose_dim', type=int, default=141,
                        help='141 for axis-angle gesture, 192 for ges+expr')
    parser.add_argument('--vae_length', type=int, default=300,
                        help='Feature length of the encoder (300 = ae_300)')
    parser.add_argument('--gesture_only', action='store_true',
                        help='Gesture only mode (saves as ges_axis_angle_*)')
    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()

    train(args)
