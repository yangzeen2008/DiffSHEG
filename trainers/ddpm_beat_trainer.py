import numpy as np
import torch
import torch.nn.functional as F
import random
import time
from models.transformer import MotionTransformer
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from collections import OrderedDict
from utils.utils import print_current_loss
from os.path import join as pjoin
import codecs as cs
import torch.distributed as dist
from tqdm import tqdm
import os
from enum import Enum
from models.respace import SpacedDiffusion, space_timesteps


from datasets import data_tools

import wandb
import json
import librosa

from mmcv.runner import get_dist_info
from models.gaussian_diffusion import (
    GaussianDiffusion,
    get_named_beta_schedule,
    create_named_schedule_sampler,
    ModelMeanType,
    ModelVarType,
    LossType
)

import datasets.rotation_converter as rot_cvt
from utils.motion_metrics import (
    linear_pck_mse,
    normalized_axis_angle,
    rotation_pck_mse,
    select_axis_angle_stats,
)
from utils.test_selection import safe_result_tag
from .loss_factory import get_loss_func

import soundfile as sf
from models.flow_matching import FlowMatching
from utils.window_stitching import audio_windows_are_consecutive, make_prefix_inpaint


def _cuda_synchronize(device):
    device = torch.device(device)
    if device.type == 'cuda' and torch.cuda.is_available():
        torch.cuda.synchronize(device)

class DDPMTrainer_beat(object):

    def __init__(self, args, encoder, eval_model=None):
        self.opt = args
        self.device = args.device
        self.encoder = encoder
        self.epoch = 0
        self.eval_model = eval_model
        if eval_model is not None and 'test' not in self.opt.mode:
            self.load_fid_net(args.e_path)
            self.eval_model.eval()
            
        self.diffusion_steps = args.diffusion_steps
        sampler = 'uniform'
        beta_scheduler = 'linear'
        betas = get_named_beta_schedule(beta_scheduler, self.diffusion_steps)
        model_mean_type = {
            "epsilon": ModelMeanType.EPSILON,
            "start_x": ModelMeanType.START_X,
            "previous_x": ModelMeanType.PREVIOUS_X
        }

        if args.flow_matching:
            self.diffusion = FlowMatching(args)
            # Flow matching doesn't use the same sampler schedule
            self.sampler = None 
        else:
            self.diffusion = GaussianDiffusion(
                opt=args,
                betas=betas,
                model_mean_type=model_mean_type[args.model_mean_type],
                model_var_type=ModelVarType.FIXED_SMALL,
                loss_type=LossType.MSE
            )
            self.sampler = create_named_schedule_sampler(sampler, self.diffusion)
        
        if self.opt.ddim and not args.flow_matching:
            self.diffusion_ddim_val = SpacedDiffusion(
                use_timesteps=space_timesteps(self.diffusion_steps, 'ddim25'),
                opt=args,
                betas=betas,
                model_mean_type=model_mean_type[args.model_mean_type],
                model_var_type=ModelVarType.FIXED_SMALL,
                loss_type=LossType.MSE,
                rescale_timesteps=False,
            )

        self.sampler_name = sampler
        self.sampler_name = sampler

        self.huber_loss = get_loss_func("huber_loss")

        if args.is_train:
            self.mse_criterion = torch.nn.MSELoss(reduction='none')
        self.to(self.device)

        if self.opt.mode == 'train' and not self.opt.debug:
            import os as _os
            _os.environ["WANDB_MODE"] = "offline"   # 本地记录，不需要 API key
            if self.opt.multiprocessing_distributed:
                wandb.init(project=f"Diffusion_{self.opt.dataset_name}", group=f"DDP_{self.opt.name}")
            else:
                wandb.init(project=f"Diffusion_{self.opt.dataset_name}")
            wandb.run.name = f"{self.opt.name}"



        if self.opt.dataset_name == 'beat' and (self.opt.expression_only or self.opt.net_dim_pose in (192, 333) or self.opt.unidiffuser):
            self.face_mean = np.load(f"data/BEAT/beat_cache/{self.opt.beat_cache_name}/train/facial52/json_mean.npy")
            self.face_std = np.load(f"data/BEAT/beat_cache/{self.opt.beat_cache_name}/train/facial52/json_std.npy")
            self.facial_list = ['browDownLeft', 'browDownRight', 'browInnerUp', 'browOuterUpLeft', 
                                'browOuterUpRight', 'cheekPuff', 'cheekSquintLeft', 'cheekSquintRight', 
                                'eyeBlinkLeft', 'eyeBlinkRight', 'eyeLookDownLeft', 'eyeLookDownRight', 
                                'eyeLookInLeft', 'eyeLookInRight', 'eyeLookOutLeft', 'eyeLookOutRight', 
                                'eyeLookUpLeft', 'eyeLookUpRight', 'eyeSquintLeft', 'eyeSquintRight', 
                                'eyeWideLeft', 'eyeWideRight', 'jawForward', 'jawLeft', 'jawOpen', 
                                'jawRight', 'mouthClose', 'mouthDimpleLeft', 'mouthDimpleRight', 
                                'mouthFrownLeft', 'mouthFrownRight', 'mouthFunnel', 'mouthLeft', 
                                'mouthLowerDownLeft', 'mouthLowerDownRight', 'mouthPressLeft', 
                                'mouthPressRight', 'mouthPucker', 'mouthRight', 'mouthRollLower', 
                                'mouthRollUpper', 'mouthShrugLower', 'mouthShrugUpper', 'mouthSmileLeft', 
                                'mouthSmileRight', 'mouthStretchLeft', 'mouthStretchRight', 'mouthUpperUpLeft', 
                                'mouthUpperUpRight', 'noseSneerLeft', 'noseSneerRight']
        

    @staticmethod
    def zero_grad(opt_list):
        for opt in opt_list:
            opt.zero_grad()

    @staticmethod
    def clip_norm(network_list):
        for network in network_list:
            clip_grad_norm_(network.parameters(), 0.5)

    @staticmethod
    def step(opt_list):
        for opt in opt_list:
            opt.step()

    def forward(self, batch_data, eval_mode=False, add_cond={}, inpaint_dict=None):
        audio_emb, motions, p_id = batch_data
        if self.opt.use_single_style:
            p_id = torch.zeros_like(p_id)
            p_id[:, :1] = 1
        

        audio_emb = audio_emb.detach().to(self.device).float()
        motions = motions.detach().to(self.device).float()
        p_id = p_id.detach().to(self.device).float()
        
        self.audio_emb = audio_emb
        self.motions = motions
        x_start = motions
        B, T = x_start.shape[:2]
        
        cur_len = torch.LongTensor([T for ii in range(B)]).to(self.device)
        
        if self.opt.flow_matching:
            t = None
        else:
            t, _ = self.sampler.sample(B, x_start.device)

        # In FM mode the model receives y=None (inpainting is handled by the
        # FM sampler, not the model's internal outpainting logic).
        model_y = None if self.opt.flow_matching else inpaint_dict

        output = self.diffusion.training_losses(
            model=self.encoder,
            x_start=x_start,
            t=t,
            model_kwargs={
                "audio_emb": audio_emb,
                "length": cur_len,
                "person_id": p_id,
                "add_cond": add_cond,
                "y": model_y,
                "pe_type": self.opt.PE
            }
        )

        self.real_noise = output['target']
        self.fake_noise = output['pred']

        self.real_vel = output['target_vel']
        self.fake_vel = output['pred_vel']
        # FM provides acceleration fields; DDPM does not.
        if self.opt.flow_matching:
            self.real_acc = output['target_acc']
            self.fake_acc = output['pred_acc']
        # Always store x0 fields; FM's training_losses() populates them too.
        if self.opt.model_mean_type == 'epsilon' or self.opt.flow_matching:
            self.real_x0 = output['target_x0']
            self.fake_x0 = output['pred_x0']

        try:
            self.src_mask = self.encoder.module.generate_src_mask(T, cur_len).to(x_start.device)
        except:
            self.src_mask = self.encoder.generate_src_mask(T, cur_len).to(x_start.device)
        
        

    def generate_batch(self, audio_emb, p_id, dim_pose, add_cond={}, inpaint_dict=None):
        audio_emb = audio_emb.to(self.device)
        B = len(audio_emb)
        T = audio_emb.shape[1]
        cur_len = torch.LongTensor([T for ii in range(B)]).to(self.device)

        if self.opt.flow_matching:
            # In FM mode, inpainting is handled by sample_with_inpaint();
            # the model always receives y=None.
            fm_model_kwargs = {
                "audio_emb": audio_emb,
                "length": cur_len,
                "person_id": p_id,
                "add_cond": add_cond,
                "y": None,
                "pe_type": self.opt.PE
            }
            has_inpaint = (
                inpaint_dict is not None
                and inpaint_dict.get('outpainting_mask') is not None
            )
            if has_inpaint:
                output = self.diffusion.sample_with_inpaint(
                    self.encoder,
                    (B, T, dim_pose),
                    model_kwargs=fm_model_kwargs,
                    inpaint_dict=inpaint_dict,
                    progress=True,
                    device=self.device
                )
            else:
                output = self.diffusion.sample(
                    self.encoder,
                    (B, T, dim_pose),
                    model_kwargs=fm_model_kwargs,
                    progress=True,
                    device=self.device
                )
        elif self.opt.ddim:
            output = self.diffusion_ddim_val.ddim_sample_loop(
                self.encoder,
                (B, T, dim_pose),
                clip_denoised=False,
                progress=True,
                model_kwargs={
                    "audio_emb": audio_emb, 
                    "length": cur_len, 
                    "person_id": p_id,
                    "add_cond": add_cond,
                    "y": inpaint_dict,
                    "pe_type": self.opt.PE
                })
        else:
            output = self.diffusion.p_sample_loop(
                self.encoder,
                (B, T, dim_pose),
                clip_denoised=False,
                progress=True,
                model_kwargs={
                    "audio_emb": audio_emb, 
                    "length": cur_len, 
                    "person_id": p_id,
                    "add_cond": add_cond,
                    "y": inpaint_dict,
                    "pe_type": self.opt.PE
                })

        return output

    def backward_G(self):
        # ------ Primary prediction loss (vector field in FM, noise in DDPM) ------
        expression_mode = self.opt.expression_only or self.opt.gesCondition_expression_only
        split_pos = min(getattr(self.opt, 'split_pos', self.fake_noise.shape[-1]), self.fake_noise.shape[-1])
        if self.opt.expr_weight == 1 or expression_mode or split_pos == self.fake_noise.shape[-1]:
            loss_model_pred = self.mse_criterion(self.fake_noise, self.real_noise).mean(dim=-1)
        else:
            loss_model_pred = self.mse_criterion(self.fake_noise[..., :split_pos],
                                              self.real_noise[..., :split_pos]).mean(dim=-1) + \
                            self.mse_criterion(self.fake_noise[..., split_pos:],
                                              self.real_noise[..., split_pos:]).mean(dim=-1) * \
                            self.opt.expr_weight
        loss_model_pred = (loss_model_pred * self.src_mask).sum() / self.src_mask.sum()
        self.loss_model_pred = 1000 * loss_model_pred
        self.final_loss = self.loss_model_pred
        loss_logs = OrderedDict({})
        loss_logs['loss_model_pred'] = self.loss_model_pred.item()

        if self.opt.flow_matching:
            # ------ FM auxiliary losses: Velocity + Acceleration + Huber x0 ------
            if self.opt.add_vel_loss and self.epoch > self.opt.vel_loss_start:
                vel_weight = getattr(self.opt, 'vel_loss_weight', 100.0)
                acc_weight = getattr(self.opt, 'acc_loss_weight', 50.0)
                jerk_weight = getattr(self.opt, 'jerk_loss_weight', 0.0)
                x0_weight  = getattr(self.opt, 'x0_rec_weight', 100.0)

                # Gesture rotations live on SO(3), while expression-only
                # targets are ordinary blendshape scalars.  Keep the two
                # temporal losses in their physically appropriate spaces.
                if self.opt.expression_only or self.opt.gesCondition_expression_only:
                    real_vel, real_acc, real_jerk = self._linear_kinematics(self.real_x0)
                    fake_vel, fake_acc, fake_jerk = self._linear_kinematics(self.fake_x0)
                else:
                    real_vel, real_acc, real_jerk = self._gesture_angular_kinematics(self.real_x0)
                    fake_vel, fake_acc, fake_jerk = self._gesture_angular_kinematics(self.fake_x0)

                # 1) 1st-order angular velocity loss
                loss_vel_rec = self._mean_feature_loss(fake_vel, real_vel)
                loss_vel_rec = (loss_vel_rec * self.src_mask[:, :-1]).sum() / self.src_mask[:, :-1].sum()
                self.loss_vel_rec = vel_weight * loss_vel_rec
                loss_logs['loss_vel_rec'] = self.loss_vel_rec.item()
                self.final_loss += self.loss_vel_rec

                # 2) 2nd-order angular acceleration loss
                if acc_weight > 0:
                    loss_acc_rec = self._mean_feature_loss(fake_acc, real_acc)
                    loss_acc_rec = (loss_acc_rec * self.src_mask[:, :-2]).sum() / self.src_mask[:, :-2].sum()
                    self.loss_acc_rec = acc_weight * loss_acc_rec
                    loss_logs['loss_acc_rec'] = self.loss_acc_rec.item()
                    self.final_loss += self.loss_acc_rec

                # 3) 3rd-order jerk loss
                if jerk_weight > 0 and fake_jerk.shape[1] > 0:
                    loss_jerk_rec = self._mean_feature_loss(fake_jerk, real_jerk)
                    loss_jerk_rec = (
                        loss_jerk_rec * self.src_mask[:, :-3]
                    ).sum() / self.src_mask[:, :-3].sum()
                    self.loss_jerk_rec = jerk_weight * loss_jerk_rec
                    loss_logs['loss_jerk_rec'] = self.loss_jerk_rec.item()
                    self.final_loss += self.loss_jerk_rec

                # 4) x0 Huber reconstruction loss
                if self.opt.dataset_name == 'beat' and self.opt.sem_rep is not None:
                    loss_x0_rec = self.huber_loss(
                        self.real_x0 * (self.in_sem.unsqueeze(2) + 1),
                        self.fake_x0 * (self.in_sem.unsqueeze(2) + 1)
                    )
                else:
                    loss_x0_rec = self.huber_loss(self.real_x0, self.fake_x0)
                self.loss_x0_rec = x0_weight * loss_x0_rec
                loss_logs['loss_x0_rec'] = self.loss_x0_rec.item()
                self.final_loss += self.loss_x0_rec

            # 4) 6D orthogonalization loss (only for rot_6d + FM)
            if getattr(self.opt, 'rot_6d', False) and self.opt.ortho_loss_weight > 0:
                # Extract gesture-only 6D dims (skip expression if concatenated)
                fake_ges_6d = self.fake_x0[..., :self.opt.split_pos]  # [B, T, 282]
                B_o, T_o, D_o = fake_ges_6d.shape
                n_joints = D_o // 6
                pred_6d = fake_ges_6d.reshape(B_o * T_o, n_joints, 6)
                # Gram-Schmidt → back to 6D (the "correct" 6D)
                pred_mat = rot_cvt.rotation_6d_to_matrix(pred_6d)          # [B*T, J, 3, 3]
                pred_6d_ortho = rot_cvt.matrix_to_rotation_6d(pred_mat)    # [B*T, J, 6]
                loss_ortho = F.mse_loss(pred_6d, pred_6d_ortho.detach())
                self.loss_ortho = self.opt.ortho_loss_weight * loss_ortho
                loss_logs['loss_ortho'] = self.loss_ortho.item()
                self.final_loss += self.loss_ortho

            # 5) Diversity loss (prevents mode collapse)
            div_weight = getattr(self.opt, 'diversity_loss_weight', 0.0)
            if div_weight > 0 and self.fake_x0.shape[0] > 1:
                # Encourage pairwise distance between samples in the batch
                B_d = self.fake_x0.shape[0]
                flat = self.fake_x0.reshape(B_d, -1)  # [B, T*D]
                # Negative mean pairwise distance → minimize = maximize diversity
                self.loss_diversity = -div_weight * torch.pdist(flat).mean()
                loss_logs['loss_div'] = self.loss_diversity.item()
                self.final_loss += self.loss_diversity
        else:
            # ------ DDPM auxiliary losses ------
            if self.opt.add_vel_loss and self.epoch > self.opt.vel_loss_start:
                loss_vel_rec = self.mse_criterion(self.fake_vel, self.real_vel).mean(dim=-1)
                loss_vel_rec = (loss_vel_rec * self.src_mask[:, :-1]).sum() / self.src_mask[:, :-1].sum()
                self.loss_vel_rec = 100 * loss_vel_rec
                loss_logs['loss_vel_rec'] = self.loss_vel_rec.item()
                self.final_loss += loss_vel_rec

                if self.opt.model_mean_type == 'epsilon':
                    if self.opt.dataset_name == 'beat' and self.opt.sem_rep is not None:
                        loss_x0_rec = self.huber_loss(
                            self.real_x0 * (self.in_sem.unsqueeze(2) + 1),
                            self.fake_x0 * (self.in_sem.unsqueeze(2) + 1)
                        )
                    else:
                        loss_x0_rec = self.huber_loss(self.real_x0, self.fake_x0)
                    self.loss_x0_rec = 100 * loss_x0_rec
                    loss_logs['loss_x0_rec'] = self.loss_x0_rec.item()
                    self.final_loss += self.loss_x0_rec

        loss_logs['final_loss'] = self.final_loss.item()
        return loss_logs

    def update(self, loss_divisor=1, step_optimizer=True):
        loss_logs = self.backward_G()
        (self.final_loss / float(loss_divisor)).backward()
        if step_optimizer:
            self.clip_norm([self.encoder])
            self.step([self.opt_encoder])
            self.zero_grad([self.opt_encoder])

        return loss_logs

    def to(self, device):
        if self.opt.is_train:
            self.mse_criterion.to(device)
        self.encoder = self.encoder.to(device)

    def train_mode(self):
        self.encoder.train()

    def eval_mode(self):
        self.encoder.eval()

    def save(self, file_name, ep, total_it, fgd, mse, pck, best_fgd, best_mse, best_pck):
        config = {}
        for key, value in vars(self.opt).items():
            if isinstance(value, (str, int, float, bool, type(None), list, tuple, dict)):
                config[key] = value
            else:
                config[key] = str(value)
        state = {
            'opt_encoder': self.opt_encoder.state_dict(),
            'ep': ep,
            'total_it': total_it,
            'FGD': fgd,
            'best_fgd': best_fgd,
            'MSE': mse,
            'best_mse': best_mse,
            'PCK': pck,
            'best_pck': best_pck,
            'config': config,
        }
        try:
            state['encoder'] = self.encoder.module.state_dict()
        except:
            state['encoder'] = self.encoder.state_dict()
        # Keep the previous checkpoint recoverable until the replacement has
        # been written completely.  This matters for multi-GB optimizer states.
        tmp_name = f"{file_name}.tmp-{os.getpid()}"
        try:
            torch.save(state, tmp_name)
            os.replace(tmp_name, file_name)
        finally:
            if os.path.exists(tmp_name):
                os.remove(tmp_name)
        return

    def load(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.device)
        saved_config = checkpoint.get('config')

        if (
            self.opt.mode == 'train'
            and self.opt.resume
            and not saved_config
            and not getattr(self.opt, 'allow_legacy_checkpoint', False)
        ):
            raise RuntimeError(
                "Refusing to resume a legacy checkpoint without saved config. "
                "Start a fresh experiment, or pass --allow_legacy_checkpoint "
                "only if legacy behavior is intentional."
            )

        if self.opt.flow_matching and self.opt.fm_expression_condition == 'auto':
            saved_condition = (saved_config or {}).get('fm_expression_condition')
            # Checkpoints predating config snapshots were trained with vector-
            # field conditioning. New checkpoints record x0 explicitly.
            self.opt.fm_expression_condition = (
                saved_condition if saved_condition in ('x0', 'velocity') else 'velocity'
            )
            print(f"FM expression conditioning: {self.opt.fm_expression_condition}")
        elif self.opt.flow_matching and saved_config:
            saved_condition = saved_config.get('fm_expression_condition')
            if (
                self.opt.mode == 'train'
                and saved_condition in ('x0', 'velocity')
                and saved_condition != self.opt.fm_expression_condition
            ):
                raise RuntimeError(
                    "Checkpoint FM conditioning mismatch: "
                    f"checkpoint={saved_condition}, requested={self.opt.fm_expression_condition}"
                )

        if self.opt.mode == 'train' and saved_config:
            critical_fields = (
                'dataset_name', 'n_poses', 'net_dim_pose', 'split_pos',
                'rot_6d', 'flow_matching', 'unidiffuser', 'motion_cache_ids',
                'hubert_cache_bindings', 'axis_angle', 'addHubert',
                'encode_hubert', 'vel_loss_weight', 'acc_loss_weight',
                'jerk_loss_weight', 'x0_rec_weight',
            )
            mismatches = []
            for field in critical_fields:
                if field in saved_config and hasattr(self.opt, field):
                    current = getattr(self.opt, field)
                    if saved_config[field] != current:
                        mismatches.append(
                            f"{field}: checkpoint={saved_config[field]!r}, current={current!r}"
                        )
            if mismatches:
                raise RuntimeError(
                    "Checkpoint configuration mismatch:\n  " + "\n  ".join(mismatches)
                )

        if self.opt.PE == "pe_sinu_repeat":
            mm = checkpoint['encoder']['PE.pe'][:, :self.n_poses, :]
            checkpoint['encoder']['PE.pe'] = torch.cat((mm, mm, mm, mm), -2)
        if self.opt.is_train:
            self.opt_encoder.load_state_dict(checkpoint['opt_encoder'])
        
        target_encoder = self.encoder.module if hasattr(self.encoder, 'module') else self.encoder
        load_result = target_encoder.load_state_dict(checkpoint['encoder'], strict=False)
        missing_keys = list(load_result.missing_keys)
        unexpected_keys = list(load_result.unexpected_keys)
        if missing_keys or unexpected_keys:
            message = (
                f"Checkpoint model keys do not match: missing={missing_keys}, "
                f"unexpected={unexpected_keys}"
            )
            if getattr(self.opt, 'allow_partial_checkpoint', False):
                print("WARNING:", message)
            else:
                raise RuntimeError(message)


        return checkpoint['ep'], checkpoint.get('total_it', 0), \
                checkpoint.get('best_fgd', 99999), checkpoint.get('best_mse', 99999), \
                checkpoint.get('best_pck', 0)
    
    def load_fid_net(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.device)
        try:
            state_dict = checkpoint['model_state']
        except:
            state_dict = checkpoint['state_dict']
        try:
            self.eval_model.load_state_dict(state_dict, strict=True)
        except:
            try:
                self.eval_model.load_state_dict({k.replace('module.', ''):v for k,v in state_dict.items()}, strict=True)
            except:
                self.eval_model.module.load_state_dict(state_dict, strict=True)
    

    def one_hot(self, ids, dim):
        flat_ids = ids.long().reshape(-1)
        return F.one_hot(flat_ids, num_classes=dim).float()

    def _gesture_rotation_matrices(self, motion):
        if getattr(self.opt, 'rot_6d', False):
            joint_count = self.opt.split_pos // 6
            gesture = motion[..., :self.opt.split_pos].reshape(
                *motion.shape[:-1], joint_count, 6
            )
            return rot_cvt.rotation_6d_to_matrix(gesture)

        joint_count = self.opt.split_pos // 3
        gesture = motion[..., :self.opt.split_pos]
        mean = self.axis_angle_mean.to(device=gesture.device, dtype=gesture.dtype)
        std = self.axis_angle_std.to(device=gesture.device, dtype=gesture.dtype)
        axis_angle = (gesture * std + mean).reshape(
            *gesture.shape[:-1], joint_count, 3
        )
        return rot_cvt.axis_angle_to_matrix(axis_angle)

    def _gesture_angular_kinematics(self, motion):
        matrices = self._gesture_rotation_matrices(motion)
        relative = matrices[:, :-1].transpose(-1, -2) @ matrices[:, 1:]
        velocity = rot_cvt.matrix_to_axis_angle(relative)
        acceleration = velocity[:, 1:] - velocity[:, :-1]
        jerk = acceleration[:, 1:] - acceleration[:, :-1]
        return velocity, acceleration, jerk

    @staticmethod
    def _linear_kinematics(motion):
        velocity = motion[:, 1:] - motion[:, :-1]
        acceleration = velocity[:, 1:] - velocity[:, :-1]
        jerk = acceleration[:, 1:] - acceleration[:, :-1]
        return velocity, acceleration, jerk

    def _mean_feature_loss(self, prediction, target):
        """Return one loss value per batch item and frame."""
        elementwise = self.mse_criterion(prediction, target)
        return elementwise.flatten(start_dim=2).mean(dim=-1)
    

    def train(self, train_dataset, val_dataset):
        rank, world_size = get_dist_info()
        self.to(self.device)
        self.opt_encoder = optim.Adam(self.encoder.parameters(), lr=self.opt.lr)
        axis_mean, axis_std = select_axis_angle_stats(
            train_dataset.mean_pose_axis_angle,
            train_dataset.std_pose_axis_angle,
            self.opt.split_pos if not getattr(self.opt, 'rot_6d', False) else self.opt.split_pos // 2,
        )
        self.axis_angle_mean = torch.as_tensor(
            axis_mean, device=self.device, dtype=torch.float32
        )
        self.axis_angle_std = torch.as_tensor(
            np.maximum(axis_std, 1e-2),
            device=self.device,
            dtype=torch.float32,
        )
        it = 0
        cur_epoch = 0
        best_fgd = 99999
        best_mse = 99999
        best_pck = 0
        if self.opt.resume:
            model_dir = pjoin(self.opt.model_dir, self.opt.ckpt)
            saved_epoch, it, best_fgd, best_mse, best_pck = self.load(model_dir)
            cur_epoch = saved_epoch + 1
            if self.opt.reset_lr:
                for i, param_group in enumerate(self.opt_encoder.param_groups):
                    param_group['lr'] = self.opt.lr

        start_time = time.time()

        if self.opt.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=True)
        else:
            train_sampler = None
            val_sampler = None

        loader_workers = max(0, int(self.opt.workers))
        loader_kwargs = {
            'num_workers': loader_workers,
            'pin_memory': True,
        }
        if loader_workers > 0:
            loader_kwargs.update(
                persistent_workers=bool(getattr(self.opt, 'persistent_workers', False)),
                prefetch_factor=max(1, int(getattr(self.opt, 'prefetch_factor', 2))),
            )

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.opt.batch_size, shuffle=(train_sampler is None),
            sampler=train_sampler, **loader_kwargs)

        # Validation only runs periodically. Keeping another full worker pool
        # resident after its first pass wastes RAM and file descriptors.
        val_loader_kwargs = dict(loader_kwargs)
        if loader_workers > 0:
            val_loader_kwargs['persistent_workers'] = False
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=self.opt.batch_size, shuffle=False,
            sampler=val_sampler, **val_loader_kwargs)
        non_blocking_transfer = bool(
            getattr(self.opt, 'non_blocking_transfer', True)
        )

        def move_to_device(tensor):
            return tensor.to(
                self.device,
                non_blocking=non_blocking_transfer,
            )
        
        ##################################### Training #####################################
        print("Start training ...")
        logs = OrderedDict()
        
        for epoch in range(cur_epoch, self.opt.num_epochs):
            self.train_mode()
            self.epoch = epoch
            if self.opt.distributed:
                train_sampler.set_epoch(epoch)

            accumulation_steps = max(1, int(getattr(self.opt, 'grad_accum_steps', 1)))
            self.zero_grad([self.opt_encoder])
            for i, batch_data in enumerate(train_loader):
                # tt1 = time.time()
                if not self.opt.expression_only:
                    if getattr(self.opt, 'rot_6d', False):
                        tar_pose = batch_data["pose_6d"]  # [B, T, 282]
                    elif self.opt.axis_angle:
                        tar_pose = batch_data["pose_axis_angle"]
                    else:
                        tar_pose = batch_data["pose"] # torch.Size([B, 34, 141])
                    if self.opt.remove_hand and not getattr(self.opt, 'rot_6d', False):
                        tar_pose = tar_pose[..., [pp for pp in range(0,21)] + [pp for pp in range(75,87)]]
                    tar_pose = move_to_device(tar_pose)
                if not self.opt.gesture_only:
                    in_facial = move_to_device(batch_data["facial"]) if self.opt.facial_rep is not None else None  # torch.Size([B, 34, 51])
                
                self.in_sem = move_to_device(batch_data["sem"]) if self.opt.sem_rep is not None else None # torch.Size([B, 34])

                if self.opt.textExpEmoCondition_gesture_only:
                    in_word = move_to_device(batch_data["word"]) if self.opt.word_rep is not None else None # torch.Size([B, 34])
                    in_emo = move_to_device(batch_data["emo"]) if self.opt.emo_rep is not None else None # torch.Size([B, 34])
                    in_facial = torch.cat([in_facial, in_word.unsqueeze(-1), in_emo.unsqueeze(-1)], dim=-1)

                if self.opt.expression_only or self.opt.gesCondition_expression_only:
                    motions = in_facial
                elif self.opt.gesture_only or self.opt.expCondition_gesture_only != None or \
                                                    self.opt.textExpEmoCondition_gesture_only:
                    motions = tar_pose
                else:
                    motions = torch.cat((tar_pose, in_facial), dim=-1)

                audio_emb = move_to_device(batch_data["aud_feat"]) if self.opt.audio_rep is not None else None

                if self.opt.expCondition_gesture_only:
                    audio_emb = torch.cat((audio_emb, in_facial), dim=-1)
                elif self.opt.gesCondition_expression_only:
                    audio_emb = torch.cat((audio_emb, tar_pose), dim=-1)
                
                add_cond = {}

                if self.opt.addTextCond:
                    in_word = move_to_device(batch_data["word"]) if self.opt.word_rep is not None else None # torch.Size([B, 34])
                    add_cond['text'] = in_word
                if self.opt.addEmoCond:
                    in_emo = move_to_device(batch_data["emo"]) if self.opt.emo_rep is not None else None # torch.Size([B, 34])
                    add_cond['emo'] = in_emo
                if self.opt.expAddHubert or self.opt.addHubert:
                    add_cond["pretrain_aud_feat"] = move_to_device(batch_data["pretrain_aud_feat"])
                


                p_id = batch_data["id"] if self.opt.speaker_id else None # torch.Size([256, 1])
                p_id = move_to_device(self.one_hot(p_id, self.opt.speaker_dim))

                if self.opt.remove_audio:
                    audio_emb = torch.zeros_like(audio_emb).to(audio_emb.device)
                if self.opt.remove_style:
                    p_id = torch.zeros_like(p_id).to(p_id.device)
                


                batch_data = [audio_emb, motions, p_id]

                
                
                inpaint_dict = {}
                if self.opt.overlap_len > 0:
                    inpaint_dict['gt'] = motions
                    inpaint_dict['outpainting_mask'] = torch.zeros_like(motions, dtype=torch.bool,
                                                    device=motions.device)  # Do inpainting/generation in those frames
                    inpaint_dict['outpainting_mask'][..., :self.opt.overlap_len, :] = True  # True means use gt motion 
                self.forward(batch_data, add_cond=add_cond, inpaint_dict=inpaint_dict)
                group_start = (i // accumulation_steps) * accumulation_steps
                group_size = min(accumulation_steps, len(train_loader) - group_start)
                should_step = (i - group_start + 1) == group_size
                log_dict = self.update(
                    loss_divisor=group_size,
                    step_optimizer=should_step,
                )
                for k, v in log_dict.items():
                    if k not in logs:
                        logs[k] = v
                    else:
                        logs[k] += v
                it += 1
                if it % self.opt.log_every == 0:
                    mean_loss = OrderedDict({})
                    for tag, value in logs.items():
                        mean_loss[tag] = value / self.opt.log_every
                    logs = OrderedDict()
                    
                    if rank == 0:
                        print_current_loss(start_time, it, mean_loss, epoch, inner_iter=i)

                    if not self.opt.debug:
                        wandb.log(mean_loss, step=epoch)

                if self.opt.debug:
                    break

            latest_every_e = max(1, int(getattr(self.opt, 'latest_every_e', 1)))
            should_save_latest = (
                (epoch + 1) % latest_every_e == 0
                or (epoch + 1) == self.opt.num_epochs
            )
            if rank == 0 and should_save_latest:
                self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it, None, None, None, best_fgd, best_mse, best_pck)

            if (epoch+1) % self.opt.save_every_e == 0 and rank == 0:
                self.save(pjoin(self.opt.model_dir, 'ckpt_e%03d.tar'%(epoch)),
                            epoch, it, None, None, None, best_fgd, best_mse, best_pck)
            
            ################################## Evaluation ####################################
            
            
            # embed_space_evaluator = EmbeddingSpaceEvaluator(self.opt.embed_net_path, self.device)
            # if (epoch+1) % self.opt.eval_every_e == 0 and rank == 0:
            if (epoch+1) % self.opt.eval_every_e == 0:
                self.eval_mode()
                count = 0.0
                diversity = AverageMeter('diversity')
                pck = AverageMeter('pck')
                srgr = AverageMeter('srgr')
                mse = AverageMeter('mse')
                fgd = AverageMeter('fgd')
                progress = ProgressMeter(
                    len(val_loader) + (self.opt.distributed and (len(val_loader.sampler) * self.opt.world_size < len(val_loader.dataset))),
                    [diversity, pck, srgr, mse],
                    prefix='Test: ')
                with torch.no_grad():
                    for i, batch_data in tqdm(enumerate(val_loader)):
                        if not self.opt.expression_only:
                            if getattr(self.opt, 'rot_6d', False):
                                tar_pose = batch_data["pose_6d"]  # [B, T, 282]
                            elif self.opt.axis_angle:
                                tar_pose = batch_data["pose_axis_angle"]
                                tar_pose_euler = move_to_device(batch_data["pose"])
                            else:
                                tar_pose = batch_data["pose"] # torch.Size([256, 34, 141])
                            if self.opt.remove_hand and not getattr(self.opt, 'rot_6d', False):
                                tar_pose = tar_pose[..., [pp for pp in range(0,21)] + [pp for pp in range(75,87)]]
                            tar_pose = move_to_device(tar_pose)
                        if not self.opt.gesture_only:
                            in_facial = move_to_device(batch_data["facial"]) if self.opt.facial_rep is not None else None  # torch.Size([256, 34, 51])
                        
                        if self.opt.textExpEmoCondition_gesture_only:
                            in_word = move_to_device(batch_data["word"]) if self.opt.word_rep is not None else None # torch.Size([256, 34])
                            in_emo = move_to_device(batch_data["emo"]) if self.opt.emo_rep is not None else None # torch.Size([256, 34])
                            in_facial = torch.cat([in_facial, in_word.unsqueeze(-1), in_emo.unsqueeze(-1)], dim=-1)
                        

                        if self.opt.expression_only or self.opt.gesCondition_expression_only:
                            motions = in_facial
                        elif self.opt.gesture_only or self.opt.expCondition_gesture_only != None or self.opt.textExpEmoCondition_gesture_only:
                            motions = tar_pose
                        else:
                            motions = torch.cat((tar_pose, in_facial), dim=-1)

                        audio_emb = move_to_device(batch_data["aud_feat"]) if self.opt.audio_rep is not None else None

                        if self.opt.expCondition_gesture_only:
                            audio_emb = torch.cat((audio_emb, in_facial), dim=-1)
                        elif self.opt.gesCondition_expression_only:
                            audio_emb = torch.cat((audio_emb, tar_pose), dim=-1)
                        
                        add_cond = {}
                        if self.opt.addTextCond:
                            in_word = move_to_device(batch_data["word"]) if self.opt.word_rep is not None else None # torch.Size([B, 34])
                            add_cond['text'] = in_word
                        if self.opt.addEmoCond:
                            in_emo = move_to_device(batch_data["emo"]) if self.opt.emo_rep is not None else None # torch.Size([B, 34])
                            add_cond['emo'] = in_emo
                        if self.opt.expAddHubert or self.opt.addHubert:
                            add_cond["pretrain_aud_feat"] = move_to_device(batch_data["pretrain_aud_feat"])

                        p_id = batch_data["id"] if self.opt.speaker_id else None # torch.Size([256, 1])
                        p_id = move_to_device(self.one_hot(p_id, self.opt.speaker_dim))

                        if self.opt.remove_audio:
                            audio_emb = torch.zeros_like(audio_emb).to(audio_emb.device)
                        if self.opt.remove_style:
                            p_id = torch.zeros_like(p_id).to(p_id.device)

                        # Save sem for SRGR before batch_data is overwritten
                        in_sem = batch_data["sem"]  # [B, T]

                        batch_data = [audio_emb, motions, p_id]

                        if self.opt.use_single_style:
                            
                            p_id = torch.zeros_like(p_id)
                            p_id[:, :1] = 1

                        audio_emb = audio_emb.detach().float()
                        motions = motions.detach().float()
                        p_id = p_id.detach().float()
                        count += len(motions)

                        # Independent validation clips must not leak GT prefix
                        # frames into the generated sample or its metrics.
                        outputs = self.generate_batch(
                            audio_emb, p_id, self.opt.net_dim_pose, add_cond, {}
                        )
                        B, seq, C = outputs.shape

                        expression_mode = self.opt.expression_only or self.opt.gesCondition_expression_only
                        if expression_mode:
                            outputs_for_eval = outputs
                            motions_for_eval = motions
                            pck_val, mse_val, metric_errors = linear_pck_mse(
                                outputs, motions, threshold=0.5
                            )
                        else:
                            metric_kwargs = dict(
                                rot_6d=bool(getattr(self.opt, 'rot_6d', False)),
                                split_pos=self.opt.split_pos,
                                mean_axis_angle=val_dataset.mean_pose_axis_angle,
                                std_axis_angle=val_dataset.std_pose_axis_angle,
                            )
                            outputs_for_eval = normalized_axis_angle(outputs, **metric_kwargs)
                            motions_for_eval = normalized_axis_angle(motions, **metric_kwargs)
                            pck_val, mse_val, metric_errors = rotation_pck_mse(
                                outputs, motions, threshold=0.5, **metric_kwargs
                            )

                        if not self.opt.no_fgd:
                            latent_out = self.eval_model(outputs_for_eval[:, :34].float())
                            latent_ori = self.eval_model(motions_for_eval[:, :34].float())
                            #print(latent_out,latent_ori)
                            if i == 0:
                                latent_out_all = latent_out.cpu().numpy()
                                latent_ori_all = latent_ori.cpu().numpy()
                            else:
                                latent_out_all = np.concatenate([latent_out_all, latent_out.cpu().numpy()], axis=0)
                                latent_ori_all = np.concatenate([latent_ori_all, latent_ori.cpu().numpy()], axis=0)

                        

                        correct = (metric_errors < 0.5).cpu().numpy()
                        outputs_np = outputs_for_eval.cpu().numpy()
                        outputs, motions = outputs.cpu(), motions.cpu()
                        
                        # SRGR: semantic-weighted PCK (BEAT paper, Liu et al. 2022)
                        # sem: [B, T] with values 0-1 (0.1=beat gesture, 0.2-1.0=semantic)
                        in_sem_np = in_sem.numpy()  # [B, T]
                        # Mean PCK per frame (average over joints): [B, T]
                        pck_per_frame = np.mean(correct, axis=2)  # [B, T]
                        # SRGR = weighted average: sum(sem * pck_per_frame) / sum(sem)
                        sem_sum = np.sum(in_sem_np)
                        if sem_sum > 0:
                            srgr_val = np.sum(in_sem_np * pck_per_frame) / sem_sum
                        else:
                            srgr_val = pck_val  # fallback to PCK if no sem data
                        
                        ### diversity
                        B_div = 50 ## In Ye et al. (ECCV'22), Batch size is 50 when evaluating diversity
                        if B < B_div:
                            B_div = B
                        # out_split = outputs.split(B_div, dim=0)
                        if B_div >= 2:
                            out_split = np.split(outputs_np, np.arange(B_div, B, B_div), axis=0)
                            for idx in range(B // B_div):
                                div_val = 0.0
                                for ii in range(0, B_div):
                                    for jj in range(ii+1, B_div):
                                        dif = out_split[idx][ii] - out_split[idx][jj]
                                        div_val += np.mean(np.absolute(dif)) ### TODO: check: np.mean() or np.sum()
                                div_val = div_val * 2 / (B_div * (B_div - 1))
                                diversity.update(div_val, B_div)

                        
                        # update in averagemeter
                        mse.update(mse_val, B)
                        pck.update(pck_val, B)
                        srgr.update(srgr_val, B)

                        if self.opt.debug or (self.opt.max_eval_samples != -1 and pck.count >= self.opt.max_eval_samples):
                            break

                if self.opt.distributed:  #  all_reduce() is for aggreagating results on each rank
                    diversity.all_reduce()
                    mse.all_reduce()
                    pck.all_reduce()
                    srgr.all_reduce()
                
                if rank == 0:
                    if not self.opt.debug:
                        wandb.log({"MSE": mse.avg, "PCK": pck.avg, "SRGR": srgr.avg, "Diversity": diversity.avg}, step = epoch)
                    print(f"[Validation]: Epoch: {epoch}, MSE: {mse.avg}, PCK: {pck.avg}, SRGR: {srgr.avg}, Diversity: {diversity.avg}")
                
                if not self.opt.no_fgd:
                    fgd_val = data_tools.FIDCalculator.frechet_distance(latent_out_all, latent_ori_all)
                    fgd.update(fgd_val)
                    if self.opt.distributed:
                        fgd.all_reduce()
                    if rank == 0:
                        if not self.opt.debug:
                            wandb.log({"FGD": fgd.avg}, step = epoch)
                        print(f"[Validation]: Epoch: {epoch}, FGD: {fgd.avg}")

                if best_fgd > fgd.avg and rank == 0 and self.eval_model != None:
                    best_fgd = fgd.avg
                    self.save(pjoin(self.opt.model_dir, 'fgd_best.tar'), epoch, it, fgd.avg, mse.avg, pck.avg, best_fgd, best_mse, best_pck)
                if best_mse > mse.avg and rank == 0:
                    best_mse = mse.avg
                    self.save(pjoin(self.opt.model_dir, 'mse_best.tar'), epoch, it, fgd.avg, mse.avg, pck.avg, best_fgd, best_mse, best_pck)
                if best_pck < pck.avg and rank == 0:
                    best_pck = pck.avg
                    self.save(pjoin(self.opt.model_dir, 'pck_best.tar'), epoch, it, fgd.avg, mse.avg, pck.avg, best_fgd, best_mse, best_pck)
                # print("Finished Validation")


    def test(self, test_dataset):
        rank, world_size = get_dist_info()
        self.to(self.device)
        self.opt.is_train = False
        cur_epoch = 0

        model_dir = pjoin(self.opt.model_dir, self.opt.ckpt)
        cur_epoch, it, _, _, _ = self.load(model_dir)

        if self.opt.distributed:
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False, drop_last=False)
        else:
            test_sampler = None
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.opt.batch_size, shuffle=False,
            num_workers=self.opt.workers, pin_memory=True, sampler=test_sampler)

        logs = OrderedDict()

        self.encoder.eval()
        ckpt_epoch = f"ckpt_e{cur_epoch}"
        if 'fgd_best' in model_dir:
            ckpt_epoch = f"BestFGD_e{cur_epoch}"
        elif 'mse_best' in model_dir:
            ckpt_epoch = f"BestMSE_e{cur_epoch}"
        elif 'pck_best' in model_dir:
            ckpt_epoch = f"BestPCK_e{cur_epoch}"
        if self.opt.ddim:
            ckpt_epoch = ckpt_epoch + f"_{self.opt.timestep_respacing}" 

        results_dir = pjoin("results", f"{self.opt.dataset_name}_{self.opt.n_poses}", self.opt.mode, self.opt.name, ckpt_epoch)

        if self.opt.fix_very_first:
            results_dir = results_dir.replace(self.opt.name, f"{self.opt.name}/fixStart{self.opt.overlap_len}_fix_very_first")
        else:
            results_dir = results_dir.replace(self.opt.name, f"{self.opt.name}/fixStart{self.opt.overlap_len}")

        if self.opt.classifier_free:
            results_dir = results_dir + f"_ClsFreeCondScale{self.opt.cond_scale}"
            
            
        if not os.path.exists(results_dir):
            os.makedirs(results_dir, exist_ok=True)
        
        middle_name = self.opt.mode
        if self.opt.test_on_trainset:
            results_dir = results_dir.replace(self.opt.mode, "test_on_trainset")
            middle_name = "test_on_trainset"
        elif self.opt.test_on_val:
            results_dir = results_dir.replace(self.opt.mode, "test_on_val")
            middle_name = "test_on_val"

        if self.opt.usePredExpr:
            results_dir = results_dir.replace(middle_name, middle_name + "_usePredExpr")
        if self.opt.output_gt:
            results_dir = results_dir.replace(middle_name, middle_name + "_GT")
        if getattr(self.opt, 'sequential_test_windows', False):
            results_dir = results_dir + f"_sequential_overlap{self.opt.overlap_len}"
        result_tag = safe_result_tag(getattr(self.opt, 'test_result_tag', None))
        if result_tag:
            results_dir = results_dir + f"_{result_tag}"

        
        if not os.path.exists(results_dir):
            os.makedirs(results_dir, exist_ok=True)
        if self.opt.expression_only:
            json_dir = os.path.join(results_dir, "face_json")
            os.makedirs(json_dir, exist_ok=True)

        if self.opt.unidiffuser or self.opt.net_dim_pose == 192:
            ori_results_dir = results_dir
            results_dir = os.path.join(ori_results_dir, "gesture")
            results_dir_expr = os.path.join(ori_results_dir, "expression")
            results_dir_aud = os.path.join(ori_results_dir, "audio")

            json_dir = os.path.join(results_dir_expr, "face_json")
            os.makedirs(json_dir, exist_ok=True)

            os.makedirs(results_dir, exist_ok=True)
            os.makedirs(results_dir_expr, exist_ok=True)
            os.makedirs(results_dir_aud, exist_ok=True)

            if self.opt.visualize_unify_x0_step:
                self.opt.unify_x0_step_path = os.path.join(results_dir_expr, "visualization_unify_x0")
                os.makedirs(self.opt.unify_x0_step_path, exist_ok=True)


        count = 0
        previous_sequential_output = None
        previous_sequential_audio = None
        previous_sequential_person = None
        for i, batch_data in enumerate(test_loader):
            if not self.opt.expression_only:
                if getattr(self.opt, 'rot_6d', False):
                    tar_pose = batch_data["pose_6d"]  # [B, T, 282]
                elif self.opt.axis_angle:
                    tar_pose = batch_data["pose_axis_angle"]
                else:
                    tar_pose = batch_data["pose"] # torch.Size([256, 34, 141])
                if self.opt.remove_hand and not getattr(self.opt, 'rot_6d', False):
                    tar_pose = tar_pose[..., [pp for pp in range(0,21)] + [pp for pp in range(75,87)]]
                tar_pose = tar_pose.to(self.device)
            if not self.opt.gesture_only:
                in_facial = batch_data["facial"].detach().to(self.device) if self.opt.facial_rep is not None else None  # torch.Size([256, 34, 51])
            
            
            if self.opt.textExpEmoCondition_gesture_only:
                in_word = batch_data["word"].to(self.device) if self.opt.word_rep is not None else None # torch.Size([256, 34])
                in_emo = batch_data["emo"].to(self.device) if self.opt.emo_rep is not None else None # torch.Size([256, 34])
                in_facial = torch.cat([in_facial, in_word.unsqueeze(-1), in_emo.unsqueeze(-1)], dim=-1)
            # in_sem = batch_data["sem"].detach().to(self.device) if self.opt.sem_rep is not None else None # torch.Size([256, 34])
            if self.opt.expression_only or self.opt.gesCondition_expression_only:
                motions = in_facial
            elif self.opt.gesture_only or self.opt.expCondition_gesture_only != None or self.opt.textExpEmoCondition_gesture_only:
                motions = tar_pose
            else:
                motions = torch.cat((tar_pose, in_facial), dim=-1)
            audio_emb = batch_data["aud_feat"].detach().to(self.device) if self.opt.audio_rep is not None else None # 
            audio_raw = batch_data["audio"].detach().to(self.device)
            if self.opt.expCondition_gesture_only:
                audio_emb = torch.cat((audio_emb, in_facial), dim=-1)
            elif self.opt.gesCondition_expression_only:
                audio_emb = torch.cat((audio_emb, tar_pose), dim=-1)

            add_cond = {}
            if self.opt.expAddHubert or self.opt.addHubert:
                add_cond["pretrain_aud_feat"] = batch_data["pretrain_aud_feat"].to(self.device)


            p_id = batch_data["id"] if self.opt.speaker_id else None # torch.Size([256, 1])
            p_id = self.one_hot(p_id, self.opt.speaker_dim).detach().to(self.device)

            if self.opt.remove_audio:
                audio_emb = torch.zeros_like(audio_emb).to(audio_emb.device)
            if self.opt.remove_style:
                p_id = torch.zeros_like(p_id).to(p_id.device)
            
            # if self.opt.visualize_unify_x0_step:
            #     self.x0_save_path = os.path.join(result_dir, "unify_x0_step")

            batch_data = [audio_emb, motions, p_id]

            if not self.opt.output_gt:
                ###### debug
                if self.opt.use_single_style:
                    p_id = torch.zeros_like(p_id)
                    p_id[:, :1] = 1
                ###### debug end

                # A standalone test clip must be generated without copying a
                # ground-truth prefix into the result.
                if getattr(self.opt, 'sequential_test_windows', False):
                    sequential_outputs = []
                    for sample_index in range(audio_emb.shape[0]):
                        sample_audio_emb = audio_emb[sample_index:sample_index + 1]
                        sample_motion = motions[sample_index:sample_index + 1]
                        sample_person = p_id[sample_index:sample_index + 1]
                        sample_audio_raw = audio_raw[sample_index:sample_index + 1]
                        sample_add_cond = {
                            key: value[sample_index:sample_index + 1]
                            for key, value in add_cond.items()
                        }

                        same_person = (
                            previous_sequential_person is not None
                            and torch.equal(
                                previous_sequential_person.detach().cpu(),
                                sample_person.detach().cpu(),
                            )
                        )
                        consecutive_audio = audio_windows_are_consecutive(
                            previous_sequential_audio,
                            sample_audio_raw,
                            pose_stride=self.opt.stride,
                            pose_fps=self.opt.pose_fps,
                            audio_fps=16000,
                        )
                        chain_window = same_person and consecutive_audio
                        inpaint_dict = make_prefix_inpaint(
                            sample_motion,
                            previous_sequential_output if chain_window else None,
                            self.opt.overlap_len,
                        )
                        sample_output = self.generate_batch(
                            sample_audio_emb,
                            sample_person,
                            self.opt.net_dim_pose,
                            sample_add_cond,
                            inpaint_dict,
                        )
                        sequential_outputs.append(sample_output)
                        previous_sequential_output = sample_output.detach()
                        previous_sequential_audio = sample_audio_raw.detach()
                        previous_sequential_person = sample_person.detach()

                    outputs = torch.cat(sequential_outputs, dim=0)
                else:
                    outputs = self.generate_batch(
                        audio_emb, p_id, self.opt.net_dim_pose, add_cond, {}
                    )
                
                outputs = outputs.cpu().numpy()

            else:
                outputs = motions.cpu().numpy()
            
            if getattr(self.opt, 'rot_6d', False):
                # 6D outputs → Gram-Schmidt → axis_angle → euler degrees → normalize with euler stats
                outputs = torch.from_numpy(outputs)
                B, T, D = outputs.shape
                n_j = self.opt.split_pos // 6  # 47 joints
                ges_6d = outputs[..., :self.opt.split_pos]  # [B, T, 282]
                mat = rot_cvt.rotation_6d_to_matrix(ges_6d.reshape(B * T, n_j, 6))  # [B*T, J, 3, 3]
                aa = rot_cvt.matrix_to_axis_angle(mat).reshape(B, T, n_j * 3)       # [B, T, 141]
                euler_out = rot_cvt.axis_angle_to_euler_angles(aa.reshape(B, T, n_j, 3)).reshape(B, T, n_j * 3)
                euler_deg = euler_out * (180 / np.pi)
                outputs_euler = (euler_deg - test_dataset.mean_pose) / test_dataset.std_pose
                # If expression dims exist, keep them as-is (already in correct space)
                if D > self.opt.split_pos:
                    expr_part = outputs[..., self.opt.split_pos:]
                    outputs = torch.cat([outputs_euler, expr_part], dim=-1).numpy()
                else:
                    outputs = outputs_euler.numpy()
            elif self.opt.axis_angle:
                outputs = torch.from_numpy(outputs)
                denorm_out = outputs * test_dataset.motion_std + test_dataset.motion_mean
                B, T, C = denorm_out.shape
                euler_out = rot_cvt.axis_angle_to_euler_angles(denorm_out[..., :self.opt.split_pos].reshape(B, T, self.opt.split_pos//3, 3)).reshape(B,T,self.opt.split_pos)
                euler_out = euler_out * (180 / np.pi)
                euler_norm = (euler_out - test_dataset.mean_pose) / test_dataset.std_pose
                # If expression dims exist, keep them
                if C > self.opt.split_pos:
                    expr_part = denorm_out[..., self.opt.split_pos:]
                    outputs = torch.cat([euler_norm, expr_part], dim=-1).numpy()
                else:
                    outputs = euler_norm.numpy()

            if self.opt.unidiffuser or self.opt.net_dim_pose == 192:
                outputs, out_expression = np.split(outputs, [self.opt.split_pos], axis=-1)

            for idx, out in enumerate(outputs):
                if self.opt.distributed:
                    np.save(pjoin(results_dir,  "%05d" % count + f"_rank{rank}.npy"), out)
                    if self.opt.expression_only:
                        self.write_face_json(out, pjoin(json_dir, "%05d" % count + f"_rank{rank}.json"))
                    if self.opt.unidiffuser or self.opt.net_dim_pose == 192:
                        np.save(pjoin(results_dir_expr,  "%05d" % count + f"_rank{rank}.npy"), out_expression[idx])
                        sf.write(pjoin(results_dir_aud,  "%05d" % count + f"_rank{rank}.wav"), audio_raw[idx].cpu().numpy(), 16000)
                        self.write_face_json(out_expression[idx], pjoin(json_dir, "%05d" % count + f"_rank{rank}.json"))
                else:
                    np.save(pjoin(results_dir, "%05d.npy" % count), out)
                    if self.opt.expression_only:
                        self.write_face_json(out, pjoin(json_dir, "%05d.json" % count))
                    if self.opt.unidiffuser or self.opt.net_dim_pose == 192:
                        np.save(pjoin(results_dir_expr, "%05d.npy" % count), out_expression[idx])
                        sf.write(pjoin(results_dir_aud,  "%05d.wav" % count), audio_raw[idx].cpu().numpy(), 16000)
                        self.write_face_json(out_expression[idx], pjoin(json_dir, "%05d" % count + f"_rank{rank}.json"))
                        

                count += 1
            if self.opt.debug:
                break
        
        return results_dir


    def test_arbitrary_len(self, test_dataset):
        rank, world_size = get_dist_info()
        self.to(self.device)
        self.opt.is_train = False
        cur_epoch = 0

        model_dir = pjoin(self.opt.model_dir, self.opt.ckpt)
        cur_epoch, it, _, _, _ = self.load(model_dir)

        if self.opt.distributed:
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False, drop_last=False)
        else:
            test_sampler = None
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.opt.batch_size, shuffle=False,
            num_workers=self.opt.workers, pin_memory=True, sampler=test_sampler)

        logs = OrderedDict()

        self.encoder.eval()
        ckpt_epoch = f"ckpt_e{cur_epoch}"
        if 'fgd_best' in model_dir:
            ckpt_epoch = f"BestFGD_e{cur_epoch}"
        elif 'mse_best' in model_dir:
            ckpt_epoch = f"BestMSE_e{cur_epoch}"
        elif 'pck_best' in model_dir:
            ckpt_epoch = f"BestPCK_e{cur_epoch}"
        if self.opt.ddim:
            ckpt_epoch = ckpt_epoch + f"_{self.opt.timestep_respacing}"
            if self.opt.addBlend:
                ckpt_epoch = ckpt_epoch + f"_lastStepInterp"

        results_dir = pjoin("results", f"{self.opt.dataset_name}_{self.opt.n_poses}", self.opt.mode, self.opt.name, ckpt_epoch)

        if self.opt.rename:
            results_dir = pjoin("results", f"{self.opt.dataset_name}_{self.opt.n_poses}", self.opt.mode, self.opt.rename, ckpt_epoch)

        middle_name = self.opt.mode
        # if self.opt.overlap_len > 0:
        if self.opt.fix_very_first:
            results_dir = results_dir.replace(self.opt.name, f"{self.opt.name}/fixStart{self.opt.overlap_len}_fix_very_first")
            middle_name = f"{self.opt.name}/fixStart{self.opt.overlap_len}_fix_very_first"
        else:
            results_dir = results_dir.replace(self.opt.name, f"{self.opt.name}/fixStart{self.opt.overlap_len}")
            middle_name = f"{self.opt.name}/fixStart{self.opt.overlap_len}"
        

        if self.opt.usePredExpr:
            results_dir = results_dir.replace(middle_name, middle_name + "_usePredExpr")

        if self.opt.output_gt:
            results_dir = results_dir.replace(middle_name, middle_name + "_GT")
        

        if self.opt.unidiffuser or self.opt.net_dim_pose == 192:
            ori_results_dir = results_dir
            if os.path.exists(ori_results_dir):
                for nn in range(1,999):
                    ori_results_dir = results_dir + "_" + f"{nn:03d}"
                    if not os.path.exists(ori_results_dir):
                        break
            results_dir = os.path.join(ori_results_dir, "gesture")
            results_dir_expr = os.path.join(ori_results_dir, "expression")

            torch.distributed.barrier()
            os.makedirs(results_dir, exist_ok=True)
            os.makedirs(results_dir_expr, exist_ok=True)
            os.makedirs(os.path.join(results_dir_expr, "face_json"), exist_ok=True)
        else:
            ori_results_dir = results_dir
            if os.path.exists(ori_results_dir):
                for nn in range(1,999):
                    ori_results_dir = results_dir + "_" + f"{nn:03d}"
                    if not os.path.exists(ori_results_dir):
                        break
            results_dir = ori_results_dir
            torch.distributed.barrier()
            os.makedirs(results_dir, exist_ok=True)
        if not self.opt.expression_only:
            os.makedirs(os.path.join(results_dir, "bvh"), exist_ok=True)
        
        
        def get_windows(x, size, step):
            if isinstance(x, dict):
                out = {}
                for key in x.keys():
                    out[key] = get_windows(x[key], size, step)
                out_dict_list = []
                for i in range(len(out[list(out.keys())[0]])):
                    out_dict_list.append({key: out[key][i] for key in out.keys()})
                return out_dict_list
            else:
                seq_len = x.shape[1]
                if seq_len <= size:
                    return [x]
                else:
                    win_num = (seq_len - (size-step)) / float(step)
                    out = [x[:, mm*step : mm*step + size, ...] for mm in range(int(win_num))]
                    if win_num - int(win_num) != 0:
                        out.append(x[:, int(win_num)*step:, ...])  
                    return out
                
        count = 0
        for i, batch_data in enumerate(test_loader):
            if not self.opt.expression_only:
                if self.opt.axis_angle:
                    tar_pose = batch_data["pose_axis_angle"]
                else:
                    tar_pose = batch_data["pose"] # torch.Size([256, 34, 141])
                if self.opt.remove_hand:
                    tar_pose = tar_pose[..., [pp for pp in range(0,21)] + [pp for pp in range(75,87)]]
                tar_pose = tar_pose.to(self.device)
            if not self.opt.gesture_only:
                in_facial = batch_data["facial"].detach().to(self.device) if self.opt.facial_rep is not None else None  # torch.Size([256, 34, 51])
            if self.opt.expression_only or self.opt.gesCondition_expression_only:
                motions = in_facial
            elif self.opt.gesture_only or self.opt.expCondition_gesture_only != None or self.opt.textExpEmoCondition_gesture_only:
                motions = tar_pose
            else:
                motions = torch.cat((tar_pose, in_facial), dim=-1)
            audio_emb = batch_data["aud_feat"].detach().to(self.device) if self.opt.audio_rep is not None else None # 
            if self.opt.expCondition_gesture_only:
                audio_emb = torch.cat((audio_emb, in_facial), dim=-1)
            elif self.opt.gesCondition_expression_only:
                audio_emb = torch.cat((audio_emb, tar_pose), dim=-1)
            
            add_cond = {}
            if self.opt.expAddHubert or self.opt.addHubert:
                add_cond["pretrain_aud_feat"] = batch_data["pretrain_aud_feat"].to(self.device)

            p_id = batch_data["id"] if self.opt.speaker_id else None # torch.Size([256, 1])
            p_id = self.one_hot(p_id, self.opt.speaker_dim).detach().to(self.device)

            if self.opt.remove_audio:
                audio_emb = torch.zeros_like(audio_emb).to(audio_emb.device)
            if self.opt.remove_style:
                p_id = torch.zeros_like(p_id).to(p_id.device)

            
            batch_data = [audio_emb, motions, p_id]
            

            if not self.opt.output_gt:
                window_step = self.opt.n_poses - self.opt.overlap_len
                audio_emb_list = get_windows(audio_emb, self.opt.n_poses, window_step)
                motions_list = get_windows(motions, self.opt.n_poses, window_step)
                if add_cond not in [None, {}]:
                    add_cond_list = get_windows(add_cond, self.opt.n_poses, window_step)
                    
                
                out_motions = []
                for ii, [audio_emb, motions] in enumerate(zip(audio_emb_list, motions_list)):
                    print(f"Rank {rank}: Video {i+1} / {len(test_loader)}, Clip {ii+1} / {len(audio_emb_list)} ")
                    if add_cond not in [None, {}]:
                        add_cond = add_cond_list[ii]
                    inpaint_dict = {}
                    inpaint_dict['clip_idx'] = ii
                    if self.opt.overlap_len > 0:
                        inpaint_dict['gt'] = torch.zeros_like(motions)
                        inpaint_dict['outpainting_mask'] = torch.zeros_like(motions, dtype=torch.bool,
                                                        device=motions.device)  # Do inpainting/generation in those frames
                        # m_lens[0] = motions.shape[1]
                        
                        if ii == 0:
                            if self.opt.fix_very_first:
                                inpaint_dict['outpainting_mask'][..., :self.opt.overlap_len, :] = True  # True means use gt motion 
                                inpaint_dict['gt'][:, :self.opt.overlap_len, ...] = motions[:, -self.opt.overlap_len:, ...]
                            else:
                                pass
                        elif ii > 0:
                            inpaint_dict['outpainting_mask'][..., :self.opt.overlap_len, :] = True  # True means use gt motion 
                            inpaint_dict['gt'][:, :self.opt.overlap_len, ...] = outputs[:, -self.opt.overlap_len:, ...]
                            if self.opt.same_overlap_noisy and not self.opt.flow_matching:
                                inpaint_dict['previous_noisy_tail'] = previous_noisy_tail
                    
                    outputs = self.generate_batch(audio_emb, p_id, self.opt.net_dim_pose, add_cond, inpaint_dict)
                    if self.opt.same_overlap_noisy and not self.opt.flow_matching:
                        previous_noisy_tail = outputs["saved_noisy_tail"]
                        outputs = outputs["sample"]

                    outputs_np = outputs.cpu().numpy()
                    if ii == len(motions_list) - 1:
                        out_motions.append(outputs_np)
                    else:
                        out_motions.append(outputs_np[:, :window_step])
                    
                    if self.opt.debug:
                        break

                out_motions = np.concatenate(out_motions, 1)
                
            else:
                out_motions = motions.cpu().numpy()
            
            if self.opt.unidiffuser or self.opt.net_dim_pose == 192:
                out_motions, out_expression = np.split(out_motions, [self.opt.split_pos], axis=-1)

            if self.opt.axis_angle:
                axis_angle_out_path = os.path.join(results_dir, "axis_angle")
                os.makedirs(axis_angle_out_path, exist_ok=True)
                if self.opt.distributed:
                    np.save(pjoin(axis_angle_out_path,  "%05d" % count + f"_rank{rank}.npy"), out_motions)
                else:
                    np.save(pjoin(axis_angle_out_path, "%05d.npy" % count), out_motions)

                out_motions = torch.from_numpy(out_motions)
                denorm_out = out_motions * test_dataset.std_pose_axis_angle + test_dataset.mean_pose_axis_angle
                B, T, C = denorm_out.shape
                euler_out = rot_cvt.axis_angle_to_euler_angles(denorm_out.reshape(B, T, C//3, 3)).reshape(B,T,C)
                euler_out = euler_out * (180 / np.pi)
                out_motions = (euler_out - test_dataset.mean_pose) / test_dataset.std_pose
                out_motions = out_motions.numpy()
            

            if self.opt.ablation == "reverse_ges2exp":
                self.opt.expression_dim, self.opt.dim_pose = self.opt.dim_pose, self.opt.expression_dim
                
            if self.opt.distributed:
                if self.opt.expression_only:
                    np.save(pjoin(results_dir,  "%05d" % count + f"_rank{rank}.npy"), out_motions)
                else:
                    np.save(pjoin(results_dir,  "%05d" % count + f"_rank{rank}.npy"), out_motions)
                    out_denorm_euler = euler_out.reshape(-1, self.opt.dim_pose).numpy()
                    self.result2target_vis(out_denorm_euler, os.path.join(results_dir, 'bvh'), "%05d" % count + f"_rank{rank}.bvh")
                    if self.opt.unidiffuser or self.opt.net_dim_pose == 192:
                        np.save(pjoin(results_dir_expr,  "%05d" % count + f"_rank{rank}.npy"), out_expression)
                        self.write_face_json(out_expression, pjoin(results_dir_expr, 'face_json', "%05d" % count + f"_rank{rank}.json"))
            else:
                if self.opt.expression_only:
                    np.save(pjoin(results_dir, "%05d.npy" % count), out_motions)
                else:
                    np.save(pjoin(results_dir, "%05d.npy" % count), out_motions)
                    out_denorm_euler = euler_out.reshape(-1, self.opt.dim_pose).numpy()
                    self.result2target_vis(out_denorm_euler, os.path.join(results_dir, 'bvh'), "%05d.bvh" % count)
                    if self.opt.unidiffuser or self.opt.net_dim_pose == 192:
                        np.save(pjoin(results_dir_expr, "%05d.npy" % count), out_expression)
                        self.write_face_json(out_expression, pjoin(results_dir_expr, 'face_json', "%05d.json" % count))
            count += 1

            if self.opt.ablation == "reverse_ges2exp":
                self.opt.expression_dim, self.opt.dim_pose = self.opt.dim_pose, self.opt.expression_dim



        torch.distributed.barrier()
        if rank == 0:
            results_dir = os.path.join(os.getcwd(), results_dir)
            if self.opt.unidiffuser or self.opt.net_dim_pose == 192:
                results_dir_expr = os.path.join(os.getcwd(), results_dir_expr)

            if self.opt.expression_only or self.opt.net_dim_pose == 192:
                pred_filename_list = os.listdir(results_dir_expr); pred_filename_list.sort()
                self.gen_face_testset(results_dir_expr, pred_filename_list)
                gpu_ids = os.environ.get("CUDA_VISIBLE_DEVICES")
                try: 
                    print("Running face test")
                    os.chdir("0_BEAT_ori/codes/audio2pose/")
                    os.system(f"CUDA_VISIBLE_DEVICES={gpu_ids} python test_face_ddpm_cjm.py --config configs/ddpm_beat_4english_15_141_test_face.yaml --ddpm_result_path {results_dir_expr} --port 34253 --stride 34")
                    os.chdir("../../../")
                except:
                    print("Failed to run face test")
                    
            print("Running pose test")
            os.chdir("0_BEAT_ori/codes/audio2pose/")
            os.system(f"CUDA_VISIBLE_DEVICES={gpu_ids} python test_ddpm_cjm.py --config configs/ddpm_beat_4english_15_141_test.yaml --ddpm_result_path {results_dir} --port 12734 --stride 34")
            if self.opt.net_dim_pose == self.opt.dim_pose + self.opt.expression_dim:
                print("Running GesExpr test")
                os.system(f"CUDA_VISIBLE_DEVICES={gpu_ids} python test_ddpm_cjm_GesExpr.py --config configs/ddpm_beat_4english_15_141_test_GesExpr.yaml --gesture_face_path {os.path.join(results_dir, 'axis_angle')} {results_dir_expr} --port 45263 --stride 34")
                
            os.chdir("../../../")
            print("Finished")
        return results_dir
    
    def test_custom_aud(self, test_audio_path, test_dataset):
        rank, world_size = get_dist_info()
        self.to(self.device)
        self.opt.is_train = False
        cur_epoch = 0

        model_dir = pjoin(self.opt.model_dir, self.opt.ckpt)
        cur_epoch, it, _, _, _ = self.load(model_dir)

        if self.opt.expAddHubert or self.opt.addHubert:
            from utils.hubert import load_hubert_components
            wav2vec2_processor, hubert_model = load_hubert_components(device=self.device)

        if os.path.isdir(test_audio_path):
            aud_list = os.listdir(test_audio_path)
            aud_list = [aud for aud in aud_list if aud.endswith(".wav") or aud.endswith(".mp3")]
            aud_list.sort()
        else:
            aud_list = [os.path.basename(test_audio_path)]
            test_audio_path = os.path.dirname(test_audio_path)
        
        def get_windows(x, size, step):
            if isinstance(x, dict):
                out = {}
                for key in x.keys():
                    out[key] = get_windows(x[key], size, step)
                out_dict_list = []
                for i in range(len(out[list(out.keys())[0]])):
                    out_dict_list.append({key: out[key][i] for key in out.keys()})
                return out_dict_list
            else:
                seq_len = x.shape[1]
                if seq_len <= size:
                    return [x]
                else:
                    win_num = (seq_len - (size-step)) / float(step)
                    out = [x[:, mm*step : mm*step + size, ...] for mm in range(int(win_num))]
                    if win_num - int(win_num) != 0:
                        out.append(x[:, int(win_num)*step:, ...])  
                    return out
                

        # p_id = 2 # [2, 4, 6, 8]
        

        logs = OrderedDict()
        self.encoder.eval()


        ckpt_epoch = f"ckpt_e{cur_epoch}"
        if 'fgd_best' in model_dir:
            ckpt_epoch = f"BestFGD_e{cur_epoch}"
        elif 'mse_best' in model_dir:
            ckpt_epoch = f"BestMSE_e{cur_epoch}"
        elif 'pck_best' in model_dir:
            ckpt_epoch = f"BestPCK_e{cur_epoch}"
        if self.opt.ddim:
            ckpt_epoch = ckpt_epoch + f"_{self.opt.timestep_respacing}"
            if self.opt.addBlend:
                ckpt_epoch = ckpt_epoch + f"_lastStepInterp"
        


        for p_id_ori in [2,4,6,8]:
            results_dir = pjoin("results", f"{self.opt.dataset_name}_{self.opt.n_poses}", self.opt.mode, self.opt.name, ckpt_epoch, f"pid_{p_id_ori}")

            if self.opt.rename:
                results_dir = pjoin("results", f"{self.opt.dataset_name}_{self.opt.n_poses}", self.opt.mode, self.opt.rename, ckpt_epoch, f"pid_{p_id_ori}")

            middle_name = self.opt.mode
            # if self.opt.overlap_len > 0:
            if self.opt.fix_very_first:
                results_dir = results_dir.replace(self.opt.name, f"{self.opt.name}/fixStart{self.opt.overlap_len}_fix_very_first")
                middle_name = f"{self.opt.name}/fixStart{self.opt.overlap_len}_fix_very_first"
            else:
                results_dir = results_dir.replace(self.opt.name, f"{self.opt.name}/fixStart{self.opt.overlap_len}")
                middle_name = f"{self.opt.name}/fixStart{self.opt.overlap_len}"
            

            if self.opt.usePredExpr:
                results_dir = results_dir.replace(middle_name, middle_name + "_usePredExpr")

            if self.opt.output_gt:
                results_dir = results_dir.replace(middle_name, middle_name + "_GT")
            
                
            if not os.path.exists(results_dir):
                os.makedirs(results_dir, exist_ok=True)

            if self.opt.unidiffuser or self.opt.net_dim_pose == 192:
                ori_results_dir = results_dir
                results_dir = os.path.join(ori_results_dir, "gesture")
                results_dir_expr = os.path.join(ori_results_dir, "expression")

                os.makedirs(pjoin(results_dir_expr, 'face_json'), exist_ok=True)
                os.makedirs(pjoin(results_dir, 'bvh'), exist_ok=True)
                os.makedirs(results_dir, exist_ok=True)
                os.makedirs(results_dir_expr, exist_ok=True)
        

            p_id = p_id_ori - 1
            p_id = torch.ones((len(aud_list), 1)) * p_id
            p_id = self.one_hot(p_id, self.opt.speaker_dim).detach().to(self.device)
            
            sr = 16000
            
            for i, name in enumerate(aud_list):
                preprocessing_time = 0.0
                generation_time = 0.0
                aud_path = os.path.join(test_audio_path, name)
                if name.endswith((".wav", ".mp3")):
                    # HuBERT expects real 16 kHz samples. librosa.load() would
                    # otherwise silently resample to its 22.05 kHz default.
                    aud_ori, sr = librosa.load(aud_path, sr=16000, mono=True)
                elif name.endswith(".npy"):
                    aud_ori = np.load(aud_path)
                    sr = 16000
                
                aud = librosa.resample(aud_ori, orig_sr=sr, target_sr=18000)


                time_start = time.perf_counter()
                mel = librosa.feature.melspectrogram(y=aud, sr=18000, hop_length=1200, n_mels=128)

                time_1 = time.perf_counter()
                preprocessing_time += time_1 - time_start

                mel = mel[..., :-1]
                audio_emb = torch.from_numpy(np.swapaxes(mel, -1, -2))
                audio_emb = audio_emb.unsqueeze(0).to(self.device)
                B, N, _ = audio_emb.shape
                C = self.opt.net_dim_pose
                motions = torch.zeros((B, audio_emb.shape[-2], C)).to(self.device)
                ## test_single_audio END

                window_step = self.opt.n_poses - self.opt.overlap_len
                audio_emb_list = get_windows(audio_emb, self.opt.n_poses, window_step)
                motions_list = get_windows(motions, self.opt.n_poses, window_step)

                add_cond = {}
                if self.opt.expAddHubert or self.opt.addHubert:
                    _cuda_synchronize(self.device)
                    time_2 = time.perf_counter()
                    add_cond["pretrain_aud_feat"] = get_hubert_from_16k_speech_long(hubert_model, wav2vec2_processor, torch.from_numpy(aud_ori).unsqueeze(0).to(self.device), device=self.device)
                    add_cond["pretrain_aud_feat"] = F.interpolate(add_cond["pretrain_aud_feat"].swapaxes(-1,-2).unsqueeze(0), size=audio_emb.shape[-2], mode='linear', align_corners=True).swapaxes(-1,-2)
                    _cuda_synchronize(self.device)
                    time_3 = time.perf_counter()
                    preprocessing_time += time_3 - time_2
                # Put dict values into self.deivce
                if isinstance(add_cond, dict):
                    for key in add_cond.keys():
                        add_cond[key] = add_cond[key].to(self.device)


                if add_cond not in [None, {}]:
                    add_cond_list = get_windows(add_cond, self.opt.n_poses, window_step)
                
                
                out_motions = []
                for ii, [audio_emb, motions] in enumerate(zip(audio_emb_list, motions_list)):
                    print(f"Rank {rank}, Style: pid_{p_id_ori}: Name: {os.path.basename(aud_path)}, Video {i+1} / {len(aud_list)}, Clip {ii+1} / {len(audio_emb_list)} ")
                    if add_cond not in [None, {}]:
                        add_cond = add_cond_list[ii]
                    inpaint_dict = {}
                    if self.opt.overlap_len > 0:
                        inpaint_dict['gt'] = torch.zeros_like(motions)
                        inpaint_dict['outpainting_mask'] = torch.zeros_like(motions, dtype=torch.bool,
                                                        device=motions.device)  # Do outpainting/generation in those frames
                        
                        if ii == 0:
                            if self.opt.fix_very_first:
                                inpaint_dict['outpainting_mask'][..., :self.opt.overlap_len, :] = True  # True means use gt motion 
                                inpaint_dict['gt'][:, :self.opt.overlap_len, ...] = motions[:, -self.opt.overlap_len:, ...]
                            else:
                                pass
                        elif ii > 0:
                            inpaint_dict['outpainting_mask'][..., :self.opt.overlap_len, :] = True  # True means use gt motion 
                            inpaint_dict['gt'][:, :self.opt.overlap_len, ...] = outputs[:, -self.opt.overlap_len:, ...]

                    
                    _cuda_synchronize(self.device)
                    time_4 = time.perf_counter()
                    outputs = self.generate_batch(audio_emb, p_id, self.opt.net_dim_pose, add_cond, inpaint_dict)
                    _cuda_synchronize(self.device)
                    time_end = time.perf_counter()
                    generation_time += time_end - time_4

                    outputs_np = outputs.cpu().numpy()
                    if ii == len(motions_list) - 1:
                        out_motions.append(outputs_np)
                    else:
                        out_motions.append(outputs_np[:, :window_step])
                    
                    if self.opt.debug:
                        break

                out_motions = np.concatenate(out_motions, 1)
                total_time = preprocessing_time + generation_time
                frame_count = out_motions.shape[1]
                print(
                    f"End-to-end time: {total_time:.3f}s; frames: {frame_count}; "
                    f"FPS: {frame_count / max(total_time, 1e-9):.2f}"
                )
                print(
                    f"Generation time: {generation_time:.3f}s; "
                    f"sampling FPS: {frame_count / max(generation_time, 1e-9):.2f}; "
                    f"audio preprocessing: {preprocessing_time:.3f}s"
                )
                if self.opt.flow_matching:
                    nfe = self.diffusion.sample_steps * self.diffusion.nfe_per_step
                    print(f"FM solver: {self.diffusion.solver}; steps: {self.diffusion.sample_steps}; NFE/window: {nfe}")
                    
                
                if self.opt.unidiffuser or self.opt.net_dim_pose == 192:
                    out_motions, out_expression = np.split(out_motions, [self.opt.split_pos], axis=-1)


                if getattr(self.opt, 'rot_6d', False):
                    B, T, C = out_motions.shape
                    n_j = C // 6
                    mat_o = rot_cvt.rotation_6d_to_matrix(torch.from_numpy(out_motions).reshape(B * T, n_j, 6))
                    aa_o = rot_cvt.matrix_to_axis_angle(mat_o).reshape(B, T, n_j * 3)
                    
                    # For FGD metric saving, save normalized axis-angle
                    std_safe = np.maximum(test_dataset.std_pose_axis_angle, 1e-2)
                    mean_t = torch.from_numpy(test_dataset.mean_pose_axis_angle)
                    std_t = torch.from_numpy(std_safe)
                    aa_norm_o = (aa_o - mean_t) / std_t
                    axis_angle_out_path = os.path.join(results_dir, "axis_angle")
                    os.makedirs(axis_angle_out_path, exist_ok=True)
                    np.save(pjoin(axis_angle_out_path, f"{name.split('.')[0]}.npy"), aa_norm_o.numpy())
                    
                    # Use raw axis-angle directly to convert to Euler angles without any scale distortion
                    euler_out = rot_cvt.axis_angle_to_euler_angles(aa_o.reshape(B, T, n_j, 3)).reshape(B, T, n_j * 3)
                    euler_out = euler_out * (180 / np.pi)
                    out_motions = (euler_out - test_dataset.mean_pose) / test_dataset.std_pose
                    out_motions = out_motions.numpy()
                elif self.opt.axis_angle:
                    axis_angle_out_path = os.path.join(results_dir, "axis_angle")
                    os.makedirs(axis_angle_out_path, exist_ok=True)
                    np.save(pjoin(axis_angle_out_path, f"{name.split('.')[0]}.npy"), out_motions)

                    out_motions = torch.from_numpy(out_motions)
                    denorm_out = out_motions * test_dataset.std_pose_axis_angle + test_dataset.mean_pose_axis_angle
                    B, T, C = denorm_out.shape
                    euler_out = rot_cvt.axis_angle_to_euler_angles(denorm_out.reshape(B, T, C//3, 3)).reshape(B,T,C)
                    euler_out = euler_out * (180 / np.pi)
                    out_motions = (euler_out - test_dataset.mean_pose) / test_dataset.std_pose
                    out_motions = out_motions.numpy()
                

                np.save(pjoin(results_dir, f"{name.split('.')[0]}.npy"), out_motions)
                dim_pose_eval = 141 if getattr(self.opt, 'rot_6d', False) else self.opt.dim_pose
                out_denorm_euler = euler_out.reshape(-1, dim_pose_eval).numpy()
                self.result2target_vis(out_denorm_euler, os.path.join(results_dir, 'bvh'), f"{name.split('.')[0]}.bvh")
                if self.opt.unidiffuser or self.opt.net_dim_pose == 192:
                    np.save(pjoin(results_dir_expr, f"{name.split('.')[0]}.npy"), out_expression)
                    self.write_face_json(out_expression, pjoin(results_dir_expr, 'face_json', f"{name.split('.')[0]}.json"))



        print("Finished")
        return results_dir
    
    def gen_face_testset(self, results_dir, pred_filename_list):
        filename_list = os.listdir(f"data/BEAT/beat_cache/{self.opt.beat_cache_name}/test/facial52")
        filename_list.sort()
        pred_filename_list = [name for name in pred_filename_list if name.endswith(".npy")]
        
        json_dir = os.path.join(results_dir, "face_json_test")
        os.makedirs(json_dir, exist_ok=True)
        
        for filename, pred_filename in tqdm(zip(filename_list, pred_filename_list)):
            out_dict = {}
            out_dict["names"] = self.facial_list
            out_dict["frames"] = []
            
            pred = np.load(os.path.join(results_dir, pred_filename)).squeeze()
            for jj, ff in enumerate(pred):
                frame_dict = {}
                ff = ff * self.face_std + self.face_mean
                frame_dict["weights"] = ff.tolist()
                frame_dict["time"] = jj * (1/15)
                frame_dict["rotation"] = []
                out_dict["frames"].append(frame_dict)
            with open(os.path.join(json_dir, filename), "w") as f:
                json.dump(out_dict, f, indent=4)

    def write_face_json(self, pred, out_path):
        out_dict = {}
        out_dict["names"] = self.facial_list
        out_dict["frames"] = []
        for jj, ff in enumerate(pred.squeeze()):
            frame_dict = {}
            ff = ff * self.face_std + self.face_mean
            frame_dict["weights"] = ff.tolist()
            frame_dict["time"] = jj * (1/15)
            frame_dict["rotation"] = []
            out_dict["frames"].append(frame_dict)
        with open(out_path, "w") as f:
            json.dump(out_dict, f, indent=4)

    def result2target_vis(self, denorm_rot_euler, save_dir, bvh_name):
        gt_bvh_path=f'data/BEAT/beat_cache/{self.opt.beat_cache_name}/test/bvh_rot_vis/2_scott_0_1_1.bvh'
        ori_list = data_tools.joints_list["beat_joints"]
        target_list = data_tools.joints_list["spine_neck_141"]
        file_content_length = 431
            
        #print(short_name)
        wirte_file =  open(os.path.join(save_dir, bvh_name),'w+')
        with open(gt_bvh_path,'r') as pose_data_pre:
            pose_data_pre_file = pose_data_pre.readlines()
            for j, line in enumerate(pose_data_pre_file[0:file_content_length]):
                    wirte_file.write(line)
            offset_data = pose_data_pre_file[file_content_length]
            offset_data = np.fromstring(offset_data, dtype=float, sep=' ')
        wirte_file.close()

        wirte_file = open(os.path.join(save_dir, bvh_name),'r')
        ori_lines = wirte_file.readlines()
        ori_lines[file_content_length-2] = 'Frames: ' + str(len(denorm_rot_euler)) + '\n'
        wirte_file.close() 



        wirte_file = open(os.path.join(save_dir, bvh_name),'w+')
        wirte_file.writelines(i for i in ori_lines[:file_content_length])    
        wirte_file.close() 

        # Preserve equivalent continuous Euler branches for BVH consumers.
        denorm_rot_euler = np.rad2deg(
            np.unwrap(np.deg2rad(denorm_rot_euler), axis=0)
        )

        with open(os.path.join(save_dir, bvh_name),'a+') as wirte_file: 
            data_each_file = []
            for j, data in enumerate(denorm_rot_euler):
                data_rotation = offset_data.copy()
                for iii, (k, v) in enumerate(target_list.items()): # here is 147 rotations by 3
                    data_rotation[ori_list[k][1]-v:ori_list[k][1]] = data[iii*3:iii*3+3]
                data_each_file.append(data_rotation)
        
            for line_data in data_each_file:
                line_data = np.array2string(line_data, max_line_width=np.inf, precision=6, suppress_small=False, separator=' ')
                wirte_file.write(line_data[1:-2]+'\n')

@torch.no_grad()
def get_hubert_from_16k_speech_long(hubert_model, wav2vec2_processor, speech, device="cuda:0"):
    hubert_model = hubert_model.to(device)
    # if speech.ndim ==2:
    #     speech = speech[:, 0] # [T, 2] ==> [T,]
    input_values_all = wav2vec2_processor(speech, return_tensors="pt", sampling_rate=16000).input_values.squeeze(0) # [1, T]
    input_values_all = input_values_all.to(device)
    # For long audio sequence, due to the memory limitation, we cannot process them in one run
    # HuBERT process the wav with a CNN of stride [5,2,2,2,2,2], making a stride of 320
    # Besides, the kernel is [10,3,3,3,3,2,2], making 400 a fundamental unit to get 1 time step.
    # So the CNN is euqal to a big Conv1D with kernel k=400 and stride s=320
    # We have the equation to calculate out time step: T = floor((t-k)/s)
    # To prevent overlap, we set each clip length of (K+S*(N-1)), where N is the expected length T of this clip
    # The start point of next clip should roll back with a length of (kernel-stride) so it is stride * N
    kernel = 400
    stride = 320
    clip_length = stride * 1000
    num_iter = input_values_all.shape[1] // clip_length
    expected_T = (input_values_all.shape[1] - (kernel-stride)) // stride
    res_lst = []
    for i in range(num_iter):
        if i == 0:
            start_idx = 0
            end_idx = clip_length - stride + kernel
        else:
            start_idx = clip_length * i
            end_idx = start_idx + (clip_length - stride + kernel)
        input_values = input_values_all[:, start_idx: end_idx]
        hidden_states = hubert_model.forward(input_values).last_hidden_state # [B=1, T=pts//320, hid=1024]
        res_lst.append(hidden_states[0])
    if num_iter > 0:
        input_values = input_values_all[:, clip_length * num_iter:]
    else:
        input_values = input_values_all
    # if input_values.shape[1] != 0:
    if input_values.shape[1] >= kernel: # if the last batch is shorter than kernel_size, skip it            
        hidden_states = hubert_model(input_values).last_hidden_state # [B=1, T=pts//320, hid=1024]
        res_lst.append(hidden_states[0])
    
    ret = torch.cat(res_lst, dim=0).cpu() # [T, 1024]
    # assert ret.shape[0] == expected_T
    assert abs(ret.shape[0] - expected_T) <= 1
    if ret.shape[0] < expected_T:
        ret = torch.nn.functional.pad(ret, (0,0,0,expected_T-ret.shape[0]))
    else:
        ret = ret[:expected_T]
    return ret


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3
    
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
