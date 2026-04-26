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
from .loss_factory import get_loss_func
import soundfile as sf
from models.flow_matching import FlowMatching


class DDPMTrainer(object):

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

        self.huber_loss = get_loss_func("huber_loss")

        if args.is_train:
            self.mse_criterion = torch.nn.MSELoss(reduction='none')
        self.to(self.device)

        if self.opt.mode == 'train' and not self.opt.debug:
            if self.opt.multiprocessing_distributed:
                wandb.init(project=f"Diffusion_{self.opt.dataset_name}", group=f"DDP_{self.opt.name}") 
            else:
                wandb.init(project=f"Diffusion_{self.opt.dataset_name}")
            wandb.run.name = f"{self.opt.name}"
        
        # Generic facial list handling or initialization if needed
        self.facial_list = []
        if hasattr(self.opt, 'facial_list'):
            self.facial_list = self.opt.facial_list


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

        # In FM mode the model receives y=None; inpainting is handled by
        # the FM sampler rather than the model's internal outpainting logic.
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
        if self.opt.expr_weight == 1:
            loss_model_pred = self.mse_criterion(self.fake_noise, self.real_noise).mean(dim=-1)
        else:
            loss_model_pred = self.mse_criterion(self.fake_noise[..., :self.opt.lower_dim],
                                              self.real_noise[..., :self.opt.lower_dim]).mean(dim=-1) + \
                            self.mse_criterion(self.fake_noise[..., self.opt.lower_dim:],
                                              self.real_noise[..., self.opt.lower_dim:]).mean(dim=-1) * \
                            self.opt.expr_weight
        loss_model_pred = self.mse_criterion(self.fake_noise, self.real_noise).mean(dim=-1)
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
                x0_weight  = getattr(self.opt, 'x0_rec_weight', 100.0)

                # 1) 1st-order velocity loss (motion smoothness)
                loss_vel_rec = self.mse_criterion(self.fake_vel, self.real_vel).mean(dim=-1)
                loss_vel_rec = (loss_vel_rec * self.src_mask[:, :-1]).sum() / self.src_mask[:, :-1].sum()
                self.loss_vel_rec = vel_weight * loss_vel_rec
                loss_logs['loss_vel_rec'] = self.loss_vel_rec.item()
                self.final_loss += self.loss_vel_rec

                # 2) 2nd-order acceleration loss (jerk suppression)
                if acc_weight > 0:
                    loss_acc_rec = self.mse_criterion(self.fake_acc, self.real_acc).mean(dim=-1)
                    loss_acc_rec = (loss_acc_rec * self.src_mask[:, :-2]).sum() / self.src_mask[:, :-2].sum()
                    self.loss_acc_rec = acc_weight * loss_acc_rec
                    loss_logs['loss_acc_rec'] = self.loss_acc_rec.item()
                    self.final_loss += self.loss_acc_rec

                # 3) x0 Huber reconstruction loss (smooth action reconstruction)
                if hasattr(self, 'in_sem') and self.in_sem is not None and \
                        self.opt.sem_rep is not None:
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
                fake_ges_6d = self.fake_x0[..., :self.opt.split_pos]  # [B, T, 282]
                B_o, T_o, D_o = fake_ges_6d.shape
                n_joints = D_o // 6
                pred_6d = fake_ges_6d.reshape(B_o * T_o, n_joints, 6)
                pred_mat = rot_cvt.rotation_6d_to_matrix(pred_6d)
                pred_6d_ortho = rot_cvt.matrix_to_rotation_6d(pred_mat)
                loss_ortho = F.mse_loss(pred_6d, pred_6d_ortho.detach())
                self.loss_ortho = self.opt.ortho_loss_weight * loss_ortho
                loss_logs['loss_ortho'] = self.loss_ortho.item()
                self.final_loss += self.loss_ortho
        else:
            # ------ DDPM auxiliary losses ------
            if self.opt.add_vel_loss and self.epoch > self.opt.vel_loss_start:
                loss_vel_rec = self.mse_criterion(self.fake_vel, self.real_vel).mean(dim=-1)
                loss_vel_rec = (loss_vel_rec * self.src_mask[:, :-1]).sum() / self.src_mask[:, :-1].sum()
                self.loss_vel_rec = 100 * loss_vel_rec
                loss_logs['loss_vel_rec'] = self.loss_vel_rec.item()
                self.final_loss += loss_vel_rec

                if self.opt.model_mean_type == 'epsilon':
                    if hasattr(self, 'in_sem') and self.in_sem is not None and \
                            self.opt.sem_rep is not None:
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

    def update(self):
        self.zero_grad([self.opt_encoder])
        loss_logs = self.backward_G()
        self.final_loss.backward()
        self.clip_norm([self.encoder])
        self.step([self.opt_encoder])

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
        state = {
            'opt_encoder': self.opt_encoder.state_dict(),
            'ep': ep,
            'total_it': total_it,
            'FGD': fgd,
            'best_fgd': best_fgd,
            'MSE': mse,
            'best_mse': best_mse,
            'PCK': pck,
            'best_pck': best_pck
        }
        try:
            state['encoder'] = self.encoder.module.state_dict()
        except:
            state['encoder'] = self.encoder.state_dict()
        torch.save(state, file_name)
        return

    def load(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.device)

        if self.opt.PE == "pe_sinu_repeat":
            mm = checkpoint['encoder']['PE.pe'][:, :self.opt.n_poses, :]
            checkpoint['encoder']['PE.pe'] = torch.cat((mm, mm, mm, mm), -2)
        if self.opt.is_train:
            self.opt_encoder.load_state_dict(checkpoint['opt_encoder'])
        
        try:
            self.encoder.module.load_state_dict(checkpoint['encoder'], strict=False)
        except:
            self.encoder.load_state_dict(checkpoint['encoder'], strict=False)


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
        ones_eye = torch.eye(dim)
        return (ones_eye[ids.long()].squeeze())
    

    def train(self, train_dataset, val_dataset):
        rank, world_size = get_dist_info()
        self.to(self.device)
        self.opt_encoder = optim.Adam(self.encoder.parameters(), lr=self.opt.lr)
        it = 0
        cur_epoch = 0
        best_fgd = 99999
        best_mse = 99999
        best_pck = 0
        if self.opt.resume:
            model_dir = pjoin(self.opt.model_dir, 'latest.tar')
            cur_epoch, it, best_fgd, best_mse, best_pck = self.load(model_dir)
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

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.opt.batch_size, shuffle=(train_sampler is None),
            num_workers=self.opt.workers, pin_memory=True, sampler=train_sampler)

        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=self.opt.batch_size, shuffle=False,
            num_workers=self.opt.workers, pin_memory=True, sampler=val_sampler)
        
        ##################################### Training #####################################
        print("Start training ...")
        logs = OrderedDict()
        
        for epoch in range(cur_epoch, self.opt.num_epochs):
            self.train_mode()
            self.epoch = epoch
            if self.opt.distributed:
                train_sampler.set_epoch(epoch)

            for i, batch_data in enumerate(train_loader):
                # NOTE: This part presumes specific batch_data keys/structure.
                # Assuming generic structure similar to BEAT/Talkshow but adapted.
                # You may need to adapt this logic below if keys differ.
                
                if not self.opt.expression_only:
                    if self.opt.axis_angle:
                        tar_pose = batch_data["pose_axis_angle"]
                    else:
                        tar_pose = batch_data["pose"] 
                    if self.opt.remove_hand:
                        # Assuming hand indices are same or adjust as needed
                        tar_pose = tar_pose[..., [pp for pp in range(0,21)] + [pp for pp in range(75,87)]]
                    tar_pose = tar_pose.to(self.device)

                if not self.opt.gesture_only:
                    in_facial = batch_data["facial"].to(self.device) if self.opt.facial_rep is not None else None 
                
                self.in_sem = batch_data["sem"].to(self.device) if self.opt.sem_rep is not None else None 

                if self.opt.textExpEmoCondition_gesture_only:
                    in_word = batch_data["word"].to(self.device) if self.opt.word_rep is not None else None 
                    in_emo = batch_data["emo"].to(self.device) if self.opt.emo_rep is not None else None 
                    in_facial = torch.cat([in_facial, in_word.unsqueeze(-1), in_emo.unsqueeze(-1)], dim=-1)

                if self.opt.expression_only or self.opt.gesCondition_expression_only:
                    motions = in_facial
                elif self.opt.gesture_only or self.opt.expCondition_gesture_only != None or \
                                                    self.opt.textExpEmoCondition_gesture_only:
                    motions = tar_pose
                else:
                    motions = torch.cat((tar_pose, in_facial), dim=-1)

                audio_emb = batch_data["aud_feat"].to(self.device) if self.opt.audio_rep is not None else None 

                if self.opt.expCondition_gesture_only:
                    audio_emb = torch.cat((audio_emb, in_facial), dim=-1)
                elif self.opt.gesCondition_expression_only:
                    audio_emb = torch.cat((audio_emb, tar_pose), dim=-1)
                
                add_cond = {}

                if self.opt.addTextCond:
                    in_word = batch_data["word"].to(self.device) if self.opt.word_rep is not None else None 
                    add_cond['text'] = in_word
                if self.opt.addEmoCond:
                    in_emo = batch_data["emo"].to(self.device) if self.opt.emo_rep is not None else None 
                    add_cond['emo'] = in_emo
                if self.opt.expAddHubert or self.opt.addHubert:
                    add_cond["pretrain_aud_feat"] = batch_data["pretrain_aud_feat"].to(self.device)
                

                p_id = batch_data["id"] if self.opt.speaker_id else None 
                if p_id is not None:
                     p_id = self.one_hot(p_id, self.opt.speaker_dim).to(self.device)

                if self.opt.remove_audio:
                    audio_emb = torch.zeros_like(audio_emb).to(audio_emb.device)
                if self.opt.remove_style:
                    if p_id is not None:
                         p_id = torch.zeros_like(p_id).to(p_id.device)
                

                batch_data = [audio_emb, motions, p_id]

                
                
                inpaint_dict = {}
                if self.opt.overlap_len > 0:
                    inpaint_dict['gt'] = motions
                    inpaint_dict['outpainting_mask'] = torch.zeros_like(motions, dtype=torch.bool,
                                                    device=motions.device)  # Do inpainting/generation in those frames
                    inpaint_dict['outpainting_mask'][..., :self.opt.overlap_len, :] = True  # True means use gt motion 
                self.forward(batch_data, add_cond=add_cond, inpaint_dict=inpaint_dict)
                log_dict = self.update()
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

            if rank == 0:
                self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it, None, None, None, best_fgd, best_mse, best_pck)

            if (epoch+1) % self.opt.save_every_e == 0 and rank == 0:
                self.save(pjoin(self.opt.model_dir, 'ckpt_e%03d.tar'%(epoch)),
                            epoch, it, None, None, None, best_fgd, best_mse, best_pck)
            
            ################################## Evaluation ####################################
            
            
            if (epoch+1) % self.opt.eval_every_e == 0:
                self.eval_mode()
                count = 0.0
                diversity = AverageMeter('diversity')
                pck = AverageMeter('pck')
                mse = AverageMeter('mse')
                fgd = AverageMeter('fgd')
                progress = ProgressMeter(
                    len(val_loader) + (self.opt.distributed and (len(val_loader.sampler) * self.opt.world_size < len(val_loader.dataset))),
                    [diversity, pck, mse],
                    prefix='Test: ')
                with torch.no_grad():
                    for i, batch_data in tqdm(enumerate(val_loader)):
                        if not self.opt.expression_only:
                            if self.opt.axis_angle:
                                tar_pose = batch_data["pose_axis_angle"]
                                tar_pose_euler = batch_data["pose"].to(self.device)
                            else:
                                tar_pose = batch_data["pose"] 
                            if self.opt.remove_hand:
                                tar_pose = tar_pose[..., [pp for pp in range(0,21)] + [pp for pp in range(75,87)]]
                            tar_pose = tar_pose.to(self.device)
                        if not self.opt.gesture_only:
                            in_facial = batch_data["facial"].to(self.device) if self.opt.facial_rep is not None else None  
                        
                        if self.opt.textExpEmoCondition_gesture_only:
                            in_word = batch_data["word"].to(self.device) if self.opt.word_rep is not None else None 
                            in_emo = batch_data["emo"].to(self.device) if self.opt.emo_rep is not None else None 
                            in_facial = torch.cat([in_facial, in_word.unsqueeze(-1), in_emo.unsqueeze(-1)], dim=-1)
                        

                        if self.opt.expression_only or self.opt.gesCondition_expression_only:
                            motions = in_facial
                        elif self.opt.gesture_only or self.opt.expCondition_gesture_only != None or self.opt.textExpEmoCondition_gesture_only:
                            motions = tar_pose
                        else:
                            motions = torch.cat((tar_pose, in_facial), dim=-1)

                        audio_emb = batch_data["aud_feat"].to(self.device) if self.opt.audio_rep is not None else None

                        if self.opt.expCondition_gesture_only:
                            audio_emb = torch.cat((audio_emb, in_facial), dim=-1)
                        elif self.opt.gesCondition_expression_only:
                            audio_emb = torch.cat((audio_emb, tar_pose), dim=-1)
                        
                        add_cond = {}
                        if self.opt.addTextCond:
                            in_word = batch_data["word"].to(self.device) if self.opt.word_rep is not None else None 
                            add_cond['text'] = in_word
                        if self.opt.addEmoCond:
                            in_emo = batch_data["emo"].to(self.device) if self.opt.emo_rep is not None else None 
                            add_cond['emo'] = in_emo
                        if self.opt.expAddHubert or self.opt.addHubert:
                            add_cond["pretrain_aud_feat"] = batch_data["pretrain_aud_feat"].to(self.device)

                        p_id = batch_data["id"] if self.opt.speaker_id else None 
                        if p_id is not None:
                             p_id = self.one_hot(p_id, self.opt.speaker_dim).to(self.device)

                        if self.opt.remove_audio:
                            audio_emb = torch.zeros_like(audio_emb).to(audio_emb.device)
                        if self.opt.remove_style:
                            if p_id is not None:
                                p_id = torch.zeros_like(p_id).to(p_id.device)

                        batch_data = [audio_emb, motions, p_id]

                        if self.opt.use_single_style:
                            if p_id is not None:
                                p_id = torch.zeros_like(p_id)
                                p_id[:, :1] = 1

                        audio_emb = audio_emb.detach().float()
                        motions = motions.detach().float()
                        if p_id is not None:
                             p_id = p_id.detach().float()
                        count += len(motions)

                        inpaint_dict = {}
                        if self.opt.overlap_len > 0:
                            inpaint_dict['gt'] = motions
                            inpaint_dict['outpainting_mask'] = torch.zeros_like(motions, dtype=torch.bool,
                                                            device=motions.device)  # Do inpainting/generation in those frames
                            inpaint_dict['outpainting_mask'][..., :self.opt.overlap_len, :] = True  # True means use gt motion 

                        outputs = self.generate_batch(audio_emb, p_id, self.opt.net_dim_pose, add_cond, inpaint_dict)
                        B, seq, C = outputs.shape

                        if not self.opt.no_fgd and self.eval_model is not None:
                            latent_out = self.eval_model(outputs[:, :34, :].float())
                            latent_ori = self.eval_model(motions[:, :34, :].float())
                            #print(latent_out,latent_ori)
                            if i == 0:
                                latent_out_all = latent_out.cpu().numpy()
                                latent_ori_all = latent_ori.cpu().numpy()
                            else:
                                latent_out_all = np.concatenate([latent_out_all, latent_out.cpu().numpy()], axis=0)
                                latent_ori_all = np.concatenate([latent_ori_all, latent_ori.cpu().numpy()], axis=0)

                        

                        ## to cpu
                        outputs, motions = outputs.cpu(), motions.cpu()
                        
                        motions = motions.reshape(B, seq, C//3, 3).numpy()
                        outputs = outputs.reshape(B, seq, C//3, 3).numpy()


                        ### MSE & PCK
                        
                        diff = outputs - motions
                        diff_square = diff ** 2
                        correct = np.sum(diff_square, axis=3)
                        correct = np.sqrt(correct) < 0.5
                        pck_val = np.mean(correct)
                        mse_val = np.mean(diff_square)
                        
                        ### diversity
                        B_div = 50 ## In Ye et al. (ECCV'22), Batch size is 50 when evaluating diversity
                        if B < B_div:
                            B_div = B
                        # out_split = outputs.split(B_div, dim=0)
                        out_split = np.split(outputs, np.arange(B_div, B, B_div), axis=0)
                        for idx in range(B // B_div):
                            div_val = 0.0
                            for ii in range(0, B_div):
                                for jj in range(ii+1, B_div):
                                    dif = out_split[idx][ii,:,:,:] - out_split[idx][jj,:,:,:]
                                    div_val += np.mean(np.absolute(dif)) ### TODO: check: np.mean() or np.sum()
                            div_val = div_val * 2 / (B_div * (B_div - 1)) 
                            # update in averagemeter
                            diversity.update(div_val, B_div)

                        
                        # update in averagemeter
                        mse.update(mse_val, B)
                        pck.update(pck_val, B)

                        if self.opt.debug or (self.opt.max_eval_samples != -1 and pck.count >= self.opt.max_eval_samples):
                            break

                if self.opt.distributed:  #  all_reduce() is for aggreagating results on each rank
                    diversity.all_reduce()
                    mse.all_reduce()
                    pck.all_reduce()
                
                if rank == 0:
                    if not self.opt.debug:
                        wandb.log({"MSE": mse.avg, "PCK": pck.avg, "Diversity": diversity.avg}, step = epoch)
                    print(f"[Validation]: Epoch: {epoch}, MSE: {mse.avg}, PCK: {pck.avg}, Diversity: {diversity.avg}")
                
                if not self.opt.no_fgd and self.eval_model is not None:
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

        
        if not os.path.exists(results_dir):
            os.makedirs(results_dir, exist_ok=True)
        if self.opt.expression_only:
            json_dir = os.path.join(results_dir, "face_json")
            os.makedirs(json_dir, exist_ok=True)

        count = 0
        for i, batch_data in enumerate(test_loader):
            if not self.opt.expression_only:
                if self.opt.axis_angle:
                    tar_pose = batch_data["pose_axis_angle"]
                else:
                    tar_pose = batch_data["pose"] 
                if self.opt.remove_hand:
                    tar_pose = tar_pose[..., [pp for pp in range(0,21)] + [pp for pp in range(75,87)]]
                tar_pose = tar_pose.to(self.device)
            if not self.opt.gesture_only:
                in_facial = batch_data["facial"].detach().to(self.device) if self.opt.facial_rep is not None else None  
            
            
            if self.opt.textExpEmoCondition_gesture_only:
                in_word = batch_data["word"].to(self.device) if self.opt.word_rep is not None else None 
                in_emo = batch_data["emo"].to(self.device) if self.opt.emo_rep is not None else None 
                in_facial = torch.cat([in_facial, in_word.unsqueeze(-1), in_emo.unsqueeze(-1)], dim=-1)
            # in_sem = batch_data["sem"].detach().to(self.device) if self.opt.sem_rep is not None else None 
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


            p_id = batch_data["id"] if self.opt.speaker_id else None 
            if p_id is not None:
                p_id = self.one_hot(p_id, self.opt.speaker_dim).detach().to(self.device)

            if self.opt.remove_audio:
                audio_emb = torch.zeros_like(audio_emb).to(audio_emb.device)
            if self.opt.remove_style:
                 if p_id is not None:
                    p_id = torch.zeros_like(p_id).to(p_id.device)
            
            
            batch_data = [audio_emb, motions, p_id]

            if not self.opt.output_gt:
                ###### debug
                if self.opt.use_single_style:
                    if p_id is not None:
                        p_id = torch.zeros_like(p_id)
                        p_id[:, :1] = 1
                ###### debug end

                inpaint_dict = {}
                if self.opt.overlap_len > 0:
                    inpaint_dict['gt'] = motions
                    inpaint_dict['outpainting_mask'] = torch.zeros_like(motions, dtype=torch.bool,
                                                    device=motions.device)  # Do inpainting/generation in those frames
                    inpaint_dict['outpainting_mask'][..., :self.opt.overlap_len, :] = True  # True means use gt motion 

                
                outputs = self.generate_batch(audio_emb, p_id, self.opt.net_dim_pose, add_cond, inpaint_dict)
                
                outputs = outputs.cpu().numpy()

            else:
                outputs = motions.cpu().numpy()
            
            # Inv standardize if dataset has those methods
            if hasattr(test_dataset, 'inv_standardize'):
                 if self.opt.expression_only or self.opt.gesCondition_expression_only:
                     outputs = test_dataset.inv_standardize(outputs, test_dataset.expression_mean, test_dataset.expression_std)
                 elif self.opt.gesture_only or self.opt.expCondition_gesture_only:
                     outputs = test_dataset.inv_standardize(outputs, test_dataset.pose_mean, test_dataset.pose_std)
                 else:
                     outputs = test_dataset.inv_standardize(outputs, test_dataset.motion_mean, test_dataset.motion_std)

            for out in outputs:
                np.save(pjoin(results_dir, "%05d.npy" % count), out)
                count += 1

            if self.opt.debug:
                break
        
        return results_dir


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
