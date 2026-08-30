import os

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

import datasets.rotation_converter as rot_cvt
from utils.motion_metrics import select_axis_angle_stats


class FlowMatching:
    """
    Optimal Transport Conditional Flow Matching (OT-CFM).

    Convention:
      - x_0 = clean data,  x_1 = noise ~ N(0, I)
      - x_t = (1 - t) * x_0 + t * x_1   (t=0 → data,  t=1 → noise)
      - target vector field: v_t = x_1 - x_0  (a constant w.r.t. t!)
      - Sampling ODE: from t=1 (noise) to t=0 (data) via Euler steps
    """

    def __init__(self, opt):
        self.opt = opt
        self.sigma_min = 1e-4
        # Sampling steps configurable via opt (default 50)
        self.sample_steps = getattr(opt, 'fm_sample_steps', 50)
        # Keep the old fm_use_rk4 attribute as a compatibility fallback, but
        # expose the solver explicitly so experiment labels match the code.
        solver = getattr(opt, 'fm_solver', None)
        if solver is None:
            solver = 'rk4' if getattr(opt, 'fm_use_rk4', True) else 'euler'
        self.solver = solver
        self.use_rk4 = solver == 'rk4'
        self.nfe_per_step = 4 if self.use_rk4 else 1
        self.transition_blend = max(0, int(getattr(opt, 'fm_transition_blend', 7)))
        self.gesture_dim = int(getattr(opt, 'split_pos', 0) or 0)
        self.axis_angle_mean = None
        self.axis_angle_std = None
        if (
            bool(getattr(opt, 'axis_angle', False))
            and not bool(getattr(opt, 'rot_6d', False))
            and self.gesture_dim >= 3
        ):
            stats_root = getattr(opt, 'mean_pose_path', None)
            if stats_root:
                mean_path = os.path.join(stats_root, 'axis_angle_mean.npy')
                std_path = os.path.join(stats_root, 'axis_angle_std.npy')
                if os.path.isfile(mean_path) and os.path.isfile(std_path):
                    self.set_axis_angle_stats(
                        np.load(mean_path),
                        np.load(std_path),
                        self.gesture_dim,
                    )

    def set_axis_angle_stats(self, mean, std, gesture_dim=None):
        """Bind normalization statistics used by SO(3) release blending."""
        if gesture_dim is not None:
            self.gesture_dim = int(gesture_dim)
        selected_mean, selected_std = select_axis_angle_stats(
            mean, std, self.gesture_dim
        )
        self.axis_angle_mean = np.asarray(selected_mean, dtype=np.float32)
        self.axis_angle_std = np.maximum(
            np.asarray(selected_std, dtype=np.float32), 1e-2
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_losses(self, model, x_start, t, model_kwargs):
        """
        Compute OT-CFM training loss for one batch.

        The `t` argument from the DDPM sampler is *ignored*; we
        re-sample a continuous t ~ U[0, 1] internally, which is the
        correct practice for flow matching.

        Return dict matches the interface of GaussianDiffusion.training_losses().
        """
        x0 = x_start                            # [B, T, D]  clean data
        x1 = torch.randn_like(x0)               # [B, T, D]  noise

        batch_size = x0.shape[0]

        # Continuous time t ~ U[0, 1]
        t_continuous = torch.rand(batch_size, device=x0.device)

        # Expand t for broadcasting: [B, 1, 1]
        t_expand = t_continuous.view(batch_size, *([1] * (len(x0.shape) - 1)))

        # 1. Noisy sample via linear interpolation
        xt = (1.0 - t_expand) * x0 + t_expand * x1

        # 2. Constant target vector field (OT-CFM)
        target_v = x1 - x0

        # 3. Scale continuous t to [0, 1000] for the sinusoidal time embedding
        #    (MotionTransformer's timestep_embedding accepts floats natively)
        t_scaled = t_continuous * 1000.0

        # 4. Model forward — predicts the vector field
        model_output = model(xt, timesteps=t_scaled, **model_kwargs)

        # 5. MSE loss (per-sample mean over non-batch dims)
        loss = F.mse_loss(model_output, target_v, reduction='none')
        loss = loss.mean(dim=list(range(1, len(loss.shape))))  # → [B]

        # 6. Recover predicted x0 from the predicted velocity field:
        #    x_0 ≈ x_t - t * v_pred
        x0_pred = xt - t_expand * model_output

        # 7. Compute 1st-order velocity (finite difference along time axis)
        #    vel[t] = x0[t+1] - x0[t],  shape: [B, T-1, D]
        target_vel = x0[:, 1:, :] - x0[:, :-1, :]
        pred_vel   = x0_pred[:, 1:, :] - x0_pred[:, :-1, :]

        # 8. Compute 2nd-order acceleration (2nd finite difference)
        #    acc[t] = vel[t+1] - vel[t] = x0[t+2] - 2*x0[t+1] + x0[t],  shape: [B, T-2, D]
        target_acc = target_vel[:, 1:, :] - target_vel[:, :-1, :]
        pred_acc   = pred_vel[:, 1:, :] - pred_vel[:, :-1, :]

        return {
            "loss": loss,
            "target": target_v,                          # predicted target (vector field)
            "pred": model_output,                        # model prediction  (vector field)
            # 1st-order velocity (motion smoothness)
            "target_vel": target_vel,
            "pred_vel": pred_vel,
            # 2nd-order acceleration (motion jerk suppression)
            "target_acc": target_acc,
            "pred_acc": pred_acc,
            # x0 fields: useful for optional Huber auxiliary loss
            "target_x0": x0,
            "pred_x0": x0_pred,
        }

    # ------------------------------------------------------------------
    # Sampling (no inpainting)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(self, model, shape, model_kwargs, progress=True, device=None):
        """
        Euler ODE sampler — generates motion from pure noise.

        Args:
            model:         conditioned MotionTransformer / UniDiffuser
            shape:         output tensor shape (B, T, D)
            model_kwargs:  conditioning dict with audio_emb, length, etc.
            progress:      show tqdm progress bar
            device:        target device; inferred from model_kwargs tensors if None
        """
        device = self._infer_device(model_kwargs, device)

        b = shape[0]
        img = torch.randn(*shape, device=device)        # start: pure noise, t=1

        steps = self.sample_steps
        # t from 1.0 → 0.0
        times = torch.linspace(1.0, 0.0, steps + 1, device=device)

        iterator = range(steps)
        if progress:
            iterator = tqdm(iterator, desc='FM sampling', total=steps)

        def _velocity(x, t_val):
            t_b = torch.full((b,), t_val, device=device)
            return model(x, timesteps=t_b * 1000.0, **model_kwargs)

        for i in iterator:
            t_curr = times[i].item()
            t_next = times[i + 1].item()
            d_t = t_next - t_curr

            if self.use_rk4:
                # 4th-order Runge-Kutta
                k1 = _velocity(img, t_curr)
                k2 = _velocity(img + 0.5 * d_t * k1, t_curr + 0.5 * d_t)
                k3 = _velocity(img + 0.5 * d_t * k2, t_curr + 0.5 * d_t)
                k4 = _velocity(img + d_t * k3, t_next)
                img = img + (d_t / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            else:
                # Euler (original)
                v_pred = _velocity(img, t_curr)
                img = img + v_pred * d_t

        return img  # [B, T, D]

    # ------------------------------------------------------------------
    # Sampling with inpainting (overlap / outpainting)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample_with_inpaint(
        self, model, shape, model_kwargs, inpaint_dict,
        progress=True, device=None
    ):
        """
        FM RePaint-style sampler for overlap/inpainting scenarios.

        In FM there is no q_sample(), so we implement inpainting by
        replacing the known region with the FM-interpolated GT at each
        step:  gt_noisy = (1 - t) * gt + t * fixed_noise

        Args:
            inpaint_dict: dict with:
                - 'gt':              ground-truth motion [B, T, D]
                - 'outpainting_mask': bool tensor [B, T, D];
                                      True  → use GT (known region)
                                      False → generate freely
        """
        device = self._infer_device(model_kwargs, device)

        b = shape[0]
        img = torch.randn(*shape, device=device)

        gt   = inpaint_dict.get('gt', None)
        mask = inpaint_dict.get('outpainting_mask', None)  # bool [B, T, D]

        # Fix the noise for the GT region so the trajectory is consistent
        noise_for_gt = torch.randn_like(gt) if gt is not None else None

        steps = self.sample_steps
        times = torch.linspace(1.0, 0.0, steps + 1, device=device)

        iterator = range(steps)
        if progress:
            iterator = tqdm(iterator, desc='FM inpaint sampling', total=steps)

        def _velocity_inpaint(x, t_val):
            t_b = torch.full((b,), t_val, device=device)
            return model(x, timesteps=t_b * 1000.0, **model_kwargs)

        def _apply_inpaint(x, t_val):
            if gt is not None and mask is not None:
                gt_noisy = (1.0 - t_val) * gt + t_val * noise_for_gt
                return torch.where(mask, gt_noisy, x)
            return x

        for i in iterator:
            t_curr = times[i].item()
            t_next = times[i + 1].item()
            d_t = t_next - t_curr

            img = _apply_inpaint(img, t_curr)

            if self.use_rk4:
                k1 = _velocity_inpaint(img, t_curr)
                k2 = _velocity_inpaint(_apply_inpaint(img + 0.5*d_t*k1, t_curr + 0.5*d_t), t_curr + 0.5*d_t)
                k3 = _velocity_inpaint(_apply_inpaint(img + 0.5*d_t*k2, t_curr + 0.5*d_t), t_curr + 0.5*d_t)
                k4 = _velocity_inpaint(_apply_inpaint(img + d_t*k3, t_next), t_next)
                img = img + (d_t / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            else:
                v_pred = _velocity_inpaint(img, t_curr)
                img = img + v_pred * d_t

        # Final hard-replace, followed by a short C1-style release blend. Hard
        # inpainting alone guarantees matching positions but creates a velocity
        # impulse at the first unconstrained frame of every long-form window.
        if gt is not None and mask is not None:
            img = torch.where(mask, gt, img)
            img = self._blend_inpaint_release(img, gt, mask)

        return img  # [B, T, D]

    def _blend_inpaint_release(self, sample, gt, mask):
        """Blend a known prefix into free motion without long rotation arcs.

        The current long-form pipeline uses a contiguous known prefix. For each
        batch item, extrapolate the last known velocity for a few frames and
        smoothly hand control back to the generated sample. Axis-angle gesture
        channels are blended on SO(3); linear channels such as expression
        coefficients retain the component-wise Hermite release. This preserves
        exact inpainted frames while avoiding axis-angle branch flips.
        """
        if self.transition_blend <= 0 or sample.shape[1] < 3:
            return sample

        known_frames = mask.all(dim=-1)
        blended = sample.clone()
        for batch_index in range(sample.shape[0]):
            prefix_len = 0
            for frame_index in range(sample.shape[1]):
                if not bool(known_frames[batch_index, frame_index].item()):
                    break
                prefix_len += 1

            if prefix_len < 2 or prefix_len >= sample.shape[1]:
                continue

            blend_len = min(self.transition_blend, sample.shape[1] - prefix_len)
            if blend_len <= 0:
                continue

            free_slice = slice(prefix_len, prefix_len + blend_len)
            last = gt[batch_index, prefix_len - 1]
            entry_velocity = last - gt[batch_index, prefix_len - 2]
            first_free = last + entry_velocity
            rotation_blend = self._can_blend_axis_angle(sample)
            linear_start = self.gesture_dim if rotation_blend else 0

            if linear_start < sample.shape[-1]:
                self._blend_linear_release(
                    blended,
                    sample,
                    first_free,
                    entry_velocity,
                    batch_index,
                    prefix_len,
                    blend_len,
                    linear_start,
                )

            if rotation_blend:
                self._blend_axis_angle_release(
                    blended,
                    sample,
                    gt,
                    batch_index,
                    prefix_len,
                    blend_len,
                )

        return blended

    def _can_blend_axis_angle(self, sample):
        return (
            self.axis_angle_mean is not None
            and self.axis_angle_std is not None
            and self.gesture_dim >= 3
            and self.gesture_dim % 3 == 0
            and self.gesture_dim <= sample.shape[-1]
        )

    @staticmethod
    def _blend_linear_release(
        blended,
        sample,
        first_free,
        entry_velocity,
        batch_index,
        prefix_len,
        blend_len,
        linear_start,
    ):
        """Hermite release for Euclidean channels such as blendshapes."""
        linear_slice = slice(linear_start, sample.shape[-1])
        if blend_len == 1:
            blended[batch_index, prefix_len, linear_slice] = first_free[linear_slice]
            return

        target_index = prefix_len + blend_len - 1
        target = sample[batch_index, target_index, linear_slice]
        target_velocity = (
            sample[batch_index, target_index, linear_slice]
            - sample[batch_index, target_index - 1, linear_slice]
        )
        interval_count = blend_len - 1
        first = first_free[linear_slice]
        displacement = target - first
        tangent_limit = 3.0 * displacement.abs() / float(interval_count)

        def _limit_tangent(tangent):
            same_direction = tangent * displacement > 0
            magnitude = torch.minimum(tangent.abs(), tangent_limit)
            return torch.where(
                same_direction,
                displacement.sign() * magnitude,
                torch.zeros_like(tangent),
            )

        entry_tangent = _limit_tangent(entry_velocity[linear_slice])
        target_tangent = _limit_tangent(target_velocity)
        phase = torch.linspace(
            0.0, 1.0, blend_len, device=sample.device, dtype=sample.dtype
        ).unsqueeze(-1)
        phase2 = phase * phase
        phase3 = phase2 * phase
        h00 = 2.0 * phase3 - 3.0 * phase2 + 1.0
        h10 = phase3 - 2.0 * phase2 + phase
        h01 = -2.0 * phase3 + 3.0 * phase2
        h11 = phase3 - phase2
        blended[
            batch_index,
            prefix_len:prefix_len + blend_len,
            linear_slice,
        ] = (
            h00 * first
            + h10 * float(interval_count) * entry_tangent
            + h01 * target
            + h11 * float(interval_count) * target_tangent
        )

    def _blend_axis_angle_release(
        self,
        blended,
        sample,
        gt,
        batch_index,
        prefix_len,
        blend_len,
    ):
        """Release gesture joints along monotone shortest paths on SO(3)."""
        joint_count = self.gesture_dim // 3
        mean = torch.as_tensor(
            self.axis_angle_mean, device=sample.device, dtype=sample.dtype
        ).reshape(joint_count, 3)
        std = torch.as_tensor(
            self.axis_angle_std, device=sample.device, dtype=sample.dtype
        ).reshape(joint_count, 3)

        def _matrices(values):
            raw = values[..., :self.gesture_dim].reshape(-1, joint_count, 3)
            return rot_cvt.axis_angle_to_matrix(raw * std + mean)

        sample_matrices = _matrices(sample[batch_index])
        known_matrices = _matrices(gt[batch_index, prefix_len - 2:prefix_len])
        previous_rotation, last_rotation = known_matrices.unbind(dim=0)
        entry_delta_matrix = previous_rotation.transpose(-1, -2) @ last_rotation
        first_rotation = last_rotation @ entry_delta_matrix

        target_index = prefix_len + blend_len - 1
        target_rotation = sample_matrices[target_index]
        if blend_len == 1:
            release_rotations = first_rotation.unsqueeze(0)
        else:
            relative_target = first_rotation.transpose(-1, -2) @ target_rotation
            target_delta = rot_cvt.matrix_to_axis_angle(relative_target)
            entry_delta = rot_cvt.matrix_to_axis_angle(entry_delta_matrix)
            if target_index + 1 < sample.shape[1]:
                exit_delta_matrix = (
                    target_rotation.transpose(-1, -2)
                    @ sample_matrices[target_index + 1]
                )
            else:
                exit_delta_matrix = (
                    sample_matrices[target_index - 1].transpose(-1, -2)
                    @ target_rotation
                )
            exit_delta = rot_cvt.matrix_to_axis_angle(exit_delta_matrix)

            interval_count = blend_len - 1
            distance_sq = (target_delta * target_delta).sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-8)
            entry_slope = (
                float(interval_count)
                * (entry_delta * target_delta).sum(dim=-1, keepdim=True)
                / distance_sq
            ).clamp(0.0, 3.0)
            exit_slope = (
                float(interval_count)
                * (exit_delta * target_delta).sum(dim=-1, keepdim=True)
                / distance_sq
            ).clamp(0.0, 3.0)

            phase = torch.linspace(
                0.0, 1.0, blend_len, device=sample.device, dtype=sample.dtype
            ).reshape(blend_len, 1, 1)
            phase2 = phase * phase
            phase3 = phase2 * phase
            phase = (
                (phase3 - 2.0 * phase2 + phase) * entry_slope
                + (-2.0 * phase3 + 3.0 * phase2)
                + (phase3 - phase2) * exit_slope
            ).clamp(0.0, 1.0)
            phase = torch.cummax(phase, dim=0).values
            release_delta = rot_cvt.axis_angle_to_matrix(
                phase * target_delta.unsqueeze(0)
            )
            release_rotations = first_rotation.unsqueeze(0) @ release_delta

        # Canonicalize every newly generated rotation. The physical motion is
        # unchanged, but future inpainted prefixes no longer inherit an
        # arbitrary non-principal axis-angle branch from the sampler.
        free_matrices = sample_matrices[prefix_len:].clone()
        free_matrices[:blend_len] = release_rotations
        free_axis_angle = rot_cvt.matrix_to_axis_angle(free_matrices)
        free_normalized = ((free_axis_angle - mean) / std).reshape(
            free_axis_angle.shape[0], self.gesture_dim
        )
        blended[batch_index, prefix_len:, :self.gesture_dim] = free_normalized

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_device(model_kwargs, device=None):
        """Infer torch.device from the first tensor found in model_kwargs."""
        if device is not None:
            return device
        for v in model_kwargs.values():
            if isinstance(v, torch.Tensor):
                return v.device
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
