import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


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

        for i in iterator:
            t_curr = times[i]
            t_next = times[i + 1]

            t_batch = torch.full((b,), t_curr.item(), device=device)
            t_batch_scaled = t_batch * 1000.0

            v_pred = model(img, timesteps=t_batch_scaled, **model_kwargs)

            # Euler update: dt is negative (moving from noise → data)
            d_t = t_next - t_curr
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

        for i in iterator:
            t_curr = times[i]
            t_next = times[i + 1]
            t_val  = t_curr.item()

            # --- Inpaint: force known region onto the FM trajectory ---
            if gt is not None and mask is not None:
                gt_noisy = (1.0 - t_val) * gt + t_val * noise_for_gt
                img = torch.where(mask, gt_noisy, img)

            t_batch = torch.full((b,), t_val, device=device)
            t_batch_scaled = t_batch * 1000.0

            v_pred = model(img, timesteps=t_batch_scaled, **model_kwargs)

            d_t = t_next - t_curr
            img = img + v_pred * d_t

        # Final hard-replace: ensure generated output matches GT in known region
        if gt is not None and mask is not None:
            img = torch.where(mask, gt, img)

        return img  # [B, T, D]

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
