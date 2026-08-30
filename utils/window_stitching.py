"""Utilities for continuity-safe long-form and cached-window inference."""

from __future__ import annotations

import math

import torch


def resolve_overlap_len(n_poses, stride, overlap_len, *, auto_overlap, sequential):
    """Resolve a safe overlap while preserving explicit non-sequential tests."""
    overlap = int(overlap_len)
    if sequential and auto_overlap and overlap <= 0:
        overlap = int(n_poses) - int(stride)
    if overlap < 0 or overlap >= int(n_poses):
        raise ValueError(
            f"overlap_len must satisfy 0 <= overlap_len < n_poses; "
            f"got overlap_len={overlap}, n_poses={n_poses}"
        )
    return overlap


def audio_windows_are_consecutive(
    previous,
    current,
    *,
    pose_stride,
    pose_fps,
    audio_fps,
    atol=1e-7,
):
    """Return True only when two fixed windows share the expected raw audio."""
    if previous is None or current is None:
        return False
    previous = previous.reshape(-1)
    current = current.reshape(-1)
    if previous.numel() != current.numel():
        return False

    exact_step = float(pose_stride) * float(audio_fps) / float(pose_fps)
    candidate_steps = {int(math.floor(exact_step)), int(math.ceil(exact_step))}
    for step_samples in candidate_steps:
        overlap_samples = previous.numel() - step_samples
        if overlap_samples <= 0:
            continue
        if torch.allclose(
            previous[-overlap_samples:],
            current[:overlap_samples],
            rtol=0.0,
            atol=atol,
        ):
            return True
    return False


def make_prefix_inpaint(template, previous_output, overlap_len):
    """Build an inpainting request that exactly reuses the prior window tail."""
    if previous_output is None or overlap_len <= 0:
        return {}
    if template.shape[0] != previous_output.shape[0]:
        raise ValueError("template and previous_output batch sizes must match")
    if template.shape[-1] != previous_output.shape[-1]:
        raise ValueError("template and previous_output feature sizes must match")
    overlap = min(int(overlap_len), template.shape[1], previous_output.shape[1])
    known = torch.zeros_like(template)
    mask = torch.zeros_like(template, dtype=torch.bool, device=template.device)
    known[:, :overlap] = previous_output[:, -overlap:].to(template)
    mask[:, :overlap] = True
    return {"gt": known, "outpainting_mask": mask}
