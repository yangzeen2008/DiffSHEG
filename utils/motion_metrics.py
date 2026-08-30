"""Rotation-aware metrics shared by training validation and standalone eval."""

from __future__ import annotations

import numpy as np
import torch

import datasets.rotation_converter as rot_cvt


def _stat_tensor(values, reference):
    return torch.as_tensor(values, device=reference.device, dtype=reference.dtype)


def select_axis_angle_stats(mean_axis_angle, std_axis_angle, gesture_dim):
    """Select statistics matching the gesture representation in use."""
    mean = np.asarray(mean_axis_angle)
    std = np.asarray(std_axis_angle)
    if gesture_dim == len(mean):
        return mean, std
    if gesture_dim == 33 and len(mean) >= 87:
        # remove_hand keeps the first seven body joints and four joints from
        # the original hand-free tail (the same indices used by the trainer).
        indices = list(range(0, 21)) + list(range(75, 87))
        return mean[indices], std[indices]
    return mean[:gesture_dim], std[:gesture_dim]


def gesture_to_rotation_matrix(motion, *, rot_6d, split_pos, mean_axis_angle, std_axis_angle):
    """Convert the gesture portion of a motion tensor to SO(3) matrices."""
    if rot_6d:
        gesture_dim = min(split_pos, motion.shape[-1])
        joint_count = gesture_dim // 6
        gesture = motion[..., :gesture_dim].reshape(*motion.shape[:-1], joint_count, 6)
        return rot_cvt.rotation_6d_to_matrix(gesture)

    gesture_dim = min(split_pos, motion.shape[-1])
    gesture_dim -= gesture_dim % 3
    joint_count = gesture_dim // 3
    normalized = motion[..., :gesture_dim]
    mean_values, std_values = select_axis_angle_stats(
        mean_axis_angle, std_axis_angle, gesture_dim
    )
    mean = _stat_tensor(mean_values, normalized)
    std = _stat_tensor(np.maximum(std_values, 1e-2), normalized)
    axis_angle = (normalized * std + mean).reshape(
        *normalized.shape[:-1], joint_count, 3
    )
    return rot_cvt.axis_angle_to_matrix(axis_angle)


def normalized_axis_angle(motion, *, rot_6d, split_pos, mean_axis_angle, std_axis_angle):
    """Return gesture-only normalized axis-angle features for FGD/diversity."""
    if not rot_6d:
        gesture_dim = min(split_pos, motion.shape[-1])
        gesture_dim -= gesture_dim % 3
        return motion[..., :gesture_dim]

    joint_count = min(split_pos, motion.shape[-1]) // 6
    gesture_dim = joint_count * 3
    matrices = gesture_to_rotation_matrix(
        motion,
        rot_6d=True,
        split_pos=split_pos,
        mean_axis_angle=mean_axis_angle,
        std_axis_angle=std_axis_angle,
    )
    axis_angle = rot_cvt.matrix_to_axis_angle(matrices).reshape(
        *motion.shape[:-1], gesture_dim
    )
    mean_values, std_values = select_axis_angle_stats(
        mean_axis_angle, std_axis_angle, gesture_dim
    )
    mean = _stat_tensor(mean_values, axis_angle)
    std = _stat_tensor(np.maximum(std_values, 1e-2), axis_angle)
    return (axis_angle - mean) / std


def rotation_geodesic_errors(prediction, target, *, rot_6d, split_pos, mean_axis_angle, std_axis_angle):
    """Per-joint geodesic rotation error in radians, shape [B, T, J]."""
    pred_matrix = gesture_to_rotation_matrix(
        prediction,
        rot_6d=rot_6d,
        split_pos=split_pos,
        mean_axis_angle=mean_axis_angle,
        std_axis_angle=std_axis_angle,
    )
    target_matrix = gesture_to_rotation_matrix(
        target,
        rot_6d=rot_6d,
        split_pos=split_pos,
        mean_axis_angle=mean_axis_angle,
        std_axis_angle=std_axis_angle,
    )
    relative = pred_matrix.transpose(-1, -2) @ target_matrix
    rotation_vectors = rot_cvt.matrix_to_axis_angle(relative)
    return torch.linalg.vector_norm(rotation_vectors, dim=-1)


def rotation_pck_mse(prediction, target, *, threshold=0.5, **kwargs):
    """Return rotation PCK and squared geodesic error (not position PCK)."""
    errors = rotation_geodesic_errors(prediction, target, **kwargs)
    pck = (errors < threshold).float().mean().item()
    mse = errors.square().mean().item()
    return pck, mse, errors


def linear_pck_mse(prediction, target, *, threshold=0.5):
    """Coefficient-wise accuracy and MSE for expression-only models."""
    errors = (prediction - target).abs()
    pck = (errors < threshold).float().mean().item()
    mse = errors.square().mean().item()
    return pck, mse, errors
