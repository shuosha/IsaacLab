# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import torch
import math

import isaacsim.core.utils.torch as torch_utils

def quat_geodesic_angle(q1: torch.Tensor, q2: torch.Tensor, eps: float = 1e-8):
    """
    q1, q2: (..., 4) float tensors in [w, x, y, z] or [x, y, z, w]—either is fine
            as long as both use the same convention.
    Returns: (...,) radians in [0, pi]
    """
    # normalize
    q1 = q1 / (q1.norm(dim=-1, keepdim=True).clamp_min(eps))
    q2 = q2 / (q2.norm(dim=-1, keepdim=True).clamp_min(eps))

    # dot, handle sign ambiguity
    dot = torch.sum(q1 * q2, dim=-1).abs().clamp(-1 + eps, 1 - eps)

    return 2.0 * torch.arccos(dot)

def get_keypoint_offsets(num_keypoints, device):
    """Get uniformly-spaced keypoints along a line of unit length, centered at 0."""
    keypoint_offsets = torch.zeros((num_keypoints, 3), device=device)
    keypoint_offsets[:, -1] = torch.linspace(0.0, 1.0, num_keypoints, device=device) - 0.5
    return keypoint_offsets


def get_deriv_gains(prop_gains, rot_deriv_scale=1.0):
    """Set robot gains using critical damping."""
    deriv_gains = 2 * torch.sqrt(prop_gains)
    deriv_gains[:, 3:6] /= rot_deriv_scale
    return deriv_gains


def wrap_yaw(angle):
    """Ensure yaw stays within range."""
    return torch.where(angle > np.deg2rad(235), angle - 2 * np.pi, angle)


def set_friction(asset, value, num_envs):
    """Update material properties for a given asset."""
    materials = asset.root_physx_view.get_material_properties()
    materials[..., 0] = value  # Static friction.
    materials[..., 1] = value  # Dynamic friction.
    env_ids = torch.arange(num_envs, device="cpu")
    asset.root_physx_view.set_material_properties(materials, env_ids)


def set_body_inertias(robot, num_envs):
    """Note: this is to account for the asset_options.armature parameter in IGE."""
    inertias = robot.root_physx_view.get_inertias()
    offset = torch.zeros_like(inertias)
    offset[:, :, [0, 4, 8]] += 0.01
    new_inertias = inertias + offset
    robot.root_physx_view.set_inertias(new_inertias, torch.arange(num_envs))


def get_held_base_pos_local(task_name, fixed_asset_cfg, num_envs, device):
    """Get transform between asset default frame and geometric base frame."""
    held_base_x_offset = 0.0
    if task_name == "peg_insert":
        held_base_z_offset = 0.0
    elif task_name == "gear_mesh":
        gear_base_offset = fixed_asset_cfg.medium_gear_base_offset
        held_base_x_offset = gear_base_offset[0]
        held_base_z_offset = gear_base_offset[2]
    elif task_name == "nut_thread":
        held_base_z_offset = 0.0
    else:
        raise NotImplementedError("Task not implemented")

    held_base_pos_local = torch.tensor([0.0, 0.0, 0.0], device=device).repeat((num_envs, 1))
    held_base_pos_local[:, 0] = held_base_x_offset
    held_base_pos_local[:, 2] = held_base_z_offset

    return held_base_pos_local


def get_held_base_pose(held_pos, held_quat, task_name, fixed_asset_cfg, num_envs, device):
    """Get current poses for keypoint and success computation."""
    held_base_pos_local = get_held_base_pos_local(task_name, fixed_asset_cfg, num_envs, device)
    held_base_quat_local = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).unsqueeze(0).repeat(num_envs, 1)

    held_base_quat, held_base_pos = torch_utils.tf_combine(
        held_quat, held_pos, held_base_quat_local, held_base_pos_local
    )
    return held_base_pos, held_base_quat


def get_target_held_base_pose(fixed_pos, fixed_quat, task_name, fixed_asset_cfg, num_envs, device):
    """Get target poses for keypoint and success computation."""
    fixed_success_pos_local = torch.zeros((num_envs, 3), device=device)
    if task_name == "peg_insert":
        fixed_success_pos_local[:, 2] = 0.0
    elif task_name == "gear_mesh":
        gear_base_offset = fixed_asset_cfg.medium_gear_base_offset
        fixed_success_pos_local[:, 0] = gear_base_offset[0]
        fixed_success_pos_local[:, 2] = gear_base_offset[2]
    elif task_name == "nut_thread":
        fixed_success_pos_local[:, 2] = 0.04
    else:
        raise NotImplementedError("Task not implemented")
    fixed_success_quat_local = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).unsqueeze(0).repeat(num_envs, 1)

    target_held_base_quat, target_held_base_pos = torch_utils.tf_combine(
        fixed_quat, fixed_pos, fixed_success_quat_local, fixed_success_pos_local
    )
    return target_held_base_pos, target_held_base_quat


def squashing_fn(x, a, b):
    """Compute bounded reward function."""
    return 1 / (torch.exp(a * x) + b + torch.exp(-a * x))


def collapse_obs_dict(obs_dict, obs_order):
    """Stack observations in given order."""
    obs_tensors = [obs_dict[obs_name] for obs_name in obs_order]
    obs_tensors = torch.cat(obs_tensors, dim=-1)
    return obs_tensors

def quat_rotate_x_axis(q: torch.Tensor) -> torch.Tensor:
    """
    q: (N,4) quaternions in wxyz
    returns: (N,3) unit vectors = x-axis of the rotated frame in world coords
    """
    # normalize to be safe
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    w, x, y, z = q.unbind(dim=-1)

    # Rotate the x-axis (1,0,0) using closed-form quaternion rotation
    # v' = q * (0,1,0,0) * q_conj
    # Resulting vector:
    vx = 1 - 2*(y*y + z*z)
    vy = 2*(x*y + w*z)
    vz = 2*(x*z - w*y)

    v = torch.stack([vx, vy, vz], dim=-1)
    return v

def x_axis_diff_ge_n_deg(q1: torch.Tensor,
                         q2: torch.Tensor,
                         n_deg: float) -> torch.Tensor:
    """
    q1, q2: (N,4) quats in wxyz
    n_deg: float in [0,180]; angle threshold in degrees
    returns: (N,) bool tensor, True if x axes differ by at least n_deg
    """
    v1 = quat_rotate_x_axis(q1)          # (N,3)
    v2 = quat_rotate_x_axis(q2)          # (N,3)

    # dot between unit vectors
    dot = (v1 * v2).sum(dim=-1).clamp(-1.0, 1.0)

    angle_rad = torch.arccos(dot)        # (N,)
    angle_deg = angle_rad * (180.0 / math.pi)

    return angle_deg >= n_deg

from huggingface_hub import hf_hub_download
from pathlib import Path

def resolve_hf_file(repo_id: str, filename: str, repo_type = "dataset", revision: str | None = None) -> str:
    p = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type=repo_type,
        revision=revision,
    )
    return str(Path(p))

from pathlib import Path
from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download

def resolve_hf_path(
    repo_id: str,
    path: str,
    repo_type: str = "dataset",
    revision: str | None = None,
    cache_dir: str | None = None,
) -> str:
    """
    If `path` is a file in the repo -> download that file and return its local path.
    If `path` is a directory prefix -> snapshot_download only files under that prefix
    and return the local directory path corresponding to that prefix.
    """
    path = path.strip("/")
    files = list_repo_files(repo_id=repo_id, repo_type=repo_type, revision=revision)

    # File case
    if path in files:
        p = hf_hub_download(
            repo_id=repo_id,
            filename=path,
            repo_type=repo_type,
            revision=revision,
            cache_dir=cache_dir,
        )
        return str(Path(p))

    # Directory (prefix) case
    prefix = path + "/"
    matched = [f for f in files if f.startswith(prefix)]
    if not matched:
        raise FileNotFoundError(f"'{path}' not found as file or directory in {repo_id} ({repo_type})")

    # Download only what's under the prefix
    repo_local = snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        cache_dir=cache_dir,
        allow_patterns=[prefix + "*"],
    )

    # Return the local directory corresponding to that prefix
    return str(Path(repo_local) / prefix)
