# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Factory: control module.

Imported by base, environment, and task classes. Not directly executed.
"""

import math
import torch

import isaacsim.core.utils.torch as torch_utils

from isaaclab.utils.math import axis_angle_from_quat


def compute_dof_torque(
    cfg,
    dof_pos,
    dof_vel,
    fingertip_midpoint_pos,
    fingertip_midpoint_quat,
    fingertip_midpoint_linvel,
    fingertip_midpoint_angvel,
    jacobian,
    arm_mass_matrix,
    ctrl_target_fingertip_midpoint_pos,
    ctrl_target_fingertip_midpoint_quat,
    task_prop_gains,
    task_deriv_gains,
    device,
    dead_zone_thresholds=None,
    ):
    """Compute Franka DOF torque to move fingertips towards target pose."""
    # References:
    # 1) https://ethz.ch/content/dam/ethz/special-interest/mavt/robotics-n-intelligent-systems/rsl-dam/documents/RobotDynamics2018/RD_HS2018script.pdf
    # 2) Modern Robotics

    num_envs = cfg.scene.num_envs
    dof_torque = torch.zeros((num_envs, dof_pos.shape[1]), device=device)
    task_wrench = torch.zeros((num_envs, 6), device=device)

    pos_error, axis_angle_error = get_pose_error(
        fingertip_midpoint_pos=fingertip_midpoint_pos,
        fingertip_midpoint_quat=fingertip_midpoint_quat,
        ctrl_target_fingertip_midpoint_pos=ctrl_target_fingertip_midpoint_pos,
        ctrl_target_fingertip_midpoint_quat=ctrl_target_fingertip_midpoint_quat,
        jacobian_type="geometric",
        rot_error_type="axis_angle",
    )
    delta_fingertip_pose = torch.cat((pos_error, axis_angle_error), dim=1)

    # Set tau = k_p * task_pos_error - k_d * task_vel_error (building towards eq. 3.96-3.98)
    task_wrench_motion = _apply_task_space_gains(
        delta_fingertip_pose=delta_fingertip_pose,
        fingertip_midpoint_linvel=fingertip_midpoint_linvel,
        fingertip_midpoint_angvel=fingertip_midpoint_angvel,
        task_prop_gains=task_prop_gains,
        task_deriv_gains=task_deriv_gains,
    )
    task_wrench += task_wrench_motion

    # Offset task_wrench motion by random amount to simulate unreliability at low forces.
    # Check if absolute value is less than specified amount. If so, 0 out, otherwise, subtract.
    if dead_zone_thresholds is not None:
        task_wrench = torch.where(
            task_wrench.abs() < dead_zone_thresholds,
            torch.zeros_like(task_wrench),
            task_wrench.sign() * (task_wrench.abs() - dead_zone_thresholds),
        )

    # Set tau = J^T * tau, i.e., map tau into joint space as desired
    jacobian_T = torch.transpose(jacobian, dim0=1, dim1=2)
    dof_torque[:, 0:7] = (jacobian_T @ task_wrench.unsqueeze(-1)).squeeze(-1)

    # adapted from https://gitlab-master.nvidia.com/carbon-gym/carbgym/-/blob/b4bbc66f4e31b1a1bee61dbaafc0766bbfbf0f58/python/examples/franka_cube_ik_osc.py#L70-78
    # roboticsproceedings.org/rss07/p31.pdf

    # useful tensors
    arm_mass_matrix_inv = torch.inverse(arm_mass_matrix)
    jacobian_T = torch.transpose(jacobian, dim0=1, dim1=2)
    arm_mass_matrix_task = torch.inverse(
        jacobian @ torch.inverse(arm_mass_matrix) @ jacobian_T
    )  # ETH eq. 3.86; geometric Jacobian is assumed
    j_eef_inv = arm_mass_matrix_task @ jacobian @ arm_mass_matrix_inv
    default_dof_pos_tensor = torch.tensor(cfg.ctrl.default_dof_pos_tensor, device=device).repeat((num_envs, 1))
    # nullspace computation
    distance_to_default_dof_pos = default_dof_pos_tensor - dof_pos[:, :7]
    distance_to_default_dof_pos = (distance_to_default_dof_pos + math.pi) % (
        2 * math.pi
    ) - math.pi  # normalize to [-pi, pi]
    u_null = cfg.ctrl.kd_null * -dof_vel[:, :7] + cfg.ctrl.kp_null * distance_to_default_dof_pos
    u_null = arm_mass_matrix @ u_null.unsqueeze(-1)
    torque_null = (torch.eye(7, device=device).unsqueeze(0) - torch.transpose(jacobian, 1, 2) @ j_eef_inv) @ u_null
    dof_torque[:, 0:7] += torque_null.squeeze(-1)

    # TODO: Verify it's okay to no longer do gripper control here.
    dof_torque = torch.clamp(dof_torque, min=-100.0, max=100.0)
    return dof_torque, task_wrench

def get_pose_error(
    fingertip_midpoint_pos,
    fingertip_midpoint_quat,
    ctrl_target_fingertip_midpoint_pos,
    ctrl_target_fingertip_midpoint_quat,
    jacobian_type,
    rot_error_type,
    ):
    """Compute task-space error between target Franka fingertip pose and current pose."""
    # Reference: https://ethz.ch/content/dam/ethz/special-interest/mavt/robotics-n-intelligent-systems/rsl-dam/documents/RobotDynamics2018/RD_HS2018script.pdf

    # Compute pos error 
    pos_error = ctrl_target_fingertip_midpoint_pos - fingertip_midpoint_pos 

    # Compute rot error
    if jacobian_type == "geometric":  # See example 2.9.8; note use of J_g and transformation between rotation vectors
        # Compute quat error (i.e., difference quat)
        # Reference: https://personal.utdallas.edu/~sxb027100/dock/quat.html

        # Check for shortest path using quaternion dot product.
        quat_dot = (ctrl_target_fingertip_midpoint_quat * fingertip_midpoint_quat).sum(dim=1, keepdim=True)
        ctrl_target_fingertip_midpoint_quat = torch.where(
            quat_dot.expand(-1, 4) >= 0, ctrl_target_fingertip_midpoint_quat, -ctrl_target_fingertip_midpoint_quat
        )

        fingertip_midpoint_quat_norm = torch_utils.quat_mul(
            fingertip_midpoint_quat, torch_utils.quat_conjugate(fingertip_midpoint_quat)
        )[
            :, 0
        ]  # scalar component
        fingertip_midpoint_quat_inv = torch_utils.quat_conjugate(
            fingertip_midpoint_quat
        ) / fingertip_midpoint_quat_norm.unsqueeze(-1)
        quat_error = torch_utils.quat_mul(ctrl_target_fingertip_midpoint_quat, fingertip_midpoint_quat_inv)

        # Convert to axis-angle error
        axis_angle_error = axis_angle_from_quat(quat_error)

    if rot_error_type == "quat":
        return pos_error, quat_error
    elif rot_error_type == "axis_angle":
        return pos_error, axis_angle_error

def get_delta_dof_pos(delta_pose, ik_method, jacobian, device):
    """Get delta Franka DOF position from delta pose using specified IK method."""
    # References:
    # 1) https://www.cs.cmu.edu/~15464-s13/lectures/lecture6/iksurvey.pdf
    # 2) https://ethz.ch/content/dam/ethz/special-interest/mavt/robotics-n-intelligent-systems/rsl-dam/documents/RobotDynamics2018/RD_HS2018script.pdf (p. 47)

    if ik_method == "pinv":  # Jacobian pseudoinverse
        k_val = 1.0
        jacobian_pinv = torch.linalg.pinv(jacobian)
        delta_dof_pos = k_val * jacobian_pinv @ delta_pose.unsqueeze(-1)
        delta_dof_pos = delta_dof_pos.squeeze(-1)

    elif ik_method == "trans":  # Jacobian transpose
        k_val = 1.0
        jacobian_T = torch.transpose(jacobian, dim0=1, dim1=2)
        delta_dof_pos = k_val * jacobian_T @ delta_pose.unsqueeze(-1)
        delta_dof_pos = delta_dof_pos.squeeze(-1)

    elif ik_method == "dls":  # damped least squares (Levenberg-Marquardt)
        lambda_val = 0.1  # 0.1
        jacobian_T = torch.transpose(jacobian, dim0=1, dim1=2)
        lambda_matrix = (lambda_val**2) * torch.eye(n=jacobian.shape[1], device=device)
        delta_dof_pos = jacobian_T @ torch.inverse(jacobian @ jacobian_T + lambda_matrix) @ delta_pose.unsqueeze(-1)
        delta_dof_pos = delta_dof_pos.squeeze(-1)

    elif ik_method == "svd":  # adaptive SVD
        k_val = 1.0
        U, S, Vh = torch.linalg.svd(jacobian)
        S_inv = 1.0 / S
        min_singular_value = 1.0e-5
        S_inv = torch.where(S > min_singular_value, S_inv, torch.zeros_like(S_inv))
        jacobian_pinv = (
            torch.transpose(Vh, dim0=1, dim1=2)[:, :, :6] @ torch.diag_embed(S_inv) @ torch.transpose(U, dim0=1, dim1=2)
        )
        delta_dof_pos = k_val * jacobian_pinv @ delta_pose.unsqueeze(-1)
        delta_dof_pos = delta_dof_pos.squeeze(-1)

    return delta_dof_pos


def _apply_task_space_gains(
    delta_fingertip_pose, fingertip_midpoint_linvel, fingertip_midpoint_angvel, task_prop_gains, task_deriv_gains
    ):
    """Interpret PD gains as task-space gains. Apply to task-space error."""

    task_wrench = torch.zeros_like(delta_fingertip_pose)

    # Apply gains to lin error components
    lin_error = delta_fingertip_pose[:, 0:3]
    task_wrench[:, 0:3] = task_prop_gains[:, 0:3] * lin_error + task_deriv_gains[:, 0:3] * (
        0.0 - fingertip_midpoint_linvel
    )

    # Apply gains to rot error components
    rot_error = delta_fingertip_pose[:, 3:6]
    task_wrench[:, 3:6] = task_prop_gains[:, 3:6] * rot_error + task_deriv_gains[:, 3:6] * (
        0.0 - fingertip_midpoint_angvel
    )
    return task_wrench

def get_task_space_error(
    cur_pos, cur_quat,
    tgt_pos, tgt_quat,
    jacobian_type="geometric",
    rot_error_type="axis_angle",
):
    """
    Returns errors as (current - target) for BOTH position and orientation.
    Assumes unit quaternions. Uses shortest-arc convention.
    """
    # 1) Position error: current - target  (matches admittance F_sd = K*e + D*xdot_ref)
    pos_error = cur_pos - tgt_pos

    # 2) Orientation error: current - target
    # Shortest path: flip target if dot<0
    dot = (tgt_quat * cur_quat).sum(dim=1, keepdim=True)
    tgt = torch.where(dot >= 0, tgt_quat, -tgt_quat)

    # For unit quats, inverse == conjugate
    tgt_inv = torch_utils.quat_conjugate(tgt)
    # q_rel = cur ∘ tgt^{-1}  (gives “current − target”)
    q_rel  = torch_utils.quat_mul(cur_quat, tgt_inv)

    if rot_error_type == "quat":
        rot_error = q_rel  # expresses current relative to target
    else:
        # axis-angle from q_rel (already “current − target”)
        rot_error = axis_angle_from_quat(q_rel)

    return pos_error, rot_error

def compute_dof_state_admittance(
    dof_pos,
    eef_pos, eef_quat,
    jacobian,
    ctrl_target_eef_pos, 
    ctrl_target_eef_quat,
    dt, device,
    xdot_ref,                    # (B,6)
    F_ext=None,                  # (B,6)
    Kx=200.0, Dx=None, mx=1.0,   # each: float or (B,)
    Kr=50.0,  Dr=None, mr=0.1,
    lam=1e-2,
    rot_scale=0.25,
    alpha=1.0,
):
    B, _ = dof_pos.shape
    n = 7

    if F_ext is None:
        F_ext = torch.zeros(B, 6, device=device)
    F_ext = F_ext * alpha

    # task-space error
    pos_err, aa_err = get_task_space_error(
        eef_pos, eef_quat,
        ctrl_target_eef_pos, ctrl_target_eef_quat,
        jacobian_type="geometric", rot_error_type="axis_angle",
    )
    e_task = torch.cat((pos_err, rot_scale * aa_err), dim=1)  # (B,6)

    # per-env scalars (B,)
    def to_B(x):
        x = torch.as_tensor(x, device=device, dtype=torch.float32)
        return x.expand(B) if x.ndim == 0 else x

    Kx, mx, Kr, mr = map(to_B, (Kx, mx, Kr, mr))
    if Dx is None: 
        Dx = 2.0 * torch.sqrt(Kx * mx)
    else:          
        Dx = to_B(Dx)
    if Dr is None: 
        Dr = 2.0 * torch.sqrt(Kr * mr)
    else:          
        Dr = to_B(Dr)

    lam = to_B(lam)                 # (B,)
    lam2 = (lam ** 2).view(B, 1, 1) # (B,1,1)

    # build 6D vectors from (B,)
    K = torch.stack([Kx, Kx, Kx, Kr, Kr, Kr], dim=1)    # (B,6)
    D = torch.stack([Dx, Dx, Dx, Dr, Dr, Dr], dim=1)    # (B,6)
    M = torch.stack([mx, mx, mx, mr, mr, mr], dim=1)    # (B,6)

    # admittance update
    F_sd  = K * e_task + D * xdot_ref
    xddot = (F_ext - F_sd) / M
    xdot_ref = xdot_ref + dt * xddot

    # damped pseudoinverse
    JT  = jacobian.transpose(1, 2)          # (B,n,6)
    I6  = torch.eye(6, device=device).expand(B, 6, 6)
    J_pinv = torch.bmm(JT, torch.linalg.inv(torch.bmm(jacobian, JT) + lam2 * I6))
    qd_next = torch.bmm(J_pinv, xdot_ref.unsqueeze(-1)).squeeze(-1)

    q_next = dof_pos[:, :n] + dt * qd_next
    return q_next, qd_next, xddot, e_task, xdot_ref

_DEBUG_COUNTER = {"k": 0}

def compute_dof_state_admittance_debug(
    *args, print_every=1, name="adm", **kwargs
):
    out = compute_dof_state_admittance(*args, **kwargs)
    q_next, qd_next, xddot, e_task, xdot_ref = out

    with torch.no_grad():
        # Unpack inputs we need for diagnostics
        (
            cfg, dof_pos, dof_vel, eef_pos, eef_quat, eef_linvel, eef_angvel,
            jacobian, ctrl_target_eef_pos, ctrl_target_eef_quat, dt, device,
            xdot_ref_in, F_ext, Kx, Dx, mx, Kr, Dr, mr, lam, rot_scale,
            v_task_limits, qd_limit
        ) = _recover_args_for_debug(*args, **kwargs)

        B = dof_pos.shape[0]
        JT = jacobian.transpose(1, 2)
        JJt = torch.bmm(jacobian, JT)
        # Condition number
        cond = torch.linalg.cond(JJt)  # (B,)

        # Quaternion sanity
        q_norm_err = (eef_quat.norm(dim=1) - 1.0).abs().mean()

        # Error split
        pos_err = e_task[:, :3]
        rot_err = e_task[:, 3:]

        # Mapping residual: are we tracking xdot_ref?
        xdot_map = torch.bmm(jacobian, qd_next.unsqueeze(-1)).squeeze(-1)
        resid = (xdot_ref - xdot_map).norm(dim=1)  # per-env
        resid_rel = resid / (xdot_ref.norm(dim=1).clamp_min(1e-8))

        # Clamping stats
        lin_max, ang_max = v_task_limits
        xref_sat_lin = (xdot_ref[:, :3].abs() >= lin_max - 1e-12).float().mean()
        xref_sat_rot = (xdot_ref[:, 3:].abs() >= ang_max - 1e-12).float().mean()
        qd_sat = (qd_next.abs() >= qd_limit - 1e-12).float().mean()

        k = _DEBUG_COUNTER["k"]
        if k % print_every == 0:
            print(
                f"[{name}] step={k} "
                f"|e_lin|={pos_err.norm(dim=1).mean():.3e} e_lin_x={pos_err[:,0].mean():+.3e} "
                f"|e_rot|={rot_err.norm(dim=1).mean():.3e} "
                f"|xddot|={xddot.norm(dim=1).mean():.3e} "
                f"|xdot_ref|={xdot_ref.norm(dim=1).mean():.3e} "
                f"|qdot|={qd_next.norm(dim=1).mean():.3e} "
                f"cond(JJ^T)={cond.mean():.3e} "
                f"resid=||J qdot - xdot_ref||={resid.mean():.3e} "
                f"rel={resid_rel.mean():.3e} "
                f"xref_sat(lin,rot)=({xref_sat_lin:.2f},{xref_sat_rot:.2f}) "
                f"qd_sat={qd_sat:.2f} "
                f"|q|-1={q_norm_err:.2e}"
            )
        _DEBUG_COUNTER["k"] = k + 1

    return out

def _recover_args_for_debug(
    cfg, dof_pos, dof_vel, eef_pos, eef_quat, eef_linvel, eef_angvel,
    jacobian, ctrl_target_eef_pos, ctrl_target_eef_quat, dt, device, xdot_ref,
    F_ext=None, Kx=200.0, Dx=None, mx=5.0, Kr=50.0, Dr=None, mr=0.2,
    lam=1e-3, rot_scale=1.0, v_task_limits=(0.5, 1.0), qd_limit=1.5
):
    return (
        cfg, dof_pos, dof_vel, eef_pos, eef_quat, eef_linvel, eef_angvel,
        jacobian, ctrl_target_eef_pos, ctrl_target_eef_quat, dt, device, xdot_ref,
        F_ext, Kx, Dx, mx, Kr, Dr, mr, lam, rot_scale, v_task_limits, qd_limit
    )

