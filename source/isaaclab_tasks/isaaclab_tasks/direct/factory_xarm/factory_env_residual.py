# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import torch
import math
import os
from pathlib import Path

import carb
import isaacsim.core.utils.torch as torch_utils

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import TiledCamera, ContactSensor
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import axis_angle_from_quat
from isaaclab.markers import VisualizationMarkers

from . import factory_control, factory_utils
from .factory_env_cfg import OBS_DIM_CFG, STATE_DIM_CFG, FactoryEnvCfg

from .nn_buffer import NearestNeighborBuffer
from .ring_buffer import LastKPoints
import pytorch_kinematics as pk
import sapien.core as sapien

class FactoryEnvResidual(DirectRLEnv):
    cfg: FactoryEnvCfg

    def __init__(self, cfg: FactoryEnvCfg, render_mode: str | None = None, **kwargs):        
        # Update number of obs/states
        cfg.observation_space = sum([OBS_DIM_CFG[obs] for obs in cfg.residual_obs_order])
        cfg.state_space = sum([STATE_DIM_CFG[state] for state in cfg.residual_state_order])
        cfg.action_space = cfg.residual_action_space # 7
        cfg.observation_space += cfg.action_space
        cfg.state_space += cfg.action_space
        self.cfg_task = cfg.task

        super().__init__(cfg, render_mode, **kwargs)

        # factory_utils.set_body_inertias(self._robot, self.scene.num_envs)
        self._init_tensors()
        self._set_default_dynamics_parameters()
        self._init_residual_policy_buffers()

    def build_init_state(self, data: str, dtype=torch.float32):
        """
        Returns:
        init: (num_eps, 13) torch tensor
        ep_keys: list[str] episode ordering used
        """
        data = np.load(data, allow_pickle=True).item()
        ep_keys = sorted(data.keys())
        num_eps = len(ep_keys)

        init = torch.empty((num_eps, 13), device=self.device, dtype=dtype)

        for i, k in enumerate(ep_keys):
            ep = data[k]

            pos  = torch.as_tensor(ep["obs.fingertip_pos"][0], device=self.device, dtype=dtype)   # (3,)
            quat = torch.as_tensor(ep["obs.fingertip_quat"][0], device=self.device, dtype=dtype)  # (4,)

            rel_fixed = torch.as_tensor(ep["obs.fingertip_pos_rel_fixed"][0], device=self.device, dtype=dtype)  # (3,)
            rel_held  = torch.as_tensor(ep["obs.fingertip_pos_rel_held"][0],  device=self.device, dtype=dtype)  # (3,)
            a = pos - rel_fixed  # (3,)
            b = pos - rel_held   # (3,)

            init[i] = torch.cat([pos, quat, a, b], dim=0)  # (13,)

        return init.unsqueeze(0).expand(self.num_envs, -1, -1)

    def _init_residual_policy_buffers(self):
        """Initialize buffers specific to residual policy."""
        self.verbose = self.cfg.env_options.verbose
        self.teleop_mode = self.cfg.env_options.teleop_mode
        self.vis_options = self.cfg.env_options.vis_options

        # base policy
        if self.cfg.env_options.base_model == "bc":
            from lerobot.rrl.dp_wrapper import DPWrapper
            self.base_policy = DPWrapper(factory_utils.resolve_hf_path(self.cfg_task.hf_repo, self.cfg_task.diffusion_path))
        elif self.cfg.env_options.base_model == "nn" or self.cfg.env_options.base_model == "noisy_nn":
            self.base_policy = NearestNeighborBuffer(
                factory_utils.resolve_hf_file(self.cfg_task.hf_repo, self.cfg_task.train_data_hf_file), 
                self.num_envs, 
                min_horizon=self.cfg.base_rand.horizon[0], 
                max_horizon=self.cfg.base_rand.horizon[1], 
                device=self.device, 
                pad=True, # type: ignore
                offline_base=self.cfg.env_options.offline_base,
            )
            if self.cfg.env_options.base_model == "noisy_nn":
                self.add_noise_to_base = True

        self.base_actions = torch.zeros((self.num_envs, 8), device=self.device)

        # initial states
        self.initial_poses = self.build_init_state(factory_utils.resolve_hf_file(self.cfg_task.hf_repo, self.cfg_task.train_data_hf_file)) # (num_envs, num_eps, 13)
        self.total_episodes: int = self.initial_poses.shape[1]

        if self.cfg.env_options.step_eps:
            self.episode_idx = torch.randint(0, self.total_episodes, (self.num_envs,), device=self.device)
        else:
            self.episode_idx = torch.arange(0, self.num_envs, device=self.device) % self.total_episodes # fixed eps idx

        # ctrl params
        self.Kx = torch.tensor([self.cfg.ctrl.Kx], device=self.device).repeat(self.num_envs) # (num_envs, )
        self.Kr = torch.tensor([self.cfg.ctrl.Kr], device=self.device).repeat(self.num_envs) # (num_envs, )
        self.mx = torch.tensor([self.cfg.ctrl.mx], device=self.device).repeat(self.num_envs) # (num_envs, )
        self.mr = torch.tensor([self.cfg.ctrl.mr], device=self.device).repeat(self.num_envs) # (num_envs, )

        # abs ik for reset
        robot_dir = factory_utils.resolve_robot_dir_materialized(self.cfg_task.hf_repo, cache_dir=os.path.expanduser("~/.cache/huggingface"))
        urdf_path = str(Path(robot_dir) / "xarm7.urdf")

        chain = pk.build_chain_from_urdf(open(urdf_path, mode="rb").read())
        # urdf = factory_utils.resolve_hf_file(self.cfg_task.hf_repo, "assets/robot/xarm7.urdf")
        # chain = pk.build_chain_from_urdf(open(urdf, mode="rb").read())
        # chain.print_tree()
        self.serial_chain = pk.SerialChain(chain, "link7", "link_base")
        self.lim = torch.tensor(chain.get_joint_limits())[:, :7]
        self.abs_ik = pk.PseudoInverseIK(self.serial_chain, max_iterations=30, num_retries=1,
            joint_limits=self.lim.T,
            early_stopping_any_converged=True,
            early_stopping_no_improvement="all",
            debug=False,
            lr=0.2)

        # TODO: test sapien 
        self.robot_name = "xarm7"

        # load sapien robot
        engine = sapien.Engine()            # create once
        scene = engine.create_scene()
        loader = scene.create_urdf_loader()
        self.sapien_robot = loader.load(urdf_path)
        self.robot_model = self.sapien_robot.create_pinocchio_model()
        self.sapien_eef_idx = -1
        for link_idx, link in enumerate(self.sapien_robot.get_links()):
            if link.name == "link7":
                self.sapien_eef_idx = link_idx
                break

        # Held asset yaw rotation tracking (only for nut_thread task, only when task-engaged)
        if self.cfg_task.name == "nut_thread":
            self.prev_held_yaw = torch.zeros(self.num_envs, device=self.device)
            self.cumulative_rotation = torch.zeros(self.num_envs, device=self.device)  # cumulative yaw rotation in degrees
            self.picked_up = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self.base_last5 = LastKPoints(self.num_envs, K=5, device=self.device)
        self.residual_last5 = LastKPoints(self.num_envs, K=5, device=self.device)
        # self.bad_insert = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)


    def compute_ik_abs(
        self,
        action: torch.Tensor,
        curr_qpos: torch.Tensor,  # (N, DOF)
        ):
        tf = torch.eye(4, device=action.device, dtype=action.dtype).unsqueeze(0).repeat(action.shape[0], 1, 1)  # (N,4,4)
        tf[:, :3, :3] = torch_utils.quats_to_rot_matrices(action[:, 3:7])
        tf[:, :3, 3] = action[:, :3]

        rob_tf = pk.Transform3d(matrix=tf.to("cpu"), dtype=action.dtype)

        self.abs_ik.initial_config = curr_qpos.to("cpu").to(torch.float32)
        output = self.abs_ik.solve(rob_tf)
        converged = output.converged    # (G, R) bool
        solutions = output.solutions    # (G, R, DOF)

        first_conv = converged.float().argmax(dim=1)        # (G,)
        G, R, DOF = solutions.shape
        idx = first_conv.view(G, 1, 1).expand(-1, 1, DOF)   # (G, 1, DOF)
        qpos = solutions.gather(1, idx).squeeze(1).to(action.device)        # (G, DOF)

        return qpos

    def _set_default_dynamics_parameters(self):
        """Set parameters defining dynamic interactions."""
        self.pos_threshold = torch.tensor(self.cfg.ctrl.res_pos_action_threshold, device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.rot_threshold = torch.tensor(self.cfg.ctrl.res_rot_action_threshold, device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.gripper_threshold = torch.tensor(self.cfg.ctrl.res_gripper_action_threshold, device=self.device).repeat(
            (self.num_envs, 1)
        )

        # Set masses and frictions.
        factory_utils.set_friction(self._held_asset, self.cfg_task.held_asset_cfg.friction, self.scene.num_envs)
        factory_utils.set_friction(self._fixed_asset, self.cfg_task.fixed_asset_cfg.friction, self.scene.num_envs)
        factory_utils.set_friction(self._robot, self.cfg_task.robot_cfg.friction, self.scene.num_envs)

    def _init_tensors(self):
        """Initialize tensors once."""
        # Control targets.
        self.ctrl_target_joint_pos = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)
        self.ema_factor = self.cfg.ctrl.ema_factor
        self.dead_zone_thresholds = None

        # Fixed asset.
        self.fixed_pos_obs_frame = torch.zeros((self.num_envs, 3), device=self.device) # fixed obj pos in base frame
        self.init_fixed_pos_obs_noise = torch.zeros((self.num_envs, 3), device=self.device) # fixed obj pos noise

        # Held asset.
        self.held_pos_obs_frame = torch.zeros((self.num_envs, 3), device=self.device)
        self.init_held_pos_obs_noise = torch.zeros((self.num_envs, 3), device=self.device)

        # traj geom augmentation
        self.xy_translation_noise = torch.zeros((self.num_envs, 2), device=self.device)
        self.yaw_rotation_noise = torch.zeros((self.num_envs, 1), device=self.device)

        self.held_center_pos_local = torch.zeros((self.num_envs, 3), device=self.device) # center2held transform
        if self.cfg_task.name == "gear_mesh":
            self.held_center_pos_local[:, 0] += self.cfg_task.fixed_asset_cfg.medium_gear_base_offset[0]
            self.held_center_pos_local[:, 2] += 0.0175 # offset
            self.held_center_pos_local[:, 2] += 0.0125 - 0.005 # offset from center to top of gear

        elif self.cfg_task.name == "peg_insert":
            self.held_center_pos_local[:, 2] += self.cfg_task.held_asset_cfg.height 
            self.held_center_pos_local[:, 2] -= 0.02

        # Computer body indices.
        self.left_finger_body_idx = self._robot.body_names.index("left_finger") 
        self.right_finger_body_idx = self._robot.body_names.index("right_finger")
        self.eef_body_idx = self._robot.body_names.index("link7")
        self.sim_fingertip2eef = torch.tensor([self.cfg.sim_fingertip2eef], device=self.device).repeat(self.num_envs, 1)
        self.real_fingertip2eef = torch.tensor([self.cfg.real_fingertip2eef], device=self.device).repeat(self.num_envs, 1)
        self.arm_dof_idx, _ = self._robot.find_joints("joint.*")
        self.gripper_dof_idx, _ = self._robot.find_joints("gripper")

        self.eef_vel = torch.zeros((self.num_envs, 6), device=self.device)
        self.task_velocities = torch.zeros((self.num_envs, 6), device=self.device)

        # Tensors for finite-differencing.
        self.last_update_timestamp = 0.0  # Note: This is for finite differencing body velocities.
        self.prev_fingertip_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.prev_fingertip_quat = (
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        )

        self.ep_succeeded = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self.ep_success_times = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)

        self.rolling_success_rate = 0.0
        self.ema_alpha = 0.002 # 350 eps half life for 450 ts eps

        self.eps_task_succeeded = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)

        self.residual_actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self.prev_actions = torch.zeros_like(self.residual_actions)
        self.env_actions = torch.zeros((self.num_envs, 8), device=self.device)

        self.rew_sum = None

        self.base_noise_state = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self.noise_gate = torch.zeros(self.num_envs, 1, device=self.device)
        self.noise_amp  = torch.zeros(self.num_envs, 1, device=self.device)
        
        self.starting_qpos = None
        self.curr_decimation = 0

        self.add_noise_to_base = False

    def _setup_scene(self):
        """Initialize simulation scene."""
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(), translation=(0.0, 0.0, -1.05))

        # spawn a usd file of a table into the scene
        cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd")
        cfg.func(
            "/World/envs/env_.*/Table", cfg, translation=(0.55, 0.0, -0.0015), orientation=(0.70711, 0.0, 0.0, 0.70711)
        )

        self._robot = Articulation(self.cfg.robot)
        self._fixed_asset = Articulation(self.cfg_task.fixed_asset) # type: ignore
        self._held_asset = Articulation(self.cfg_task.held_asset) # type: ignore
        if self.cfg_task.name == "gear_mesh":
            self._small_gear_asset = Articulation(self.cfg_task.small_gear_cfg) # type: ignore
            self._large_gear_asset = Articulation(self.cfg_task.large_gear_cfg) # type: ignore

        self.measure_force = self.cfg.env_options.measure_force
        self.enable_cameras = self.cfg.env_options.enable_cameras

        if self.measure_force:
            self.eef_contact_sensor = ContactSensor(self.cfg.eef_contact_sensor_cfg)
            self.scene.sensors["eef_contact_sensor"] = self.eef_contact_sensor

            # self.fixed_asset_contact_sensor = ContactSensor(self.cfg_task.fixed_asset_contact_sensor_cfg)
            # self.scene.sensors["fixed_asset_contact_sensor"] = self.fixed_asset_contact_sensor

            self.held_asset_contact_sensor = ContactSensor(self.cfg_task.held_asset_contact_sensor_cfg)
            self.scene.sensors["held_asset_contact_sensor"] = self.held_asset_contact_sensor

        if self.enable_cameras:
            self.front_camera = TiledCamera(self.cfg.front_camera_cfg)
            self.scene.sensors["front_camera"] = self.front_camera
            self.left_camera = TiledCamera(self.cfg.left_camera_cfg)
            self.scene.sensors["left_camera"] = self.left_camera
            self.right_camera = TiledCamera(self.cfg.right_camera_cfg)
            self.scene.sensors["right_camera"] = self.right_camera

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            # we need to explicitly filter collisions for CPU simulation
            self.scene.filter_collisions()

        self.scene.articulations["robot"] = self._robot
        self.scene.articulations["fixed_asset"] = self._fixed_asset
        self.scene.articulations["held_asset"] = self._held_asset
        if self.cfg_task.name == "gear_mesh":
            self.scene.articulations["small_gear"] = self._small_gear_asset
            self.scene.articulations["large_gear"] = self._large_gear_asset

        cfg = self.cfg.frame_marker_cfg
        cfg.prim_path = "/Visuals/fingertip_marker"
        self.fingertip_marker = VisualizationMarkers(cfg)
        cfg.prim_path = "/Visuals/base_fingertip_marker"
        self.base_fingertip_marker = VisualizationMarkers(cfg)
        cfg.prim_path = "/Visuals/fixed_asset_marker"
        self.fixed_asset_marker = VisualizationMarkers(cfg)
        cfg.prim_path = "/Visuals/held_asset_marker"
        self.held_asset_marker = VisualizationMarkers(cfg)

        self.red_sphere_marker = VisualizationMarkers(self.cfg.red_sphere_cfg)
        self.blue_sphere_marker = VisualizationMarkers(self.cfg.blue_sphere_cfg)
        self.green_sphere_marker = VisualizationMarkers(self.cfg.green_sphere_cfg)
        self.yellow_sphere_marker = VisualizationMarkers(self.cfg.yellow_sphere_cfg)
        self.orange_sphere_marker = VisualizationMarkers(self.cfg.orange_sphere_cfg)

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _compute_nn_base_actions(self):
        sim_fingertip_pos = self.fingertip_midpoint_pos.clone()
        sim_fingertip_pos[:, :2] -= self.xy_translation_noise
        sim_eef_quat = self.fingertip_midpoint_quat.clone()
        sim_eef_quat = torch_utils.quat_mul(
            sim_eef_quat,
            torch_utils.quat_from_euler_xyz(
                roll=torch.zeros((self.num_envs,), device=self.device),
                pitch=torch.zeros((self.num_envs,), device=self.device),
                yaw=-self.yaw_rotation_noise.squeeze(-1),
            ),
        )

        self.base_actions = self.base_policy.get_actions(self.episode_idx, sim_fingertip_pos, sim_eef_quat, self.gripper, verbose=False) # (num_envs, residual_action_dim) at eef
        
        if self.vis_options["training_data"]:
            self.obs_base, quat_base, _ = self.base_policy.get_closest_obs(self.episode_idx, sim_fingertip_pos, sim_eef_quat, self.gripper, verbose=False)
            self.obs_base[:, :2] += self.xy_translation_noise

        self.base_actions[:, :2] += self.xy_translation_noise
        self.base_actions[:, 3:7] = torch_utils.quat_mul(
            self.base_actions[:, 3:7],
            torch_utils.quat_from_euler_xyz(
                roll=torch.zeros((self.num_envs,), device=self.device),
                pitch=torch.zeros((self.num_envs,), device=self.device),
                yaw=self.yaw_rotation_noise.squeeze(-1),
            )
        )

    def _compute_bc_base_actions(self):
        bc_obs = {
            "observation.state": torch.cat([
                self.fingertip_midpoint_pos,
                self.fingertip_midpoint_quat,
                self.gripper,
                self.ee_linvel_fd,
                self.ee_angvel_fd,
            ], dim=-1),
            "observation.environment_state": torch.cat([
                self.fingertip_midpoint_pos - self.fixed_pos_obs_frame,
                self.fingertip_midpoint_pos - self.held_pos_obs_frame,
            ], dim=-1),
        }

        self.base_actions = self.base_policy.act(bc_obs)  # (num_envs, 8)

    def _compute_intermediate_values(self, dt):
        """Get values computed from raw tensors. This includes adding noise."""
        # TODO: A lot of these can probably only be set once?
        self.fixed_pos = self._fixed_asset.data.root_pos_w - self.scene.env_origins
        self.fixed_quat = self._fixed_asset.data.root_quat_w

        self.held_pos = self._held_asset.data.root_pos_w - self.scene.env_origins
        self.held_quat = self._held_asset.data.root_quat_w

        self.held_pos_obs_frame = torch_utils.tf_combine(
            self.held_quat,
            self.held_pos,
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1),
            self.held_center_pos_local,
        )[1]

        self.eef_pos = self._robot.data.body_pos_w[:, self.eef_body_idx] - self.scene.env_origins
        self.fingertip_midpoint_quat = self._robot.data.body_quat_w[:, self.eef_body_idx]
        self.fingertip_midpoint_pos = torch_utils.tf_combine(
            self.fingertip_midpoint_quat,
            self.eef_pos,
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1),
            self.sim_fingertip2eef,
        )[1]

        self.gripper = self._robot.data.joint_pos[:, self.gripper_dof_idx[0:1]] / 1.6 # (num_envs, 1)
        if self.cfg_task.name == "gear_mesh":
            self.gripper = torch.clamp(self.gripper, 0.0, 0.695)
        elif self.cfg_task.name == "peg_insert":
            self.gripper = torch.clamp(self.gripper, 0.0, 0.94)
        elif self.cfg_task.name == "nut_thread":
            self.gripper = torch.clamp(self.gripper, 0.0, 0.40)

        self.fingertip_midpoint_linvel = self._robot.data.body_lin_vel_w[:, self.eef_body_idx] # NOTE: actually eef vels
        self.fingertip_midpoint_angvel = self._robot.data.body_ang_vel_w[:, self.eef_body_idx]

        jacobians = self._robot.root_physx_view.get_jacobians() # (num_envs, num_bodies, 6, num_dofs)
        self.eef_jacobian = jacobians[:, self.eef_body_idx - 1, 0:6, 0:7] # (num_envs, 6, arm_idx), origin at body idx

        if self.measure_force:
            self.eef_force = self.eef_contact_sensor.data.net_forces_w.squeeze(1) # (num_envs, 3)
            self.F_ext = torch.cat([self.eef_force, torch.zeros((self.num_envs, 3), device=self.device)], dim=-1) # (num_envs, 6)

            # self.fixed_asset_force = self.fixed_asset_contact_sensor.data.net_forces_w.squeeze(1) # (num_envs, 3)
            self.held_asset_force = self.held_asset_contact_sensor.data.net_forces_w.squeeze(1) # (num_envs, 3)

        if self.enable_cameras:
            self.front_rgb = self.front_camera.data.output["rgb"] # (num_envs, H, W, 3) (0-255)
            self.left_rgb = self.left_camera.data.output["rgb"] # (num_envs, H, W, 3) (0-255)
            self.right_rgb = self.right_camera.data.output["rgb"] # (num_envs, H, W, 3) (0-255)

        self.joint_pos = self._robot.data.joint_pos.clone()
        self.joint_vel = self._robot.data.joint_vel.clone()

        # Finite-differencing results in more reliable velocity estimates.
        self.ee_linvel_fd = (self.fingertip_midpoint_pos - self.prev_fingertip_pos) / dt
        self.prev_fingertip_pos = self.fingertip_midpoint_pos.clone()

        # Add state differences if velocity isn't being added.
        rot_diff_quat = torch_utils.quat_mul(
            self.fingertip_midpoint_quat, torch_utils.quat_conjugate(self.prev_fingertip_quat)
        )
        rot_diff_quat *= torch.sign(rot_diff_quat[:, 0]).unsqueeze(-1)
        rot_diff_aa = axis_angle_from_quat(rot_diff_quat)
        self.ee_angvel_fd = rot_diff_aa / dt
        self.prev_fingertip_quat = self.fingertip_midpoint_quat.clone()

        self.last_update_timestamp = self._robot._data._sim_timestamp

        if self.cfg_task.name == "nut_thread":
            above = torch.where(self.held_pos[:, 2] > 0.03, torch.ones_like(self.picked_up), torch.zeros_like(self.picked_up)).bool()
            self.picked_up = torch.logical_or(self.picked_up, above)

    def _add_noise_to_base(self):
        N, d = self.num_envs, self.cfg.action_space
        dev = self.device

        # --- (1) Smooth noise process n_t (low-pass filtered uniform) ---
        alpha = self.cfg.base_rand.noise_smooth_alpha          # e.g. 0.98-0.995
        lo, hi = self.cfg.base_rand.base_action_noise_range

        eps = torch.empty(N, d, device=dev).uniform_(lo, hi)
        eps[:, -1] = 0.0  # no noise on gripper
        self.base_noise_state = alpha * self.base_noise_state + (1.0 - alpha) * eps

        # --- (2) Smooth per-step "noisy?" gate g_t in [0,1] ---
        p_on = self.cfg.base_rand.noise_on_prob                # noise probability per step
        g_tgt = (torch.rand(N, 1, device=dev) < p_on).float()

        beta_g = self.cfg.base_rand.noise_gate_smooth_beta     # e.g. 0.95-0.995
        self.noise_gate = beta_g * self.noise_gate + (1.0 - beta_g) * g_tgt

        # --- Apply ---
        noise = self.base_noise_state * self.noise_gate
        self.base_actions = self._apply_residual(noise, self.base_actions)

    def _get_factory_obs_state_dict(self):
        """Populate dictionaries for the policy and critic."""
        if self.cfg.env_options.obs_dmr:
            noisy_fixed_pos = self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise
            noisy_held_pos = self.held_pos_obs_frame + self.init_held_pos_obs_noise 
        else:
            noisy_fixed_pos = self.fixed_pos_obs_frame
            noisy_held_pos = self.held_pos_obs_frame

        prev_actions = self.residual_actions.clone()
        
        # self.red_sphere_marker.visualize(self.env_actions[:,:3] + self.scene.env_origins)
        # self.blue_sphere_marker.visualize(self.base_actions[:, :3] + self.scene.env_origins)
        # self.green_sphere_marker.visualize(self.fingertip_midpoint_pos + self.scene.env_origins)
 
        obs_dict = {
            "fingertip_pos": self.fingertip_midpoint_pos,
            "fingertip_pos_rel_fixed": self.fingertip_midpoint_pos - noisy_fixed_pos,
            "fingertip_pos_rel_held": self.fingertip_midpoint_pos - noisy_held_pos,
            "fingertip_quat": self.fingertip_midpoint_quat,
            "gripper": self.gripper,
            "ee_linvel": self.ee_linvel_fd,
            "ee_angvel": self.ee_angvel_fd,
            "prev_actions": prev_actions,
            "base_fingertip_pos": self.base_actions[:, :3],
            "base_fingertip_quat": self.base_actions[:, 3:7],
            "base_gripper": self.base_actions[:, 7:8],
        }

        state_dict = {
            "fingertip_pos": self.fingertip_midpoint_pos,
            "fingertip_pos_rel_fixed": self.fingertip_midpoint_pos - self.fixed_pos_obs_frame,
            "fingertip_pos_rel_held": self.fingertip_midpoint_pos - self.held_pos_obs_frame,
            "fingertip_quat": self.fingertip_midpoint_quat,
            "gripper": self.gripper,
            "ee_linvel": self.fingertip_midpoint_linvel,
            "ee_angvel": self.fingertip_midpoint_angvel,
            "joint_pos": self.joint_pos[:, 0:7],
            "held_pos": self.held_pos,
            "held_pos_rel_fixed": self.held_pos - self.fixed_pos_obs_frame,
            "held_quat": self.held_quat,
            "fixed_pos": self.fixed_pos,
            "fixed_quat": self.fixed_quat,
            "pos_threshold": self.pos_threshold,
            "rot_threshold": self.rot_threshold,
            "prev_actions": prev_actions,
            "base_fingertip_pos": self.base_actions[:, :3],
            "base_fingertip_quat": self.base_actions[:, 3:7],
            "base_gripper": self.base_actions[:, 7:8],
        }

        return obs_dict, state_dict

    def _get_observations(self):
        """Get actor/critic inputs using asymmetric critic."""
        if not self.teleop_mode and (self.cfg.env_options.base_model == "nn" or self.cfg.env_options.base_model == "noisy_nn"):
            self._compute_nn_base_actions()
        elif not self.teleop_mode and self.cfg.env_options.base_model == "bc":
            self._compute_bc_base_actions()
        if self.add_noise_to_base:
            self._add_noise_to_base()
        obs_dict, state_dict = self._get_factory_obs_state_dict()

        obs_tensors = factory_utils.collapse_obs_dict(obs_dict, self.cfg.residual_obs_order + ["prev_actions"])
        state_tensors = factory_utils.collapse_obs_dict(state_dict, self.cfg.residual_state_order + ["prev_actions"])

        if obs_tensors.isnan().any() or state_tensors.isnan().any():
            import pdb; pdb.set_trace()

        return {"policy": obs_tensors, "critic": state_tensors}

    def _reset_buffers(self, env_ids):
        """Reset buffers."""
        self.ep_succeeded[env_ids] = 0
        self.ep_success_times[env_ids] = 0
        self.eps_task_succeeded[env_ids] = 0
        self.first_success[env_ids] = 0

    def _apply_residual(self, residual_actions, base_actions):
        # Interpret actions as target pos displacements and set pos target
        pos_actions = residual_actions[:, 0:3] * self.pos_threshold 
        ctrl_target_fingertip_midpoint_pos = base_actions[:, 0:3] + pos_actions

        # Interpret actions as target rot (axis-angle) displacements
        rot_actions = residual_actions[:, 3:6]
        # if self.cfg_task.unidirectional_rot:
        #     rot_actions[:, 2] = -(rot_actions[:, 2] + 1.0) * 0.5  # [-1, 0]
        rot_actions = rot_actions * self.rot_threshold

        # Convert to quat and set rot target
        angle = torch.norm(rot_actions, p=2, dim=-1)
        axis = rot_actions / angle.unsqueeze(-1)

        rot_actions_quat = torch_utils.quat_from_angle_axis(angle, axis)
        rot_actions_quat = torch.where(
            angle.unsqueeze(-1).repeat(1, 4) > 1e-6,
            rot_actions_quat,
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1),
        )

        ctrl_target_fingertip_midpoint_quat = torch_utils.quat_mul(rot_actions_quat, base_actions[:, 3:7])

        grip_actions = residual_actions[:, 6:7] * self.gripper_threshold
        ctrl_target_gripper_dof_pos = torch.clamp(base_actions[:, 7:8] + grip_actions, 0.0, 1.0)

        return torch.cat([ctrl_target_fingertip_midpoint_pos, ctrl_target_fingertip_midpoint_quat, ctrl_target_gripper_dof_pos], dim=-1)

    def compute_fk_sapien_links(self, qpos, link_idx):
        fk = self.robot_model.compute_forward_kinematics(qpos)
        link_pose_ls = []
        for i in link_idx:
            link_pose_ls.append(self.robot_model.get_link_pose(i).to_transformation_matrix())
        return link_pose_ls

    def compute_ik_sapien(self, initial_qpos, tf, verbose=False):
        """
        Compute IK using sapien
        initial_qpos: (N, ) numpy array
        cartesian: (6, ) numpy array, x,y,z in meters, r,p,y in radians
        """
        # tf_mat = np.eye(4)
        # tf_mat[:3, :3] = transforms3d.euler.euler2mat(ai=cartesian[3], aj=cartesian[4], ak=cartesian[5], axes='sxyz')
        # tf_mat[:3, 3] = cartesian[0:3]
        pose = sapien.Pose.from_transformation_matrix(tf)

        if 'xarm7' in self.robot_name:
            active_qmask = np.array([True, True, True, True, True, True, True])
        qpos = self.robot_model.compute_inverse_kinematics(
            link_index=self.sapien_eef_idx, 
            pose=pose,
            initial_qpos=initial_qpos, 
            active_qmask=active_qmask, 
            )
        if verbose:
            print('ik qpos:', qpos)

        # verify ik
        fk_pose = self.compute_fk_sapien_links(qpos[0], [self.sapien_eef_idx])[0]
        
        if verbose:
            print('target pose for IK:', tf)
            print('fk pose for IK:', fk_pose)
        
        pose_diff = np.linalg.norm(fk_pose[:3, 3] - tf[:3, 3])
        rot_diff = np.linalg.norm(fk_pose[:3, :3] - tf[:3, :3])
        
        if pose_diff > 0.01 or rot_diff > 0.01:
            print('ik pose diff:', pose_diff)
            print('ik rot diff:', rot_diff)
            import pdb; pdb.set_trace()
            return initial_qpos
        return qpos[0]

    def _pre_physics_step(self, action):
        """Apply policy actions with smoothing."""
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self._reset_buffers(env_ids)

        self.residual_actions = self.ema_factor * action.clone().to(self.device) + (1 - self.ema_factor) * self.residual_actions

        self.env_actions = self._apply_residual(self.residual_actions, self.base_actions)

        ctrl_target_fingertip_midpoint_pos = self.env_actions[:, 0:3].clone()
        ctrl_target_fingertip_midpoint_quat = self.env_actions[:, 3:7].clone()

        ctrl_target_eef_pos = torch_utils.tf_combine(
            ctrl_target_fingertip_midpoint_quat,
            ctrl_target_fingertip_midpoint_pos,
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1),
            -self.sim_fingertip2eef,
        )[1]

        cartesian_target, self.task_velocities = factory_control.adm_ctrl_task_space(
            pos=self.eef_pos, quat=self.fingertip_midpoint_quat,
            pos_g=ctrl_target_eef_pos, quat_g=ctrl_target_fingertip_midpoint_quat,
            v=self.task_velocities, F_ext=self.F_ext, dt=self.physics_dt,
            kx=self.Kx,kr=self.Kr,mx=self.mx,mr=self.mr,dx=1.*torch.sqrt(self.Kx * self.mx),dr=1.*torch.sqrt(self.Kr * self.mr),
        )

        if self.cfg.ctrl.ik == "pk":
            self.qpos_targets = self.compute_ik_abs(cartesian_target, self.joint_pos[:, 0:7])
        elif self.cfg.ctrl.ik == "sapien":
            self.qpos_targets = torch.zeros((self.num_envs, 7), device=self.device)
            for i in range(self.num_envs):
                curr_qpos = self.joint_pos[i, 0:7].cpu().numpy()
                tf = np.eye(4)
                tf[:3, :3] = torch_utils.quats_to_rot_matrices(cartesian_target[i, 3:7]).cpu().numpy()  
                tf[:3, 3] = cartesian_target[i, :3].cpu().numpy()
                ik_qpos = self.compute_ik_sapien(
                    initial_qpos=curr_qpos,
                    tf=tf,
                    verbose=False,
                )
                self.qpos_targets[i, 0:7] = torch.tensor(ik_qpos, device=self.device)

    def _apply_action(self):
        """Apply actions for policy as delta targets from current position."""
        # Note: We use finite-differenced velocities for control and observations.
        # Check if we need to re-compute velocities within the decimation loop.
        if self.last_update_timestamp < self._robot._data._sim_timestamp:
            self._compute_intermediate_values(dt=self.physics_dt)

        ctrl_target_gripper_dof_pos = self.env_actions[:, 7:8].clone() * 1.6

        if self.cfg_task.name == "gear_mesh":
            ctrl_target_gripper_dof_pos = torch.clamp(ctrl_target_gripper_dof_pos, max=1.2)
        elif self.cfg_task.name == "nut_thread":
            ctrl_target_gripper_dof_pos = torch.clamp(ctrl_target_gripper_dof_pos, max=0.75)

        if self.starting_qpos is None:
            self.starting_qpos = self.joint_pos[:, :7].clone()
        ratio = (self.curr_decimation+1) / self.cfg.decimation # 1/8 to 8/8
        qpos_target = ratio * self.qpos_targets + (1.0 - ratio) * self.starting_qpos
        self.curr_decimation += 1

        if self.curr_decimation == self.cfg.decimation:
            self.starting_qpos = None
            self.curr_decimation = 0

        # TODO: add interpolation?
        self._robot.set_joint_position_target(qpos_target,            joint_ids=self.arm_dof_idx)
        self._robot.set_joint_position_target(ctrl_target_gripper_dof_pos,  joint_ids=self.gripper_dof_idx)

        # self.generate_ctrl_signals(
        #     ctrl_target_eef_pos=ctrl_target_eef_pos,
        #     ctrl_target_fingertip_midpoint_quat=ctrl_target_fingertip_midpoint_quat,
        #     ctrl_target_gripper_dof_pos=ctrl_target_gripper_dof_pos,
        # )

    def generate_ctrl_signals(
        self, 
        ctrl_target_eef_pos, 
        ctrl_target_fingertip_midpoint_quat,
        ctrl_target_gripper_dof_pos, # (num_envs, 1)
        ):

        self.arm_joint_pose_target, self.joint_vel_target, x_acc, _, self.eef_vel = factory_control.compute_dof_state_admittance(
            dof_pos=self.joint_pos,
            eef_pos=self.eef_pos,
            eef_quat=self.fingertip_midpoint_quat,
            jacobian=self.eef_jacobian,
            ctrl_target_eef_pos=ctrl_target_eef_pos,
            ctrl_target_eef_quat=ctrl_target_fingertip_midpoint_quat,
            xdot_ref=self.eef_vel,
            dt=self.physics_dt,
            F_ext=self.F_ext if self.measure_force else None, # NOTE: external wrench at eef frame
            device=self.device,
            Kx=self.Kx, Kr=self.Kr, mx=self.mx, mr=self.mr, Dx=None, Dr=None, lam=self.lam, rot_scale=self.cfg.ctrl.rot_scale,
        )

        self._robot.set_joint_position_target(self.arm_joint_pose_target, joint_ids=self.arm_dof_idx)
        self._robot.set_joint_position_target(ctrl_target_gripper_dof_pos, joint_ids=self.gripper_dof_idx)
        # self._robot.set_joint_velocity_target(self.joint_vel_target)

    def _get_dones(self):
        """Check which environments are terminated.

        For Factory reset logic, it is important that all environments
        stay in sync (i.e., _get_dones should return all true or all false).
        """
        self._compute_intermediate_values(dt=self.physics_dt)
        self._visualize_markers()

        time_out = self.episode_length_buf >= self.max_episode_length - 1 

        task_success = self._check_success()
        self.first_success = task_success & (~self.eps_task_succeeded.to(torch.bool))
        self.eps_task_succeeded[task_success] = 1

        if self.cfg.env_options.step_eps:
            dist_threshold = 0.2
            terminated = torch.norm(self.fingertip_midpoint_pos - self.held_pos_obs_frame, dim=1) > dist_threshold # to eliminate the case where held asset falls far away
            terminated |= self.eps_task_succeeded.bool()

            # if self.cfg_task.name == "gear_mesh" or self.cfg_task.name == "peg_insert":
            #     terminated |= self.bad_insert.bool()
            #     self.bad_insert[:] = 0

            if self.cfg_task.name == "peg_insert":
                unit_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
                tilt_degrees = factory_utils.quat_geodesic_angle(self.held_quat, unit_quat) * 180.0 / math.pi
                terminated |= torch.where(tilt_degrees > 60.0, torch.ones_like(terminated), torch.zeros_like(terminated)).bool()

            if self.cfg_task.name == "nut_thread":
                on_ground = self.held_pos[:, 2] < 0.02
                dropped = torch.logical_and(on_ground, self.picked_up)
                terminated |= dropped

            assert not (self.first_success & (~terminated)).any()

        else:
            terminated = time_out.clone()

        done = torch.logical_or(time_out, terminated)
        s = self.eps_task_succeeded[done].float()  # shape [n_finished]
        n = s.numel()
        if n > 0:
            alpha = self.ema_alpha
            one_minus = 1.0 - alpha

            # weights: (1-α)^{n-1}, ..., (1-α)^0
            exponents = torch.arange(n - 1, -1, -1, device=s.device, dtype=torch.float32)
            weights = one_minus ** exponents  # shape [n]

            weighted_sum = (weights * s).sum()

            # closed-form EMA update over n new episodes
            self.rolling_success_rate = (
                (one_minus ** n) * self.rolling_success_rate + alpha * weighted_sum
            )

        self.extras["rolling_avg_succ_rate"] = float(self.rolling_success_rate)   

        self.base_last5.push(self.base_actions[:, :3])
        self.residual_last5.push(self.env_actions[:, :3])

        return terminated, time_out

    def _get_curr_successes(self, success_threshold):
        """Get success mask at current timestep."""
        curr_successes = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)

        held_base_pos, held_base_quat = factory_utils.get_held_base_pose(
            self.held_pos, self.held_quat, self.cfg_task.name, self.cfg_task.fixed_asset_cfg, self.num_envs, self.device
        )
        target_held_base_pos, target_held_base_quat = factory_utils.get_target_held_base_pose(
            self.fixed_pos,
            self.fixed_quat,
            self.cfg_task.name,
            self.cfg_task.fixed_asset_cfg,
            self.num_envs,
            self.device,
        )

        xy_dist = torch.linalg.vector_norm(target_held_base_pos[:, 0:2] - held_base_pos[:, 0:2], dim=1)
        z_disp = held_base_pos[:, 2] - target_held_base_pos[:, 2]
        is_centered = torch.where(xy_dist < 0.0025, torch.ones_like(curr_successes), torch.zeros_like(curr_successes))
        # Height threshold to target
        fixed_cfg = self.cfg_task.fixed_asset_cfg
        if self.cfg_task.name == "peg_insert" or self.cfg_task.name == "gear_mesh":
            height_threshold = fixed_cfg.height * success_threshold
        elif self.cfg_task.name == "nut_thread":
            height_threshold = success_threshold # type: ignore
        else:
            raise NotImplementedError("Task not implemented")
        is_close_or_below = torch.where(
            z_disp < height_threshold, torch.ones_like(curr_successes), torch.zeros_like(curr_successes)
        )
        curr_successes = torch.logical_and(is_centered, is_close_or_below)

        return curr_successes

    def _log_factory_metrics(self, rew_dict, curr_successes):
        """Keep track of episode statistics and log rewards."""
        # Only log episode success rates at the end of an episode.
        # if torch.any(self.reset_buf): # NOTE: only if eps reset at same time
        #     self.extras["eoe_success_rate"] = torch.count_nonzero(curr_successes) / self.num_envs
        #     print(f"End-of-Eps Success Rate: {self.extras['eoe_success_rate'].item()*100:.1f}%")

        # Get the time at which an episode first succeeds.
        first_success = torch.logical_and(curr_successes, torch.logical_not(self.ep_succeeded))
        self.ep_succeeded[curr_successes] = 1

        first_success_ids = first_success.nonzero(as_tuple=False).squeeze(-1)
        self.ep_success_times[first_success_ids] = self.episode_length_buf[first_success_ids]
        nonzero_success_ids = self.ep_success_times.nonzero(as_tuple=False).squeeze(-1)

        if len(nonzero_success_ids) > 0:  # Only log for successful episodes.
            success_times = self.ep_success_times[nonzero_success_ids].sum() / len(nonzero_success_ids)
            self.extras["success_times"] = success_times

        for rew_name, rew in rew_dict.items():
            self.extras[f"logs_rew_{rew_name}"] = rew.mean()

    def _check_success(self):
        # -------------------------
        # Base success (task-specific threshold)
        # -------------------------
        task_success = self._get_curr_successes(
            success_threshold=self.cfg_task.success_threshold
        ).to(torch.bool)

        # -------------------------
        # Held pose + target pose (task frame)
        # -------------------------
        held_pos, held_quat = factory_utils.get_held_base_pose(
            self.held_pos, self.held_quat,
            self.cfg_task.name, self.cfg_task.fixed_asset_cfg,
            self.num_envs, self.device
        )
        target_pos, target_quat = factory_utils.get_target_held_base_pose(
            self.fixed_pos, self.fixed_quat,
            self.cfg_task.name, self.cfg_task.fixed_asset_cfg,
            self.num_envs, self.device
        )

        # -------------------------
        # XY alignment shaping + (optional) precondition for nut_thread
        # -------------------------
        xy_dist_held_fixed = torch.linalg.vector_norm(target_pos[:, :2] - held_pos[:, :2], dim=1)          # (N,)
        xy_dist_eef_held = torch.linalg.vector_norm(self.fingertip_midpoint_pos[:, :2] - held_pos[:, :2], dim=1)  # (N,)
        z_disp  = (held_pos[:, 2] - target_pos[:, 2]) 

        # -------------------------
        # nut_thread: track accumulated rotation ONLY when precondition holds AND gripper closed
        # -------------------------
        if self.cfg_task.name == "nut_thread":
            # check nut thread aligned precondition
            xy_center_thresh = 0.015
            xy_centered = (xy_dist_held_fixed < xy_center_thresh)

            z_close_thresh = 0.01
            z_close_or_below = (z_disp < z_close_thresh)

            thread_precondition = (xy_centered & z_close_or_below)

            # gripper condition for counting rotation
            gripper_closed = (self.gripper.squeeze(-1) >= self.cfg_task.close_gripper)

            # yaw unwrap increment (degrees)
            _, _, yaw = torch_utils.get_euler_xyz(self.held_quat)
            yaw = (yaw + math.pi) % (2 * math.pi) - math.pi

            dyaw = yaw - self.prev_held_yaw
            dyaw = (dyaw + math.pi) % (2 * math.pi) - math.pi
            dyaw_deg = dyaw * (180.0 / math.pi)

            # accumulate only when in valid regime
            acc_mask = thread_precondition & gripper_closed
            self.cumulative_rotation[acc_mask] += torch.abs(dyaw_deg[acc_mask])

            # always update prev yaw to avoid jumps when mask toggles
            self.prev_held_yaw = yaw.clone()

            # override task success: need >= target rotation
            rot_target_deg = 180.0  # change to 360.0 if you actually want a full turn
            task_success = (self.cumulative_rotation >= rot_target_deg)
        
        return task_success

    def _get_rewards(self):
        """Compute dense shaping + terminal reward + diagnostics."""
        # -------------------------
        # Held pose + target pose (task frame)
        # -------------------------
        held_pos, held_quat = factory_utils.get_held_base_pose(
            self.held_pos, self.held_quat,
            self.cfg_task.name, self.cfg_task.fixed_asset_cfg,
            self.num_envs, self.device
        )
        target_pos, target_quat = factory_utils.get_target_held_base_pose(
            self.fixed_pos, self.fixed_quat,
            self.cfg_task.name, self.cfg_task.fixed_asset_cfg,
            self.num_envs, self.device
        )

        # -------------------------
        # XY alignment shaping + (optional) precondition for nut_thread
        # -------------------------
        xy_dist_held_fixed = torch.linalg.vector_norm(target_pos[:, :2] - held_pos[:, :2], dim=1)          # (N,)
        xy_dist_eef_held = torch.linalg.vector_norm(self.fingertip_midpoint_pos[:, :2] - held_pos[:, :2], dim=1)  # (N,)
        z_disp  = (held_pos[:, 2] - target_pos[:, 2])                            # (N,)

        xy_align_thresh  = 0.005
        xy_aligned  = (xy_dist_held_fixed < xy_align_thresh).float() + (xy_dist_eef_held < xy_align_thresh).float()
        
        # -------------------------
        # Tilt penalty (high priority)
        # -------------------------
        a_local = torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(self.num_envs, 3)
        a_world = torch_utils.quat_rotate(self.env_actions[:, 3:7], a_local)     # (N,3)
        z_down  = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand_as(a_world)
        cos = (a_world * z_down).sum(dim=-1).clamp(-1.0, 1.0)
        tilt_rad = torch.acos(cos)                                              # (N,)
        tilt_penalty = tilt_rad

        # -------------------------
        # Contact-force penalty (soft)
        # -------------------------
        F = torch.norm(self.held_asset_force, dim=1)
        F0, F1 = 10.0, 30.0
        force_penalty = torch.clamp((F - F0).clamp_min(0.0) / (F1 - F0), max=1.0)

        # -------------------------
        # Action-norm penalty (soft)
        # -------------------------
        action_norm = torch.norm(self.residual_actions, dim=1) / math.sqrt(self.cfg.action_space)

        # -------------------------
        # Action-smoothing penalty (soft)
        # -------------------------
        action_smoothing = torch.norm(self.prev_actions - self.residual_actions, dim=1)

        rew_dict = {
            "action_norm": -action_norm * self.cfg.env_options.action_norm_reward_scale,
            "tilt_penalty": -tilt_penalty * self.cfg.env_options.tilt_penalty_reward_scale,
            "force_penalty": -force_penalty * self.cfg.env_options.force_penalty_reward_scale,
            "action_smoothing": -action_smoothing * self.cfg.env_options.action_smoothing_reward_scale,
            "xy_align": xy_aligned.float() * self.cfg.env_options.xy_aligned_reward_scale,
            "terminated": -(self.reset_terminated & (~self.eps_task_succeeded)).float() * self.cfg.env_options.termination_reward_scale,
            "task_success": self.first_success.float() * self.cfg.env_options.task_success_reward_scale,
        }
        rew_buf = torch.zeros_like(rew_dict["task_success"])

        GREEN = "\033[92m"
        RED   = "\033[91m"
        RESET = "\033[0m"

        if self.rew_sum is None:
            self.rew_sum = {k: 0.0 for k in rew_dict.keys()}

        for rew_name, rew in rew_dict.items():
            self.rew_sum[rew_name] += rew.mean().item()

        log_rew_int = 100
        if self.common_step_counter % log_rew_int == 0:
            mean_rew = 0.0
            print("\n" + "=" * 50)
            print(f" Iter {self.common_step_counter // log_rew_int}")
            print("=" * 50)
            for rew_name, rew in self.rew_sum.items():
                val = rew / log_rew_int
                color = GREEN if val >= 0 else RED
                print(f"{rew_name}: {color}{val:.4f}{RESET}")
                self.rew_sum[rew_name] = 0.0
                mean_rew += val
            print("mean rew: ", mean_rew)
            print()   # trailing blank line

        for rew_name, rew in rew_dict.items():
            rew_buf += rew_dict[rew_name]

        self.prev_actions = self.residual_actions.clone()
        self._log_factory_metrics(rew_dict, self.first_success)

        if self.vis_options["rewards"] == True:
            try:
                if not hasattr(self, 'draw'):
                    from isaacsim.util.debug_draw import _debug_draw
                    self.draw = _debug_draw.acquire_debug_draw_interface()
                self.draw.clear_lines()

                goal_fixed_pos = (target_held_base_pos + self.scene.env_origins).cpu().numpy().tolist()
                goal_held_pos = (held_base_pos + self.scene.env_origins).cpu().numpy().tolist()

                grasp_pos = (self.held_pos_obs_frame + self.scene.env_origins).cpu().numpy().tolist()
                fingertip_pos = (self.fingertip_midpoint_pos + self.scene.env_origins).cpu().numpy().tolist()

                sizes = [10] * self.num_envs 
                pink_color = [(1.0, 0.75, 0.8, 1.0)] * self.num_envs

                self.draw.draw_lines(goal_fixed_pos, goal_held_pos, pink_color, sizes)
                self.draw.draw_lines(grasp_pos, fingertip_pos, pink_color, sizes)

                curr_pos_list = (self.fingertip_midpoint_pos + self.scene.env_origins).cpu().numpy().tolist()
                base_pos_list = (self.base_actions[:, :3] + self.scene.env_origins).cpu().numpy().tolist()
                env_pos_list = (self.env_actions[:, :3] + self.scene.env_origins).cpu().numpy().tolist()

                sizes = [5] * self.num_envs 
                red_color = [(1, 0, 0, 1)] * self.num_envs
                blue_color = [(0, 0, 1, 1)] * self.num_envs
                green_color = [(0, 1, 0, 1)] * self.num_envs

                self.draw.draw_lines(curr_pos_list, base_pos_list, blue_color, sizes)
                self.draw.draw_lines(base_pos_list, env_pos_list, red_color, sizes)
                self.draw.draw_lines(curr_pos_list, env_pos_list, green_color, sizes)

            except Exception as e:
                print("Vis reward error: ", e)
                pass

        self.log("==============================")

        return rew_buf

    def _reset_idx(self, env_ids):
        """We assume all envs will always be reset at the same time."""
        super()._reset_idx(env_ids)

        if self.enable_cameras:
            self.front_camera.reset(env_ids=env_ids)
            self.left_camera.reset(env_ids=env_ids)
            self.right_camera.reset(env_ids=env_ids)

        if self.cfg.env_options.ctrl_dmr:
            self.Kx[env_ids] = self.cfg.ctrl.Kx_dmr_range[0] + (self.cfg.ctrl.Kx_dmr_range[1] - self.cfg.ctrl.Kx_dmr_range[0]) * torch.rand(len(env_ids), device=self.device)
            self.Kr[env_ids] = self.cfg.ctrl.Kr_dmr_range[0] + (self.cfg.ctrl.Kr_dmr_range[1] - self.cfg.ctrl.Kr_dmr_range[0]) * torch.rand(len(env_ids), device=self.device)
            self.mx[env_ids] = self.cfg.ctrl.mx_dmr_range[0] + (self.cfg.ctrl.mx_dmr_range[1] - self.cfg.ctrl.mx_dmr_range[0]) * torch.rand(len(env_ids), device=self.device)
            self.mr[env_ids] = self.cfg.ctrl.mr_dmr_range[0] + (self.cfg.ctrl.mr_dmr_range[1] - self.cfg.ctrl.mr_dmr_range[0]) * torch.rand(len(env_ids), device=self.device)
        self.task_velocities[env_ids] = 0.0

        # move to next episode
        if self.cfg.env_options.step_eps:
            self.episode_idx[env_ids] = (self.episode_idx[env_ids] + 1) % self.total_episodes 

        # object position noises
        fixed_asset_pos_noise = torch.randn((len(env_ids), 3), dtype=torch.float32, device=self.device)
        fixed_asset_pos_rand = torch.tensor(self.cfg.obs_rand.fixed_asset_pos, dtype=torch.float32, device=self.device)
        fixed_asset_pos_noise = fixed_asset_pos_noise @ torch.diag(fixed_asset_pos_rand)
        self.init_fixed_pos_obs_noise[env_ids] = fixed_asset_pos_noise

        held_asset_pos_noise = torch.randn((len(env_ids), 3), dtype=torch.float32, device=self.device)
        held_asset_pos_rand = torch.tensor(self.cfg.obs_rand.held_asset_pos, dtype=torch.float32, device=self.device)
        held_asset_pos_noise = held_asset_pos_noise @ torch.diag(held_asset_pos_rand)
        self.init_held_pos_obs_noise[env_ids] = held_asset_pos_noise

        if self.cfg.env_options.data_aug:
            translation_noise = torch.randn((len(env_ids), 2), device=self.device) * self.cfg.obs_rand.pos_aug
            yaw_rotation_noise = torch.randn((len(env_ids), ), device=self.device) * math.radians(self.cfg.obs_rand.rot_aug)
            fixed_height_noise = torch.randn((len(env_ids), ), device=self.device) * self.cfg.obs_rand.height_aug
        else:
            translation_noise = torch.randn((len(env_ids), 2), device=self.device) * 0.0
            yaw_rotation_noise = torch.randn((len(env_ids), ), device=self.device) * 0.0
            fixed_height_noise = torch.randn((len(env_ids), ), device=self.device) * 0.0

        self.xy_translation_noise[env_ids] = translation_noise
        self.yaw_rotation_noise[env_ids] = yaw_rotation_noise.unsqueeze(-1) # in local frame

        held_pos = self.initial_poses[env_ids, self.episode_idx[env_ids], -3:] # (num_resets, 3)
        held_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device).repeat(len(env_ids), 1)

        held_pos = torch_utils.tf_combine(
            held_quat,
            held_pos,
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(len(env_ids), 1),
            -self.held_center_pos_local[env_ids],
        )[1]
        
        held_pos[:, :2] += translation_noise
        held_quat = torch_utils.quat_mul(
            held_quat,
            torch_utils.quat_from_euler_xyz(
                roll=torch.zeros((len(env_ids),), device=self.device),
                pitch=torch.zeros((len(env_ids),), device=self.device),
                yaw=yaw_rotation_noise,
            )
        )

        # Compute fixed_pos_obs_frame
        fixed_tip_pos_local = torch.zeros((len(env_ids), 3), device=self.device)
        fixed_tip_pos_local[:, 2] += self.cfg_task.fixed_asset_cfg.height
        fixed_tip_pos_local[:, 2] += self.cfg_task.fixed_asset_cfg.base_height
        if self.cfg_task.name == "gear_mesh":
            fixed_tip_pos_local[:, 0] = self.cfg_task.fixed_asset_cfg.medium_gear_base_offset[0] # type: ignore

        fixed_pos = self.initial_poses[env_ids, self.episode_idx[env_ids], -6:-3]
        fixed_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device).repeat(len(env_ids), 1)

        fixed_pos = torch_utils.tf_combine(
            fixed_quat,
            fixed_pos,
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(len(env_ids), 1),
            -fixed_tip_pos_local,
        )[1]

        fixed_pos[:, :2] += translation_noise
        fixed_pos[:, 2] += fixed_height_noise
        fixed_quat = torch_utils.quat_mul(
            fixed_quat,
            torch_utils.quat_from_euler_xyz(
                roll=torch.zeros((len(env_ids),), device=self.device),
                pitch=torch.zeros((len(env_ids),), device=self.device),
                yaw=yaw_rotation_noise,
            ),
        )

        # reset assets
        self._set_assets_state( 
            held_pos=held_pos,
            held_quat=held_quat,
            fixed_pos=fixed_pos,
            fixed_quat=fixed_quat,
            env_ids=env_ids,
        )
        
        # reset robot
        init_qpos = self._robot.data.default_joint_pos[env_ids, :7]

        # if self.cfg.env_options.data_aug:
        #     init_qpos_noise = None
        #     init_qpos = None
        #     import pdb; pdb.set_trace()
        #     # TODO: addnoise
        init_fingertip = self.initial_poses[env_ids, self.episode_idx[env_ids], :7] # (num_resets, 7)
        sim_eef = init_fingertip.clone() # (num_resets, 7)
        sim_eef[:, :2] += translation_noise
        sim_eef[:, 3:7] = torch_utils.quat_mul(
            sim_eef[:, 3:7],
            torch_utils.quat_from_euler_xyz(
                roll=torch.zeros((len(env_ids),), device=self.device),
                pitch=torch.zeros((len(env_ids),), device=self.device),
                yaw=yaw_rotation_noise,
            ),
        )
        sim_eef[:, 0:3] = torch_utils.tf_combine(
            sim_eef[:, 3:7],
            sim_eef[:, 0:3],
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(len(env_ids), 1),
            -self.sim_fingertip2eef[env_ids],
        )[1]

        noised_qpos = self.compute_ik_abs(sim_eef[:, :7], init_qpos)
        self._set_replay_default_pose(joints=noised_qpos, env_ids=env_ids) # compute intermediate values there

        # clear base policy action chunk
        if self.cfg.env_options.base_model == "bc":
            self.base_policy.reset()
        elif self.cfg.env_options.base_model == "nn" or self.cfg.env_options.base_model == "noisy_nn":
            self.base_policy.clear(env_ids)
        
        if self.add_noise_to_base:
            self.base_noise_state[env_ids] = 0.0
            self.noise_gate[env_ids] = 0.0

        _, fixed_tip_pos = torch_utils.tf_combine(
            self.fixed_quat[env_ids],
            self.fixed_pos[env_ids],
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(len(env_ids), 1),
            fixed_tip_pos_local,
        )
        self.fixed_pos_obs_frame[env_ids] = fixed_tip_pos

        # reset buffers
        self.prev_fingertip_pos[env_ids] = self.fingertip_midpoint_pos[env_ids].clone()
        self.prev_fingertip_quat[env_ids] = self.fingertip_midpoint_quat[env_ids].clone()

        # Set initial actions to involve no-movement. Needed for EMA/correct penalties.
        self.residual_actions[env_ids] = torch.zeros_like(self.residual_actions[env_ids])
        self.prev_actions[env_ids] = torch.zeros_like(self.residual_actions[env_ids])
        self.env_actions[env_ids] = torch.zeros_like(self.env_actions[env_ids])

        # Zero initial velocity.
        self.ee_angvel_fd[env_ids, :] = 0.0
        self.ee_linvel_fd[env_ids, :] = 0.0

        self.base_last5.clear_envs(env_ids)
        self.residual_last5.clear_envs(env_ids)

        # Reset held asset yaw rotation tracking (only for nut_thread task)
        if self.cfg_task.name == "nut_thread":
            _, _, held_yaw0 = torch_utils.get_euler_xyz(self.held_quat[env_ids])
            held_yaw0 = (held_yaw0 + math.pi) % (2 * math.pi) - math.pi  # wrap to [-pi, pi]
            self.prev_held_yaw[env_ids] = held_yaw0
            self.cumulative_rotation[env_ids] = 0.0
            self.picked_up[env_ids] = 0

        if self.vis_options["training_data"] == True:
            self.obs_traj = []
            self.act_traj = []
            for env_id in env_ids:
                obs_pos, obs_quat, act_pos, act_quat = self.base_policy.get_episode_traj(self.episode_idx[env_id].item()) # (eps_len, 3)
                obs_pos[:, :2] += self.xy_translation_noise[env_id] # broadcast
                obs_pos += self.scene.env_origins[env_id]
                act_pos[:, :2] += self.xy_translation_noise[env_id] 
                act_pos += self.scene.env_origins[env_id]
                
                self.obs_traj.extend(obs_pos.cpu().numpy().tolist())
                self.act_traj.extend(act_pos.cpu().numpy().tolist())

            yellow_color = [(1, 1, 0, 1)] * len(self.obs_traj)
            purple_color = [(1, 0, 1, 1)] * len(self.act_traj)

            try: 
                if not hasattr(self, 'data_vis'):
                    from isaacsim.util.debug_draw import _debug_draw
                    self.data_vis = _debug_draw.acquire_debug_draw_interface()
                self.data_vis.clear_points()

                self.data_vis.draw_points(self.act_traj, purple_color, [5]*len(self.act_traj))
                self.data_vis.draw_points(self.obs_traj, yellow_color, [5]*len(self.obs_traj))

            except Exception as e:
                print("Visualize data error: ", e)
                pass

        # if self.teleop_mode:
        #     base_fingertip_pos = self.load_all_episode_act_pos(factory_utils.resolve_hf_file(self.cfg_task.hf_repo, self.cfg_task.action_data_hf_file))
        #     base_fingertip_pos = base_fingertip_pos.reshape(-1, 3)

        #     from isaacsim.util.debug_draw import _debug_draw
        #     draw = _debug_draw.acquire_debug_draw_interface()
        #     draw.clear_lines()
        #     base_fingertip_pos = (base_fingertip_pos + self.scene.env_origins).cpu().numpy().tolist()
        #     purple_color = [(1, 0, 1, 1)]* len(base_fingertip_pos)

        #     draw.draw_points(base_fingertip_pos, purple_color, [5]*len(base_fingertip_pos))

    def load_all_episode_act_pos(self, path, pad=True):
        """
        Load ALL episodes' action positions from a .npz file.

        Returns:
            list_of_trajs : list of (T, 3) tensors
        """
        # TODO: deprecated
        flat = np.load(path, allow_pickle=True)
        flat = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in flat.items()}

        eps = sorted({k.split("/", 1)[0] for k in flat})
        pos_traj = []
        quat_traj = []

        # compute max length if padding is needed
        lengths = [len(flat[f"{e}/action.eef_pos"]) for e in eps]
        maxT = max(lengths)

        for e in eps:
            pos = flat[f"{e}/action.eef_pos"].to(self.device)  # (T, 3)
            quat = flat[f"{e}/action.eef_quat"].to(self.device)  # (T, 4)

            pos = torch_utils.tf_combine( # NOTE: real eef != sim eef
                quat,
                pos,
                torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device).repeat(quat.shape[0], 1),
                torch.tensor([[0, 0, 0.225]], device=self.device).repeat(quat.shape[0], 1),
            )[1]

            if pad and pos.shape[0] < maxT:
                pad_len = maxT - pos.shape[0]
                pos = torch.cat([pos, pos[-1:].repeat(pad_len, 1)], dim=0)
                quat = torch.cat([quat, quat[-1:].repeat(pad_len, 1)], dim=0)
            pos_traj.append(pos)
            quat_traj.append(quat)

        return torch.stack(pos_traj)

    def _set_replay_default_pose(self, joints, env_ids):
        """Set xarm to various given joint position."""
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        gripper_pos = 0.0
        joint_pos[:, 8:] = gripper_pos / 2.0
        joint_pos[:, 7] = gripper_pos
        joint_pos[:, :7] = joints
        joint_vel = torch.zeros_like(joint_pos)
        joint_effort = torch.zeros_like(joint_pos)
        self.ctrl_target_joint_pos[env_ids, :] = joint_pos
        self._robot.set_joint_position_target(self.ctrl_target_joint_pos[env_ids], env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self._robot.reset(env_ids=env_ids)
        self._robot.set_joint_effort_target(joint_effort, env_ids=env_ids)

        self.step_sim_no_action()

    def _set_assets_state(self, held_pos, held_quat, fixed_pos, fixed_quat, env_ids):
        """Set the assets position and orientation."""

        # Disable gravity.
        physics_sim_view = sim_utils.SimulationContext.instance().physics_sim_view
        physics_sim_view.set_gravity(carb.Float3(0.0, 0.0, 0.0))

        # Set fixed base state.
        fixed_state = torch.zeros((len(env_ids), 13), device=self.device)
        fixed_state[:, 0:3] = fixed_pos + self.scene.env_origins[env_ids]
        fixed_state[:, 3:7] = fixed_quat

        self._fixed_asset.write_root_pose_to_sim(fixed_state[:, 0:7], env_ids=env_ids)
        self._fixed_asset.write_root_velocity_to_sim(fixed_state[:, 7:], env_ids=env_ids)
        self._fixed_asset.reset()

        self.step_sim_no_action()

        if self.cfg_task.name == "gear_mesh":
            # Set small and large gear states.
            small_gear_state = self._small_gear_asset.data.default_root_state.clone()[env_ids]
            small_gear_state[:, 0:7] = fixed_state[:, 0:7]
            small_gear_state[:, 7:] = 0.0  # vel
            self._small_gear_asset.write_root_pose_to_sim(small_gear_state[:, 0:7], env_ids=env_ids)
            self._small_gear_asset.write_root_velocity_to_sim(small_gear_state[:, 7:], env_ids=env_ids)
            self._small_gear_asset.reset()

            large_gear_state = self._large_gear_asset.data.default_root_state.clone()[env_ids]
            large_gear_state[:, 0:7] = fixed_state[:, 0:7]
            large_gear_state[:, 7:] = 0.0  # vel
            self._large_gear_asset.write_root_pose_to_sim(large_gear_state[:, 0:7], env_ids=env_ids)
            self._large_gear_asset.write_root_velocity_to_sim(large_gear_state[:, 7:], env_ids=env_ids)
            self._large_gear_asset.reset(env_ids=env_ids)

        # Set held gear state.
        held_state = torch.zeros((len(env_ids), 13), device=self.device)
        held_state[:, 0:3] = held_pos + self.scene.env_origins[env_ids]
        held_state[:, 3:7] = held_quat
        held_state[:, 7:] = 0.0
        self._held_asset.write_root_pose_to_sim(held_state[:, 0:7], env_ids=env_ids)
        self._held_asset.write_root_velocity_to_sim(held_state[:, 7:], env_ids=env_ids)
        self._held_asset.reset(env_ids=env_ids)

        self.step_sim_no_action()

        physics_sim_view.set_gravity(carb.Float3(*self.cfg.sim.gravity))

    def step_sim_no_action(self):
        """Step the simulation without an action. Used for resets only.

        This method should only be called during resets when all environments
        reset at the same time.
        """
        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        self.scene.update(dt=self.physics_dt)
        self._compute_intermediate_values(dt=self.physics_dt)

    def log(self, string):
        if self.verbose:
            print(f"{string}")

    def _visualize_markers(self):
        if self.vis_options["object_obs"] == True:
            self.held_asset_marker.visualize(self.held_pos_obs_frame + self.scene.env_origins, self.held_quat)
            self.fixed_asset_marker.visualize(self.fixed_pos_obs_frame + self.scene.env_origins, self.fixed_quat)
            # self.fingertip_marker.visualize(self.fingertip_midpoint_pos + self.scene.env_origins, self.fingertip_midpoint_quat)

        if self.vis_options["training_data"] == True:
            self.orange_sphere_marker.visualize(self.obs_base + self.scene.env_origins)
