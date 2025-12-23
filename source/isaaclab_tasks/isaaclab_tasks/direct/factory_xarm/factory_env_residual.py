# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import torch
import math

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

    def _init_residual_policy_buffers(self):
        """Initialize buffers specific to residual policy."""
        self.verbose = self.cfg.env_options.verbose
        self.teleop_mode = self.cfg.env_options.teleop_mode

        self.vis_options = self.cfg.env_options.vis_options

        if self.vis_options["eval_mode"]:
            self.base_actions_agent = NearestNeighborBuffer(
                factory_utils.resolve_hf_file(self.cfg_task.hf_repo, self.cfg_task.action_data_hf_file), 
                self.num_envs, 
                min_horizon=1, 
                max_horizon=15, 
                device=self.device, 
                pad=True # type: ignore
            )
        else:
            self.base_actions_agent = NearestNeighborBuffer(
                factory_utils.resolve_hf_file(self.cfg_task.hf_repo, self.cfg_task.action_data_hf_file), 
                self.num_envs, 
                min_horizon=1, 
                max_horizon=15, 
                device=self.device, 
                pad=True # type: ignore
            )
        self.base_actions = torch.zeros((self.num_envs, 8), device=self.device)

        self.total_episodes: int = self.base_actions_agent.get_total_episodes() 
        if self.vis_options["eval_mode"]:
            # assert self.num_envs == self.total_episodes, "num_envs must equal total_episodes in eval_mode"
            self.episode_idx = torch.arange(0, self.num_envs, device=self.device) % self.total_episodes
        self.episode_idx = torch.randint(0, self.total_episodes, (self.num_envs,), device=self.device)
        # self.episode_idx = torch.arange(0, self.num_envs, device=self.device) % self.total_episodes

        # overwrite cfg
        # self.cfg.episode_length_s = self.base_actions_agent.get_max_episode_length() * (self.cfg.sim.dt * self.cfg.decimation)
        # self.max_per_eps_length = self.base_actions_agent.get_max_per_episode_length() # (num_eps, )

        self.initial_poses = torch.load(factory_utils.resolve_hf_file(self.cfg_task.hf_repo, self.cfg_task.initial_poses_hf_file)) # dict each of shape (tot_eps, dim) # type: ignore
        self.initial_poses = {k: v.unsqueeze(0).repeat((self.num_envs, 1, 1)).to(self.device) for k, v in self.initial_poses.items()} # dict each of shape (num_envs, tot_eps, dim)

        # ctrl params
        self.Kx = torch.tensor([150.0], device=self.device).repeat(self.num_envs) # (num_envs, )
        self.Kr = torch.tensor([100.0], device=self.device).repeat(self.num_envs) # (num_envs, )
        self.mx = torch.tensor([0.1], device=self.device).repeat(self.num_envs) # (num_envs, )
        self.mr = torch.tensor([0.01], device=self.device).repeat(self.num_envs) # (num_envs, )
        self.lam = torch.tensor([1e-2], device=self.device).repeat(self.num_envs) # (num_envs, )

        # abs ik for reset
        urdf = factory_utils.resolve_hf_file(self.cfg_task.hf_repo, "assets/robot/xarm7.urdf")
        chain = pk.build_chain_from_urdf(open(urdf, mode="rb").read())
        # chain.print_tree()
        self.serial_chain = pk.SerialChain(chain, "link7", "link_base")

        self.lim = torch.tensor(chain.get_joint_limits())[:, :7]

        # Held asset yaw rotation tracking (only for nut_thread task, only when task-engaged)
        if self.cfg_task.name == "nut_thread":
            self.prev_held_yaw = torch.zeros(self.num_envs, device=self.device)
            self.cumulative_rotation = torch.zeros(self.num_envs, device=self.device)  # cumulative yaw rotation in degrees
            self.picked_up = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self.base_last5 = LastKPoints(self.num_envs, K=5, device=self.device)
        self.residual_last5 = LastKPoints(self.num_envs, K=5, device=self.device)
        self.bad_insert = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)

    def compute_ik_abs(
        self,
        action: torch.Tensor,
        curr_qpos: torch.Tensor,  # (N, DOF)
    ):
        tf = torch.eye(4, device=action.device, dtype=action.dtype).unsqueeze(0).repeat(action.shape[0], 1, 1)  # (N,4,4)
        tf[:, :3, :3] = torch_utils.quats_to_rot_matrices(action[:, 3:7])
        tf[:, :3, 3] = action[:, :3]

        rob_tf = pk.Transform3d(matrix=tf.to("cpu"), dtype=action.dtype)

        self.abs_ik = pk.PseudoInverseIK(self.serial_chain, max_iterations=50, num_retries=1,
            retry_configs=curr_qpos.to("cpu").to(torch.float32),
            joint_limits=self.lim.T,
            early_stopping_any_converged=True,
            early_stopping_no_improvement="all",
            debug=False,
            lr=0.2)

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
        self.default_gains = torch.tensor(self.cfg.ctrl.default_task_prop_gains, device=self.device).repeat(
            (self.num_envs, 1)
        )

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
            self.held_center_pos_local[:, 2] += self.cfg_task.held_asset_cfg.height / 2.0 * 1.3 + 0.01

        elif self.cfg_task.name == "peg_insert":
            self.held_center_pos_local[:, 2] += self.cfg_task.held_asset_cfg.height 
            self.held_center_pos_local[:, 2] -= self.cfg_task.robot_cfg.xarm_fingerpad_length / 3.0

        elif self.cfg_task.name == "nut_thread":
            self.held_center_pos_local[:, 2] += 0.01

        # Computer body indices.
        self.left_finger_body_idx = self._robot.body_names.index("left_finger") 
        self.right_finger_body_idx = self._robot.body_names.index("right_finger")
        self.eef_body_idx = self._robot.body_names.index("link7") # TODO: change logic to fingertip == T(eef)
        self.sim_fingertip2eef = torch.tensor([self.cfg.sim_fingertip2eef], device=self.device).repeat(self.num_envs, 1)
        self.real_fingertip2eef = torch.tensor([self.cfg.real_fingertip2eef], device=self.device).repeat(self.num_envs, 1)
        self.arm_dof_idx, _ = self._robot.find_joints("joint.*")
        self.gripper_dof_idx, _ = self._robot.find_joints("gripper")

        self.eef_vel = torch.zeros((self.num_envs, 6), device=self.device)

        # Tensors for finite-differencing.
        self.last_update_timestamp = 0.0  # Note: This is for finite differencing body velocities.
        self.prev_fingertip_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.prev_fingertip_quat = (
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        )
        self.prev_joint_pos = torch.zeros((self.num_envs, 7), device=self.device)

        self.ep_succeeded = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self.ep_success_times = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)

        self.rolling_success_rate = 0.0
        self.ema_alpha = 0.002 # 350 eps half life for 450 ts eps

        self.eps_grasp_succeeded = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self.eps_task_engaged = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self.eps_task_succeeded = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)

        self.keypoints_fingertip = torch.zeros((self.num_envs, self.cfg_task.num_keypoints, 3), device=self.device)
        self.keypoints_held = torch.zeros((self.num_envs, self.cfg_task.num_keypoints, 3), device=self.device)
        self.keypoints_fixed = torch.zeros((self.num_envs, self.cfg_task.num_keypoints, 3), device=self.device)

        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self.prev_actions = torch.zeros_like(self.actions)
        self.env_actions = torch.zeros((self.num_envs, 8), device=self.device)

        # Track continuous grasp from start of closing to fully closed
        self.grasp_active = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)

        if self.cfg_task == "nut_thread":
            self.picked_up = torch.where(self.held_pos[:, 2] > 0.03, torch.ones_like(self.picked_up), torch.zeros_like(self.picked_up)).bool()

    def _setup_scene(self):
        """Initialize simulation scene."""
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(), translation=(0.0, 0.0, -1.05))

        # spawn a usd file of a table into the scene
        cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd")
        cfg.func(
            "/World/envs/env_.*/Table", cfg, translation=(0.55, 0.0, -0.006), orientation=(0.70711, 0.0, 0.0, 0.70711)
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

        if self.enable_cameras:
            self.front_camera = TiledCamera(self.cfg.front_camera_cfg)
            self.scene.sensors["front_camera"] = self.front_camera

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

    def _compute_base_actions(self):
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

        self.base_actions = self.base_actions_agent.get_actions(self.episode_idx, sim_fingertip_pos, sim_eef_quat, self.gripper) # (num_envs, residual_action_dim) at eef
        
        if self.vis_options["training_data"]:
            self.obs_base, quat_base, _ = self.base_actions_agent.get_closest_obs(self.episode_idx, sim_fingertip_pos, sim_eef_quat, self.gripper, verbose=False)
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

        self.left_finger_jacobian = jacobians[:, self.left_finger_body_idx - 1, 0:6, 0:7] # (num_envs, 6, arm_idx), origin at body idx
        self.right_finger_jacobian = jacobians[:, self.right_finger_body_idx - 1, 0:6, 0:7]
        self.fingertip_midpoint_jacobian = (self.left_finger_jacobian + self.right_finger_jacobian) * 0.5
        self.arm_mass_matrix = self._robot.root_physx_view.get_generalized_mass_matrices()[:, 0:7, 0:7]

        if self.measure_force:
            self.eef_force = self.eef_contact_sensor.data.net_forces_w.squeeze(1) # (num_envs, 3)
            self.F_ext = torch.cat([self.eef_force, torch.zeros((self.num_envs, 3), device=self.device)], dim=-1) # (num_envs, 6)

        if self.enable_cameras:
            self.front_rgb = self.front_camera.data.output["rgb"] # (num_envs, H, W, 3) (0-255)

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

        joint_diff = self.joint_pos[:, 0:7] - self.prev_joint_pos
        self.joint_vel_fd = joint_diff / dt
        self.prev_joint_pos = self.joint_pos[:, 0:7].clone()

        self.last_update_timestamp = self._robot._data._sim_timestamp

    def _get_factory_obs_state_dict(self):
        """Populate dictionaries for the policy and critic."""
        noisy_fixed_pos = self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise
        noisy_held_pos = self.held_pos_obs_frame + self.init_held_pos_obs_noise 

        prev_actions = self.actions.clone()

        # self.red_sphere_marker.visualize(self.held_pos_obs_frame + self.scene.env_origins)
        # self.blue_sphere_marker.visualize(self.fixed_pos_obs_frame + self.scene.env_origins)
 
        obs_dict = {
            "fingertip_pos": self.fingertip_midpoint_pos,
            "fingertip_pos_rel_fixed": self.fingertip_midpoint_pos - noisy_fixed_pos,
            "fingertip_pos_rel_held": self.fingertip_midpoint_pos - noisy_held_pos,
            "fingertip_quat": self.fingertip_midpoint_quat,
            "gripper": self.gripper,
            "ee_linvel": self.ee_linvel_fd,
            "ee_angvel": self.ee_angvel_fd,
            "prev_actions": prev_actions,
            "base_actions": self.base_actions,
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
            # "task_prop_gains": self.task_prop_gains,
            "pos_threshold": self.pos_threshold,
            "rot_threshold": self.rot_threshold,
            "gripper_threshold": self.gripper_threshold,
            "prev_actions": prev_actions,
            "base_actions": self.base_actions, 
        }
        return obs_dict, state_dict

    def _get_observations(self):
        """Get actor/critic inputs using asymmetric critic."""
        if not self.teleop_mode:
            self._compute_base_actions()
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
        self.eps_grasp_succeeded[env_ids] = 0
        self.eps_task_engaged[env_ids] = 0
        self.eps_task_succeeded[env_ids] = 0
        self.grasp_active[env_ids] = 0

    def _pre_physics_step(self, action):
        """Apply policy actions with smoothing."""
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self._reset_buffers(env_ids)

        self.actions = self.ema_factor * action.clone().to(self.device) + (1 - self.ema_factor) * self.actions

    def _apply_action(self):
        """Apply actions for policy as delta targets from current position."""
        # Note: We use finite-differenced velocities for control and observations.
        # Check if we need to re-compute velocities within the decimation loop.
        if self.last_update_timestamp < self._robot._data._sim_timestamp:
            self._compute_intermediate_values(dt=self.physics_dt)

        # Interpret actions as target pos displacements and set pos target
        pos_actions = self.actions[:, 0:3] * self.pos_threshold 

        # Interpret actions as target rot (axis-angle) displacements
        rot_actions = self.actions[:, 3:6]
        # if self.cfg_task.unidirectional_rot:
        #     rot_actions[:, 2] = -(rot_actions[:, 2] + 1.0) * 0.5  # [-1, 0]
        rot_actions = rot_actions * self.rot_threshold

        ctrl_target_fingertip_midpoint_pos = self.base_actions[:, 0:3] + pos_actions

        # Convert to quat and set rot target
        angle = torch.norm(rot_actions, p=2, dim=-1)
        axis = rot_actions / angle.unsqueeze(-1)

        rot_actions_quat = torch_utils.quat_from_angle_axis(angle, axis)
        rot_actions_quat = torch.where(
            angle.unsqueeze(-1).repeat(1, 4) > 1e-6,
            rot_actions_quat,
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1),
        )
        ctrl_target_fingertip_midpoint_quat = torch_utils.quat_mul(rot_actions_quat, self.base_actions[:, 3:7])

        target_euler_xyz = torch.stack(torch_utils.get_euler_xyz(ctrl_target_fingertip_midpoint_quat), dim=1)
        target_euler_xyz[:, 0] = 3.14159  # Restrict actions to be upright.
        target_euler_xyz[:, 1] = 0.0

        ctrl_target_fingertip_midpoint_quat = torch_utils.quat_from_euler_xyz(
            roll=target_euler_xyz[:, 0], pitch=target_euler_xyz[:, 1], yaw=target_euler_xyz[:, 2]
        )

        ctrl_target_eef_pos = torch_utils.tf_combine(
            ctrl_target_fingertip_midpoint_quat,
            ctrl_target_fingertip_midpoint_pos,
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1),
            -self.sim_fingertip2eef,
        )[1]

        # gripper_action = torch.abs(self.actions[:, 6:7])
        ctrl_target_gripper_dof_pos = torch.clamp(self.base_actions[:, 7:8], 0.0, 1.0) * 1.6
        if self.cfg_task.name == "gear_mesh":
            ctrl_target_gripper_dof_pos = torch.clamp(ctrl_target_gripper_dof_pos, max=1.2)
        elif self.cfg_task.name == "nut_thread":
            ctrl_target_gripper_dof_pos = torch.clamp(ctrl_target_gripper_dof_pos, max=0.75)

        self.env_actions = torch.cat([ctrl_target_fingertip_midpoint_pos, ctrl_target_fingertip_midpoint_quat, ctrl_target_gripper_dof_pos], dim=-1)

        self.generate_ctrl_signals(
            ctrl_target_eef_pos=ctrl_target_eef_pos,
            ctrl_target_fingertip_midpoint_quat=ctrl_target_fingertip_midpoint_quat,
            ctrl_target_gripper_dof_pos=ctrl_target_gripper_dof_pos,
        )

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
            Kx=self.Kx, Kr=self.Kr, mx=self.mx, mr=self.mr, Dx=None, Dr=None, lam=self.lam, rot_scale=1.25,
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
        # time_out = self.episode_length_buf >= self.max_per_eps_length[self.episode_idx] - 1

        time_out = self.episode_length_buf >= self.max_episode_length - 1 # TODO: efficiency problem -> per eps max length speeds up learning
        # print("timestep: ", self.episode_length_buf[0].item(), "/", self.max_episode_length)
        dist_threshold = 0.2
        terminated = torch.norm(self.fingertip_midpoint_pos - self.held_pos_obs_frame, dim=1) > dist_threshold # to eliminate the case where held asset falls far away
        terminated |= self.ep_succeeded.bool()

        if self.cfg_task.name == "gear_mesh" or self.cfg_task.name == "peg_insert":
            terminated |= self.bad_insert.bool()
            self.bad_insert[:] = 0

        if self.cfg_task.name == "peg_insert":
            unit_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
            tilt_degrees = factory_utils.quat_geodesic_angle(self.held_quat, unit_quat) * 180.0 / math.pi
            terminated |= torch.where(tilt_degrees > 30.0, torch.ones_like(terminated), torch.zeros_like(terminated)).bool()

        elif self.cfg_task.name == "nut_thread":
            on_ground = self.held_pos[:, 2] < 0.02
            dropped = torch.logical_and(on_ground, self.picked_up)
            terminated |= dropped

        done = torch.logical_or(time_out, terminated)
        s = self.ep_succeeded[done].float()  # shape [n_finished]
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

        # print("xy dist:", xy_dist)
        # print("z disp:", z_disp)

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

    def _get_rewards(self):
        """Update rewards and compute success statistics."""
        # Get successful and failed envs at current timestep
        task_successes = self._get_curr_successes(
            success_threshold=self.cfg_task.success_threshold
        )

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

        # Relaxed Rewards:
        xy_dist = torch.linalg.vector_norm(target_held_base_pos - held_base_pos, dim=1)
        z_disp = held_base_pos[:, 2] - target_held_base_pos[:, 2]

        xy_threshold = 0.015
        is_centered = torch.where(xy_dist < xy_threshold, torch.ones_like(task_successes), torch.zeros_like(task_successes))

        if self.cfg_task.name == "gear_mesh":
            height_threshold = self.cfg_task.fixed_asset_cfg.height * 1.2
        elif self.cfg_task.name == "peg_insert":
            height_threshold = self.cfg_task.fixed_asset_cfg.height * 1.5
        elif self.cfg_task.name == "nut_thread":
            height_threshold = 0.01
        is_close_or_below = torch.where(
            z_disp < height_threshold, torch.ones_like(task_successes), torch.zeros_like(task_successes)
        )
        self.log(f"xy_dist: {xy_dist.mean().item():.4f}, goal {xy_threshold}")
        self.log(f"z_disp: {z_disp.mean().item():.4f}, goal {height_threshold}")
        task_engaged = torch.logical_and(is_centered, is_close_or_below)

        grasp_dist = torch.linalg.vector_norm(self.held_pos_obs_frame - self.fingertip_midpoint_pos, dim=1)
        grasp_dist_satisfied = grasp_dist < 0.01
        gripper_closing = self.gripper.squeeze(-1) > 0.1
        gripper_closed = self.gripper.squeeze(-1) >= self.cfg_task.close_gripper
        
        # Activate grasp tracking when gripper starts closing with good grasp distance
        self.grasp_active = torch.where(
            torch.logical_and(gripper_closing, grasp_dist_satisfied),
            torch.ones_like(self.grasp_active),
            self.grasp_active
        )
        
        # Deactivate if grasp distance lost while closing
        self.grasp_active = torch.where(
            torch.logical_and(gripper_closing, ~grasp_dist_satisfied),
            torch.zeros_like(self.grasp_active),
            self.grasp_active
        )
        
        # Success = maintained grasp from start of closing until fully closed
        grasp_successes = torch.logical_and(self.grasp_active, gripper_closed)
        close_gripper = gripper_closed

        # Track held asset yaw rotation only for nut_thread task, only when task-engaged
        if self.cfg_task.name == "nut_thread":
            # Extract yaw from held asset quaternion
            _, _, curr_held_yaw = torch_utils.get_euler_xyz(self.held_quat)
            curr_held_yaw = (curr_held_yaw + math.pi) % (2 * math.pi) - math.pi  # wrap to [-pi, pi]
            
            # Compute smallest signed delta between prev and current yaw
            yaw_delta = curr_held_yaw - self.prev_held_yaw
            yaw_delta = (yaw_delta + math.pi) % (2 * math.pi) - math.pi  # wrap delta to [-pi, pi]
            yaw_delta_deg = yaw_delta * 180.0 / math.pi
            
            # Only accumulate yaw rotation when task_engaged
            acc_condition = torch.logical_or(task_engaged, grasp_successes)
            self.cumulative_rotation[acc_condition] += torch.abs(yaw_delta_deg[acc_condition])
            
            # Always update previous yaw to avoid jumps when task_engaged becomes True again
            self.prev_held_yaw = curr_held_yaw.clone()
            
            # Compute rotation reward: 1 if rotation >= 180°
            task_successes = torch.where(
                self.cumulative_rotation >= 180.0,
                torch.ones_like(task_successes), torch.zeros_like(task_successes)
            )

        action_delta = torch.norm(self.env_actions[:, :3] - self.base_actions[:, :3], dim=1) / 10 # meter / 10

        grasp_successes = torch.logical_and(grasp_successes, close_gripper)
        if self.cfg_task.name == "gear_mesh" or self.cfg_task.name == "peg_insert":
            task_successes = torch.logical_and(task_successes, grasp_successes)
            self.bad_insert = torch.logical_and(task_successes, ~grasp_successes)
        self.log(f"Grasp success: {grasp_successes.float().mean().item():.3f}")
        self.log(f"Task engaged: {task_engaged.float().mean().item():.3f}")
        self.log(f"Task success: {task_successes.float().mean().item():.3f}")

        first_task_engaged = torch.logical_and(task_engaged, torch.logical_not(self.eps_task_engaged))
        self.eps_task_engaged[task_engaged] = 1
        task_engaged = torch.logical_and(task_engaged, first_task_engaged)
        first_grasp_succeeded = torch.logical_and(grasp_successes, torch.logical_not(self.eps_grasp_succeeded))
        self.eps_grasp_succeeded[grasp_successes] = 1
        grasp_successes = torch.logical_and(grasp_successes, first_grasp_succeeded)
        first_task_succeeded = torch.logical_and(task_successes, torch.logical_not(self.eps_task_succeeded))
        self.eps_task_succeeded[task_successes] = 1
        task_successes = torch.logical_and(task_successes, first_task_succeeded)

        # Visualize failed envs with red sphere markers
        failed_envs = ~task_successes
        if failed_envs.any() and self.vis_options["failed_envs"] == True:
            failed_positions = torch.tensor([0.5, 0.5, 0.5], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
            failed_positions = failed_positions + self.scene.env_origins
            # Only visualize positions for failed envs
            failed_positions[~failed_envs] = torch.tensor([0.0, 0.0, -10.0], device=self.device)  # Move successful envs off-screen
            self.red_sphere_marker.visualize(failed_positions)

        rew_dict = {
            "action_delta": -action_delta,
            "grasp_success": grasp_successes.float(),
            "task_engaged": task_engaged.float(),
            "task_success": task_successes.float(),
        }
        # print("Rewards: ", {k: v.mean().item() for k, v in rew_dict.items()})
        rew_buf = torch.zeros_like(rew_dict["task_success"])
        for rew_name, rew in rew_dict.items():
            rew_buf += rew_dict[rew_name]

        self.prev_actions = self.actions.clone()

        self._log_factory_metrics(rew_dict, task_successes)

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

        self.Kx[env_ids] = self.cfg.ctrl.Kx_dmr_range[0] + (self.cfg.ctrl.Kx_dmr_range[1] - self.cfg.ctrl.Kx_dmr_range[0]) * torch.rand(len(env_ids), device=self.device)
        self.Kr[env_ids] = self.cfg.ctrl.Kr_dmr_range[0] + (self.cfg.ctrl.Kr_dmr_range[1] - self.cfg.ctrl.Kr_dmr_range[0]) * torch.rand(len(env_ids), device=self.device)
        self.mx[env_ids] = self.cfg.ctrl.mx_dmr_range[0] + (self.cfg.ctrl.mx_dmr_range[1] - self.cfg.ctrl.mx_dmr_range[0]) * torch.rand(len(env_ids), device=self.device)
        self.mr[env_ids] = self.cfg.ctrl.mr_dmr_range[0] + (self.cfg.ctrl.mr_dmr_range[1] - self.cfg.ctrl.mr_dmr_range[0]) * torch.rand(len(env_ids), device=self.device)
        self.lam[env_ids] = self.cfg.ctrl.lam_dmr_range[0] + (self.cfg.ctrl.lam_dmr_range[1] - self.cfg.ctrl.lam_dmr_range[0]) * torch.rand(len(env_ids), device=self.device)

        # move to next episode
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

        if self.vis_options["eval_mode"]:
            translation_noise = torch.randn((len(env_ids), 2), device=self.device) * 0.0
            yaw_rotation_noise = torch.randn((len(env_ids), ), device=self.device) * 0.0
        else:
            translation_noise = torch.randn((len(env_ids), 2), device=self.device) * 0.05
            yaw_rotation_noise = torch.randn((len(env_ids), ), device=self.device) * math.radians(5.0)

        self.xy_translation_noise[env_ids] = translation_noise
        self.yaw_rotation_noise[env_ids] = yaw_rotation_noise.unsqueeze(-1) # in local frame

        held_pos = self.initial_poses["held_pos"][env_ids, self.episode_idx[env_ids]] # (num_resets, 3)
        held_pos[:, :2] += translation_noise

        held_quat = self.initial_poses["held_quat"][env_ids, self.episode_idx[env_ids]]
        held_quat = torch_utils.quat_mul(
            held_quat,
            torch_utils.quat_from_euler_xyz(
                roll=torch.zeros((len(env_ids),), device=self.device),
                pitch=torch.zeros((len(env_ids),), device=self.device),
                yaw=yaw_rotation_noise,
            )
        )

        fixed_pos = self.initial_poses["fixed_pos"][env_ids, self.episode_idx[env_ids]]
        fixed_pos[:, :2] += translation_noise

        fixed_quat = self.initial_poses["fixed_quat"][env_ids, self.episode_idx[env_ids]]
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
        init_qpos = self.initial_poses["sim_init_qpos"][env_ids, self.episode_idx[env_ids]]
        init_fingertip = self.initial_poses["real_fingertip_pos"][env_ids, self.episode_idx[env_ids]] # (num_resets, 7)
        sim_eef = init_fingertip.clone() # (num_resets, 7)
        sim_eef[:, 0:3] = torch_utils.tf_combine(
            sim_eef[:, 3:7],
            sim_eef[:, 0:3],
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(len(env_ids), 1),
            -self.sim_fingertip2eef[env_ids],
        )[1]
        sim_eef[:, :2] += translation_noise
        sim_eef[:, 3:7] = torch_utils.quat_mul(
            sim_eef[:, 3:7],
            torch_utils.quat_from_euler_xyz(
                roll=torch.zeros((len(env_ids),), device=self.device),
                pitch=torch.zeros((len(env_ids),), device=self.device),
                yaw=yaw_rotation_noise,
            ),
        )
        noised_qpos = self.compute_ik_abs(sim_eef[:, :7], init_qpos)
        self._set_replay_default_pose(joints=noised_qpos, env_ids=env_ids) # compute intermediate values there

        self.base_actions_agent.clear(env_ids)
        if not self.teleop_mode:
            self._compute_base_actions()
        
        # Compute fixed_pos_obs_frame
        fixed_tip_pos_local = torch.zeros((len(env_ids), 3), device=self.device)
        fixed_tip_pos_local[:, 2] += self.cfg_task.fixed_asset_cfg.height
        fixed_tip_pos_local[:, 2] += self.cfg_task.fixed_asset_cfg.base_height
        if self.cfg_task.name == "gear_mesh":
            fixed_tip_pos_local[:, 0] = self.cfg_task.fixed_asset_cfg.medium_gear_base_offset[0] # type: ignore

        _, fixed_tip_pos = torch_utils.tf_combine(
            self.fixed_quat[env_ids],
            self.fixed_pos[env_ids],
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(len(env_ids), 1),
            fixed_tip_pos_local,
        )
        self.fixed_pos_obs_frame[env_ids] = fixed_tip_pos

        # reset buffers
        self.prev_joint_pos[env_ids] = self.joint_pos[env_ids, 0:7].clone()
        self.prev_fingertip_pos[env_ids] = self.fingertip_midpoint_pos[env_ids].clone()
        self.prev_fingertip_quat[env_ids] = self.fingertip_midpoint_quat[env_ids].clone()

        # Set initial actions to involve no-movement. Needed for EMA/correct penalties.
        self.actions[env_ids] = torch.zeros_like(self.actions[env_ids])
        self.prev_actions[env_ids] = torch.zeros_like(self.actions[env_ids])
        self.env_actions[env_ids] = torch.zeros_like(self.env_actions[env_ids])

        # Zero initial velocity.
        self.ee_angvel_fd[env_ids, :] = 0.0
        self.ee_linvel_fd[env_ids, :] = 0.0

        self.base_last5.clear_envs(env_ids)
        # self.base_last5.push(self.base_actions[env_ids, :3])  # initialize with current base actions
        self.residual_last5.clear_envs(env_ids)
        # self.residual_last5.push(self.base_actions[env_ids, :3])  # initialize with current residual actions

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
                obs_pos, obs_quat, act_pos, act_quat = self.base_actions_agent.get_episode_traj(self.episode_idx[env_id].item()) # (eps_len, 3)
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

        if self.teleop_mode:
            base_fingertip_pos = self.load_all_episode_act_pos(factory_utils.resolve_hf_file(self.cfg_task.hf_repo, self.cfg_task.action_data_hf_file))
            base_fingertip_pos = base_fingertip_pos.reshape(-1, 3)

            from isaacsim.util.debug_draw import _debug_draw
            draw = _debug_draw.acquire_debug_draw_interface()
            draw.clear_lines()
            base_fingertip_pos = (base_fingertip_pos + self.scene.env_origins).cpu().numpy().tolist()
            purple_color = [(1, 0, 1, 1)]* len(base_fingertip_pos)

            draw.draw_points(base_fingertip_pos, purple_color, [5]*len(base_fingertip_pos))

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

    def _set_franka_to_default_pose(self, joints, env_ids):
        """Return Franka to its default joint position."""
        gripper_width = self.cfg_task.held_asset_cfg.diameter / 2 * 1.25 # 0.005 m
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        gripper_pos = 0.0
        # NOTE: has to set finger pos = 0.5 gripper pos by design of urdf
        joint_pos[:, 8:] = gripper_pos / 2.0 
        joint_pos[:, 7] = gripper_pos
        joint_pos[:, :7] = torch.tensor(joints, device=self.device)[None, :]
        joint_vel = torch.zeros_like(joint_pos)
        joint_effort = torch.zeros_like(joint_pos)
        self.ctrl_target_joint_pos[env_ids, :] = joint_pos
        self._robot.set_joint_position_target(self.ctrl_target_joint_pos[env_ids], env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self._robot.reset()
        self._robot.set_joint_effort_target(joint_effort, env_ids=env_ids)

        self.step_sim_no_action()

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
