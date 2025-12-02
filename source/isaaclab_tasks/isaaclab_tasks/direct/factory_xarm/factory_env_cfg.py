# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import json
import numpy as np
import torch
import os

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, TiledCameraCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import quat_from_matrix
from isaaclab.markers import VisualizationMarkersCfg

from .factory_tasks_cfg import ASSET_DIR, FactoryTask, GearMesh, NutThread, PegInsert

OBS_DIM_CFG = {
    "fingertip_pos": 3,
    "fingertip_pos_rel_fixed": 3,
    "fingertip_pos_rel_held": 3,
    "fingertip_quat": 4,
    "gripper": 1,
    "ee_linvel": 3,
    "ee_angvel": 3,
    "base_actions": 8,
}

STATE_DIM_CFG = {
    "fingertip_pos": 3,
    "fingertip_pos_rel_fixed": 3,
    "fingertip_pos_rel_held": 3,
    "fingertip_quat": 4,
    "gripper": 1,
    "ee_linvel": 3,
    "ee_angvel": 3,
    "joint_pos": 7,
    "held_pos": 3,
    "held_pos_rel_fixed": 3,
    "held_quat": 4,
    "fixed_pos": 3,
    "fixed_quat": 4,
    "task_prop_gains": 6,
    "ema_factor": 1,
    "pos_threshold": 3,
    "rot_threshold": 3,
    "gripper_threshold": 1,
    "base_actions": 8,
}

intr_path = "logs/data/calibration/251119_rrl_1cam/intrinsics.json"
with open(intr_path, "r") as f:
    intr = json.load(f)

extr_path = "logs/data/calibration/251119_rrl_1cam/extrinsics.json"
with open(extr_path, "r") as f:
    extr = json.load(f)

H, W = intr["front"]["height"], intr["front"]["width"]

FRONT_INTR = [
    intr["front"]["fx"], 0.0, intr["front"]["ppx"],
    0.0, intr["front"]["fy"], intr["front"]["ppy"],
    0.0, 0.0, 1.0
]

FRONT_PINHOLE_CFG = sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
    intrinsic_matrix=FRONT_INTR,
    height=H,
    width=W,
    focal_length=1.93
)

FRONT_PINHOLE_CFG.horizontal_aperture_offset = -0.04

front2base = np.array(extr["cam2base"]["front2base"]).reshape(4, 4)

R_front2base = front2base[:3, :3]
q_front2base = quat_from_matrix(torch.from_numpy(R_front2base)).tolist() # wxyz
t_front2base = front2base[:3, 3].tolist()

@configclass
class ObsRandCfg:
    fixed_asset_pos = [0.001, 0.001, 0.001]
    held_asset_pos = [0.001, 0.001, 0.001]

@configclass
class BaseActionRandCfg:
    horizon = [7, 30]

@configclass
class CtrlCfg:
    ema_factor = 0.2

    pos_action_bounds = [0.05, 0.05, 0.05]
    rot_action_bounds = [1.0, 1.0, 1.0]

    pos_action_threshold = [0.05, 0.05, 0.05]
    rot_action_threshold = [0.097, 0.097, 0.097]
    gripper_action_threshold = [0.1]

    res_pos_action_threshold = [0.03, 0.03, 0.03] # 0.25mm -> action 0.5mm
    res_rot_action_threshold = [0.097, 0.097, 0.097]
    res_gripper_action_threshold = [0.1]

    Kx_dmr_range = [100, 300]
    Kr_dmr_range = [20, 50]
    mx_dmr_range = [0.2, 0.05]
    mr_dmr_range = [0.02, 0.005]
    lam_dmr_range = [1e-3, 1e-2]


    reset_joints = [0.035, -0.323, 0.0, 0.523, 0.0, 1.31, 0.0]
    reset_task_prop_gains = [300, 300, 300, 20, 20, 20]
    reset_rot_deriv_scale = 10.0
    default_task_prop_gains = [100, 100, 100, 0, 0, 0]

    # Null space parameters.
    default_dof_pos_tensor = [0.035, -0.323, 0.0, 0.523, 0.0, 1.31, 0.0]
    kp_null = 10.0
    kd_null = 6.3246

    # Admittance control parameters
    K = [200.0, 200.0, 200.0, 50.0, 50.0, 50.0]
    D = [63.0, 63.0, 63.0, 6.3, 6.3, 6.3]
    M = [5.0, 5.0, 5.0, 0.2, 0.2, 0.2]
    lam = 1e-2
    rot_scale = 0.25
    v_task_limits = (0.25, 0.6)
    qd_limit=1.5

@configclass
class FactoryEnvCfg(DirectRLEnvCfg):
    decimation = 8
    action_space = 6 # TODO: 7 for residual
    residual_action_space = 7
    # num_*: will be overwritten to correspond to obs_order, state_order.
    observation_space = 21
    state_space = 72
    obs_order: list = ["fingertip_pos_rel_fixed", "fingertip_quat", "ee_linvel", "ee_angvel"]
    state_order: list = [
        "fingertip_pos",
        "fingertip_quat",
        "ee_linvel",
        "ee_angvel",
        "joint_pos",
        "held_pos",
        "held_pos_rel_fixed",
        "held_quat",
        "fixed_pos",
        "fixed_quat",
    ]

    # for replay
    obs_order_no_task: list = ["fingertip_pos", "fingertip_quat"]

    # for residual policies
    residual_obs_order: list = [
        "fingertip_pos",
        "fingertip_quat", 
        "gripper",
        "fingertip_pos_rel_fixed", 
        "fingertip_pos_rel_held", 
        "ee_linvel", 
        "ee_angvel", 
        "base_actions"
    ]

    residual_state_order: list = [
        "fingertip_pos",
        "fingertip_quat",
        "gripper",
        "ee_linvel",
        "ee_angvel",
        "joint_pos",
        "held_pos",
        "held_pos_rel_fixed",
        "held_quat",
        "fixed_pos",
        "fixed_quat",
        "base_actions"
    ]

    residual_obs_order_no_base: list = [
        "fingertip_pos",
        "fingertip_quat", 
        "gripper",
        "fingertip_pos_rel_fixed", 
        "fingertip_pos_rel_held", 
        "ee_linvel", 
        "ee_angvel", 
    ]
    residual_state_order_no_base: list = [
        "fingertip_pos",
        "fingertip_quat",
        "gripper",
        "ee_linvel",
        "ee_angvel",
        "joint_pos",
        "held_pos",
        "held_pos_rel_fixed",
        "held_quat",
        "fixed_pos",
        "fixed_quat",
        "base_actions"
    ]

    task_name: str = "peg_insert"  # peg_insert, gear_mesh, nut_thread
    task: FactoryTask = FactoryTask()
    obs_rand: ObsRandCfg = ObsRandCfg()
    ctrl: CtrlCfg = CtrlCfg()
    base_rand: BaseActionRandCfg = BaseActionRandCfg()

    episode_length_s = 10.0  # Probably need to override.
    sim: SimulationCfg = SimulationCfg(
        device="cuda:0",
        dt=1 / 120,
        gravity=(0.0, 0.0, -9.81),
        physx=PhysxCfg(
            solver_type=1,
            max_position_iteration_count=192,  # Important to avoid interpenetration.
            max_velocity_iteration_count=1,
            bounce_threshold_velocity=0.2,
            friction_offset_threshold=0.01,
            friction_correlation_distance=0.00625,
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**23,
            gpu_collision_stack_size=2**28,
            gpu_max_num_partitions=1,  # Important for stable simulation.
        ),
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=128, env_spacing=2.0, clone_in_fabric=False)
    
    measure_force = False
    enable_cameras = False
    sparse_rewards = False

    XARM_USD_PATH = "source/isaaclab_tasks/isaaclab_tasks/direct/factory_xarm/assets/xarm7_gripper.usd"
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=XARM_USD_PATH,
            activate_contact_sensors=measure_force,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=3666.0,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
                max_contact_impulse=1e32,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,      
                solver_position_iteration_count=192, 
                solver_velocity_iteration_count=1
            ),
            joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),    
            # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ), 
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "joint1": 0.035,
                "joint2": -0.785, #-45
                "joint3": 0.0,
                "joint4": 0.523, # 30
                "joint5": 0.0,
                "joint6": 1.31, # 75
                "joint7": 0.0,
                "gripper": 0.0, # 0.0 to 1.7
                "left_driver_joint": 0.0,
                "left_inner_knuckle_joint": 0.0,
                "left_finger_joint": 0.0,
                "right_driver_joint": 0.0,
                "right_inner_knuckle_joint": 0.0,
                "right_finger_joint": 0.0,
            },
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["joint[1-7]"],
                # effort_limit=50.0,
                # velocity_limit=3.14,
                stiffness=200,#80.0, 
                damping=20,#10.0,
            ),
            "xarm_hand": ImplicitActuatorCfg(
                joint_names_expr=["gripper"], 
                # effort_limit_sim=40.0,
                # velocity_limit_sim=0.04,
                stiffness=5.0, # 5
                damping=0.0, #0.0
            ),
        },
    )
    
    sim_fingertip2eef = [0.0, 0.0, 0.17]
    real_fingertip2eef = [0.0, 0.0, 0.23]

    eef_contact_sensor_cfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/robot/link7",
        update_period=0.0,
        history_length=6,
        debug_vis=False,
        # filter_prim_paths_expr=["{ENV_REGEX_NS}/Cube"],
    )

    front_camera_cfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/front_camera",
        offset=TiledCameraCfg.OffsetCfg(pos=t_front2base, rot=q_front2base, convention="ros"), # z-down; x-forward # greater angle = towards gripper
        height=H,
        width=W,
        data_types=[
            "rgb",
            # "distance_to_image_plane",
            ],
        spawn=FRONT_PINHOLE_CFG,
    )

    frame_marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/myMarkers",
        markers={
            "frame": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                scale=(0.05, 0.05, 0.05))
            }
        )
    
    keypoints_marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/myMarkers",
        markers={
            "sphere": sim_utils.SphereCfg(
                radius=0.006,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            ),
            "sphere": sim_utils.SphereCfg(
                radius=0.006,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            ),
            "sphere": sim_utils.SphereCfg(
                radius=0.006,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            ),
            "sphere": sim_utils.SphereCfg(
                radius=0.006,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            ),
        }
    )

    red_sphere_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/red_sphere_marker",
        markers={
            "sphere": sim_utils.SphereCfg(
                radius=0.004,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            )
        }
    )

    blue_sphere_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/blue_sphere_marker",
        markers={
            "sphere": sim_utils.SphereCfg(
                radius=0.004,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
            )
        }
    )

    green_sphere_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/green_sphere_marker",
        markers={
            "sphere": sim_utils.SphereCfg(
                radius=0.004,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            )
        }
    )


@configclass
class FactoryTaskPegInsertCfg(FactoryEnvCfg):
    task_name = "peg_insert"
    task = PegInsert()
    episode_length_s = 10.0


@configclass
class FactoryTaskGearMeshCfg(FactoryEnvCfg):
    task_name = "gear_mesh"
    task = GearMesh()
    episode_length_s = 20.0


@configclass
class FactoryTaskNutThreadCfg(FactoryEnvCfg):
    task_name = "nut_thread"
    task = NutThread()
    episode_length_s = 30.0
