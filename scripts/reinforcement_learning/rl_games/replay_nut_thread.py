# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RL-Games."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import cv2

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from RL-Games.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=7, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rl_games_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import math
import os
import random
import time
import torch
import numpy as np

np.set_printoptions(precision=3, suppress=True)
torch.set_printoptions(precision=3, sci_mode=False)

from rl_games.common import env_configurations, vecenv
from rl_games.common.player import BasePlayer
from rl_games.torch_runner import Runner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab.utils.math import quat_from_matrix
from scipy.spatial.transform import Rotation as R

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
import isaacsim.core.utils.torch as torch_utils

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)
def load_npz_dict(path):
    data = np.load(path, allow_pickle=True)
    return (
        data["data"].item() if "data" in data.files
        else {k: data[k].item() for k in data.files}
    )

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



@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Play with RL-Games agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    agent_cfg["params"]["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["params"]["seed"]
    # set the environment seed (after multi-gpu config for updated rank from agent seed)
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg["params"]["seed"]

    # wrap around environment for rl-games
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    obs_groups = agent_cfg["params"]["env"].get("obs_groups")
    concate_obs_groups = agent_cfg["params"]["env"].get("concate_obs_groups", True)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_root_path, log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rl-games
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions, obs_groups, concate_obs_groups)

    # register the environment to rl-games registry
    # note: in agents configuration: environment name must be "rlgpu"
    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    # set number of actors into agent config
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    # create runner from rl-games
    runner = Runner()
    runner.load(agent_cfg)

    dt = env.unwrapped.step_dt

    input_path = "logs/data/1209_teleop_nut_thread_7"
    output_path = input_path

    # load traj & object data
    eps_idx = list(range(7))
    assert env.unwrapped.num_envs == len(eps_idx), "Number of envs must match number of trajs to replay."
    obj_states_path: str = f"{input_path}/obj_states/object_states.npz"
    obj_data = np.load(obj_states_path, allow_pickle=True)

    held2base_pos = torch.zeros((len(eps_idx), 3)).to(env.device)
    held2base_quat = torch.zeros((len(eps_idx), 4)).to(env.device)
    fixed2base_pos = torch.zeros((len(eps_idx), 3)).to(env.device)
    fixed2base_quat = torch.zeros((len(eps_idx), 4)).to(env.device)

    robot_states_path: str = f"{input_path}/robot_states/robot_trajectories.npz"
    robot_data = np.load(robot_states_path, allow_pickle=True)

    init_robot_qpos_path = f"{input_path}/robot_states/init_qpos_sim.npy"
    init_robot_qpos_data = np.load(init_robot_qpos_path, allow_pickle=True)

    max_ts = -1
    for i, ep in enumerate(eps_idx):
        eps_idx_key = f"episode_{eps_idx[i]:04d}"
        max_ts = max(max_ts, robot_data[f"{eps_idx_key}/obs.eef_pos"].shape[0])

    # --- padding bookkeeping ---
    # valid_mask: (max_ts, num_envs) bool tensor, True for valid timesteps, False for padded timesteps
    valid_mask = torch.zeros((max_ts, len(eps_idx)), dtype=torch.bool).to(env.device)

    real_eef_pos_targets = torch.zeros((max_ts, len(eps_idx), 3)).to(env.device)
    real_quat_targets = torch.zeros((max_ts, len(eps_idx), 4)).to(env.device)
    real_gripper_targets = torch.zeros((max_ts, len(eps_idx), 1)).to(env.device)

    for i, ep in enumerate(eps_idx):
        eps_idx_key = f"episode_{eps_idx[i]:04d}"
        held2base_mat = torch.from_numpy(obj_data[f"{eps_idx_key}"][0]).to(env.device).reshape(4,4)
        fixed2base_mat = torch.from_numpy(obj_data[f"{eps_idx_key}"][1]).to(env.device).reshape(4,4)

        held2base_pos[i] = held2base_mat[:3, 3]
        held2base_pos[i, 2] = 0.015
        held2base_quat[i] = torch.tensor([1.0, 0.0, 0.0, 0.0]).to(env.device)#torch_utils.rot_matrices_to_quats(held2base_mat[:3, :3])
        # NOTE: need formal sys id
        fixed2base_pos[i] = fixed2base_mat[:3, 3]
        fixed2base_pos[i, 2] = 0.035
        fixed2base_quat[i] = torch.tensor([1.0, 0.0, 0.0, 0.0]).to(env.device)#torch_utils.rot_matrices_to_quats(fixed2base_mat[:3, :3])

        # --- simple padding: repeat the last valid frame ---
        pos_np   = robot_data[f"{eps_idx_key}/action.eef_pos"]
        quat_np  = robot_data[f"{eps_idx_key}/action.eef_quat"]
        grip_np  = robot_data[f"{eps_idx_key}/action.gripper"]
        L = pos_np.shape[0]

        pos_t  = torch.from_numpy(pos_np).to(env.device)
        quat_t = torch.from_numpy(quat_np).to(env.device)
        grip_t = torch.from_numpy(grip_np).to(env.device)

        real_eef_pos_targets[:L, i] = pos_t
        real_quat_targets[:L, i] = quat_t
        real_gripper_targets[:L, i] = grip_t

        # pad remaining timesteps by repeating the last frame
        if L < max_ts:
            valid_mask[:L, i] = True
            last_pos = pos_t[-1].expand(max_ts - L, -1)
            last_quat = quat_t[-1].expand(max_ts - L, -1)
            last_grip = grip_t[-1].expand(max_ts - L, -1)
            real_eef_pos_targets[L:, i] = last_pos
            real_quat_targets[L:, i] = last_quat
            real_gripper_targets[L:, i] = last_grip
        else:
            valid_mask[:, i] = True

    # import pdb; pdb.set_trace()

    real_init_qpos = torch.from_numpy(init_robot_qpos_data).to(env.device)
    real_init_eef = torch.cat([
        real_eef_pos_targets[0,:,:],
        real_quat_targets[0,:,:]
    ], dim=-1)
    # print("real init qpos:", real_init_qpos.shape, real_init_qpos)

    # if save initial states
    if True:
        init_states_output_path = f"{input_path}/initial_poses"
        os.makedirs(init_states_output_path, exist_ok=True)
        init_poses = {
            "eef": real_init_eef,  # (num_eps, 7)
            "robot": real_init_qpos, # (num_eps, 7)
            "held_pos": held2base_pos, # (num_eps, 3)
            "held_quat": held2base_quat, # (num_eps, 4)
            "fixed_pos": fixed2base_pos, # (num_eps, 3)
            "fixed_quat": fixed2base_quat, # (num_eps, 4)
        }
        torch.save(init_poses, os.path.join(init_states_output_path, "initial_poses.pt"))
        print(f"[INFO] Saved initial poses to: {os.path.join(init_states_output_path, 'initial_poses.pt')}")

    T = max_ts
    # reset environment
    obs = env.reset()

    # set to initial pose
    env.unwrapped._set_replay_default_pose(real_init_qpos, env_ids=torch.arange(len(eps_idx), device=env.device))
    env.unwrapped._set_assets_state(
        held_pos=held2base_pos, 
        held_quat=held2base_quat, 
        fixed_pos=fixed2base_pos, 
        fixed_quat=fixed2base_quat,
        env_ids=torch.arange(len(eps_idx), device=env.device)
    )
    obs = env.unwrapped._get_observations()
    obs = obs["policy"]

    obs_eef_pos = []
    obs_eef_quat = []
    obs_gripper = []

    act_eef_pos = []
    act_eef_quat = []
    act_gripper = []

    debug_dir = f"{output_path}/debug"
    os.makedirs(debug_dir, exist_ok=True)
    for i in range(len(eps_idx)):
        out_path = f"{output_path}/episode_{eps_idx[i]:04d}"
        os.makedirs(out_path, exist_ok=True)
        cam_path = os.path.join(out_path, "camera_1", "rgb")
        os.makedirs(cam_path, exist_ok=True)

    timestep = 0
    # simulate environment
    # note: We simplified the logic in rl-games player.py (:func:`BasePlayer.run()`) function in an
    #   attempt to have complete control over environment stepping. However, this removes other
    #   operations such as masking that is used for multi-agent learning by RL-Games.
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            eef_real_pos = real_eef_pos_targets[timestep]
            eef_quat = real_quat_targets[timestep]
            gripper_pos = real_gripper_targets[timestep]
            eef_sim_pos = torch_utils.tf_combine(
                eef_quat,
                eef_real_pos,
                torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=env.device).repeat(len(eps_idx),1),
                torch.tensor([[0.0, 0.0, 0.23]], device=env.device).repeat(len(eps_idx),1)
            )[1]
            actions = torch.cat([eef_sim_pos, eef_quat, gripper_pos], dim=-1)

            obs_eef = torch_utils.tf_combine(
                obs[:,3:7],
                obs[:,:3],
                torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=env.device).repeat(len(eps_idx),1),
                torch.tensor([[0.0, 0.0, -0.23]], device=env.device).repeat(len(eps_idx),1)
            )[1]

            obs_eef_pos.append(obs_eef.cpu().numpy())
            obs_eef_quat.append(obs[:,3:7].cpu().numpy())

            act_eef_pos.append(eef_real_pos.cpu().numpy())
            act_eef_quat.append(eef_quat.cpu().numpy())

            for i in range(len(eps_idx)):
                if not valid_mask[timestep, i]:
                    continue  # skip padded timesteps
                cam_path = f"{output_path}/episode_{i:04d}/camera_1/rgb"
                img = env.unwrapped.front_rgb[i].cpu().numpy()
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(cam_path, f"{timestep:06d}.jpg"), img_bgr)

            if len(eps_idx) == 1:
                print("------------ Step Info (single env) -----------")
                print("Currently at timestep:", timestep, "/", T)
                print("curr task space pose:", obs[:,:7].cpu().numpy())
                print("goal task space pose:", actions.cpu().numpy())
                print("---------------------------------")
            else:
                print("------------ Step Info (multi env) -----------")
                print("Currently at timestep:", timestep, "/", T)
                print("pos err:", torch.norm(obs[:,:3]-actions[:,:3], dim=-1).cpu().numpy())
                print("rot err:", quat_geodesic_angle(obs[:,3:7], actions[:,3:7]).cpu().numpy())
                # print("grip err:", torch.abs(obs[:,7:8]-actions[:,7:8]).cpu().numpy())
                print("---------------------------------")

            # env stepping
            obs, _, dones, _ = env.step(actions)
            obs = obs["obs"]
            timestep += 1

            if timestep >= T:
                break

        if args_cli.video:
            # exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()

    arr_obs_pos  = np.array(obs_eef_pos)        # (T, N, 3)
    arr_obs_quat = np.array(obs_eef_quat)       # (T, N, 4)
    arr_act_pos  = np.array(act_eef_pos)        # (T, N, 3)
    arr_act_quat = np.array(act_eef_quat)       # (T, N, 4)

    out_data = {}
    for i, ep in enumerate(eps_idx):
        k = f"episode_{ep:04d}"
        out_data[f"{k}/obs.eef_pos"]    = arr_obs_pos[:,  i, :]
        out_data[f"{k}/obs.eef_quat"]   = arr_obs_quat[:, i, :]
        out_data[f"{k}/action.eef_pos"] = arr_act_pos[:,  i, :]
        out_data[f"{k}/action.eef_quat"]= arr_act_quat[:, i, :]

    np.savez_compressed(os.path.join(debug_dir, "robot_trajectories.npz"), **out_data)
    print(f"[INFO] Saved simulated trajectories to: {os.path.join(debug_dir,'robot_trajectories.npz')}")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
