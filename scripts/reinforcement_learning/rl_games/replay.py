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
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rl_games_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--out", type=str, required=True, help="Output directory to save the simulated data.")
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
from isaaclab.utils.math import (
    apply_delta_pose,
    combine_frame_transforms,
    compute_pose_error,
    matrix_from_quat,
    quat_mul,
    quat_from_matrix,
    subtract_frame_transforms,
)

from huggingface_hub import hf_hub_download
from pathlib import Path

# PLACEHOLDER: Extension template (do not remove this comment)
def load_npz_dict(path):
    data = np.load(path, allow_pickle=True)
    return (
        data["data"].item() if "data" in data.files
        else {k: data[k].item() for k in data.files}
    )

def resolve_hf_file(repo_id: str, filename: str, repo_type = "dataset", revision: str | None = None) -> str:
    p = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type=repo_type,
        revision=revision,
    )
    return str(Path(p))

@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Play with RL-Games agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # load traj & object data
    output_path = os.path.join(args_cli.out, "sim_replay")
    teleop_data = np.load(resolve_hf_file(env_cfg.task.hf_repo, env_cfg.task.train_data_hf_file), allow_pickle=True).item()

    num_envs = len(teleop_data)
    eps_idx = list(range(num_envs))

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.env_options.ctrl_dmr = False
    env_cfg.env_options.obs_dmr = False
    env_cfg.env_options.data_aug = False
    env_cfg.env_options.step_eps = False
    env_cfg.env_options.offline_base = True
    env_cfg.env_options.measure_force = True
    env_cfg.env_options.enable_cameras = True
    env_cfg.env_options.verbose = True

    lengths = [len(teleop_data[f"episode_{e:04d}"]["obs.gripper"]) for e in eps_idx]
    horizon = env_cfg.episode_length_s * 15

    T = min(max(lengths), horizon-1)

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

    # reset environment
    obs = env.reset()
    obs = obs["obs"]

    actions = torch.zeros((env.unwrapped.num_envs, env_cfg.residual_action_space), device=rl_device)

    N = len(eps_idx)

    obs_fingertip_pos  = [[] for _ in range(N)]
    obs_fingertip_quat = [[] for _ in range(N)]
    obs_gripper  = [[] for _ in range(N)]

    obs_fingertip_pos_rel_fixed  = [[] for _ in range(N)]
    obs_fingertip_pos_rel_held = [[] for _ in range(N)]
    obs_ee_linvel_fd = [[] for _ in range(N)]
    obs_ee_angvel_fd = [[] for _ in range(N)]

    act_fingertip_pos  = [[] for _ in range(N)]
    act_fingertip_quat = [[] for _ in range(N)]
    act_gripper  = [[] for _ in range(N)]

    for i in range(len(eps_idx)):
        out_path = f"{output_path}/episode_{eps_idx[i]:04d}"
        os.makedirs(out_path, exist_ok=True)
        cam_path = os.path.join(out_path, "camera_0", "rgb")
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

            base_actions = env.unwrapped.base_actions
            for i in range(len(eps_idx)):
                if timestep >= lengths[i]:
                    continue
                cam_path = f"{output_path}/episode_{eps_idx[i]:04d}/camera_0/rgb"
                img = env.unwrapped.front_rgb[i].cpu().numpy()
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                ok = cv2.imwrite(os.path.join(cam_path, f"{timestep:06d}.jpg"), img_bgr)
                if not ok:
                    print(f"[ERROR] imwrite failed: ep={eps_idx[i]} t={timestep} path={cam_path}")

                obs_fingertip_pos[i].append(obs[i, :3].cpu().numpy())
                obs_fingertip_quat[i].append(obs[i, 3:7].cpu().numpy())
                obs_gripper[i].append(obs[i, 7:8].cpu().numpy())

                obs_fingertip_pos_rel_fixed[i].append(obs[i, 8:11].cpu().numpy())
                obs_fingertip_pos_rel_held[i].append(obs[i, 11:14].cpu().numpy())

                obs_ee_linvel_fd[i].append(obs[i, 14:17].cpu().numpy())
                obs_ee_angvel_fd[i].append(obs[i, 17:20].cpu().numpy())

                act_fingertip_pos[i].append(obs[i, 20:23].cpu().numpy())
                if env_cfg.task.name == "nut_thread":
                    act_fingertip_quat[i].append(base_actions[i, 3:7].cpu().numpy())
                else:
                    act_fingertip_quat[i].append(obs[i, 23:27].cpu().numpy())
                act_gripper[i].append(base_actions[i, 7:8].cpu().numpy())

            if len(eps_idx) == 1:
                print("------------ Step Info (single env) -----------")
                print("Currently at timestep:", timestep, "/", T)
                print("curr task space pose:", obs[:,:7].cpu().numpy())
                print("goal task space pose:", actions.cpu().numpy())
                print("---------------------------------")
            else:
                print("------------ Step Info (multi env) -----------")
                print("Currently at timestep:", timestep, "/", T)
                # print("pos err:", torch.norm(obs[:,:3]-actions[:,:3], dim=-1).cpu().numpy())
                # print("rot err:", quat_geodesic_angle(obs[:,3:7], actions[:,3:7]).cpu().numpy())
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

    out_data = {}
    for i, ep in enumerate(eps_idx):
        k = f"episode_{ep:04d}"
        out_data[k] = {
            "obs.fingertip_pos":     np.stack(obs_fingertip_pos[i], axis=0),      # (Li, 3)
            "obs.fingertip_quat":    np.stack(obs_fingertip_quat[i], axis=0),     # (Li, 4)
            "obs.gripper":     np.stack(obs_gripper[i], axis=0),      # (Li, 1)
            "obs.fingertip_pos_rel_fixed":  np.stack(obs_fingertip_pos_rel_fixed[i], axis=0),      # (Li, 3)
            "obs.fingertip_pos_rel_held":  np.stack(obs_fingertip_pos_rel_held[i], axis=0),      # (Li, 3)
            "obs.ee_linvel_fd":  np.stack(obs_ee_linvel_fd[i], axis=0),      # (Li, 3)
            "obs.ee_angvel_fd":  np.stack(obs_ee_angvel_fd[i], axis=0),      # (Li, 3)
            "action.fingertip_pos":  np.stack(act_fingertip_pos[i], axis=0),      # (Li, 3)
            "action.fingertip_quat": np.stack(act_fingertip_quat[i], axis=0),     # (Li, 4)
            "action.gripper":  np.stack(act_gripper[i], axis=0),      # (Li, 1)
        }

    os.makedirs(os.path.join(args_cli.out, "data"), exist_ok=True)
    np.save(os.path.join(args_cli.out, "data", "sim_replay_data.npy"), out_data)
    print(f"[INFO] Saved simulated trajectories to: {os.path.join(args_cli.out, 'data', 'sim_replay_data.npy')}")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
