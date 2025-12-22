# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to visualize failure cases by running play.py and storing all images for one episode per environment."""

"""Launch Isaac Sim Simulator first."""

import argparse
import json
import sys
import cv2
import os

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Visualize failure cases by running an RL agent and storing images.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rl_games_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--use_last_checkpoint",
    action="store_true",
    help="When no checkpoint provided, use the last saved model. Otherwise use the best saved model.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--output_name", type=str, default="no_name", help="Name of the output directory to save images.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
# always enable cameras to record video
args_cli, hydra_args = parser.parse_known_args()
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
import random
import time
import torch
import numpy as np

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

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)
def draw_trail(img, points, color="red", radius=3):
    """
    img: (H,W,3) BGR uint8 image
    points: iterable of 5 (u,v) points (float or int)
    color: "red" or "purple"
    """

    if color == "red":
        # BGR gradient: yellow -> orange -> red
        palette = [
            np.array([0, 255, 255], dtype=np.uint8),   # yellow
            np.array([0, 200, 255], dtype=np.uint8),   # yellow-orange
            np.array([0, 140, 255], dtype=np.uint8),   # orange
            np.array([0, 80, 220],  dtype=np.uint8),   # orange-red
            np.array([0, 0, 200],   dtype=np.uint8),   # red
        ]

    elif color == "purple":
        # BGR gradient: light ocean blue -> blue -> purple
        palette = [
            np.array([255, 220, 160], dtype=np.uint8), # light ocean blue
            np.array([255, 170, 120], dtype=np.uint8), # cyan-blue
            np.array([200, 100, 80],  dtype=np.uint8), # blue
            np.array([160, 60, 120],  dtype=np.uint8), # blue-purple
            np.array([120, 0, 160],   dtype=np.uint8), # purple
        ]

    else:
        raise ValueError("color must be 'red' or 'purple'")

    for (u, v), col in zip(points, palette):
        if u < 0 or v < 0:
            continue
        cv2.circle(img, (int(u), int(v)), radius, col.tolist(), thickness=-1)

    return img

def project_points(points_cam, K):
    """
    points_cam: (N,3) array of 3D points in camera frame
    K: (3,3) camera intrinsics matrix

    returns:
        points_img: (N,2) pixel coordinates
    """
    X = points_cam[:, 0]
    Y = points_cam[:, 1]
    Z = points_cam[:, 2]

    # avoid divide-by-zero
    Z = np.clip(Z, 1e-6, None)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    u = fx * X / Z + cx
    v = fy * Y / Z + cy

    return np.stack([u, v], axis=1)

def transform_points(T, p):  # T: (4,4), p: (N,3)
    p_h = np.concatenate([p, np.ones((p.shape[0], 1))], axis=1)  # (N,4)
    out = (T @ p_h.T).T                                          # (N,4)
    return out[:, :3]


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Play with RL-Games agent and store images for one episode per environment."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.env_options.vis_options = {
        "action_goals": False,      # red, blue, green triangle
        "training_data": False,     # yellow + purple
        "rewards": False,           # pink circles/shapes
        "object_obs": False,        # coordinate frames
        "failed_envs": False,      # red tint
        "eval_mode": True,        
    }
    env_cfg.env_options.verbose = False
    env_cfg.env_options.enable_cameras = True

    intr_mat = env_cfg.intr
    intr_mat = np.array(intr_mat).reshape(3, 3)

    extr_mat = env_cfg.extr
    extr_mat = np.array(extr_mat).reshape(4, 4)

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    agent_cfg["params"]["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["params"]["seed"]
    # set the environment seed (after multi-gpu config for updated rank from agent seed)
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg["params"]["seed"]

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rl_games", agent_cfg["params"]["config"]["name"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # find checkpoint
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rl_games", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint is None:
        # specify directory for logging runs
        run_dir = agent_cfg["params"]["config"].get("full_experiment_name", ".*")
        # specify name of checkpoint
        if args_cli.use_last_checkpoint:
            checkpoint_file = ".*"
        else:
            # this loads the best checkpoint
            checkpoint_file = f"{agent_cfg['params']['config']['name']}.pth"
        # get path to previous checkpoint
        resume_path = get_checkpoint_path(log_root_path, run_dir, checkpoint_file, other_dirs=["nn"])
    else:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    log_dir = os.path.dirname(os.path.dirname(resume_path))

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

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

    # load previously trained model
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = resume_path
    print(f"[INFO]: Loading model checkpoint from: {agent_cfg['params']['load_path']}")

    # set number of actors into agent config
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    # create runner from rl-games
    runner = Runner()
    runner.load(agent_cfg)
    # obtain the agent from the runner
    agent: BasePlayer = runner.create_player()
    agent.restore(resume_path)
    agent.reset()

    dt = env.unwrapped.step_dt

    # Create output directory
    output_path = os.path.abspath(f"logs/visualize_failure_cases/{args_cli.output_name}")
    os.makedirs(output_path, exist_ok=True)
    print(f"[INFO] Saving images to: {output_path}")

    # Track episode completion per environment
    num_envs = env.unwrapped.num_envs
    episode_done = torch.zeros(num_envs, dtype=torch.bool, device=env.unwrapped.device)
    timesteps = [0] * num_envs
    # Track ep_succeeded flags for each episode (captured when episode completes)
    ep_succeeded_dict = {}
    
    # Get camera attributes from environment
    env_unwrapped = env.unwrapped

    # Create directories for each environment episode
    for env_id in range(num_envs):
        episode_dir = os.path.join(output_path, f"episode_{env_id:04d}")
        cam_dir = os.path.join(episode_dir, f"camera_0", "rgb")
        os.makedirs(cam_dir, exist_ok=True)

    # reset environment
    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    timestep = 0
    # required: enables the flag for batched observations
    _ = agent.get_batch_size(obs, 1)
    # initialize RNN states if used
    if agent.is_rnn:
        agent.init_rnn()
    
    # simulate environment
    # note: We simplified the logic in rl-games player.py (:func:`BasePlayer.run()`) function in an
    #   attempt to have complete control over environment stepping. However, this removes other
    #   operations such as masking that is used for multi-agent learning by RL-Games.
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # convert obs to agent format
            obs = agent.obs_to_torch(obs)
            # agent stepping
            actions = agent.get_action(obs, is_deterministic=agent.is_deterministic)
            # env stepping
            obs, _, dones, _ = env.step(actions)
            img_tensor = env.unwrapped.front_rgb # (num_envs, H, W, C) tensor

            base_last5 = env_unwrapped.base_last5.get_points().cpu().numpy().reshape(5 * env_unwrapped.num_envs, 3)
            base_last5 = transform_points(np.linalg.inv(extr_mat), base_last5)
            base_last5_2d = project_points(base_last5, intr_mat).reshape(5, env_unwrapped.num_envs, 2)
            residual_last5 = env_unwrapped.residual_last5.get_points().cpu().numpy().reshape(5 * env_unwrapped.num_envs, 3)
            residual_last5 = transform_points(np.linalg.inv(extr_mat), residual_last5)
            residual_last5_2d = project_points(residual_last5, intr_mat).reshape(5, env_unwrapped.num_envs, 2)

            # Store images for environments that haven't completed their episode
            for env_id in range(num_envs):
                # Check done status FIRST
                done = dones[env_id].item()
                
                if done:
                    # Capture ep_succeeded flag before marking as done
                    ep_succeeded = bool(env_unwrapped.ep_succeeded[env_id].item())
                    ep_succeeded_dict[f"episode_{env_id:04d}"] = ep_succeeded
                    episode_done[env_id] = True
                    
                    # Apply success/failure filter to the last saved image
                    if timesteps[env_id] > 0:
                        last_img_path = os.path.join(output_path, f"episode_{env_id:04d}", 
                                                     f"camera_0", "rgb", f"{timesteps[env_id]-1:06d}.jpg")
                        if os.path.exists(last_img_path):
                            last_img = cv2.imread(last_img_path)
                            if last_img is not None:
                                # Create color overlay (light green for success, light red for failure)
                                overlay = last_img.copy()
                                color = (100, 255, 100) if ep_succeeded else (100, 100, 255)  # BGR format
                                overlay[:] = color
                                # Blend: 85% original, 15% color filter
                                filtered = cv2.addWeighted(last_img, 0.85, overlay, 0.15, 0)
                                cv2.imwrite(last_img_path, filtered)
                    
                    print(f"[INFO] Episode {env_id} completed at timestep {timesteps[env_id]}, succeeded: {ep_succeeded}")
                    continue  # Skip saving this frame (it's the reset frame)

                if not episode_done[env_id]:
                    # Save images for this environment (before checking done)
                    episode_dir = os.path.join(output_path, f"episode_{env_id:04d}")
                    cam_dir = os.path.join(episode_dir, f"camera_0", "rgb")
                    img_path = os.path.join(cam_dir, f"{timesteps[env_id]:06d}.jpg")

                    img = img_tensor[env_id].cpu().numpy()
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    img_labeled = draw_trail(img_bgr, base_last5_2d[:, env_id], color="purple", radius=3)
                    img_labeled = draw_trail(img_labeled, residual_last5_2d[:, env_id], color="red", radius=3)
                    
                    cv2.imwrite(img_path, img_labeled)

                    timesteps[env_id] += 1
                        
            # Check if all episodes are done
            if torch.all(episode_done):
                print("[INFO] All episodes completed.")
                break

    # Ensure all episodes are in the dict (in case some didn't complete)
    for env_id in range(num_envs):
        if f"episode_{env_id:04d}" not in ep_succeeded_dict:
            print(f"[WARNING] Episode {env_id} did not complete. Setting ep_succeeded to False.")
            ep_succeeded_dict[f"episode_{env_id:04d}"] = False
    
    # Save meta.json
    meta_path = os.path.join(output_path, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(ep_succeeded_dict, f, indent=2)
    print(f"[INFO] Saved metadata to: {meta_path}")
    print(f"[INFO] Success rate: {sum(ep_succeeded_dict.values())}/{len(ep_succeeded_dict)}")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
