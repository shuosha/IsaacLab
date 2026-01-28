# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RL-Games."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import cv2
import json
from pathlib import Path

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from RL-Games.")
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
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--base", choices=["noisy_nn", "nn", "bc_teleop", "bc_expert", "laggy_bc_expert", "noisy_bc_expert"], default=None, help="Base model type: nn (neural network) or bc (behavior cloning).")
parser.add_argument("--eval_episodes", type=int, default=200, help="Number of evaluation episodes.")
parser.add_argument("--log_path", type=str, default=None, help="Path to save evaluation logs.")

parser.add_argument("--policy_path", type=str, required=True, help="Path to the trained policy checkpoint.")
parser.add_argument("--imitation_only", action="store_true", default=False, help="Use imitation only policy.")
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
from tqdm import tqdm

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

from lerobot.rrl.dp_wrapper import DPWrapper

# PLACEHOLDER: Extension template (do not remove this comment)

def transform_obs(obs):
    """transfer from env flatten tensor to lerobot format"""
    state = torch.cat((obs[:, :8], obs[:, 14:20]), dim=-1) # (batch, 14)
    environment_state = obs[:, 8:14]  # (batch, 6)
    return {
        "observation.state": state,
        "observation.environment_state": environment_state,
    }

def transform_action(action):
    """transfer from lerobot format to env flatten tensor"""

    return {
        "action": action
    }

@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Play with RL-Games agent."""
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
    }
    assert args_cli.base is not None, "Base model type must be specified."
    env_cfg.env_options.base_model = args_cli.base
    env_cfg.env_options.ctrl_dmr = True
    env_cfg.env_options.obs_dmr = True
    env_cfg.env_options.data_aug = True
    env_cfg.env_options.step_eps = True
    env_cfg.env_options.offline_base = False
    # env_cfg.env_options.measure_force = True
    # env_cfg.env_options.enable_cameras = True
    env_cfg.env_options.verbose = False

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

    policy = DPWrapper(args_cli.policy_path)

    # reset environment
    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    actions = obs[:,:8].clone()

    imitation_only = args_cli.imitation_only
    tot_eps = 0
    tot_suc = 0
    tot_rew = 0.0
    terminal_eps = args_cli.eval_episodes
    pbar = tqdm(total=terminal_eps, desc="Rollouts", unit="ep")
    timestep = 0
    ep_return = torch.zeros(env.unwrapped.num_envs, dtype=torch.float32, device=env.device)

    # simulate environment
    # note: We simplified the logic in rl-games player.py (:func:`BasePlayer.run()`) function in an
    #   attempt to have complete control over environment stepping. However, this removes other
    #   operations such as masking that is used for multi-agent learning by RL-Games.
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
                        # base actions
            base_actions = env.unwrapped.base_actions

            # convert obs to agent format
            obs_lerobot = transform_obs(obs)
            if imitation_only:
                actions = policy.act(obs_lerobot)
            else:
                actions = policy.act(obs_lerobot, ref_action=base_actions)
            # env stepping
            obs, rew, dones, _ = env.step(actions)
            if isinstance(obs, dict):
                obs = obs["obs"]
            ep_return += rew # (num_envs, )
            if dones.any():
                policy.reset()
                completed = dones.sum().item()
                succeeded = env.unwrapped.eps_task_succeeded[dones].sum().item()
                rew = ep_return[dones].cpu().numpy()
                ep_return[dones] = 0.0

                tot_eps += completed
                tot_suc += succeeded
                tot_rew += rew.sum()

                update_val = min(completed, max(0, terminal_eps - pbar.n))
                if update_val > 0:
                    pbar.update(update_val)

                success_rate = tot_suc / tot_eps if tot_eps > 0 else 0.0
                pbar.set_postfix({"success_rate": f"{success_rate:.3f}"})

            if tot_eps >= terminal_eps:
                print("final success rate:", tot_suc / tot_eps)
                break
        if args_cli.video:
            timestep += 1
            # exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # # time delay for real-time evaluation
        # sleep_time = dt - (time.time() - start_time)
        # if args_cli.real_time and sleep_time > 0:
        #     time.sleep(sleep_time)

    pbar.close()
    # close the simulator
    env.close()

    if args_cli.log_path is not None:
        out_dict = {}

        # ensure directory exists
        os.makedirs(os.path.dirname(args_cli.log_path), exist_ok=True)

        # Load existing JSON if possible
        if os.path.exists(args_cli.log_path):
            try:
                with open(args_cli.log_path, "r") as f:
                    content = f.read().strip()
                    out_dict = json.loads(content) if content else {}
            except json.JSONDecodeError:
                # File exists but is empty or corrupted/truncated
                print(f"[WARN] log_path not valid JSON (starting fresh): {args_cli.log_path}")
                out_dict = {}
        
        name = Path(args_cli.policy_path).name # GearMesh_bc_expert
        parts = name.split("_")
        task_name = parts[0]
        train_base = "_".join(parts[1:])   # "bc_teleop"
        eval_base = f"{args_cli.base}"
        ckpt_name = f"diffusion_t:{train_base}_e:{eval_base}"
        
        out_dict.setdefault(task_name, {})
        out_dict[task_name][ckpt_name] = {
            "eval_episodes": int(terminal_eps),
            "success_rate": float(tot_suc / tot_eps) if tot_eps > 0 else 0.0,
            "mean_eps_return": float(tot_rew / tot_eps) if tot_eps > 0 else 0.0,
            "base_model": args_cli.base,
        }

        with open(args_cli.log_path, "w") as f:
            json.dump(out_dict, f, indent=2)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
