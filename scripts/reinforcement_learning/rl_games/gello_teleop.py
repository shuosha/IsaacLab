# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RL-Games."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from RL-Games.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
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
import cv2

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

from pynput import keyboard
from keyboard_agent import KeyboardTeleop

sys.path.insert(0, "scripts/reinforcement_learning/gello")
from agents.gello_agent import GelloAgent, DynamixelRobotConfig
from listener import GelloListener
from kinematics_utils import KinHelper

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

key_states = {
    "r": False,
    "p": False
}

def on_press(key):
        try:
            key_char = key.char.lower() if key.char else key.char
            if key_char in key_states:
                key_states[key_char] = True
        except AttributeError:
            pass

def on_release(key):
    try:
        key_char = key.char.lower() if key.char else key.char
        if key_char in key_states:
            key_states[key_char] = False
    except AttributeError:
        if key == keyboard.Key.esc:
            return False

@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Play with RL-Games agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.teleop_mode = True
    env_cfg.visualize_markers = True

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

    # reset environment
    obs = env.reset()

    gello_listener = GelloListener(gello_port='/dev/ttyUSB0')
    gello_listener.start()
    keyboard_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    keyboard_listener.start()
    kin_helper = KinHelper(robot_name="xarm7")

    # set to initial pose
    obs = env.unwrapped._get_observations()
    obs = obs["policy"]

    obs_eef_pos = []
    obs_eef_quat = []
    obs_gripper = []

    act_eef_pos = []
    act_eef_quat = []
    act_gripper = []

    name = "gearmesh_old_policy"
    out_path = f"logs/teleop/{name}"
    os.makedirs(out_path, exist_ok=True)
    cam_path = os.path.join(out_path, "camera_1", "rgb")
    os.makedirs(cam_path, exist_ok=True)

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
    use_policy = False
    while simulation_app.is_running():
        t_loop_start = time.time()

        with torch.inference_mode():
            # --- checkpoint A ---
            t0 = time.time()

            obs = agent.obs_to_torch(obs)

            qpos = gello_listener.get()
            fk = kin_helper.compute_fk_sapien_links(qpos[:7], [kin_helper.sapien_eef_idx])[0]
            pos = torch.from_numpy(fk[:3, 3]).unsqueeze(0).to(env.device)
            rot_mat = torch.from_numpy(fk[:3, :3]).unsqueeze(0).to(env.device)
            quat = quat_from_matrix(rot_mat)
            gripper = torch.tensor([[qpos[-1] * 1.6]], device=env.device)

            pos = torch_utils.tf_combine(
                quat, pos,
                torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=env.device),
                torch.tensor([[0.0, 0.0, 0.17]], device=env.device),
            )[1]

            base_state = torch.cat([pos, quat, gripper], dim=-1).to(torch.float32)

            curr_state = obs[:, :8].clone()
            curr_state[:, :3] += env.unwrapped.scene.env_origins[:, :3]

            # --- checkpoint B ---
            t1 = time.time()

            if key_states["r"]:
                print("[INFO] Resetting environment.")
                obs = env.unwrapped._reset_idx(torch.arange(env.unwrapped.num_envs).to(env.device))
                obs = env.unwrapped._get_observations()["policy"]
                timestep = 0
                continue

            if key_states["p"]:
                use_policy = not use_policy
                print(f"[INFO] Toggling policy usage to: {use_policy}")
                time.sleep(0.5)

            env.unwrapped.base_actions = base_state

            # agent stepping
            actions = agent.get_action(obs, is_deterministic=agent.is_deterministic) * float(use_policy)

            # store logs
            obs_eef_pos.append(obs[:, :3].cpu().numpy())
            obs_eef_quat.append(obs[:, 3:7].cpu().numpy())
            act_eef_pos.append(actions[:, :3].cpu().numpy())
            act_eef_quat.append(actions[:, 3:7].cpu().numpy())

            # --- checkpoint C ---
            t2 = time.time()

            # env step
            obs, _, dones, _ = env.step(actions)
            timestep += 1

            # --- checkpoint D ---
            t3 = time.time()

        # --- print timing ---
        total = t3 - t_loop_start
        # print(
        #     f"[t={t_loop_start:.3f}] "
        #     f"obs+fk={(t1 - t0)*1000:.2f}ms | "
        #     f"policy={(t2 - t1)*1000:.2f}ms | "
        #     f"env.step={(t3 - t2)*1000:.2f}ms | "
        #     f"loop={total*1000:.2f}ms ({1.0/total:.1f} Hz)"
        # )

        # stop after one video
        if args_cli.video and timestep == args_cli.video_length:
            break

        # try real-time pacing
        sleep_time = dt - (time.time() - t_loop_start)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)


    # close the simulator
    env.close()

    arr_obs_pos  = np.array(obs_eef_pos)        # (T, N, 3)
    arr_obs_quat = np.array(obs_eef_quat)       # (T, N, 4)
    arr_act_pos  = np.array(act_eef_pos)        # (T, N, 3)
    arr_act_quat = np.array(act_eef_quat)       # (T, N, 4)

    out_data = {}
    k = name
    out_data[f"{k}/obs.eef_pos"]    = arr_obs_pos[:,  0, :]
    out_data[f"{k}/obs.eef_quat"]   = arr_obs_quat[:, 0, :]
    out_data[f"{k}/action.eef_pos"] = arr_act_pos[:,  0, :]
    out_data[f"{k}/action.eef_quat"]= arr_act_quat[:, 0, :]

    robot_path = os.path.join(out_path, "robot_data")
    os.makedirs(robot_path, exist_ok=True)
    np.savez_compressed(os.path.join(robot_path, "sim_trajs.npz"), **out_data)
    print(f"[INFO] Saved simulated trajectories to: {os.path.join(robot_path,'sim_trajs.npz')}")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
