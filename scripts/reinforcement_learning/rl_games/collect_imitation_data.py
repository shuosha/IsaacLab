# ----------------------------
# Main (IsaacLab + rl-games)
# ----------------------------
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
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--num_episodes", type=int, required=True)

# launch args
AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()

# Always enable cameras since you want images for debugging
args_cli.enable_cameras = True

# Clear sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# Launch Omniverse
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""


import gymnasium as gym
import math
import random
import time
import torch
import numpy as np
from pathlib import Path
from PIL import Image

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
# ----------------------------
# Data collector
# ----------------------------
class DataCollector:
    """
    Images:
      - always saved for every played episode (debug/visualization)
      - stored under output_dir/episodes_played/episode_{id:04d}/camera_0/rgb/{t:06d}.png

    Robot state/action:
      - buffered in RAM per env
      - flushed only if the episode ends AND success==True
      - stored under output_dir/episodes_collected/episode_{id:04d}/robot/{t:06d}.json

    Meta:
      - append-only jsonl at output_dir/meta.jsonl
      - one line per played episode (success or fail)
    """

    def __init__(self, output_dir: str, num_envs: int):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.num_envs = num_envs

        self.played_root = self.output_dir / "episodes_played"
        self.collected_root = self.output_dir / "episodes_collected"
        self.played_root.mkdir(parents=True, exist_ok=True)
        self.collected_root.mkdir(parents=True, exist_ok=True)

        self.meta_path = self.output_dir / "meta.jsonl"

        # global counters
        self.episodes_played = 0
        self.episodes_collected = 0

        # per-env episode assignment for played episodes
        self.curr_played_id = np.full((num_envs,), -1, dtype=np.int64)

        # per-env buffers for state/action (only flushed on success)
        self.buffers = [[] for _ in range(num_envs)]
        self.t_in_ep = np.zeros((num_envs,), dtype=np.int64)

        # initialize an episode id for each env
        for env_id in range(num_envs):
            self._start_new_played_episode(env_id)

    def _played_episode_dir(self, played_id: int) -> Path:
        return self.played_root / f"episode_{played_id:04d}"

    def _collected_episode_dir(self, collected_id: int) -> Path:
        return self.collected_root / f"episode_{collected_id:04d}"

    def _ensure_played_dirs(self, played_id: int):
        ep_dir = self._played_episode_dir(played_id)
        (ep_dir / "camera_0" / "rgb").mkdir(parents=True, exist_ok=True)

    def _start_new_played_episode(self, env_id: int):
        played_id = int(self.episodes_played)
        self.episodes_played += 1

        self.curr_played_id[env_id] = played_id
        self.buffers[env_id] = []
        self.t_in_ep[env_id] = 0

        self._ensure_played_dirs(played_id)

    def save_step(
        self,
        env_id: int,
        obs_vec: np.ndarray,
        action_vec: np.ndarray,
        img_bgr: np.ndarray,
    ):
        """
        Save one timestep:
          - write image to played episode folder
          - buffer obs/action in RAM
        """
        played_id = int(self.curr_played_id[env_id])
        t = int(self.t_in_ep[env_id])

        # --- Save image for debugging ---
        # expected img_bgr: (H, W, 3) uint8
        img_path = self._played_episode_dir(played_id) / "camera_0" / "rgb" / f"{t:06d}.jpg"
        cv2.imwrite(img_path, img_bgr)

        # --- Buffer state/action (RAM only) ---
        obs_np = np.asarray(obs_vec).reshape(-1)
        act_np = np.asarray(action_vec).reshape(-1)

        self.buffers[env_id].append(
            {
                "obs": obs_np.astype(np.float32, copy=False),
                "action": act_np.astype(np.float32, copy=False),
            }
        )

        self.t_in_ep[env_id] += 1

    def end_episode(self, env_id: int, success: bool):
        """
        Called exactly when dones[env_id] is True.
          - if success: flush robot data to collected_root and increment episodes_collected
          - always: append meta entry
          - always: clear buffer and start a new played episode id for this env
        """
        played_id = int(self.curr_played_id[env_id])
        num_steps = int(self.t_in_ep[env_id])

        collected_id = None
        if success:
            collected_id = int(self.episodes_collected)
            self.episodes_collected += 1

            ep_dir = self._collected_episode_dir(collected_id)
            robot_dir = ep_dir / "robot"
            robot_dir.mkdir(parents=True, exist_ok=True)

            # dump buffered timesteps
            for t, entry in enumerate(self.buffers[env_id]):
                out = {
                    "obs": entry["obs"].tolist(),
                    "action": entry["action"].tolist(),
                }
                with open(robot_dir / f"{t:06d}.json", "w") as f:
                    json.dump(out, f, indent=2)

        # meta append (always)
        meta = {
            "episode_played": played_id,
            "env_id": int(env_id),
            "success": bool(success),
            "num_steps": num_steps,
            "episode_collected": collected_id,  # null if failed
        }
        with open(self.meta_path, "a") as f:
            f.write(json.dumps(meta) + "\n")

        # reset per-env and start next played episode
        self._start_new_played_episode(env_id)

@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
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
    env_cfg.env_options.verbose = False
    env_cfg.env_options.enable_cameras = True

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

    env_unwrapped = env.unwrapped
    num_envs = env_unwrapped.num_envs
    dt = env_unwrapped.step_dt

    # Collector
    collector = DataCollector(args_cli.output_dir, num_envs)

    # Reset env
    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    obs = agent.obs_to_torch(obs)

    # required: enables the flag for batched observations
    _ = agent.get_batch_size(obs, 1)
    if agent.is_rnn:
        agent.init_rnn()

    actions = agent.get_action(obs, is_deterministic=agent.is_deterministic)

    max_episodes = int(args_cli.num_episodes)

    while simulation_app.is_running() and collector.episodes_collected < max_episodes:
        start_time = time.time()

        with torch.inference_mode():
            obs, _, dones, infos = env.step(actions)
            obs = agent.obs_to_torch(obs)
            actions = agent.get_action(obs, is_deterministic=agent.is_deterministic)

            # Pull tensors from unwrapped env
            img_tensor = env_unwrapped.front_rgb          # (N, H, W, 3) uint8
            env_actions_tensor = env_unwrapped.env_actions  # (N, A)
            success_tensor = env_unwrapped.ep_succeeded    # (N,) bool/0-1

            obs_np_all = obs[:, :-14].detach().cpu().numpy() # exclude base action and prev residual action
            act_np_all = env_actions_tensor.detach().cpu().numpy()
            img_np_all = img_tensor.detach().cpu().numpy()
            dones_np = dones.detach().cpu().numpy()
            succ_np = success_tensor.detach().cpu().numpy()

            for env_id in range(num_envs):
                done = bool(dones_np[env_id].item())
                success = bool(succ_np[env_id].item())

                obs_np = obs_np_all[env_id].reshape(-1)
                act_np = act_np_all[env_id].reshape(-1)

                img = img_np_all[env_id]
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                collector.save_step(env_id, obs_np, act_np, img_bgr)

                if done:
                    collector.end_episode(env_id, success)

    print(f"[INFO] Collected {collector.episodes_collected} successful episodes to {args_cli.output_dir}")
    print(f"[INFO] Played {collector.episodes_played} total episodes (images saved for all played episodes).")
    print(f"[INFO] Meta log: {str(Path(args_cli.output_dir) / 'meta.jsonl')}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
