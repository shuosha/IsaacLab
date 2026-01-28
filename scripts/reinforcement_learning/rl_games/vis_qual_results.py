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
parser.add_argument("--base", choices=["nn", "bc", "noisy_nn"], default="nn", help="Base model type: nn (neural network) or bc (behavior cloning).")
parser.add_argument("--base-only", action="store_true", default=False, help="If 1, use only base model actions without residuals.")
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

# ----------------------------
# Clean overlay helpers (replace draw_trail + dot logic)
# ----------------------------
import numpy as np
import cv2
import math

# If your quats are (x,y,z,w), set this to "xyzw"
QUAT_ORDER = "wxyz"

def wrap_to_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi

def quat_to_yaw_np(q: np.ndarray, order: str = "wxyz") -> float:
    """
    q: shape (4,), yaw-only is assumed but works generally.
    Returns yaw in radians in [-pi, pi].
    """
    q = np.asarray(q, dtype=np.float64).reshape(4,)
    if order == "wxyz":
        w, x, y, z = q
    elif order == "xyzw":
        x, y, z, w = q
    else:
        raise ValueError("order must be 'wxyz' or 'xyzw'")

    # yaw (Z) from quaternion
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return wrap_to_pi(math.atan2(siny_cosp, cosy_cosp))

def project_base_points_to_uv(points_base: np.ndarray, intr_mat: np.ndarray, extr_mat: np.ndarray):
    """
    points_base: (N,3) in base frame
    intr_mat: (3,3)
    extr_mat: (4,4) cam2base  (same as your env_cfg.extr)
    Returns:
      uvs: (N,2) float
      valid: (N,) bool (Z>0 in cam frame)
    """
    # base -> cam
    T_base_cam = np.linalg.inv(extr_mat)
    pts_cam = transform_points(T_base_cam, points_base)  # (N,3)
    valid = pts_cam[:, 2] > 1e-6
    uvs = project_points(pts_cam, intr_mat)              # (N,2)
    return uvs, valid

def draw_line_or_arrow(img, p0, p1, bgr, thickness=2, arrow=True):
    p0 = (int(round(p0[0])), int(round(p0[1])))
    p1 = (int(round(p1[0])), int(round(p1[1])))
    if arrow:
        cv2.arrowedLine(img, p0, p1, bgr, thickness=thickness, tipLength=0.15)
    else:
        cv2.line(img, p0, p1, bgr, thickness=thickness)
    return img

def clamp_point(pt, W, H, margin=2):
    x = int(np.clip(pt[0], margin, W - 1 - margin))
    y = int(np.clip(pt[1], margin, H - 1 - margin))
    return (x, y)

def draw_triangle_overlay(img_bgr, curr_uv, base_uv, net_uv, arrow=True):
    """
    Edges:
      curr -> base (blue)
      base -> net  (red)   [this is your "base -> res" edge]
      curr -> net  (green)
    """
    BLUE  = (255, 0, 0)
    RED   = (0, 0, 255)
    GREEN = (0, 255, 0)

    draw_line_or_arrow(img_bgr, curr_uv, base_uv, BLUE,  thickness=2, arrow=arrow)
    draw_line_or_arrow(img_bgr, base_uv, net_uv,  RED,   thickness=2, arrow=arrow)
    draw_line_or_arrow(img_bgr, curr_uv, net_uv,  GREEN, thickness=2, arrow=arrow)
    return img_bgr

def draw_yaw_overlay_3d(
    img_bgr,
    curr_pos_base,         # (3,) in base frame
    base_yaw,              # rad
    env_yaw,               # rad
    intr_mat,              # (3,3)
    extr_mat,              # (4,4) cam2base
    r_m=0.03,              # circle radius in meters (tune)
    thickness=2,
    alpha=0.6,
    blur_ksize=5,
):
    """
    Draw yaw circle in 3D:
      - Circle plane: base XY (perp to base Z)
      - Center: curr_pos_base
      - Blue arrow: base_yaw direction on circle
      - Green arrow: env_yaw direction on circle
      - Red arc: shortest delta base->env along the circle
    """
    H, W = img_bgr.shape[:2]

    # --- Build 3D points on base XY plane ---
    center = np.asarray(curr_pos_base, dtype=np.float64).reshape(3,)
    ex = np.array([1.0, 0.0, 0.0], dtype=np.float64)  # base +X
    ey = np.array([0.0, 1.0, 0.0], dtype=np.float64)  # base +Y

    def pt_on_circle(theta):
        return center + r_m * (math.cos(theta) * ex + math.sin(theta) * ey)

    # Circle polyline points
    n_circle = 80
    thetas = np.linspace(0.0, 2.0 * math.pi, n_circle, endpoint=True)
    circle_pts_base = np.stack([pt_on_circle(t) for t in thetas], axis=0)  # (N,3)

    # Direction endpoints for arrows
    p_base_dir = pt_on_circle(base_yaw)
    p_env_dir  = pt_on_circle(env_yaw)

    # Arc points (shortest delta from base->env)
    delta = wrap_to_pi(env_yaw - base_yaw)
    n_arc = 40
    arc_thetas = np.linspace(base_yaw, base_yaw + delta, n_arc, endpoint=True)
    arc_pts_base = np.stack([pt_on_circle(t) for t in arc_thetas], axis=0)

    # Project all needed points (base->cam->uv)
    all_pts_base = np.concatenate(
        [center[None, :], circle_pts_base, p_base_dir[None, :], p_env_dir[None, :], arc_pts_base],
        axis=0,
    )
    uvs, valid = project_base_points_to_uv(all_pts_base, intr_mat, extr_mat)
    if not bool(valid.all()):
        return img_bgr

    # Unpack projected uvs
    idx = 0
    center_uv = uvs[idx]; idx += 1
    circle_uvs = uvs[idx:idx + n_circle]; idx += n_circle
    base_uv = uvs[idx]; idx += 1
    env_uv  = uvs[idx]; idx += 1
    arc_uvs = uvs[idx:idx + n_arc]; idx += n_arc

    # --- Soft/alpha drawing: draw on overlay, optionally blur, then blend ---
    overlay = np.zeros_like(img_bgr)

    RED   = (0, 0, 255)
    GRAY  = (200, 200, 200)

    # Draw circle
    circle_poly = np.round(circle_uvs).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(overlay, [circle_poly], isClosed=False, color=GRAY, thickness=1)

    # Draw curved arrow along the circumference for delta (base -> env)
    arc_poly = np.round(arc_uvs).astype(np.int32).reshape(-1, 1, 2)

    # 1) draw the arc itself
    cv2.polylines(overlay, [arc_poly], isClosed=False, color=RED, thickness=thickness)

    # 2) arrowhead aligned with arc tangent
    if n_arc >= 2:
        p_prev = arc_poly[-2, 0].astype(np.float32)
        p_last = arc_poly[-1, 0].astype(np.float32)

        v = p_last - p_prev
        n = np.linalg.norm(v)
        if n > 1e-6:
            v /= n
            head_len = 18.0  # pixels
            p_head_start = (p_last - v * head_len).astype(np.int32)
            p_head_end   = p_last.astype(np.int32)

            cv2.arrowedLine(
                overlay,
                tuple(p_head_start.tolist()),
                tuple(p_head_end.tolist()),
                RED,
                thickness=thickness,
                tipLength=0.6,
                line_type=cv2.LINE_AA,
            )

    # Optional: mark arc start/end
    cv2.circle(overlay, tuple(arc_poly[0, 0].tolist()), 3, RED, -1, lineType=cv2.LINE_AA)
    cv2.circle(overlay, tuple(arc_poly[-1, 0].tolist()), 3, RED, -1, lineType=cv2.LINE_AA)

    # Blur overlay slightly (for “soft” arrows)
    # if blur_ksize and blur_ksize > 1:
    #     if blur_ksize % 2 == 0:
    #         blur_ksize += 1
    #     overlay = cv2.GaussianBlur(overlay, (blur_ksize, blur_ksize), 1.2)

    # Alpha blend where overlay is nonzero
    mask = (overlay.sum(axis=2) > 0)
    out = img_bgr.copy()
    out[mask] = ((1.0 - alpha) * img_bgr[mask] + alpha * overlay[mask]).astype(np.uint8)
    return out


def draw_task_overlay(img_bgr, env_unwrapped, env_id: int, intr_mat: np.ndarray, extr_mat: np.ndarray):
    task_name = getattr(getattr(env_unwrapped, "cfg_task", None), "name", "")

    base_7 = env_unwrapped.base_actions[env_id, :7].detach().cpu().numpy()
    net_7  = env_unwrapped.env_actions[env_id, :7].detach().cpu().numpy()

    curr_pos = env_unwrapped.fingertip_midpoint_pos[env_id, :3].detach().cpu().numpy()
    curr_q   = env_unwrapped.fingertip_midpoint_quat[env_id, :4].detach().cpu().numpy()

    base_pos = base_7[:3]
    base_q   = base_7[3:7]
    net_pos  = net_7[:3]

    # nut_thread: ONLY yaw overlay (3D circle centered at curr_pos)
    if task_name == "nut_thread":
        base_yaw = quat_to_yaw_np(base_q, order=QUAT_ORDER)
        env_yaw  = quat_to_yaw_np(curr_q, order=QUAT_ORDER)
        return draw_yaw_overlay_3d(
            img_bgr,
            curr_pos_base=curr_pos,
            base_yaw=base_yaw,
            env_yaw=env_yaw,
            intr_mat=intr_mat,
            extr_mat=extr_mat,
            r_m=0.03,        # tune
            alpha=0.6,       # tune
            blur_ksize=5,    # tune
        )

    # peg_insert / gear_mesh: triangle overlay (unchanged)
    if task_name in ("peg_insert", "gear_mesh"):
        pts_base = np.stack([curr_pos, base_pos, net_pos], axis=0)
        uvs, valid = project_base_points_to_uv(pts_base, intr_mat, extr_mat)
        if bool(valid.all()):
            curr_uv, base_uv, net_uv = uvs[0], uvs[1], uvs[2]
            return draw_triangle_overlay(img_bgr, curr_uv, base_uv, net_uv, arrow=True)

    return img_bgr

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
    }
    env_cfg.env_options.base_model = args_cli.base if args_cli.base is not None else env_cfg.env_options.base_model
    env_cfg.env_options.ctrl_dmr = True
    env_cfg.env_options.obs_dmr = True
    env_cfg.env_options.data_aug = False
    env_cfg.env_options.step_eps = True
    env_cfg.env_options.offline_base = False
    env_cfg.env_options.eval_mode = False
    # env_cfg.env_options.measure_force = True
    env_cfg.env_options.enable_cameras = True
    env_cfg.env_options.verbose = False
    # env_cfg.base_rand.horizon = [1, 1]

    # intr_mat = env_cfg.intr
    # intr_mat = np.array(intr_mat).reshape(3, 3)

    # extr_mat = env_cfg.extr
    # extr_mat = np.array(extr_mat).reshape(4, 4)

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
    if args_cli.base_only:
        output_path = os.path.abspath(f"logs/qual_results/{args_cli.output_name}_{args_cli.base}_only")
    else:
        output_path = os.path.abspath(f"logs/qual_results/{args_cli.output_name}_on_{args_cli.base}_rollout")
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

    # --- camera naming convention: {front,left,right}_{rgb,intr,extr} ---
    CAM_NAMES = ["front"]#, "left", "right"]

    # cache intr/extr per camera (3x3 and 4x4)
    cam_K = {c: np.array(getattr(env_cfg, f"{c}_intr")).reshape(3, 3) for c in CAM_NAMES}
    cam_T = {c: np.array(getattr(env_cfg, f"{c}_extr")).reshape(4, 4) for c in CAM_NAMES}

    # Create directories for each environment episode (3 cameras)
    for env_id in range(num_envs):
        episode_dir = os.path.join(output_path, f"episode_{env_id:04d}")
        for cam_idx, cam in enumerate(CAM_NAMES):
            cam_dir = os.path.join(episode_dir, f"camera_{cam_idx}", "rgb")
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
            actions = agent.get_action(obs, is_deterministic=agent.is_deterministic) * (1 - args_cli.base_only)
            # env stepping
            obs, _, dones, _ = env.step(actions)
            rgb_tensors = {c: getattr(env_unwrapped, f"{c}_rgb") for c in CAM_NAMES}  # (num_envs,H,W,3)

            # base_last5 = env_unwrapped.base_last5.get_points().cpu().numpy().reshape(5 * env_unwrapped.num_envs, 3)
            # base_last5 = transform_points(np.linalg.inv(extr_mat), base_last5)
            # base_last5_2d = project_points(base_last5, intr_mat).reshape(5, env_unwrapped.num_envs, 2)
            # residual_last5 = env_unwrapped.residual_last5.get_points().cpu().numpy().reshape(5 * env_unwrapped.num_envs, 3)
            # residual_last5 = transform_points(np.linalg.inv(extr_mat), residual_last5)
            # residual_last5_2d = project_points(residual_last5, intr_mat).reshape(5, env_unwrapped.num_envs, 2)

            # Store images for environments that haven't completed their episode
            for env_id in range(num_envs):
                # Check done status FIRST; only handle completion once per episode
                done = dones[env_id].item()
                
                if (not episode_done[env_id]) and done:
                    # Capture ep_succeeded flag before marking as done
                    ep_succeeded = bool(env_unwrapped.ep_succeeded[env_id].item())
                    ep_succeeded_dict[f"episode_{env_id:04d}"] = ep_succeeded
                    episode_done[env_id] = True
                    
                    # Apply success/failure filter to the last saved image
                    if timesteps[env_id] > 0:
                        for cam_idx, cam in enumerate(CAM_NAMES):
                            last_img_path = os.path.join(
                                output_path, f"episode_{env_id:04d}",
                                f"camera_{cam_idx}", "rgb", f"{timesteps[env_id]-1:06d}.jpg"
                            )
                            if os.path.exists(last_img_path):
                                last_img = cv2.imread(last_img_path)
                                if last_img is not None:
                                    overlay = last_img.copy()
                                    color = (100, 255, 100) if ep_succeeded else (100, 100, 255)  # BGR
                                    overlay[:] = color
                                    filtered = cv2.addWeighted(last_img, 0.85, overlay, 0.15, 0)
                                    cv2.imwrite(last_img_path, filtered)
                    
                    print(f"[INFO] Episode {env_id} completed at timestep {timesteps[env_id]}, succeeded: {ep_succeeded}")
                    continue  # Skip saving this frame (it's the reset frame)

                if not episode_done[env_id]:
                    episode_dir = os.path.join(output_path, f"episode_{env_id:04d}")

                    for cam_idx, cam in enumerate(CAM_NAMES):
                        cam_dir = os.path.join(episode_dir, f"camera_{cam_idx}", "rgb")
                        img_path = os.path.join(cam_dir, f"{timesteps[env_id]:06d}.jpg")

                        img = rgb_tensors[cam][env_id].cpu().numpy()
                        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                        # per-camera intr/extr
                        if args_cli.base_only:
                            cv2.imwrite(img_path, img_bgr)
                        else:
                            img_labeled = draw_task_overlay(img_bgr, env_unwrapped, env_id, cam_K[cam], cam_T[cam])
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
    meta_path = os.path.join(output_path, "no_meta.json")
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
