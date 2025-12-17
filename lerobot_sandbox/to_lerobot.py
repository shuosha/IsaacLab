#!/usr/bin/env python3
import os
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset


# -------------------------------
# JSON readers
# -------------------------------
def read_obs(json_path: str) -> dict[str, torch.Tensor]:
    """Read residual observation from JSON file."""
    data = json.load(open(json_path, 'r'))
    # Residual observation space: fingertip_pos(3) + fingertip_quat(4) + gripper(1) + 
    # fingertip_pos_rel_fixed(3) + fingertip_pos_rel_held(3) + ee_linvel(3) + ee_angvel(3)
    # Total: 20
    fingertip_pos = torch.tensor(data["obs"][:3], dtype=torch.float32)  # (3,)
    fingertip_quat = torch.tensor(data["obs"][3:7], dtype=torch.float32)  # (4,) wxyz
    gripper = torch.tensor([data["obs"][7]], dtype=torch.float32).flatten()  # (1,)
    fingertip_pos_rel_fixed = torch.tensor(data["obs"][8:11], dtype=torch.float32)  # (3,)
    fingertip_pos_rel_held = torch.tensor(data["obs"][11:14], dtype=torch.float32)  # (3,)
    ee_linvel = torch.tensor(data["obs"][14:17], dtype=torch.float32)  # (3,)
    ee_angvel = torch.tensor(data["obs"][17:20], dtype=torch.float32)  # (3,)
    
    state = torch.cat([
        fingertip_pos, fingertip_quat, gripper,
        fingertip_pos_rel_fixed, fingertip_pos_rel_held,
        ee_linvel, ee_angvel
    ], dim=0)  # (20,)
    return {"state": state}


def read_act(json_path: str) -> dict[str, torch.Tensor]:
    """Read residual action from JSON file."""
    data = json.load(open(json_path, 'r'))
    # Residual action space: fingertip_pos(3) + fingertip_quat(4) + gripper(1) = 8D
    fingertip_pos = torch.tensor(data["action"][:3], dtype=torch.float32)  # (3,)
    fingertip_quat = torch.tensor(data["action"][3:7], dtype=torch.float32)  # (4,) wxyz
    gripper = torch.tensor([data["action"][7]], dtype=torch.float32).flatten()  # (1,)
    action = torch.cat([fingertip_pos, fingertip_quat, gripper], dim=0)  # (8,)
    return {"action": action}


# -------------------------------
# Feature schema builders
# -------------------------------
def build_features() -> dict:
    """Build feature schema for residual mode (state only, no images)."""
    return {
        "action": {"dtype": "float32", "shape": (8,),
                "names": {"waypoint": ["fingertip_pos_x", "fingertip_pos_y", "fingertip_pos_z",
                                       "fingertip_quat_x", "fingertip_quat_y", "fingertip_quat_z", "fingertip_quat_w",
                                       "gripper"]}},
        "observation.state": {"dtype": "float32", "shape": (20,),
                            "names": {"waypoint": ["fingertip_pos_x", "fingertip_pos_y", "fingertip_pos_z",
                                                   "fingertip_quat_x", "fingertip_quat_y", "fingertip_quat_z", "fingertip_quat_w",
                                                   "gripper",
                                                   "fingertip_pos_rel_fixed_x", "fingertip_pos_rel_fixed_y", "fingertip_pos_rel_fixed_z",
                                                   "fingertip_pos_rel_held_x", "fingertip_pos_rel_held_y", "fingertip_pos_rel_held_z",
                                                   "ee_linvel_x", "ee_linvel_y", "ee_linvel_z",
                                                   "ee_angvel_x", "ee_angvel_y", "ee_angvel_z"]}},
        "next.done": {"dtype": "bool", "shape": (1,), "names": None},
    }


# -------------------------------
# Converter
# -------------------------------
def convert_to_lerobot(input_dir: str, repo_id: str, fps: int, robot_type: str,
                       task_name: str) -> None:
    """Convert residual mode data to LeRobot dataset format (state only, no images)."""

    features = build_features()
    dataset = LeRobotDataset.create(
        repo_id=repo_id, fps=fps, features=features, robot_type=robot_type, video_backend="pyav"
    )

    root = Path(input_dir)
    episodes = sorted(root.glob("episode_*"))

    for ep in tqdm(episodes, desc="Episodes", unit="ep"):
        dataset.clear_episode_buffer()
        robot_folder = ep / "robot"
        timesteps = sorted(p.stem for p in robot_folder.glob("*.json"))

        for idx, ts in enumerate(tqdm(timesteps, desc=ep.name, leave=False)):
            jf = robot_folder / f"{ts}.json"
            act_dict = read_act(str(jf))
            obs_dict = read_obs(str(jf))

            frame = {
                "action":            act_dict["action"].numpy(),
                "observation.state": obs_dict["state"].numpy(),
                "next.done":         np.array([idx == len(timesteps) - 1], dtype=bool),
                "task":        str(task_name),
            }

            dataset.add_frame(frame)

        dataset.save_episode()

    dataset.finalize()

    meta_path = root / "meta.json"
    # meta_path.parent.mkdir(parents=True, exist_ok=True)

    with open(meta_path, 'w') as f:
        json.dump(dataset.meta.__dict__, f, default=str, indent=2)
    print(f"LeRobot dataset saved; metadata at {meta_path}")


# -------------------------------
# CLI
# -------------------------------
def main():
    parser = argparse.ArgumentParser(description="Convert residual mode robot trajectories to LeRobot dataset.")
    parser.add_argument("-i", "--input_dir", required=True, help="Root folder with episode_*/")
    parser.add_argument("--repo-id", required=True, help="HuggingFace repo ID for LeRobotDataset")
    parser.add_argument("--task_name", default="insert_rope", help="Task name for LeRobotDataset")
    parser.add_argument("--fps", type=int, default=15, help="Frames per second")
    parser.add_argument("--robot-type", default="xarm", help="Robot type identifier")
    args = parser.parse_args()

    convert_to_lerobot(
        input_dir=args.input_dir,
        repo_id=args.repo_id,
        fps=args.fps,
        robot_type=args.robot_type,
        task_name=args.task_name,
    )


if __name__ == "__main__":
    main()
