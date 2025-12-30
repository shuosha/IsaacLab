#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset


# -------------------------------
# JSON readers (original behavior)
# -------------------------------
def read_obs(json_path: str) -> dict[str, torch.Tensor]:
    """Read residual observation from JSON file."""
    data = json.load(open(json_path, "r"))
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

    state = torch.cat(
        [fingertip_pos, fingertip_quat, gripper, ee_linvel, ee_angvel], dim=0
    )  # (14,)

    environment_state = torch.cat([fingertip_pos_rel_fixed, fingertip_pos_rel_held], dim=0)  # (6,)
    return {"state": state, "environment_state": environment_state}


def read_act(json_path: str) -> dict[str, torch.Tensor]:
    """Read residual action from JSON file."""
    data = json.load(open(json_path, "r"))
    # Residual action space: fingertip_pos(3) + fingertip_quat(4) + gripper(1) = 8D
    fingertip_pos = torch.tensor(data["action"][:3], dtype=torch.float32)  # (3,)
    fingertip_quat = torch.tensor(data["action"][3:7], dtype=torch.float32)  # (4,) wxyz
    gripper = torch.tensor([data["action"][7]], dtype=torch.float32).flatten()  # (1,)
    action = torch.cat([fingertip_pos, fingertip_quat, gripper], dim=0)  # (8,)
    return {"action": action}


# -------------------------------
# NPY readers (new option)
# data = { 'episode_0000': { 'obs.fingertip_pos': (T,3), ... }, ... }
# -------------------------------
def load_npy_dict(npy_path: str) -> dict:
    data = np.load(npy_path, allow_pickle=True).item()
    if not isinstance(data, dict):
        raise TypeError("Expected a dict at top level of the .npy file.")
    return data


def read_obs_from_npy(ep: dict, t: int) -> dict[str, torch.Tensor]:
    fingertip_pos = torch.tensor(ep["obs.fingertip_pos"][t], dtype=torch.float32)  # (3,)
    fingertip_quat = torch.tensor(ep["obs.fingertip_quat"][t], dtype=torch.float32)  # (4,)
    gripper = torch.tensor(ep["obs.gripper"][t], dtype=torch.float32).flatten()  # (1,)

    fingertip_pos_rel_fixed = torch.tensor(ep["obs.fingertip_pos_rel_fixed"][t], dtype=torch.float32)  # (3,)
    fingertip_pos_rel_held = torch.tensor(ep["obs.fingertip_pos_rel_held"][t], dtype=torch.float32)  # (3,)

    ee_linvel = torch.tensor(ep["obs.ee_linvel_fd"][t], dtype=torch.float32)  # (3,)
    ee_angvel = torch.tensor(ep["obs.ee_angvel_fd"][t], dtype=torch.float32)  # (3,)

    state = torch.cat(
        [fingertip_pos, fingertip_quat, gripper, ee_linvel, ee_angvel], dim=0
    )  # (14,)

    environment_state = torch.cat([fingertip_pos_rel_fixed, fingertip_pos_rel_held], dim=0)  # (6,)
    return {"state": state, "environment_state": environment_state}


def read_act_from_npy(ep: dict, t: int) -> dict[str, torch.Tensor]:
    fingertip_pos = torch.tensor(ep["action.fingertip_pos"][t], dtype=torch.float32)  # (3,)
    fingertip_quat = torch.tensor(ep["action.fingertip_quat"][t], dtype=torch.float32)  # (4,)
    gripper = torch.tensor(ep["action.gripper"][t], dtype=torch.float32).flatten()  # (1,)
    action = torch.cat([fingertip_pos, fingertip_quat, gripper], dim=0)  # (8,)
    return {"action": action}


# -------------------------------
# Feature schema builders
# -------------------------------
def build_features() -> dict:
    """Build feature schema for residual mode (state only, no images)."""
    return {
        "action": {
            "dtype": "float32",
            "shape": (8,),
            "names": {
                "waypoint": [
                    "fingertip_pos_x",
                    "fingertip_pos_y",
                    "fingertip_pos_z",
                    "fingertip_quat_x",
                    "fingertip_quat_y",
                    "fingertip_quat_z",
                    "fingertip_quat_w",
                    "gripper",
                ]
            },
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (14,),
            "names": {
                "waypoint": [
                    "fingertip_pos_x",
                    "fingertip_pos_y",
                    "fingertip_pos_z",
                    "fingertip_quat_x",
                    "fingertip_quat_y",
                    "fingertip_quat_z",
                    "fingertip_quat_w",
                    "gripper",
                    "ee_linvel_x",
                    "ee_linvel_y",
                    "ee_linvel_z",
                    "ee_angvel_x",
                    "ee_angvel_y",
                    "ee_angvel_z",
                ]
            },
        },
        "observation.environment_state": {
            "dtype": "float32",
            "shape": (6,),
            "names": {
                "relative_positions": [
                    "fingertip_to_fixed_x",
                    "fingertip_to_fixed_y",
                    "fingertip_to_fixed_z",
                    "fingertip_to_held_x",
                    "fingertip_to_held_y",
                    "fingertip_to_held_z",
                ]
            },
        },
        "next.done": {"dtype": "bool", "shape": (1,), "names": None},
    }


# -------------------------------
# Converter
# -------------------------------
def convert_to_lerobot(
    input_dir: str,
    repo_id: str,
    fps: int,
    robot_type: str,
    task_name: str,
    npy_path: str | None = None,
) -> None:
    """Convert residual mode data to LeRobot dataset format (state only, no images)."""

    features = build_features()
    dataset = LeRobotDataset.create(
        repo_id=repo_id, fps=fps, features=features, robot_type=robot_type, video_backend="pyav"
    )

    use_npy = npy_path is not None

    if use_npy:
        npy_data = load_npy_dict(npy_path)
        episode_names = sorted(npy_data.keys())

        for ep_name in tqdm(episode_names, desc="Episodes", unit="ep"):
            dataset.clear_episode_buffer()
            ep = npy_data[ep_name]

            # Infer T from obs.fingertip_pos
            if "obs.fingertip_pos" not in ep:
                raise KeyError(f"{ep_name}: missing key 'obs.fingertip_pos'")
            T = int(ep["obs.fingertip_pos"].shape[0])

            for t in range(T):
                act_dict = read_act_from_npy(ep, t)
                obs_dict = read_obs_from_npy(ep, t)

                frame = {
                    "action": act_dict["action"].numpy(),
                    "observation.state": obs_dict["state"].numpy(),
                    "observation.environment_state": obs_dict["environment_state"].numpy(),
                    "next.done": np.array([t == T - 1], dtype=bool),
                    "task": str(task_name),
                }
                dataset.add_frame(frame)

            dataset.save_episode()

        dataset.finalize()

        # Save metadata near the npy file
        meta_path = Path(npy_path).with_suffix("")  # drop .npy
        meta_path = meta_path.parent / (meta_path.name + "_meta.json")
        with open(meta_path, "w") as f:
            json.dump(dataset.meta.__dict__, f, default=str, indent=2)
        print(f"LeRobot dataset saved; metadata at {meta_path}")
        return

    # ---- original JSON-based workflow ----
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
                "action": act_dict["action"].numpy(),
                "observation.state": obs_dict["state"].numpy(),
                "observation.environment_state": obs_dict["environment_state"].numpy(),
                "next.done": np.array([idx == len(timesteps) - 1], dtype=bool),
                "task": str(task_name),
            }
            dataset.add_frame(frame)

        dataset.save_episode()

    dataset.finalize()

    meta_path = root / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(dataset.meta.__dict__, f, default=str, indent=2)
    print(f"LeRobot dataset saved; metadata at {meta_path}")


# -------------------------------
# CLI
# -------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Convert residual mode robot trajectories to LeRobot dataset."
    )
    parser.add_argument("input_dir", help="Root folder with episode_*/ (ignored if --npy-path is set)")
    parser.add_argument("repo_id", help="HuggingFace repo ID for LeRobotDataset")
    parser.add_argument("task_name", help="Task name for LeRobotDataset")
    parser.add_argument("--fps", type=int, default=15, help="Frames per second")
    parser.add_argument("--robot-type", default="xarm", help="Robot type identifier")
    parser.add_argument(
        "--npy-path",
        type=str,
        default=None,
        help="Optional path to .npy containing {episode_name: {keys -> arrays}}",
    )
    args = parser.parse_args()

    convert_to_lerobot(
        input_dir=args.input_dir,
        repo_id=args.repo_id,
        fps=args.fps,
        robot_type=args.robot_type,
        task_name=args.task_name,
        npy_path=args.npy_path,
    )


if __name__ == "__main__":
    main()
