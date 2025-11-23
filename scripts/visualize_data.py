import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def load_episode_act_pos(path, episode_idx=0, pad=True, key="action.eef_pos"):
    """
    Load one episode's action positions (T, 3) from a .npz path.

    Args:
        path         : Path to .npz file.
        episode_idx  : Which episode to load.
        pad          : Whether to pad shorter episodes to max length.
    Returns:
        act_pos : (T, 3) tensor of eef action positions.
    """
    flat = np.load(path, allow_pickle=True)

    # Convert to {key -> tensor}
    flat = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in flat.items()}

    # Group keys by episode
    eps = sorted({k.split("/", 1)[0] for k in flat})
    if not (0 <= episode_idx < len(eps)):
        raise IndexError(f"Episode {episode_idx} not in [0, {len(eps)-1}]")

    # Extract this episode’s data
    ep_name = eps[episode_idx]
    data = {s.split("/", 1)[1]: flat[s] for s in flat if s.startswith(ep_name)}

    # Get true length
    T = len(data["obs.gripper"])
    maxT = max(len(data[key]), T)

    # Pad optionally
    if pad and T < maxT:
        pad_len = maxT - T
        data[key] = torch.cat(
            [data[key], data[key][-1:].repeat(pad_len, 1)],
            dim=0,
        )

    return data[key]    # (T, 3)


def load_all_episode_act_pos(path, pad=True):
    """
    Load ALL episodes' action positions from a .npz file.

    Returns:
        list_of_trajs : list of (T, 3) tensors
        lengths       : list of true episode lengths
    """
    flat = np.load(path, allow_pickle=True)
    flat = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in flat.items()}

    eps = sorted({k.split("/", 1)[0] for k in flat})
    list_of_trajs = []
    lengths = []

    # compute max length if padding
    lengths = [len(flat[f"{e}/action.eef_pos"]) for e in eps]
    maxT = max(lengths)

    for e, T in zip(eps, lengths):
        act = flat[f"{e}/action.eef_pos"]  # (T, 3)

        if pad and T < maxT:
            pad_len = maxT - T
            act = torch.cat([act, act[-1:].repeat(pad_len, 1)], dim=0)

        list_of_trajs.append(act)

    return list_of_trajs, lengths



def plot_two_action_trajs(path_a, path_b, ep_a=0, ep_b=0, pad=True):
    """
    Plot two action position trajectories in 3D:
      - Dataset A in red
      - Dataset B in blue
    """
    act_a = load_episode_act_pos(path_a, ep_a, pad=pad)
    act_b = load_episode_act_pos(path_b, ep_b, pad=pad)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(act_a[:, 0], act_a[:, 1], act_a[:, 2],
               c='r', s=8, label=f"A (ep {ep_a})")

    ax.scatter(act_b[:, 0], act_b[:, 1], act_b[:, 2],
               c='b', s=8, label=f"B (ep {ep_b})")

    ax.set_title("Action Trajectory Comparison (3D)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.show()

def plot_all_trajs_two_datasets(path_a, path_b, pad=True):
    """
    Plot ALL action trajectories of two datasets in 3D:
      - Dataset A = red
      - Dataset B = blue
    Also prints the mean length of episodes in each dataset.
    """

    all_a, lengths_a = load_all_episode_act_pos(path_a, pad=pad)
    all_b, lengths_b = load_all_episode_act_pos(path_b, pad=pad)

    mean_a = np.mean(lengths_a)
    mean_b = np.mean(lengths_b)

    print(f"Dataset A: {len(lengths_a)} episodes, mean length = {mean_a:.1f}")
    print(f"Dataset B: {len(lengths_b)} episodes, mean length = {mean_b:.1f}")

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    # plot dataset A in red
    for traj in all_a:
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                color="red", alpha=0.6, linewidth=1)

    # plot dataset B in blue
    for traj in all_b:
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                color="blue", alpha=0.6, linewidth=1)

    ax.set_title(
        f"Action Trajectories\n"
        f"Red mean length = {mean_a:.1f},  Blue mean length = {mean_b:.1f}\n"
        f"Red max length = {max(lengths_a)},  Blue max length = {max(lengths_b)}"
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # plot_two_action_trajs(
    #     "demo_data_a.npz",
    #     "demo_data_b.npz",
    #     ep_a=0,
    #     ep_b=0,
    #     pad=True
    # )

    plot_all_trajs_two_datasets(
        "logs/data/1119_teleop_gear_mesh_20/robot_states/robot_trajectories.npz",
        "logs/data/teleop_gear_mesh_9/robot_states/robot_trajectories.npz",
        pad=True
    )