import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.lines import Line2D
import os

def load_npy_dict(path):
    return np.load(path, allow_pickle=True).item()

def plot_two_trajs(path_a, path_b, out_path, eps_id=0):
    A = load_npy_dict(path_a)   # real
    B = load_npy_dict(path_b)   # sim

    ep = sorted(A.keys())[eps_id]    # one episode
    a = A[ep]
    b = B[ep]

    obs_pos_k  = "obs.fingertip_pos"
    obs_quat_k = "obs.eef_quat"
    obs_grip_k = "obs.gripper"
    act_pos_k  = "action.fingertip_pos"
    act_quat_k = "action.eef_quat"
    act_grip_k = "action.gripper"

    T = min(
        a[obs_pos_k].shape[0],
        b[obs_pos_k].shape[0],
        a[act_pos_k].shape[0],
    )

    def S(x): return x[:T]

    grip = S(a[obs_grip_k])[:, 0]
    grasp_idx = np.argmax(grip > 0.5) if np.any(grip > 0.5) else None

    def draw_grasp_line(ax):
        if grasp_idx is not None:
            ax.axvline(grasp_idx, color="grey", ls=":", lw=2, alpha=0.8)


    fig, axes = plt.subplots(4, 2, figsize=(18, 16), sharex=True)

    lw = 2.5  # thicker lines

    # ---- LEFT COL: eef_pos (x,y,z) + gripper ----
    pos_labels = ["x [m]", "y [m]", "z [m]"]
    for d in range(3):
        ax = axes[d, 0]
        ax.plot(S(a[act_pos_k])[:, d], color="red", lw=lw)
        ax.plot(S(a[obs_pos_k])[:, d], color="green", lw=lw)
        ax.plot(S(b[obs_pos_k])[:, d], color="gold", lw=lw, ls=":")
        draw_grasp_line(ax)
        ax.set_ylabel(pos_labels[d])

    ax = axes[3, 0]
    ax.plot(S(a[act_grip_k])[:, 0], color="red", lw=lw)
    ax.plot(S(a[obs_grip_k])[:, 0], color="green", lw=lw)
    ax.plot(S(b[obs_grip_k])[:, 0], color="gold", lw=lw, ls=":")

    draw_grasp_line(ax)
    ax.set_ylabel("openness")

    # ---- RIGHT COL: eef_quat (w,x,y,z) ----
    quat_labels = ["w", "x", "y", "z"]
    for d in range(4):
        ax = axes[d, 1]
        ax.plot(S(a[act_quat_k])[:, d], color="red", lw=lw)
        ax.plot(S(a[obs_quat_k])[:, d], color="green", lw=lw)
        ax.plot(S(b[obs_quat_k])[:, d], color="gold", lw=lw, ls=":")
        draw_grasp_line(ax)
        ax.set_ylabel(quat_labels[d])

    axes[-1, 0].set_xlabel("timestep")
    axes[-1, 1].set_xlabel("timestep")

    # ---- Single legend outside ----
    legend_handles = [
        Line2D([0], [0], color="green", lw=lw, label="real obs"),
        Line2D([0], [0], color="gold",  lw=lw, ls=":", label="sim obs"),
        Line2D([0], [0], color="red",   lw=lw, label="action"),
    ]
    fig.tight_layout(rect=[0, 0, 1, 0.92])  # reserve space for legend

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.89, 0.93),
    )
    fig.savefig(out_path, dpi=150)
    print(f"[INFO] Saved plot to: {out_path}")
    plt.close(fig)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=str, help="data path")
    parser.add_argument("--eps_id", type=int, default=None, help="Episode ID to plot")
    args = parser.parse_args()

    real_data_path = Path(args.data_path) / "data" / "real_teleop_trajectories.npy"
    sim_data_path  = Path(args.data_path) / "data" / "sim_replay_trajectories.npy"
    os.makedirs(Path(args.data_path) / "real2sim", exist_ok=True)

    if args.eps_id is None:
        for i in range(20):
            output_path = Path(args.data_path) / "real2sim" / f"task_space_real2sim_gap_eps_{i}.png"
            plot_two_trajs(real_data_path, sim_data_path, output_path, i)
    else:
        output_path = Path(args.data_path) / "real2sim" / f"task_space_real2sim_gap_eps_{args.eps_id}.png"
        plot_two_trajs(real_data_path, sim_data_path, output_path, args.eps_id)