#!/usr/bin/env python3
import argparse
import os

import torch
import matplotlib.pyplot as plt


FIELD_LAYOUT = [
    ("fingertip_pos",           3),
    ("fingertip_quat",          4),
    ("gripper",                 1),
    ("fingertip_pos_rel_fixed", 3),
    ("fingertip_pos_rel_held",  3),
    ("ee_linvel",               3),
    ("ee_angvel",               3),
    ("base_actions",           8),
    ("prev_action",             7),
]

def load_list_1x35_to_35xT(path):
    data = torch.load(path, map_location="cpu")  # list of T tensors, each (1, 35)
    if not isinstance(data, list):
        raise ValueError(f"{path} is expected to be a list, got {type(data)}")

    # [T, 1, 35] -> [T, 35]
    data = torch.cat(data, dim=0).squeeze(1)
    if data.dim() != 2 or data.shape[1] != 35:
        raise ValueError(f"Unexpected tensor shape from {path}: {data.shape}, expected [T, 35]")

    # [T, 35] -> [35, T]
    return data.transpose(0, 1)


def plot_fields(t1, t2, out_dir, label1="file1", label2="file2"):
    os.makedirs(out_dir, exist_ok=True)
    n_dims, T = t1.shape
    assert t2.shape == t1.shape, "Both tensors must have the same shape"

    start = 0
    for name, size in FIELD_LAYOUT:
        end = start + size
        if end > n_dims:
            # Safety check in case layout and tensor size mismatch
            print(f"[WARN] Skipping field '{name}' (dims {start}:{end}) beyond tensor size {n_dims}.")
            break

        seg1 = t1[start:end]  # [size, T]
        seg2 = t2[start:end]  # [size, T]

        fig, axes = plt.subplots(size, 1, figsize=(10, 2.5 * size), sharex=True)
        if size == 1:
            axes = [axes]

        timesteps = range(T)
        for i, ax in enumerate(axes):
            ax.plot(timesteps, seg1[i].numpy(), label=label1)
            ax.plot(timesteps, seg2[i].numpy(), label=label2, linestyle="--")
            ax.set_ylabel(f"{name}[{i}]")
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.legend()

        axes[-1].set_xlabel("time step")
        fig.tight_layout()
        out_path = os.path.join(out_dir, f"{name}.jpg")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

        print(f"Saved {out_path}")
        start = end


def main():
    parser = argparse.ArgumentParser(description="Compare two trajectory .pt files and plot key fields.")
    parser.add_argument("pt1", type=str, help="Path to first .pt file")
    parser.add_argument("pt2", type=str, help="Path to second .pt file")
    parser.add_argument("out_dir", type=str, help="Output directory for plots")
    args = parser.parse_args()

    t1 = load_list_1x35_to_35xT(args.pt1)
    t2 = load_list_1x35_to_35xT(args.pt2)

    label1 = os.path.splitext(os.path.basename(args.pt1))[0]
    label2 = os.path.splitext(os.path.basename(args.pt2))[0]

    plot_fields(t1, t2, args.out_dir, label1=label1, label2=label2)


if __name__ == "__main__":
    main()
