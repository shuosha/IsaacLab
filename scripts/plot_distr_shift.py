#!/usr/bin/env python3
"""
Two visualization modes for distribution shift:

1) timeseries (default):
   - Use NN alignment indices derived from base_out to align to GT curve.
   - Plot x/y/z vs aligned GT index + deviation norm.
   - Y-limits fixed from GT; outliers dropped consistently.

2) projections:
   - Scatter all points on XY, XZ, YZ planes (GT/base/net).
   - Compute 2D-histogram Jensen–Shannon divergence (JS) between:
       GT vs base_out, GT vs net_out
     for each plane. JS is symmetric and bounded.

CLI:
  python vis_shift.py base.npy rollout.npy --eps-idx 0 --out out.png
  python vis_shift.py base.npy rollout.npy --eps-idx 0 --out out.png --vis projections
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ------------------------
# IO helpers
# ------------------------
def load_ep(path: str, eps_idx: int) -> dict:
    d = np.load(path, allow_pickle=True).item()
    key = f"episode_{eps_idx:04d}"
    if key not in d:
        raise KeyError(f"{path}: missing key '{key}'. Keys example: {list(d.keys())[:5]}")
    return d[key]


# ------------------------
# Alignment helpers (timeseries mode)
# ------------------------
def nn_align_indices(base_xyz: np.ndarray, traj_xyz: np.ndarray, chunk: int = 4096) -> np.ndarray:
    """Nearest base index in 3D for each traj point (batched)."""
    idx = np.empty((traj_xyz.shape[0],), dtype=np.int64)

    base = base_xyz.astype(np.float32)
    base_norm2 = (base * base).sum(axis=1, keepdims=True)  # (T0,1)

    for s in range(0, traj_xyz.shape[0], chunk):
        q = traj_xyz[s : s + chunk].astype(np.float32)      # (B,3)
        q_norm2 = (q * q).sum(axis=1, keepdims=True).T      # (1,B)
        dist2 = base_norm2 + q_norm2 - 2.0 * (base @ q.T)   # (T0,B)
        idx[s : s + chunk] = dist2.argmin(axis=0)

    return idx


def enforce_monotone(idx: np.ndarray) -> np.ndarray:
    """Make indices non-decreasing to reduce backward jumps from NN alignment noise."""
    return np.maximum.accumulate(idx)


def compute_ylim_from_ref(ref_1d: np.ndarray, margin_ratio: float = 0.05, min_margin: float = 1e-4):
    """Fixed y-limits based on ref curve only, with a small symmetric margin."""
    lo = float(np.nanmin(ref_1d))
    hi = float(np.nanmax(ref_1d))
    span = hi - lo
    m = max(min_margin, margin_ratio * span)
    if span < 1e-12:
        m = max(min_margin, 1e-3)
    return lo - m, hi + m


def mask_in_range(y: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return (y >= lo) & (y <= hi) & np.isfinite(y)


# ------------------------
# Distribution metrics (projections mode)
# ------------------------
def hist2d_prob(points_xy: np.ndarray, xlim, ylim, bins: int = 80, eps: float = 1e-12) -> np.ndarray:
    """2D histogram -> probability matrix (bins,bins) with smoothing."""
    H, _, _ = np.histogram2d(
        points_xy[:, 0], points_xy[:, 1],
        bins=bins,
        range=[xlim, ylim],
    )
    P = H.astype(np.float64)
    P = P + eps
    P = P / P.sum()
    return P


def kl_div(P: np.ndarray, Q: np.ndarray) -> float:
    """KL(P||Q) for discrete probabilities. Assumes P,Q > 0 and sum to 1."""
    return float(np.sum(P * (np.log(P) - np.log(Q))))


def js_div(P: np.ndarray, Q: np.ndarray) -> float:
    """Jensen–Shannon divergence (natural log). Bounded in [0, ln 2]."""
    M = 0.5 * (P + Q)
    return 0.5 * kl_div(P, M) + 0.5 * kl_div(Q, M)


def subsample(arr: np.ndarray, max_n: int, rng: np.random.Generator) -> np.ndarray:
    if arr.shape[0] <= max_n:
        return arr
    idx = rng.choice(arr.shape[0], size=max_n, replace=False)
    return arr[idx]


# ------------------------
# Plotting modes
# ------------------------
def plot_timeseries(gt, base_out, net_out, eps_idx, out_path, monotone: bool, ylim_margin: float):
    T0 = gt.shape[0]
    x_gt = np.arange(T0)

    # alignment from base_out -> GT only
    idx = nn_align_indices(gt, base_out)
    if monotone:
        idx = enforce_monotone(idx)

    dev_base = np.linalg.norm(base_out - gt[idx], axis=1)
    dev_net  = np.linalg.norm(net_out  - gt[idx], axis=1)

    ylims = [compute_ylim_from_ref(gt[:, d], margin_ratio=ylim_margin) for d in range(3)]
    m_base_dims = [mask_in_range(base_out[:, d], *ylims[d]) for d in range(3)]
    m_net_dims  = [mask_in_range(net_out[:,  d], *ylims[d]) for d in range(3)]

    m_base_all = m_base_dims[0] & m_base_dims[1] & m_base_dims[2]
    m_net_all  = m_net_dims[0]  & m_net_dims[1]  & m_net_dims[2]
    m_keep = m_base_all & m_net_all

    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    dims = ["x", "y", "z"]

    for d in range(3):
        ax = axes[d]
        ylo, yhi = ylims[d]
        ax.plot(x_gt, gt[:, d], linewidth=2.0, color="gold", label="GT action.fingertip_pos")
        ax.scatter(idx[m_base_dims[d]], base_out[m_base_dims[d], d], s=6, alpha=0.6,
                   label="rollout base_fingertip_pos")
        ax.scatter(idx[m_net_dims[d]],  net_out[m_net_dims[d],  d], s=6, alpha=0.6,
                   label="rollout net_fingertip_pos")
        ax.set_ylabel(f"{dims[d]} (m)")
        ax.set_ylim([ylo, yhi])
        ax.grid(True, linewidth=0.6, alpha=0.4)

    axd = axes[3]
    axd.scatter(idx[m_keep], dev_base[m_keep], s=8, alpha=0.7, label="||base_out - GT||")
    axd.scatter(idx[m_keep], dev_net[m_keep],  s=8, alpha=0.7, label="||net_out  - GT||")
    axd.set_ylabel("deviation to GT (m)")
    axd.set_xlabel("GT trajectory index (aligned via base_out)")
    axd.grid(True, linewidth=0.6, alpha=0.4)

    axes[0].set_title(
        f"Episode {eps_idx:04d}: GT vs rollout (same warping for net; outliers dropped consistently)"
    )
    axes[0].legend(loc="upper right", fontsize=9)
    axes[3].legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_projections(gt, base_out, net_out, eps_idx, out_path, bins: int, max_pts: int, seed: int, margin_ratio: float):
    rng = np.random.default_rng(seed)

    # Axis limits from GT only (robust; drop outliers by range selection)
    xlim = compute_ylim_from_ref(gt[:, 0], margin_ratio=margin_ratio)
    ylim = compute_ylim_from_ref(gt[:, 1], margin_ratio=margin_ratio)
    zlim = compute_ylim_from_ref(gt[:, 2], margin_ratio=margin_ratio)

    # Filter points to GT-derived bounding box so outliers don't dominate
    def in_box(P):
        mx = mask_in_range(P[:, 0], *xlim)
        my = mask_in_range(P[:, 1], *ylim)
        mz = mask_in_range(P[:, 2], *zlim)
        return P[mx & my & mz]

    gt_f   = in_box(gt)
    base_f = in_box(base_out)
    net_f  = in_box(net_out)

    # Subsample for speed/clarity
    gt_s   = subsample(gt_f,   max_pts, rng)
    base_s = subsample(base_f, max_pts, rng)
    net_s  = subsample(net_f,  max_pts, rng)

    # 2D histograms + JS divergence
    # XY
    P_xy = hist2d_prob(gt_f[:, [0, 1]], xlim, ylim, bins=bins)
    Q_xy = hist2d_prob(base_f[:, [0, 1]], xlim, ylim, bins=bins)
    R_xy = hist2d_prob(net_f[:, [0, 1]],  xlim, ylim, bins=bins)
    js_xy_base = js_div(P_xy, Q_xy)
    js_xy_net  = js_div(P_xy, R_xy)

    # XZ
    P_xz = hist2d_prob(gt_f[:, [0, 2]], xlim, zlim, bins=bins)
    Q_xz = hist2d_prob(base_f[:, [0, 2]], xlim, zlim, bins=bins)
    R_xz = hist2d_prob(net_f[:, [0, 2]],  xlim, zlim, bins=bins)
    js_xz_base = js_div(P_xz, Q_xz)
    js_xz_net  = js_div(P_xz, R_xz)

    # YZ
    P_yz = hist2d_prob(gt_f[:, [1, 2]], ylim, zlim, bins=bins)
    Q_yz = hist2d_prob(base_f[:, [1, 2]], ylim, zlim, bins=bins)
    R_yz = hist2d_prob(net_f[:, [1, 2]],  ylim, zlim, bins=bins)
    js_yz_base = js_div(P_yz, Q_yz)
    js_yz_net  = js_div(P_yz, R_yz)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharex=False, sharey=False, constrained_layout=True)

    # XY
    ax = axes[0]
    ax.scatter(gt_s[:, 0],   gt_s[:, 1],   s=12, facecolors='none', edgecolors='gold', label='GT')
    ax.scatter(base_s[:, 0], base_s[:, 1], s=6, alpha=0.25, label="base_out")
    ax.scatter(net_s[:, 0],  net_s[:, 1],  s=6, alpha=0.25, label="net_out")
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title(f"XY: JS(GT,base)={js_xy_base:.3f}, JS(GT,net)={js_xy_net:.3f}")
    ax.grid(True, linewidth=0.6, alpha=0.3)

    # XZ
    ax = axes[1]
    ax.scatter(gt_s[:, 0],   gt_s[:, 2],   s=12, facecolors='none', edgecolors='gold', label='GT')
    ax.scatter(base_s[:, 0], base_s[:, 2], s=6, alpha=0.25, label="base_out")
    ax.scatter(net_s[:, 0],  net_s[:, 2],  s=6, alpha=0.25, label="net_out")
    ax.set_xlim(xlim); ax.set_ylim(zlim)
    ax.set_xlabel("x (m)"); ax.set_ylabel("z (m)")
    ax.set_title(f"XZ: JS(GT,base)={js_xz_base:.3f}, JS(GT,net)={js_xz_net:.3f}")
    ax.grid(True, linewidth=0.6, alpha=0.3)

    # YZ
    ax = axes[2]
    ax.scatter(gt_s[:, 1],   gt_s[:, 2],   s=12, facecolors='none', edgecolors='gold', label='GT')
    ax.scatter(base_s[:, 1], base_s[:, 2], s=6, alpha=0.25, label="base_out")
    ax.scatter(net_s[:, 1],  net_s[:, 2],  s=6, alpha=0.25, label="net_out")
    ax.set_xlim(ylim); ax.set_ylim(zlim)
    ax.set_xlabel("y (m)"); ax.set_ylabel("z (m)")
    ax.set_title(f"YZ: JS(GT,base)={js_yz_base:.3f}, JS(GT,net)={js_yz_net:.3f}")
    ax.grid(True, linewidth=0.6, alpha=0.3)

    # single legend for all
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.12),)

    fig.suptitle(f"Episode {eps_idx:04d}: 2D projections (GT-range clipped), JS divergence via {bins}x{bins} hist", y=1.05)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ------------------------
# Main
# ------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base_npy", type=str)
    ap.add_argument("rollout_npy", type=str)
    ap.add_argument("--eps-idx", type=int, required=True)
    ap.add_argument("--out", type=str, required=True)

    ap.add_argument("--vis", choices=["timeseries", "projections"], default="timeseries")

    # timeseries options
    ap.add_argument("--no-monotone", action="store_true")
    ap.add_argument("--ylim-margin", type=float, default=0.05)

    # projections options
    ap.add_argument("--bins", type=int, default=80, help="2D histogram bins for JS divergence")
    ap.add_argument("--max-pts", type=int, default=20000, help="Max points to scatter per set")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--proj-margin", type=float, default=0.05, help="Axis margin ratio from GT for projections")
    args = ap.parse_args()

    base_ep = load_ep(args.base_npy, args.eps_idx)
    roll_ep = load_ep(args.rollout_npy, args.eps_idx)

    gt = np.asarray(base_ep["action.fingertip_pos"], dtype=np.float32)
    base_out = np.asarray(roll_ep["base_fingertip_pos"], dtype=np.float32)
    net_out  = np.asarray(roll_ep["net_fingertip_pos"], dtype=np.float32)

    if args.vis == "timeseries":
        plot_timeseries(
            gt=gt,
            base_out=base_out,
            net_out=net_out,
            eps_idx=args.eps_idx,
            out_path=args.out,
            monotone=not args.no_monotone,
            ylim_margin=args.ylim_margin,
        )
    else:
        plot_projections(
            gt=gt,
            base_out=base_out,
            net_out=net_out,
            eps_idx=args.eps_idx,
            out_path=args.out,
            bins=args.bins,
            max_pts=args.max_pts,
            seed=args.seed,
            margin_ratio=args.proj_margin,
        )


if __name__ == "__main__":
    main()
