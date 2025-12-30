#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

try:
    from sklearn.neighbors import NearestNeighbors
except Exception:
    NearestNeighbors = None


# ---------------------------
# Camera parameters (PLACEHOLDERS)
# ---------------------------
cam2base = np.array([
      [
        0.051754931620038046,
        0.537235188103101,
        -0.8418430849729842,
        0.7328167574019903
      ],
      [
        0.9986368611596909,
        -0.03355668150370313,
        0.039979603044303084,
        -0.032323139684666755
      ],
      [
        -0.006771010716739843,
        -0.8427646775881661,
        -0.5382395857084361,
        0.22158513865539187
      ],
      [
        0.0,
        0.0,
        0.0,
        1.0
      ]
    ], dtype=np.float64)

K = np.array([
    [427.0138854980469,   0.0, 425.43218994140625],
    [0.0, 426.470458984375, 245.81968688964844],
    [0.0, 0.0, 1.0],
], dtype=np.float64)


# ---------------------------
# Data loaders
# ---------------------------
def load_points_base_from_json(data_root: Path) -> np.ndarray:
    """Load points = data['obs'][:3] from episode_*/robot/*.json."""
    pts = []
    for jf in sorted(data_root.glob("episode_*/robot/*.json")):
        d = json.load(open(jf, "r"))
        pts.append(np.asarray(d["obs"][:3], dtype=np.float32))
    if not pts:
        raise FileNotFoundError(f"No files under {data_root}/episode_*/robot/*.json")
    return np.stack(pts, axis=0)  # (N,3)


def load_points_base_from_npy(npy_path: Path) -> np.ndarray:
    """
    Load points from data.npy where:
      data = { episode_name: { 'obs.fingertip_pos': (T,3), ... } }
    """
    data = np.load(npy_path, allow_pickle=True).item()
    if not isinstance(data, dict):
        raise ValueError("Expected dict at top level of .npy")

    pts = []
    for ep in data.values():
        if "obs.fingertip_pos" not in ep:
            continue
        arr = np.asarray(ep["obs.fingertip_pos"], dtype=np.float32)  # (T,3)
        pts.append(arr)

    if not pts:
        raise ValueError("No 'obs.fingertip_pos' found in npy file")

    return np.concatenate(pts, axis=0)  # (N,3)


# ---------------------------
# Geometry + density
# ---------------------------
def project_base_to_image(points_base: np.ndarray, cam2base: np.ndarray, K: np.ndarray):
    """Project base-frame points into image pixels using cam2base + K."""
    base2cam = np.linalg.inv(cam2base)
    R = base2cam[:3, :3]
    t = base2cam[:3, 3]

    Pc = (R @ points_base.T + t[:, None]).T  # (N,3)
    X, Y, Z = Pc[:, 0], Pc[:, 1], Pc[:, 2]

    mask = Z > 1e-6
    X, Y, Z = X[mask], Y[mask], Z[mask]

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u = fx * (X / Z) + cx
    v = fy * (Y / Z) + cy
    return np.stack([u, v], axis=1)  # (M,2)


def knn_density_2d(uv: np.ndarray, k: int = 30) -> np.ndarray:
    """kNN density proxy in pixel space, normalized to [0,1]."""
    n = uv.shape[0]
    if n < max(5, k + 1):
        return np.ones((n,), dtype=np.float32)

    if NearestNeighbors is None:
        if n > 5000:
            raise RuntimeError("Install scikit-learn for large N")
        d2 = ((uv[:, None, :] - uv[None, :, :]) ** 2).sum(-1)
        dk2 = np.sort(d2, axis=1)[:, k]
        dens = 1.0 / (dk2 + 1e-12)
    else:
        nbrs = NearestNeighbors(n_neighbors=k).fit(uv)
        dists, _ = nbrs.kneighbors(uv)
        dk = dists[:, -1]
        dens = 1.0 / (dk**2 + 1e-12)
        dens = np.log(dens + 1e-12)

    lo, hi = np.percentile(dens, [5, 95])
    dens_n = np.clip((dens - lo) / (hi - lo + 1e-12), 0, 1).astype(np.float32)
    return dens_n


# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=str, help="RGB image taken by the camera")
    ap.add_argument("--data-root", type=str, default=None,
                    help="Root with episode_*/robot/*.json")
    ap.add_argument("--npy-path", type=str, default=None,
                    help="Optional path to data.npy (overrides --data-root)")
    ap.add_argument("--out", type=str, default="overlay.png")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--max-points", type=int, default=200000)
    args = ap.parse_args()

    if args.npy_path is not None:
        pts_base = load_points_base_from_npy(Path(args.npy_path))
    else:
        if args.data_root is None:
            raise ValueError("Must provide either --data-root or --npy-path")
        pts_base = load_points_base_from_json(Path(args.data_root))

    if pts_base.shape[0] > args.max_points:
        sel = np.random.choice(pts_base.shape[0], size=args.max_points, replace=False)
        pts_base = pts_base[sel]

    img = np.asarray(Image.open(args.image).convert("RGB"))
    H, W = img.shape[0], img.shape[1]

    uv = project_base_to_image(pts_base, cam2base, K)
    inb = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    uv = uv[inb]

    dens = knn_density_2d(uv, k=args.k)

    plt.figure(figsize=(10, 6))
    plt.imshow(img)
    plt.scatter(
        uv[:, 0], uv[:, 1],
        c=dens, cmap="viridis",
        s=1, alpha=0.2, linewidths=0,
    )
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(args.out, dpi=200, bbox_inches="tight", pad_inches=0)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
