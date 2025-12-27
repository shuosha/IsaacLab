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
        0.058270442368724015,
        0.5340480876382162,
        -0.8434436529111579,
        0.7303619252924132
      ],
      [
        0.9982042933467551,
        -0.04291879446520892,
        0.0417871490503764,
        -0.03347226866280275
      ],
      [
        -0.013883237744072378,
        -0.8443640311924814,
        -0.5355899910735117,
        0.22157812543698424
      ],
      [
        0.0,
        0.0,
        0.0,
        1.0
      ]
    ], dtype=np.float64)

K = np.array([
    [427.0138854980469,   0.0, 425.43218994140625],  # fx, 0, cx  <-- replace
    [  0.0, 426.470458984375, 245.81968688964844],  # 0, fy, cy  <-- replace
    [  0.0,   0.0,   1.0],
], dtype=np.float64)


def load_points_base(data_root: Path) -> np.ndarray:
    """Load points = data['obs'][:3] from episode_*/robot/*.json."""
    pts = []
    for jf in sorted(data_root.glob("episode_*/robot/*.json")):
        d = json.load(open(jf, "r"))
        pts.append(np.asarray(d["obs"][:3], dtype=np.float32))
    if not pts:
        raise FileNotFoundError(f"No files under {data_root}/episode_*/robot/*.json")
    return np.stack(pts, axis=0)  # (N,3)


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
    """kNN density proxy in pixel space: density ~ 1 / d_k^2, normalized to [0,1]."""
    n = uv.shape[0]
    if n < max(5, k + 1):
        return np.ones((n,), dtype=np.float32)

    if NearestNeighbors is None:
        # Fallback O(N^2) for small N
        if n > 5000:
            raise RuntimeError("Install scikit-learn for large N: pip install scikit-learn")
        d2 = ((uv[:, None, :] - uv[None, :, :]) ** 2).sum(-1)
        dk2 = np.sort(d2, axis=1)[:, k]
        dens = 1.0 / (dk2 + 1e-12)
    else:
        nbrs = NearestNeighbors(n_neighbors=k).fit(uv)
        dists, _ = nbrs.kneighbors(uv)
        dk = dists[:, -1]
        dens = 1.0 / (dk**2 + 1e-12)
        dens = np.log(dens + 1e-12)

    lo, hi = np.percentile(dens, [5, 95])  # widen/shift if needed
    dens_n = np.clip((dens - lo) / (hi - lo + 1e-12), 0, 1).astype(np.float32)
    return dens_n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_root", type=str, help="Root with episode_*/robot/*.json")
    ap.add_argument("image", type=str, help="RGB image taken by the camera")
    ap.add_argument("--out", type=str, default="overlay.png")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--max_points", type=int, default=200000)
    args = ap.parse_args()

    pts_base = load_points_base(Path(args.data_root))
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
        s=1, alpha=0.4, linewidths=0,
    )
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(args.out, dpi=200, bbox_inches="tight", pad_inches=0)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
