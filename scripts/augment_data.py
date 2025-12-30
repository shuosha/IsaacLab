#!/usr/bin/env python3
"""
Augment a dataset stored as a .npy dict and output a new dict with contiguous keys:

Input format:
data = {
  'episode_0000': {
    'obs.fingertip_pos': (T,3) float32,
    'obs.fingertip_quat': (T,4) float32,
    'obs.gripper': (T,1) float32,
    'obs.fingertip_pos_rel_fixed': (T,3) float32,
    'obs.fingertip_pos_rel_held': (T,3) float32,
    'obs.ee_linvel_fd': (T,3) float32,
    'obs.ee_angvel_fd': (T,3) float32,
    'action.fingertip_pos': (T,3) float32,
    'action.fingertip_quat': (T,4) float32,
    'action.gripper': (T,1) float32,
  },
  ...
}

Output keys are exactly:
  episode_0000 ... episode_{target_total-1}

Behavior:
- Copy all original episodes first (in sorted key order).
- Then add augmented episodes until reaching target_total.
- Augmented episodes are sampled ONLY from original episodes (never from augmented).
- Per augmented episode, sample constant noise:
    (dx, dy) ~ N(0, pos_aug^2)
    yaw      ~ N(0, rot_aug_rad^2)
  Apply to both obs and action:
    - pos: add (dx,dy) to [:, :2]
    - quat: q <- quat_mul(q, q_yaw)
  Other fields are copied unchanged.

Example:
  python augment_data_npy.py --in data.npy --out data_aug.npy \
    --target-total 2000 --pos-aug 0.01 --rot-aug-deg 5 --quat-order xyzw --seed 0
"""
import argparse
import copy
import math
from typing import Dict

import numpy as np


# ----------------------------
# Quaternion utilities
# ----------------------------
def quat_mul_xyzw(q: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Hamilton product for quaternions in (x,y,z,w) order. Returns q * r."""
    x1, y1, z1, w1 = np.split(q, 4, axis=-1)
    x2, y2, z2, w2 = np.split(r, 4, axis=-1)

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.concatenate([x, y, z, w], axis=-1)


def quat_mul_wxyz(q: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Hamilton product for quaternions in (w,x,y,z) order. Returns q * r."""
    w1, x1, y1, z1 = np.split(q, 4, axis=-1)
    w2, x2, y2, z2 = np.split(r, 4, axis=-1)

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.concatenate([w, x, y, z], axis=-1)


def quat_from_yaw_xyzw(yaw: float, T: int) -> np.ndarray:
    """(T,4) yaw quaternion in (x,y,z,w)."""
    half = 0.5 * yaw
    s = math.sin(half)
    c = math.cos(half)
    q = np.zeros((T, 4), dtype=np.float32)
    q[:, 2] = s  # z
    q[:, 3] = c  # w
    return q


def quat_from_yaw_wxyz(yaw: float, T: int) -> np.ndarray:
    """(T,4) yaw quaternion in (w,x,y,z)."""
    half = 0.5 * yaw
    s = math.sin(half)
    c = math.cos(half)
    q = np.zeros((T, 4), dtype=np.float32)
    q[:, 0] = c  # w
    q[:, 3] = s  # z
    return q


def normalize_quat(q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / (n + eps)


def format_episode_name(i: int, width: int) -> str:
    return f"episode_{i:0{width}d}"


# ----------------------------
# Augmentation
# ----------------------------
def augment_one_episode(
    ep: Dict[str, np.ndarray],
    pos_aug: float,
    rot_aug_rad: float,
    rng: np.random.Generator,
    quat_order: str,
) -> Dict[str, np.ndarray]:
    ep2 = copy.deepcopy(ep)

    # constant per-episode noise
    trans = rng.normal(loc=0.0, scale=pos_aug, size=(2,)).astype(np.float32)  # (dx,dy)
    yaw = float(rng.normal(loc=0.0, scale=rot_aug_rad, size=()))

    # translate positions (xy only)
    for k in ("obs.fingertip_pos", "action.fingertip_pos"):
        if k in ep2:
            arr = ep2[k].astype(np.float32, copy=False)  # (T,3)
            arr = arr.copy()
            arr[:, :2] += trans[None, :]
            ep2[k] = arr

    # yaw-rotate quaternions
    for k in ("obs.fingertip_quat", "action.fingertip_quat"):
        if k in ep2:
            q = ep2[k].astype(np.float32, copy=False)  # (T,4)
            q = q.copy()
            T = q.shape[0]

            if quat_order == "xyzw":
                qy = quat_from_yaw_xyzw(yaw, T)
                q = quat_mul_xyzw(q, qy)
            elif quat_order == "wxyz":
                qy = quat_from_yaw_wxyz(yaw, T)
                q = quat_mul_wxyz(q, qy)
            else:
                raise ValueError("quat_order must be 'xyzw' or 'wxyz'")

            ep2[k] = normalize_quat(q)

    return ep2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", help="Input data.npy")
    ap.add_argument("--target-total", type=int, help="Total episodes in output (includes originals)")
    ap.add_argument("--out", dest="out_path", default=None, help="Output data_aug.npy")
    ap.add_argument("--pos-aug", type=float, default=0.02, help="Std dev for (dx,dy) translation noise (pos units)")
    ap.add_argument("--rot-aug-deg", type=float, default=2.0, help="Std dev for yaw noise in degrees")
    ap.add_argument(
        "--quat-order",
        choices=["xyzw", "wxyz"],
        default="xyzw",
        help="Quaternion storage order in arrays (Isaac often uses xyzw).",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.out_path is None:
        args.out_path = args.in_path.replace(".npy", "_aug.npy")

    data_in = np.load(args.in_path, allow_pickle=True).item()
    if not isinstance(data_in, dict) or len(data_in) == 0:
        raise ValueError("Input .npy must contain a non-empty dict of episodes")

    orig_keys = sorted(data_in.keys())  # freeze originals
    N_orig = len(orig_keys)
    target_total = int(args.target_total)

    if target_total < N_orig:
        raise ValueError(f"target_total ({target_total}) < number of original episodes ({N_orig})")

    num_aug = target_total - N_orig
    width = max(4, len(str(target_total - 1)))

    rng = np.random.default_rng(args.seed)
    rot_aug_rad = math.radians(args.rot_aug_deg)

    # Output dict with contiguous keys
    data_out: Dict[str, Dict[str, np.ndarray]] = {}

    # 1) Copy originals (unchanged), re-keyed contiguously
    for i, k in enumerate(orig_keys):
        data_out[format_episode_name(i, width=width)] = data_in[k]

    # 2) Add augmented episodes (sample ONLY from originals) until target_total
    for j in range(num_aug):
        src_key = rng.choice(orig_keys)
        new_ep = augment_one_episode(
            data_in[src_key],
            pos_aug=args.pos_aug,
            rot_aug_rad=rot_aug_rad,
            rng=rng,
            quat_order=args.quat_order,
        )
        idx = N_orig + j
        data_out[format_episode_name(idx, width=width)] = new_ep

    assert len(data_out) == target_total, f"Expected {target_total} episodes, got {len(data_out)}"

    np.save(args.out_path, data_out, allow_pickle=True)
    print(f"Saved {len(data_out)} episodes ({N_orig} original + {num_aug} augmented) to: {args.out_path}")


if __name__ == "__main__":
    main()