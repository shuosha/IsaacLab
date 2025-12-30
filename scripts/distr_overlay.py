#!/usr/bin/env python3
"""
Minimal "occupancy overlay" visualization.

Given:
  - an empty scene image (background) .jpg
  - a root folder containing episode_XXXX/rgb/000000.jpg (first frame per episode)

For each episode:
  1) load the first RGB frame (000000.jpg)
  2) compute abs-difference vs empty background
  3) threshold to segment "foreground" (robot + objects)
  4) alpha-composite the foreground pixels onto the background

Outputs:
  - a single overlay image showing union of segmented foregrounds across episodes

Example:
  python overlay_segments.py \
    --root logs/real2sim/1229_gearmesh_20/real_teleop \
    --empty empty_scene.jpg \
    --out overlay_union.png \
    --thresh 25 --blur 3 --alpha 0.6
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


def load_rgb(p: Path) -> np.ndarray:
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8)


def find_first_frame(ep_dir: Path) -> Path:
    # Your convention: episode_xxxx/rgb/000000.jpg
    p = ep_dir / "camera_0" / "rgb" / "000000.jpg"
    if p.exists():
        return p
    # fallback: find first jpg in rgb/
    rgb_dir = ep_dir / "camera_0" / "rgb"
    if rgb_dir.exists():
        jpgs = sorted(rgb_dir.glob("*.jpg"))
        if jpgs:
            return jpgs[0]
    raise FileNotFoundError(f"Could not find first frame under {ep_dir}/camera_0/rgb/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="Root folder containing episode_*/")
    ap.add_argument("--empty", type=str, required=True, help="Empty scene/background image (.jpg)")
    ap.add_argument("--out", type=str, default="overlay_union.png")
    ap.add_argument("--thresh", type=int, default=60, help="Diff threshold in [0,255]")
    ap.add_argument("--blur", type=int, default=1, help="Optional mask blur radius (0 disables)")
    ap.add_argument("--alpha", type=float, default=0.6, help="Alpha for compositing per-episode foreground")
    ap.add_argument("--max-episodes", type=int, default=-1, help="Limit episodes for speed (-1 = all)")
    args = ap.parse_args()

    root = Path(args.root)
    empty = load_rgb(Path(args.empty))
    H, W = empty.shape[:2]

    out_img = empty.astype(np.float32)  # accumulate in float

    episodes = sorted([p for p in root.glob("episode_*") if p.is_dir()])
    if args.max_episodes > 0:
        episodes = episodes[: args.max_episodes]

    if not episodes:
        raise FileNotFoundError(f"No episode_* folders found under: {root}")

    for ep in episodes:
        frame_path = find_first_frame(ep)
        rgb = load_rgb(frame_path)

        if rgb.shape[:2] != (H, W):
            raise ValueError(
                f"Size mismatch: empty={empty.shape[:2]} but {frame_path}={rgb.shape[:2]}. "
                "Resize/crop to match before running."
            )

        # --- simple foreground segmentation by abs-diff ---
        diff = np.abs(rgb.astype(np.int16) - empty.astype(np.int16)).astype(np.uint8)  # (H,W,3)
        diff_gray = diff.max(axis=2)  # robust: max channel difference
        mask = (diff_gray >= args.thresh).astype(np.uint8) * 255  # (H,W) 0/255

        # optional blur to soften edges
        if args.blur > 0:
            mask_img = Image.fromarray(mask, mode="L").filter(ImageFilter.GaussianBlur(radius=args.blur))
            mask = np.asarray(mask_img, dtype=np.uint8)

        m = (mask.astype(np.float32) / 255.0)[..., None]  # (H,W,1) in [0,1]

        # --- alpha composite episode foreground onto running canvas ---
        # Only paint where mask is nonzero; scale by args.alpha
        a = float(args.alpha)
        out_img = out_img * (1.0 - a * m) + rgb.astype(np.float32) * (a * m)

    out_img_u8 = np.clip(out_img, 0, 255).astype(np.uint8)
    Image.fromarray(out_img_u8).save(args.out)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
