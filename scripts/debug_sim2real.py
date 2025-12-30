import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.lines import Line2D


# -------------------------
# IO helpers
# -------------------------
def load_npy_dict(path):
    return np.load(path, allow_pickle=True).item()


def assert_same_structure(A, B):
    a_eps = sorted(A.keys())
    b_eps = sorted(B.keys())
    assert a_eps == b_eps, f"Episode keys differ.\nA={a_eps}\nB={b_eps}"
    for ep in a_eps:
        ak = set(A[ep].keys())
        bk = set(B[ep].keys())
        assert ak == bk, f"Subkeys differ in {ep}.\nOnlyA={sorted(ak-bk)}\nOnlyB={sorted(bk-ak)}"


def assert_actions_equal(epA, epB, ep_name, rtol=0.0, atol=1e-7):
    a_keys = sorted([k for k in epA.keys() if k.startswith("action.")])
    b_keys = sorted([k for k in epB.keys() if k.startswith("action.")])
    assert a_keys == b_keys, f"{ep_name}: action subkeys differ."

    for k in a_keys:
        a = epA[k]
        b = epB[k]
        T = min(a.shape[0], b.shape[0])
        aT = a[:T]
        bT = b[:T]

        diff = np.abs(aT - bT)
        max_err = float(diff.max())

        if max_err > atol:
            import pdb; pdb.set_trace()
            # find timestep + dim of max error
            idx = np.unravel_index(np.argmax(diff), diff.shape)
            t = idx[0]
            d = idx[1] if diff.ndim > 1 else 0

            raise AssertionError(
                f"{ep_name}: action mismatch at key='{k}', "
                f"t={t}, dim={d}, max|Δ|={max_err}"
            )

# -------------------------
# Plot helpers
# -------------------------
def _trim_T(*arrs):
    Ts = [a.shape[0] for a in arrs if a is not None]
    T = min(Ts) if Ts else 0
    return [(a[:T] if a is not None else None) for a in arrs], T


def plot_1d(ax, a_obs_1d, b_obs_1d, a_act_1d=None, ylabel=""):
    (a_obs_1d, b_obs_1d, a_act_1d), T = _trim_T(a_obs_1d, b_obs_1d, a_act_1d)
    if T == 0:
        ax.set_axis_off()
        return

    lw = 2.5

    # action (only one, from file A)
    if a_act_1d is not None:
        ax.plot(a_act_1d, color="red", lw=lw)

    # observations (A green, B dotted yellow)
    ax.plot(a_obs_1d, color="green", lw=lw)
    ax.plot(b_obs_1d, color="gold", lw=lw, ls=":")

    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.2)


def get_obs_and_action(epA, epB, obs_key):
    """
    Returns (a_obs, b_obs, a_act_or_None). Action is taken from file A only.
    """
    assert obs_key in epA and obs_key in epB, f"Missing key: {obs_key}"
    name = obs_key.split(".", 1)[1]       # e.g., fingertip_pos
    act_key = f"action.{name}"

    a_obs = epA[obs_key]
    b_obs = epB[obs_key]
    a_act = epA.get(act_key, None)

    return a_obs, b_obs, a_act


def compute_asset_pos(ep, rel_key):
    # asset_pos = fingertip_pos - rel_pos
    return ep["obs.fingertip_pos"] - ep[rel_key]


# -------------------------
# Main plotting per-episode
# -------------------------
def plot_episode(epA, epB, out_path):
    fig, axes = plt.subplots(4, 6, figsize=(30, 14), sharex=True)

    # ---- Col 0: fingertip_pos (x,y,z) + gripper ----
    a_obs, b_obs, a_act = get_obs_and_action(epA, epB, "obs.fingertip_pos")
    for r, lab in enumerate(["x [m]", "y [m]", "z [m]"]):
        plot_1d(
            axes[r, 0],
            a_obs[:, r],
            b_obs[:, r],
            a_act_1d=(a_act[:, r] if a_act is not None else None),
            ylabel=lab,
        )

    a_obs, b_obs, a_act = get_obs_and_action(epA, epB, "obs.gripper")
    plot_1d(
        axes[3, 0],
        a_obs[:, 0],
        b_obs[:, 0],
        a_act_1d=(a_act[:, 0] if a_act is not None else None),
        ylabel="openness",
    )

    # # ---- ZOOM-IN LOGIC (NEW) ----
    # y = a_obs[:, 0]  # use real obs gripper to determine window
    # mask = (y >= 0.005) & (y <= 0.69)
    # if np.any(mask):
    #     idx = np.where(mask)[0]
    #     axes[3, 0].set_xlim(idx[0], idx[-1])

    # ---- Col 1: fingertip_quat (w,x,y,z) ----
    a_obs, b_obs, a_act = get_obs_and_action(epA, epB, "obs.fingertip_quat")
    for r, lab in enumerate(["w", "x", "y", "z"]):
        plot_1d(
            axes[r, 1],
            a_obs[:, r],
            b_obs[:, r],
            a_act_1d=(a_act[:, r] if a_act is not None else None),
            ylabel=lab,
        )

    # ---- Col 2: fixed asset absolute position (x,y,z) ----
    a_fixed = compute_asset_pos(epA, "obs.fingertip_pos_rel_fixed")
    b_fixed = compute_asset_pos(epB, "obs.fingertip_pos_rel_fixed")
    for r, lab in enumerate(["x [m]", "y [m]", "z [m]"]):
        plot_1d(axes[r, 2], a_fixed[:, r], b_fixed[:, r], ylabel=lab)
    axes[3, 2].set_axis_off()

    # ---- Col 3: held asset absolute position (x,y,z) ----
    a_held = compute_asset_pos(epA, "obs.fingertip_pos_rel_held")
    b_held = compute_asset_pos(epB, "obs.fingertip_pos_rel_held")
    for r, lab in enumerate(["x [m]", "y [m]", "z [m]"]):
        plot_1d(axes[r, 3], a_held[:, r], b_held[:, r], ylabel=lab)
    axes[3, 3].set_axis_off()

    # ---- Col 4: ee_linvel_fd (x,y,z) ----
    a_obs, b_obs, a_act = get_obs_and_action(epA, epB, "obs.ee_linvel_fd")
    for r, lab in enumerate(["x [m/s]", "y [m/s]", "z [m/s]"]):
        plot_1d(
            axes[r, 4],
            a_obs[:, r],
            b_obs[:, r],
            a_act_1d=(a_act[:, r] if a_act is not None else None),
            ylabel=lab,
        )
    axes[3, 4].set_axis_off()

    # ---- Col 5: ee_angvel_fd (r,p,y) ----
    a_obs, b_obs, a_act = get_obs_and_action(epA, epB, "obs.ee_angvel_fd")
    for r, lab in enumerate(["r [rad/s]", "p [rad/s]", "y [rad/s]"]):
        plot_1d(
            axes[r, 5],
            a_obs[:, r],
            b_obs[:, r],
            a_act_1d=(a_act[:, r] if a_act is not None else None),
            ylabel=lab,
        )
    axes[3, 5].set_axis_off()

    # x-labels (bottom visible row per column)
    axes[3, 0].set_xlabel("timestep")
    axes[3, 1].set_xlabel("timestep")
    for c in range(2, 6):
        axes[2, c].set_xlabel("timestep")

    # legend outside
    lw = 2.5
    handles = [
        Line2D([0], [0], color="green", lw=lw, label="real obs (file A)"),
        Line2D([0], [0], color="gold",  lw=lw, ls=":", label="sim obs (file B)"),
        Line2D([0], [0], color="red",   lw=lw, label="action"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.995))
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(path_a, path_b, out_dir):
    A = load_npy_dict(path_a)  # real
    B = load_npy_dict(path_b)  # sim
    assert_same_structure(A, B)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for ep in sorted(A.keys()):
        try:
            # Assert action equality BEFORE plotting this episode
            assert_actions_equal(A[ep], B[ep], ep_name=ep, rtol=0.0, atol=5e-7)
        except AssertionError as e:
            print(f"[SKIP] {ep}: {e}")
            continue

        eps_id = ep.split("_", 1)[1] if "_" in ep else ep
        out_path = out_dir / f"eps_{eps_id}_real2sim.png"
        plot_episode(A[ep], B[ep], out_path)
        print(f"[INFO] Saved: {out_path}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("path_a", type=str, help="file A .npy (real)")
    p.add_argument("path_b", type=str, help="file B .npy (sim)")
    p.add_argument("--out_dir", type=str, default="real2sim_viz", help="output dir")
    args = p.parse_args()
    main(args.path_a, args.path_b, args.out_dir)
