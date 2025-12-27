import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=str)
    ap.add_argument("out", type=str, default="peg_insert_bar_ci.png")
    ap.add_argument("--sort", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    # column names (robust to small variations)
    col_method = "Peg Insert"
    col_sr = "sr"
    col_elow = "err low"
    col_ehigh = "err high"

    if args.sort:
        df = df.sort_values(col_sr, ascending=True).reset_index(drop=True)

    methods = df[col_method].astype(str).tolist()
    sr = df[col_sr].to_numpy()
    yerr = np.vstack([
        df[col_elow].to_numpy(),
        df[col_ehigh].to_numpy()
    ])

    x = np.arange(len(methods))

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(8.5, 4.0), dpi=200)

    bar_width = 0.6
    bars = ax.bar(
        x, sr,
        width=bar_width,
        color="#4C72B0",
        edgecolor="black",
        linewidth=0.8,
        zorder=2
    )

    ax.errorbar(
        x, sr,
        yerr=yerr,
        fmt="none",
        ecolor="black",
        elinewidth=1.2,
        capsize=4,
        capthick=1.2,
        zorder=3,
        alpha=0.8
    )

    # axes & grid
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Success rate")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)

    ax.grid(
        axis="y",
        linestyle="--",
        linewidth=0.6,
        alpha=0.6,
        zorder=0
    )

    ax.set_axisbelow(True)
    ax.set_title("Peg Insert Success Rate")

    plt.tight_layout()
    plt.savefig(args.out, bbox_inches="tight")
    print(f"Saved: {args.out}")

if __name__ == "__main__":
    main()
