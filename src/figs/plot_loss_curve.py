"""
Plot the training loss curves from a run of main.py.

Basic usage:
    python plot_loss_curve.py --history ../../figs/loss_history.npz

Reads the .npz file main.py's train() saves during training and plots the losses
on a shared log-scale plot, with a vertical line marking the Adam -> LBFGS
transition.

Note LBGFS save frequency is different from Adam's.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]


def plot_loss_curve(history_path: str, out_path: str) -> None:
    """
    Load a loss history .npz (saved by main.py's train()) and plot it.

    Args:
        history_path:
            Path to the .npz file, with arrays "phase", "iter", "loss",
            "loss1", "loss2", "loss3", "loss4" (one entry per logged
            iteration).
        out_path:
            Where to save the resulting figure.
    """
    data = np.load(history_path)
    iters = data["iter"]
    phase = data["phase"]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(iters, data["loss"], label="Total", linewidth=2, color="black")
    ax.plot(iters, data["loss1"], label="Initial condition", alpha=0.8)
    ax.plot(iters, data["loss2"], label="Left boundary", alpha=0.8)
    ax.plot(iters, data["loss3"], label="Right boundary", alpha=0.8)
    ax.plot(iters, data["loss4"], label="PDE residual", alpha=0.8)

    # Mark the Adam -> LBFGS transition, if both phases were actually logged
    # (e.g. a very short/interrupted run might only have Adam entries).
    is_lbfgs = phase == "lbfgs"
    if is_lbfgs.any() and (~is_lbfgs).any():
        transition_iter = iters[is_lbfgs][0]
        ax.axvline(transition_iter, color="gray", linestyle="--", linewidth=1)
        ax.text(
            transition_iter,
            ax.get_ylim()[1],
            " LBFGS starts",
            va="top",
            fontsize=8,
            color="gray",
        )

    ax.set_yscale("log")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss (log scale)")
    ax.set_title("Training loss over Adam + LBFGS phases")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot PINN training loss curve.")
    parser.add_argument(
        "--history",
        type=str,
        default=str(REPO_ROOT / "figs" / "loss_history.npz"),
        help="Path to the loss history saved by main.py",
    )
    parser.add_argument(
        "--out", type=str, default=str(REPO_ROOT / "figs" / "loss_curve.png")
    )
    args = parser.parse_args()
    plot_loss_curve(args.history, args.out)


if __name__ == "__main__":
    main()
