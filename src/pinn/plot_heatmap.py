"""
Plot the PINN solution as a heatmap over the (x, t) domain.

Basic usage (just the PINN prediction):
    python plot_heatmap.py --checkpoint pinn.pt

If you also pass --compare, it loads an .npz file produced by eval_analytical.py
or eval_hdf5.py (containing X/T or x/t, u_pred, u_true) and plots prediction,
reference, and error side by side instead.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path

from common.classes import Domain
from pinn.model import FCN
from pinn.pinn_io import load_pinn

REPO_ROOT = Path(__file__).resolve().parents[2]


def predict_grid(
    model,
    x_min,
    x_max,
    t_min,
    t_max,
    nx,
    nt,
    device,
    rescale=False,
    x_scale=None,
    t_scale=None,
    u_scale=None,
):
    if x_scale is None:
        x_scale = x_max - x_min
    if t_scale is None:
        t_scale = t_max - t_min
    if u_scale is None:
        u_scale = x_max - x_min

    if rescale:
        x_min_in, x_max_in = 0.0, (x_max - x_min) / x_scale
        t_min_in, t_max_in = 0.0, (t_max - t_min) / t_scale
    else:
        x_min_in, x_max_in = x_min, x_max
        t_min_in, t_max_in = t_min, t_max

    x = torch.linspace(x_min_in, x_max_in, nx)
    t = torch.linspace(t_min_in, t_max_in, nt)
    X, T = torch.meshgrid(x, t, indexing="ij")
    inputs = torch.stack((X.flatten(), T.flatten()), dim=-1).to(device)
    with torch.no_grad():
        u_pred = model(inputs).cpu().view(nx, nt)

    if rescale:
        X, T, u_pred = X * x_scale + x_min, T * t_scale + t_min, u_pred * u_scale

    return X.numpy(), T.numpy(), u_pred.numpy()


def plot_single(X, T, U, title, out_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    mesh = ax.pcolormesh(T, X, U, shading="auto", cmap="viridis")
    fig.colorbar(mesh, ax=ax, label="u(x, t)")
    ax.set_xlabel("t")
    ax.set_ylabel("x")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")
    plt.close(fig)


def plot_comparison(X, T, u_pred, u_true, out_path):
    err = u_pred - u_true
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    for ax, data, title, cmap in zip(
        axes,
        [u_pred, u_true, err],
        ["PINN prediction", "Reference", "Error (pred - ref)"],
        ["viridis", "viridis", "RdBu_r"],
    ):
        is_err = title.startswith("Error")
        vmax = np.max(np.abs(data)) if is_err else None
        vmin = -vmax if is_err else None
        mesh = ax.pcolormesh(
            T, X, data, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax
        )
        fig.colorbar(mesh, ax=ax)
        ax.set_xlabel("t")
        ax.set_title(title)
    axes[0].set_ylabel("x")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot PINN solution heatmap.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to saved model checkpoint (.pt)",
    )
    parser.add_argument(
        "--x-min",
        type=float,
        default=None,
        help="Defaults to value stored in checkpoint metadata",
    )
    parser.add_argument(
        "--x-max",
        type=float,
        default=None,
        help="Defaults to value stored in checkpoint metadata",
    )
    parser.add_argument(
        "--t-min",
        type=float,
        default=None,
        help="Defaults to value stored in checkpoint metadata",
    )
    parser.add_argument(
        "--t-max",
        type=float,
        default=None,
        help="Defaults to value stored in checkpoint metadata",
    )
    parser.add_argument("--nx", type=int, default=200)
    parser.add_argument("--nt", type=int, default=200)
    parser.add_argument("--out", type=str, default="pinn_heatmap.png")
    parser.add_argument(
        "--compare",
        type=str,
        default=None,
        help="Path to an .npz file from eval_analytical.py / eval_hdf5.py with X/T (or x/t), u_pred, u_true",
    )
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, metadata = load_pinn(args.checkpoint, FCN, device)

    domain = Domain.from_metadata(metadata["domain"])
    print(domain)

    if args.compare:
        data = np.load(args.compare)
        if "X" in data:
            X, T = data["X"], data["T"]
        else:
            X, T = np.meshgrid(
                np.unique(data["x"]), np.unique(data["t"]), indexing="ij"
            )
        u_pred = data["u_pred"].reshape(X.shape)
        u_true = data["u_true"].reshape(X.shape)
        plot_comparison(X, T, u_pred, u_true, args.out)
    else:
        x_min = args.x_min if args.x_min is not None else domain.mindim(0)
        x_max = args.x_max if args.x_max is not None else domain.maxdim(0)
        t_min = args.t_min if args.t_min is not None else domain.t_min
        t_max = args.t_max if args.t_max is not None else domain.t_max
        X, T, u_pred = predict_grid(
            model, x_min, x_max, t_min, t_max, args.nx, args.nt, device, rescale=True
        )
        plot_single(
            X, T, u_pred, "PINN prediction u(x, t)", str(REPO_ROOT / "figs" / args.out)
        )


if __name__ == "__main__":
    main()
