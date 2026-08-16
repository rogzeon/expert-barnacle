"""
Evaluate a trained PINN against reference data stored in an HDF5 file.

Expected HDF5 layout by default (adjust dataset names with --x-key etc. if yours differ):
  - 'x': 1D array of x coordinates, shape (nx,)
  - 't': 1D array of t coordinates, shape (nt,)
  - 'u': 2D array of reference values, shape (nx, nt)

If your data is instead a flat point cloud (x, t, u all shape (N,)), pass --flat.

Usage:
    python eval_hdf5.py --checkpoint pinn.pt --data results.h5
"""

import argparse

import h5py
import numpy as np
import torch
from pinn import FCN  # <-- adjust import to wherever your FCN class lives
from pinn_io import load_pinn


def load_hdf5(path, x_key, t_key, u_key, flat):
    with h5py.File(path, "r") as f:
        x = np.array(f[x_key])
        t = np.array(f[t_key])
        u = np.array(f[u_key])

    if flat:
        # x, t, u are already the same shape (N,)
        return x, t, u, None
    else:
        # x: (nx,), t: (nt,), u: (nx, nt) -> build meshgrid to match u
        X, T = np.meshgrid(x, t, indexing="ij")
        assert u.shape == X.shape, (
            f"u shape {u.shape} doesn't match meshgrid(x, t) shape {X.shape}; "
            f"pass --flat if your data is already a point list."
        )
        return X.flatten(), T.flatten(), u.flatten(), u.shape


def evaluate(model, x, t, device):
    inputs = torch.tensor(np.stack((x, t), axis=-1), dtype=torch.float32).to(device)
    with torch.no_grad():
        u_pred = model(inputs).cpu().numpy().flatten()
    return u_pred


def report_errors(u_pred: np.ndarray, u_true: np.ndarray):
    diff = u_pred - u_true
    mse = np.mean(diff**2)
    mae = np.mean(np.abs(diff))
    max_err = np.max(np.abs(diff))
    l2_rel = np.linalg.norm(diff) / np.linalg.norm(u_true)

    print("=== Error metrics vs HDF5 reference data ===")
    print(f"MSE:                {mse:.6e}")
    print(f"MAE:                {mae:.6e}")
    print(f"Max abs error:      {max_err:.6e}")
    print(f"Relative L2 error:  {l2_rel:.6e}")

    return {"mse": mse, "mae": mae, "max_err": max_err, "l2_rel": l2_rel}


def main():
    parser = argparse.ArgumentParser(
        description="Test PINN against HDF5 reference data."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to saved model checkpoint (.pt)",
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to .h5/.hdf5 file with reference data",
    )
    parser.add_argument("--x-key", type=str, default="x")
    parser.add_argument("--t-key", type=str, default="t")
    parser.add_argument("--u-key", type=str, default="u")
    parser.add_argument(
        "--flat",
        action="store_true",
        help="Set if x, t, u are all flat point clouds of equal length",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optional .npz path to save x, t, u_pred, u_true",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _metadata = load_pinn(args.checkpoint, FCN, device)

    x, t, u_true, grid_shape = load_hdf5(
        args.data, args.x_key, args.t_key, args.u_key, args.flat
    )
    u_pred = evaluate(model, x, t, device)

    report_errors(u_pred, u_true)

    if args.save:
        np.savez(
            args.save, x=x, t=t, u_pred=u_pred, u_true=u_true, grid_shape=grid_shape
        )
        print(f"Saved evaluation data to {args.save}")


if __name__ == "__main__":
    main()
