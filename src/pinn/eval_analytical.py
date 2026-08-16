"""
Evaluate a trained PINN against a known analytical solution.

Fill in `analytical_solution(x, t)` below with the closed-form solution you
want to compare against, then run e.g.:

    python eval_analytical.py --checkpoint pinn.pt --nx 200 --nt 200

Requires the model to have been saved after training, e.g.:
    torch.save(pinn.state_dict(), "pinn.pt")
"""

import argparse

import numpy as np
import torch
from model import FCN, Domain  # <-- adjust import to wherever your FCN class lives
from pinn_io import load_pinn
from pinn.plot_heatmap import predict_grid


def _norm_cdf(z: torch.Tensor) -> torch.Tensor:
    """Standard normal CDF, N(z) = 0.5 * (1 + erf(z / sqrt(2)))."""
    return 0.5 * (1.0 + torch.erf(z / torch.sqrt(torch.tensor(2.0, dtype=z.dtype))))


def analytical_solution(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    TODO: replace with the analytical solution u(x, t) you want to test against.

    x, t: flattened tensors of the same shape.
    Must return a tensor of the same shape as x / t.

    Example (1D diffusion on an infinite domain with D=0.1):
        D = 0.1
        return torch.exp(-(x - 5) ** 2 / (4 * D * (t + 0.1))) / torch.sqrt(4 * np.pi * D * (t + 0.1))
    """
    S = x
    K, r, sigma, T = 0.5, 0.02, 0.02, 10

    tau = T - t
    eps = 1e-8
    tau_safe = torch.clamp(tau, min=eps)  # avoid div-by-zero / log(0) right at maturity

    d1 = (torch.log(S / K) + (r + 0.5 * sigma**2) * tau_safe) / (
        sigma * torch.sqrt(tau_safe)
    )
    d2 = d1 - sigma * torch.sqrt(tau_safe)

    price = S * _norm_cdf(d1) - K * torch.exp(-r * tau_safe) * _norm_cdf(d2)
    payoff_at_maturity = torch.clamp(S - K, min=0.0)

    # Use the exact payoff wherever tau is (numerically) zero.
    return torch.where(tau > eps, price, payoff_at_maturity)


def evaluate(
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
    X_np, T_np, u_pred_np = predict_grid(
        model,
        x_min,
        x_max,
        t_min,
        t_max,
        nx,
        nt,
        device,
        rescale=rescale,
        x_scale=x_scale,
        t_scale=t_scale,
        u_scale=u_scale,
    )
    X = torch.from_numpy(X_np)
    T = torch.from_numpy(T_np)
    u_pred = torch.from_numpy(u_pred_np)

    u_true = analytical_solution(X.flatten(), T.flatten()).view(nx, nt)

    return X, T, u_pred, u_true


def report_errors(u_pred: torch.Tensor, u_true: torch.Tensor):
    diff = u_pred - u_true
    mse = torch.mean(diff**2).item()
    mae = torch.mean(diff.abs()).item()
    max_err = diff.abs().max().item()
    l2_rel = (torch.norm(diff) / torch.norm(u_true)).item()

    print("=== Error metrics vs analytical solution ===")
    print(f"MSE:                {mse:.6e}")
    print(f"MAE:                {mae:.6e}")
    print(f"Max abs error:      {max_err:.6e}")
    print(f"Relative L2 error:  {l2_rel:.6e}")

    return {"mse": mse, "mae": mae, "max_err": max_err, "l2_rel": l2_rel}


def main():
    parser = argparse.ArgumentParser(
        description="Test PINN against an analytical solution."
    )
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
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optional .npz path to save X, T, u_pred, u_true",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, metadata = load_pinn(args.checkpoint, FCN, device)
    domain = metadata["domain"]
    domain = Domain.from_dict(metadata["domain"])

    x_min = args.x_min if args.x_min is not None else domain.mindim(0)
    x_max = args.x_max if args.x_max is not None else domain.maxdim(0)
    t_min = args.t_min if args.t_min is not None else domain.t_min
    t_max = args.t_max if args.t_max is not None else domain.t_max

    X, T, u_pred, u_true = evaluate(
        model, x_min, x_max, t_min, t_max, args.nx, args.nt, device, rescale=True
    )
    report_errors(u_pred, u_true)

    if args.save:
        np.savez(
            args.save,
            X=X.numpy(),
            T=T.numpy(),
            u_pred=u_pred.numpy(),
            u_true=u_true.numpy(),
        )
        print(f"Saved evaluation grid to {args.save}")


if __name__ == "__main__":
    main()
