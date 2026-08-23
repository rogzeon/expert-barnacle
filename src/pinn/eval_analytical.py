"""
Evaluate a trained PINN against the closed-form Black-Scholes solution.

    python eval_analytical.py --checkpoint pinn.pt --nx 200 --nt 200

The contract parameters (strike, sigma, r, option type) are read from the
checkpoint's saved metadata (see pinn_io.save_pinn / common.classes.
BlackScholesPDE.to_metadata) so they always match what the model was
actually trained on. Use --strike / --sigma / --r / --option / --t-max to
override any of them manually, e.g. to compare the model against a
*different* contract than it was trained on.

Requires a checkpoint saved with pinn_io.save_pinn(), e.g.:
    save_pinn("pinn.pt", pinn, arch, domain=domain.to_metadata(),
              pde=params.to_metadata())
"""

import argparse
import sys

import numpy as np
import torch
from model import FCN

from common.classes import BlackScholesPDE, Domain, Option
from pinn.pinn_io import load_pinn
from pinn.plot_heatmap import predict_grid


def _norm_cdf(z: torch.Tensor) -> torch.Tensor:
    """Standard normal CDF, N(z) = 0.5 * (1 + erf(z / sqrt(2)))."""
    return 0.5 * (1.0 + torch.erf(z / torch.sqrt(torch.tensor(2.0, dtype=z.dtype))))


def analytical_solution(
    x: torch.Tensor,
    t: torch.Tensor,
    params: BlackScholesPDE,
    t_max: float,
) -> torch.Tensor:
    """
    Closed-form Black-Scholes price for a European call, u(x, t).

    x, t:    flattened tensors of the same shape (asset price, time).
    params:  strike/sigma/r for the contract being priced. Pull this from
             the checkpoint (see main()) rather than hardcoding it, so the
             analytical solution always matches what the model trained on.
    t_max:   contract maturity (time to expiry is t_max - t).

    Returns a tensor of the same shape as x / t.
    """
    S = x
    K, r, sigma = params.k, params.r, params.sigma

    tau = t_max - t
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
    params: BlackScholesPDE,
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

    u_true = analytical_solution(X.flatten(), T.flatten(), params, t_max).view(nx, nt)

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


def resolve_params(metadata: dict, args, t_max: float) -> BlackScholesPDE:
    """
    Build the BlackScholesPDE to evaluate against: start from whatever the
    checkpoint has saved (expected to be BlackScholesPDE.to_metadata()'s
    output, i.e. keys "option"/"K"/"sigma"/"r"), then apply any
    --option/--strike/--sigma/--r overrides. Errors clearly if a value
    isn't available from either source.

    ``t_max`` is passed in explicitly rather than pulled from the saved PDE
    metadata: BlackScholesPDE.to_metadata() doesn't currently emit a
    top-level "t_max" key (it's only nested under "domain"), so it can't be
    resolved from `saved` alone. Callers should derive it from the
    checkpoint's domain metadata / --t-max instead (see main()).
    """
    saved = metadata.get("pde") or {}

    def resolve(name: str, cli_value, saved_key: str | None = None):
        if cli_value is not None:
            return cli_value
        key = saved_key or name
        if key in saved:
            return saved[key]
        raise ValueError(
            f"'{key}' isn't in the checkpoint's saved PDE metadata (it may "
            f"predate the pde= field in save_pinn, or predate the option "
            f"field) and no --{name} override was given. Either re-save the "
            f"checkpoint with pde=params.to_metadata(), or pass --{name} "
            f"explicitly."
        )

    option_value = resolve("option", args.option, saved_key="option")
    option = option_value if isinstance(option_value, Option) else Option[option_value]

    return BlackScholesPDE(
        option=option,
        k=resolve("strike", args.strike, saved_key="K"),
        sigma=resolve("sigma", args.sigma),
        r=resolve("r", args.r),
        t_max=t_max,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Test PINN against the analytical Black-Scholes solution."
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
    parser.add_argument(
        "--option",
        type=str,
        default=None,
        choices=[o.name for o in Option],
        help="Override the option type. Defaults to the value saved in the checkpoint.",
    )
    parser.add_argument(
        "--strike",
        type=float,
        default=None,
        help="Override the strike K. Defaults to the value saved in the checkpoint.",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=None,
        help="Override volatility sigma. Defaults to the value saved in the checkpoint.",
    )
    parser.add_argument(
        "--r",
        type=float,
        default=None,
        help="Override risk-free rate r. Defaults to the value saved in the checkpoint.",
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

    try:
        model, metadata = load_pinn(args.checkpoint, FCN, device)
        domain = Domain.from_metadata(metadata["domain"])

        x_min = args.x_min if args.x_min is not None else domain.mindim(0)
        x_max = args.x_max if args.x_max is not None else domain.maxdim(0)
        t_min = args.t_min if args.t_min is not None else domain.t_min
        t_max = args.t_max if args.t_max is not None else domain.t_max

        params = resolve_params(metadata, args, t_max)
    except (FileNotFoundError, RuntimeError, KeyError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(
        f"Evaluating against option={params.option.name}, strike={params.k}, "
        f"sigma={params.sigma}, r={params.r}, t_max={t_max}"
    )

    X, T, u_pred, u_true = evaluate(
        model,
        x_min,
        x_max,
        t_min,
        t_max,
        args.nx,
        args.nt,
        device,
        params,
        rescale=True,
    )
    report_errors(u_pred, u_true)

    if args.save:
        try:
            np.savez(
                args.save,
                X=X.numpy(),
                T=T.numpy(),
                u_pred=u_pred.numpy(),
                u_true=u_true.numpy(),
            )
        except OSError as e:
            print(
                f"Error: failed to save evaluation grid to '{args.save}': {e}",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"Saved evaluation grid to {args.save}")


if __name__ == "__main__":
    main()
