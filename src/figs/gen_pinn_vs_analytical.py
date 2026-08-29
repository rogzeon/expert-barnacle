"""
Generate figs/pinn_vs_analytical.png: a 3-panel comparison (PINN prediction |
analytical Black-Scholes solution | absolute error) for a trained forward
checkpoint (see main.py).

Basic usage:
    python gen_pinn_vs_analytical.py --checkpoint ../../models/pinn.pt
"""

import argparse
import sys
from pathlib import Path

import torch

from common.classes import Domain
from pinn.eval_analytical import evaluate, report_errors, resolve_params
from pinn.model import FCN
from pinn.pinn_io import load_pinn
from pinn.plot_heatmap import plot_comparison

REPO_ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(
        description="""Plot PINN vs. analytical Black-Scholes solution."""
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(REPO_ROOT / "models" / "pinn.pt"),
        help="Path to saved model checkpoint (.pt). Defaults to models/pinn.pt.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(REPO_ROOT / "figs" / "pinn_vs_analytical.png"),
        help="Where to save the comparison figure.",
    )
    parser.add_argument(
        "--x-min", type=float, default=None, help="Defaults to checkpoint metadata"
    )
    parser.add_argument(
        "--x-max", type=float, default=None, help="Defaults to checkpoint metadata"
    )
    parser.add_argument(
        "--t-min", type=float, default=None, help="Defaults to checkpoint metadata"
    )
    parser.add_argument(
        "--t-max", type=float, default=None, help="Defaults to checkpoint metadata"
    )
    parser.add_argument(
        "--option",
        type=str,
        default=None,
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

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plot_comparison(X.numpy(), T.numpy(), u_pred.numpy(), u_true.numpy(), args.out)


if __name__ == "__main__":
    main()
