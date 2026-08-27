"""
Solve the Black-Scholes PDE forward in time using a Physics-Informed Neural Network (PINN).

This script trains a simple neural network to approximate the
solution u(x, t) of the Black-Scholes equation given the contract
parameters (strike, sigma, r, option type) and the domain. The network
is trained based using various loss metrics which measure adherence
to both the boundary conditions and the PDE.

Command-line usage:
    python main.py -s <save_path>

The trained model is saved to the specified path.
"""

import argparse

import torch

from common.classes import BlackScholesPDE, Domain, Option
from common.pinn_helpers import (
    create_left_boundary,
    create_right_boundary,
    gen_phys_training_pts,
    get_derivatives,
    initial_condition,
)
from pinn import FCN
from pinn.pinn_io import save_pinn
from pinn.plot_heatmap import plot_single, predict_grid


# Functions below use functions above in order to generate IC.
# Modify functions above to change IC
def create_t0_boundary(x_min, x_max, num_pts, t0=0.0, **kwargs):
    """
    Helper function used to generate the meshgrid and the initial condition array.

    Args:
        x_min:
            Spatial domain lower bound.
        x_max:
            Spatial domain upper bound.
        num_pts:
            Number of points to generate along the spatial dimension.
        t0:
            The (fixed) time value at which to build the initial condition.
            Defaults to 0.0.
        **kwargs:
            PDE parameters (ex. for diffusion, could contain
            the diffusion coefficient.) to be passed to ``initial_condition()``

    Returns:
        space_x:
            A 2D tensor (from meshgrid) of spatial coordinates at t=t0.
        space_t:
            A 2D tensor (from meshgrid) of the constant t0 value, matching
            the shape of space_x.
        u0:
            The initial condition target tensor(s) returned by
            ``initial_condition()``, each reshaped to (N, 1).
    """
    # define boundary points, for the boundary loss
    t0_tf = torch.tensor(t0, dtype=torch.float32).requires_grad_(True)
    x0 = torch.linspace(x_min, x_max, num_pts).requires_grad_(True)
    space_x, space_t = torch.meshgrid(x0, t0_tf, indexing="ij")

    u0 = initial_condition(t0, x0, **kwargs)
    u0 = tuple(v.view(-1, 1) for v in u0)

    return (space_x, space_t, *u0)


def train(domain: Domain, params: BlackScholesPDE):
    """
    Trains a PINN in order to solve the Black-Scholes PDE forward in time.

    Args:
        domain:
            A ``Domain`` object which specifies the distribution of
            physics training points.
        params:
            A ``BlackScholesPDE`` object specifying observable
            parameters of the equation.

    Returns:
        pinn:
            The trained network.
    """
    # PDE paramenters
    MIN_X = domain.mindim(0)
    MAX_X = domain.maxdim(0)
    NUM_PTS_X = domain.ndim(0)
    MIN_T = domain.t_min
    MAX_T = domain.t_max
    NUM_PTS_T = domain.nt

    # PDE parameters
    SIG = params.sigma
    R = params.r
    STRIKE = params.k

    # Normalize
    X_SCALE = MAX_X - MIN_X
    T_SCALE = MAX_T - MIN_T
    U_SCALE = MAX_X - MIN_X
    _ = U_SCALE

    STRIKE = (STRIKE - MIN_X) / X_SCALE
    SIG = SIG * (T_SCALE**0.5)
    R = R * T_SCALE
    MIN_X, MAX_X = 0.0, 1.0
    MIN_T, MAX_T = 0.0, 1.0

    # Setup
    space_x, space_t, u0_target = create_t0_boundary(
        MIN_X, MAX_X, NUM_PTS_X, MAX_T, STRIKE=STRIKE
    )
    boundary_inputs = torch.stack((space_x.flatten(), space_t.flatten()), dim=-1)

    t_boundary_x_limits = torch.linspace(MIN_T, MAX_T, NUM_PTS_T).requires_grad_(True)
    bdry_left, u_bdry_left_tgt = create_left_boundary(t_boundary_x_limits, MIN_X)
    bdry_right, u_bdry_right_tgt = create_right_boundary(
        t_boundary_x_limits, MAX_X, STRIKE=STRIKE, R=R, T_MAX=MAX_T, S_MAX=MAX_X
    )

    t_phys, x_phys, phys_inputs = gen_phys_training_pts(
        MIN_T, MAX_T, MIN_X, MAX_X, NUM_PTS_T, NUM_PTS_X
    )

    print(f"Shape of physics_inputs (x,t): {phys_inputs.shape}")

    # PINN setup
    pinn = FCN(2, 1, 64, 3)
    optimiser = torch.optim.Adam(pinn.parameters(), lr=1e-3)
    lambda1, lambda2 = 1, 1
    print(boundary_inputs.std(dim=0))
    print(u0_target.var().item())
    with torch.no_grad():
        test_x = torch.linspace(0, 1, 10).unsqueeze(1)
        test_t = torch.ones_like(test_x)
        test_in = torch.cat([test_x, test_t], dim=1)
        print(pinn(test_in).flatten())

    def compute_loss():
        u_at_t0 = pinn(boundary_inputs)

        # Boundary losses for initial conditions
        loss1 = torch.mean((u_at_t0 - u0_target) ** 2)

        # Boundary losses at x=0, MAX_X
        u_bdry_left = pinn(bdry_left).view(-1, 1)
        u_bdry_right = pinn(bdry_right).view(-1, 1)

        loss2 = torch.mean((u_bdry_left.view(-1, 1) - u_bdry_left_tgt) ** 2)
        loss3 = torch.mean((u_bdry_right.view(-1, 1) - u_bdry_right_tgt) ** 2)

        u_phys = pinn(phys_inputs)

        # derivatives w.r.t. t, x
        (dudt_phys,) = get_derivatives(u_phys, t_phys, 1)
        dudx_phys, d2udx2_phys = get_derivatives(u_phys, x_phys, 2)

        x_phys_view = x_phys.view(-1, 1)
        # Physics loss
        loss4 = torch.mean(
            (
                dudt_phys
                + 1 / 2 * SIG**2 * (x_phys_view**2) * d2udx2_phys
                + R * (x_phys_view) * dudx_phys
                - R * u_phys
            )
            ** 2
        )

        # backpropagate joint loss, take optimiser step
        loss = loss1 + lambda1 * (loss2 + loss3) + lambda2 * loss4
        return loss, loss1, loss2, loss3, loss4

    for i in range(7501):
        optimiser.zero_grad()

        # loss function hyperparameters
        # inside loop in case the need arises for them to be adaptive.

        # compute boundary loss at t=0
        # Evaluate PINN output u(x, t=0)
        loss, loss1, loss2, loss3, loss4 = compute_loss()
        loss.backward(retain_graph=True)  # Added retain_graph=True here
        optimiser.step()
        if i % 500 == 0:
            print(
                f"[Adam] {i}/7500. Loss: {loss.item():.6e}. "
                f"L1: {loss1.item():.6e}. L2: {loss2.item():.6e}. "
                f"L3: {loss3.item():.6e}. L4: {loss4.item():.6e}"
            )

    lbfgs = torch.optim.LBFGS(
        pinn.parameters(),
        lr=1.0,
        max_iter=2000,
        tolerance_grad=1e-9,
        tolerance_change=1e-12,
        history_size=50,
        line_search_fn="strong_wolfe",
    )

    lbfgs_steps = [0]

    def closure():
        lbfgs.zero_grad()
        loss, loss1, loss2, loss3, loss4 = compute_loss()
        loss.backward(retain_graph=True)
        if lbfgs_steps[0] % 200 == 0:
            print(
                f"[LBFGS] step {lbfgs_steps[0]}. Loss: {loss.item():.6e}. "
                f"L1: {loss1.item():.6e}. L2: {loss2.item():.6e}. "
                f"L3: {loss3.item():.6e}. L4: {loss4.item():.6e}"
            )
        lbfgs_steps[0] += 1
        return loss

    lbfgs.step(closure)

    loss, loss1, loss2, loss3, loss4 = compute_loss()
    print(
        f"[FINAL] Loss: {loss.item():.6e}. "
        f"L1: {loss1.item():.6e}. L2: {loss2.item():.6e}. "
        f"L3: {loss3.item():.6e}. L4: {loss4.item():.6e}"
    )
    return pinn


def main():
    """
    Executes the following tasks:
        1. Parse command-line arguments
        2. set up the domain and PDE parameters,
        3. PINN training
        4. Save model to path given by ``--saveto``
        5. generate a heatmap plot of the PINN output
    """
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        print("GPU available. Running on GPU.")
    torch.set_default_device(device)

    parser = argparse.ArgumentParser(description="Plot PINN solution heatmap.")
    parser.add_argument(
        "-s",
        "--saveto",
        type=str,
        default="models/pinn.pt",
        help="Path to where the model should be saved",
    )
    args = parser.parse_args()
    domain = Domain(
        bounds=((0, 3),), resolution=(40,), t_min=0.0, t_max=10.0, _t_res=10.0
    )
    params = BlackScholesPDE(
        option=Option.EUROCALL, k=0.5, sigma=0.02, r=0.02, t_max=10
    )
    pinn = train(domain, params)
    save_pinn(
        args.saveto,
        pinn,
        pinn.metadata,
        domain=domain.to_metadata(),
        pde=params.to_metadata(),
    )
    pinn = pinn.to("cpu")
    torch.set_default_device("cpu")
    x, t, pred = predict_grid(
        pinn,
        domain.mindim(0),
        domain.maxdim(0),
        domain.t_min,
        domain.t_max,
        domain.ndim(0),
        domain.nt,
        device,
        rescale=True,
    )
    plot_single(x, t, pred, "PINN output", "figs/pinn_results.png")


if __name__ == "__main__":
    main()
