import argparse

import torch

from common.classes import BlackScholesPDE, Domain, Option
from pinn import FCN
from pinn.pinn_io import save_pinn
from pinn.plot_heatmap import plot_single, predict_grid


# The following functions create the IC and/or boundary conditions.
def initial_condition(t0, x0, **kwargs):
    """
    Constructs the initial condition off of which to train.
    Modify this function in order to change the IC.
    """
    u0_target = torch.clamp(x0 - kwargs["STRIKE"], min=0)
    # eps = 0.0025
    # k = x0 - kwargs["STRIKE"]
    # u0_target = (k + torch.sqrt(k**2 + eps)) / 2
    #   dudt0_target = torch.zeros_like(x0).view(-1, 1).requires_grad_(False)

    return (u0_target,)


# Helpers for create_left/right_boundary, should return the values at respective boundaries
def boundary_left(space_t_left, **kwargs):
    return torch.zeros_like(space_t_left.flatten())


def boundary_right(space_t_right, **kwargs):
    return kwargs["S_MAX"] - kwargs["STRIKE"] * torch.exp(
        -kwargs["R"] * (kwargs["T_MAX"] - space_t_right.flatten())
    )


# Functions below use functions above in order to generate IC.
# Modify functions above to change IC
def create_t0_boundary(x_min, x_max, num_pts, t0=0.0, **kwargs):
    """
    Helper function used to generate the meshgrid and the initial condition array.
    """
    # define boundary points, for the boundary loss
    t0_tf = torch.tensor(t0, dtype=torch.float32).requires_grad_(True)
    x0 = torch.linspace(x_min, x_max, num_pts).requires_grad_(True)
    space_x, space_t = torch.meshgrid(x0, t0_tf, indexing="ij")

    u0 = initial_condition(t0, x0, **kwargs)
    u0 = tuple(v.view(-1, 1) for v in u0)

    return (space_x, space_t, *u0)


def create_left_boundary(t_boundary_x_limits, x_min, **kwargs):
    x_left = torch.tensor(x_min, dtype=torch.float32).requires_grad_(True)
    space_x_left, space_t_left = torch.meshgrid(
        x_left, t_boundary_x_limits, indexing="ij"
    )
    boundary_inputs_left = torch.stack(
        (space_x_left.flatten(), space_t_left.flatten()), dim=-1
    )
    boundary_values_left = boundary_left(space_t_left, **kwargs).view(-1, 1)
    return boundary_inputs_left, boundary_values_left


def create_right_boundary(t_boundary_x_limits, x_max, **kwargs):
    x_right = torch.tensor(x_max, dtype=torch.float32).requires_grad_(True)
    space_x_right, space_t_right = torch.meshgrid(
        x_right, t_boundary_x_limits, indexing="ij"
    )
    boundary_inputs_right = torch.stack(
        (space_x_right.flatten(), space_t_right.flatten()), dim=-1
    )
    boundary_values_right = boundary_right(space_t_right, **kwargs).view(-1, 1)
    return boundary_inputs_right, boundary_values_right


# The following function(s) help set up training
def gen_phys_training_pts(min_t, max_t, min_x, max_x, num_pts_t, num_pts_x):
    """
    Generates physics loss training points for the PDE.
    Modify as needed for higher-dimensional PDEs.

    First return values are vectors for use with autograd,
    final return value is for input to the PINN.
    """
    t_physics_1d = torch.linspace(min_t, max_t, num_pts_t).requires_grad_(True)
    x_physics_1d = torch.linspace(min_x, max_x, num_pts_x).requires_grad_(True)
    space_x_physics, space_t_physics = torch.meshgrid(
        x_physics_1d, t_physics_1d, indexing="ij"
    )

    flattened_x_physics = space_x_physics.flatten()
    flattened_t_physics = space_t_physics.flatten()
    phys_inputs = torch.stack((flattened_x_physics, flattened_t_physics), dim=-1)

    return flattened_t_physics, flattened_x_physics, phys_inputs


def get_derivatives(output, wrt, num):
    derivs = []
    for _ in range(num):
        # First derivative w.r.t. x
        if len(derivs) == 0:
            ddx = torch.autograd.grad(
                output,
                wrt,
                torch.ones_like(output),
                create_graph=True,
            )[0].view(-1, 1)
        else:
            ddx = torch.autograd.grad(
                derivs[-1],
                wrt,
                torch.ones_like(derivs[-1]),
                create_graph=True,
            )[0].view(-1, 1)

        derivs.append(ddx)

    return tuple(derivs)


def train(domain: Domain, params: BlackScholesPDE):
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
    U_SCALE = MAX_X - MIN_X  # same convention as predict_grid's u_scale default

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
    # Domain's fields are (bounds, resolution, t_max, t_min, _t_res) in that
    # order -- the previous positional call here was
    # `Domain(((0, 3),), (40,), 0, 10, 10)`, which silently bound t_max=0
    # and t_min=10 (inverted), rather than the clearly-intended t_min=0,
    # t_max=10. Written out with keywords so this can't happen again, and so
    # it now matches the t_max=10 already passed to BlackScholesPDE below.
    domain = Domain(bounds=((0, 3),), resolution=(40,), t_min=0.0, t_max=10.0, _t_res=10.0)
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
