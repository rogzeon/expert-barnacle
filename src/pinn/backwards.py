import argparse
import matplotlib.pyplot as plt

import torch
from model import ConstrainedFCN

from common.classes import BlackScholesPDE, Domain
from common.funcs import read_black_scholes_run, modify_domain
from pinn.pinn_io import save_pinn
from pinn.plot_heatmap import plot_single, predict_grid


# The following functions create the IC and/or boundary conditions.
def initial_condition(t0: float, x0: torch.Tensor, **kwargs):
    """
    Constructs the initial condition off of which to train.
    Modify this function in order to change the IC.

    Args:
        t0:
            Initial value of t
        x0:
            Tensor representing the values of x at t=t0
        kwargs:
            PDE parameters (ex. for diffusion, could contain
            the diffusion coefficient.)

    Returns a 1D tensor representing the values of the solution at t=t0.
    """
    u0_target = torch.clamp(x0 - kwargs["STRIKE"], min=0)

    return (u0_target,)


# Helpers for create_left/right_boundary, should return the values at respective boundaries
def boundary_left(space_t_left: torch.Tensor, **kwargs) -> torch.Tensor:
    """
    Helper function that builds the x=x_min boundary of the PDE

    Args:
        space_t_left:
            A prebuilt linspace from t_min to t_max.
        kwargs: PDE parameters (ex. for diffusion, could contain
            the diffusion coefficient.)

    Returns a 1D tensor representing the x=x_min boundary.
    """
    return torch.zeros_like(space_t_left.flatten())


def boundary_right(space_t_right: torch.Tensor, **kwargs) -> torch.Tensor:
    """
    Helper function that builds the x=x_max boundary of the PDE

    Args:
        space_t_left: A prebuilt linspace from t_min to t_max.
        kwargs: PDE parameters (ex. for diffusion, could contain
            the diffusion coefficient.

    Returns a tensor representing the x=x_max boundary.
    """
    return kwargs["S_MAX"] - kwargs["STRIKE"] * torch.exp(
        -kwargs["R"] * (kwargs["T_MAX"] - space_t_right.flatten())
    )


def create_left_boundary(
    t_boundary_x_limits: torch.Tensor, x_min: float, **kwargs
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Create input points and target values for the x = ``x_min`` boundary.

    The boundary points are formed by pairing the fixed ``x_min`` with each temporal
    coordinate in ``t_boundary_x_limits``. The target values are computed by calling
    ``boundary_left()`` with the temporal grid and any additional PDE parameters
    provided in kwargs.

    Args:
        t_boundary_x_limits:
            A 1D tensor of temporal coordinates at which to evaluate the boundary,
            typically a linspace from t_min to t_max. Requires gradients enabled.
        x_min:
            The spatial coordinate of the left boundary
        **kwargs:
            PDE parameters (ex. for diffusion, could contain
            the diffusion coefficient.) to be passed to ``boundary_left()``

    Returns:
        boundary_inputs_left:
            A 2D tensor of shape (N, 2) where each row is (x, t) for the boundary
            points, with x = ``x_min`` for all points.
        boundary_values_left:
            A 2D tensor of shape (N, 1) containing the corresponding target
            boundary values from ``boundary_left()``.
    """
    x_left = torch.tensor(x_min, dtype=torch.float32).requires_grad_(True)
    space_x_left, space_t_left = torch.meshgrid(
        x_left, t_boundary_x_limits, indexing="ij"
    )
    boundary_inputs_left = torch.stack(
        (space_x_left.flatten(), space_t_left.flatten()), dim=-1
    )
    boundary_values_left = boundary_left(space_t_left, **kwargs).view(-1, 1)
    return boundary_inputs_left, boundary_values_left


def create_right_boundary(
    t_boundary_x_limits: torch.Tensor, x_max: float, **kwargs
) -> tuple[torch.Tensor, torch.Tensor]:
    x_right = torch.tensor(x_max, dtype=torch.float32).requires_grad_(True)
    space_x_right, space_t_right = torch.meshgrid(
        x_right, t_boundary_x_limits, indexing="ij"
    )
    boundary_inputs_right = torch.stack(
        (space_x_right.flatten(), space_t_right.flatten()), dim=-1
    )
    boundary_values_right = boundary_right(space_t_right, **kwargs).view(-1, 1)
    return boundary_inputs_right, boundary_values_right


def create_data_points(
    data: torch.Tensor,
    x_min: float,
    x_max: float,
    t_min: float,
    t_max: float,
    x_scale: float,
):
    """
    Build (x, t) input points and matching targets from a stored PDE run's
    solution tensor, in the same normalized [0, 1] x [0, 1] space used
    everywhere else in this module.

    Args:
        data: The solved PDE field values (forward-chronological).
        x_min: Spatial domain lower bound.
        x_max: Spatial domain upper bound.
        t_min: Temporal domain lower bound.
        t_max: Temporal domain upper bound.
        x_scale: The physical range of the spatial domain, used to normalize
            the PDE values to match the rest of the model's targets.

    Returns:
        data_inputs:
            Tensor with columns (x, t) representing the input points.
        u_real:
            Tensor containing the corresponding normalized target values.
    """
    n_times, n_x = data.shape
    x_data = torch.linspace(x_min, x_max, n_x)
    t_data = torch.linspace(t_min, t_max, n_times)
    # indexing="ij" with (t, x) so flattening matches data's (n_times, n_x)
    # row-major layout: t varies slowest, x varies fastest.
    space_t_data, space_x_data = torch.meshgrid(t_data, x_data, indexing="ij")

    data_inputs = torch.stack((space_x_data.flatten(), space_t_data.flatten()), dim=-1)
    u_real = (data.flatten() / x_scale).view(-1, 1)
    return data_inputs, u_real


# The following function(s) help set up training
def gen_phys_training_pts(
    min_t: float,
    max_t: float,
    min_x: float,
    max_x: float,
    num_pts_t: int,
    num_pts_x: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate a grid of (x, t) training points for the physics loss (PDE residual)
    evaluation, in the normalized [0, 1] x [0, 1] space used elsewhere in
    this module.

    Args:
        min_t:
            Temporal domain lower bound.
        max_t:
            Temporal domain upper bound.
        min_x:
            Spatial domain lower bound.
        max_x:
            Spatial domain upper bound.
        num_pts_t:
            Number of points along the temporal dimension.
        num_pts_x:
            Number of points along the spatial dimension.

    Returns:
        flattened_t_physics:
            1D tensor of temporal coordinates with gradients enabled.
        flattened_x_physics:
            1D tensor of spatial coordinates with gradients enabled.
        phys_inputs:
            2D tensor with columns (x, t) suitable for PINN input.
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


def get_derivatives(
    output: torch.Tensor, wrt: torch.Tensor, num: int
) -> tuple[torch.Tensor, ...]:
    """
    Simple function that outputs the first num derivatives
    of the ``output`` tensor w.r.t the ``wrt`` tensor, and returns them
    as a tuple, so that they can be nicely unpacked.

    Args:
        output:
            A tensor whose derivative(s) will be calculated.
        wrt:
            The tensor that output is being
            differentiated with respect to.
        num:
            The number of derivatives to return

    Returns:
        derivs:
            A tuple containing the first num derivatives of ``output``.

    """
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


def train(
    domain: Domain, params: BlackScholesPDE, data: torch.Tensor
) -> tuple[ConstrainedFCN, float, list[float]]:
    """
    Trains a PINN in order to calculate sigma.

    Args:
        domain:
            A ``Domain`` object which specifies the distribution of
            physics training points.
        params:
            A ``BlackScholesPDE`` object specifying observable
            parameters of the equation.
        data:
            a ``Tensor`` containing the data from which to estimate
            sigma.

    Returns:
        pinn:
            The trained network.
        sig_est_final:
            The final estimate of sigma.
        sig_history:
            A list containing all estimates for sigma
            throughout training.

    """
    torch.manual_seed(123)
    # PDE paramenters
    MIN_X = domain.mindim(0)
    MAX_X = domain.maxdim(0)
    NUM_PTS_X = domain.ndim(0)
    MIN_T = domain.t_min
    MAX_T = domain.t_max
    NUM_PTS_T = domain.nt

    # Unnormalized contract parameters.
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
    t_boundary_x_limits = torch.linspace(MIN_T, MAX_T, NUM_PTS_T).requires_grad_(True)
    bdry_left, u_bdry_left_tgt = create_left_boundary(t_boundary_x_limits, MIN_X)
    bdry_right, u_bdry_right_tgt = create_right_boundary(
        t_boundary_x_limits, MAX_X, STRIKE=STRIKE, R=R, T_MAX=MAX_T, S_MAX=MAX_X
    )

    # Data points/targets straight from the stored PDE solve. This is the
    # actual solution the PDE solver computed, so fitting to it grounds the
    # PINN in real data rather than relying only on the IC/BC/physics terms.
    data_inputs, u_real = create_data_points(data, MIN_X, MAX_X, MIN_T, MAX_T, X_SCALE)

    t_phys, x_phys, phys_inputs = gen_phys_training_pts(
        MIN_T, MAX_T, MIN_X, MAX_X, NUM_PTS_T, NUM_PTS_X
    )

    print(f"Shape of physics_inputs (x,t): {phys_inputs.shape}")
    print(f"Shape of data_inputs (x,t): {data_inputs.shape}")

    # PINN setup
    KINK_WIDTH = 0.05
    pinn = ConstrainedFCN(
        input_dim=2,
        output_dim=1,
        hidden_dim=64,
        num_layers=3,
        strike=STRIKE,
        t_max=MAX_T,
        kink_width=KINK_WIDTH,
    )

    # Initial guess for sigma
    sig_init = 0.05
    sig_init_scaled = sig_init * (T_SCALE**0.5)

    # Train against the logarithm of sigma, in order to improve
    log_sig_est = torch.nn.Parameter(
        torch.log(torch.tensor([sig_init_scaled])).view(-1, 1)
    )

    sig_history = []

    # Sigma & the network require different learning rates,
    # (and separate training cycles appear to produce better results),
    # so they have separate optimisers.
    optim_net = torch.optim.Adam(
        [
            {"params": pinn.parameters(), "lr": 5e-4},
        ]
    )
    # optim_sig = torch.optim.Adam([log_sig_est], 0.9, (0.55, 0.99))
    optim_sig = torch.optim.SGD([log_sig_est], 1, 0.8)

    # lambda2 (physics loss weight) is raised relative to lambda1/lambda3, as
    # the physics residual is smaller in scale than the data/boundary losses.
    lambda1, lambda2, lambda3 = 1, 5, 2

    # sigma_phase determines whether sigma or the PINN are being trained
    # during the training loop.
    sigma_phase = False

    def compute_loss():
        # Data loss: fit the PINN to the actual solved PDE run.
        u_data_pred = pinn(data_inputs)
        data_loss = torch.mean((u_data_pred - u_real) ** 2)

        # Boundary losses (left/right spatial boundaries).
        u_bdry_left_pred = pinn(bdry_left)
        boundary_left_loss = torch.mean((u_bdry_left_pred - u_bdry_left_tgt) ** 2)

        u_bdry_right_pred = pinn(bdry_right)
        boundary_right_loss = torch.mean((u_bdry_right_pred - u_bdry_right_tgt) ** 2)

        u_phys = pinn(phys_inputs)

        # derivatives w.r.t. t, x
        (dudt_phys,) = get_derivatives(u_phys, t_phys, 1)
        dudx_phys, d2udx2_phys = get_derivatives(u_phys, x_phys, 2)

        x_phys_view = x_phys.view(-1, 1)
        sig_est = torch.exp(log_sig_est)
        # Physics loss
        loss_phys = torch.mean(
            (
                dudt_phys
                + 1 / 2 * sig_est**2 * (x_phys_view**2) * d2udx2_phys
                + R * (x_phys_view) * dudx_phys
                - R * u_phys
            )
            ** 2
        )

        # backpropagate joint loss, take optimiser step
        loss = (
            lambda3 * data_loss
            + lambda1 * (boundary_left_loss + boundary_right_loss)
            + lambda2 * loss_phys
        )
        return loss, data_loss, boundary_left_loss, boundary_right_loss, loss_phys

    for i in range(200001):
        if not sigma_phase:
            optim_net.zero_grad()

        else:
            optim_sig.zero_grad()
        loss, loss1, loss2, loss3, loss4 = compute_loss()
        loss.backward(retain_graph=True)  # Added retain_graph=True here
        if not sigma_phase:
            optim_net.step()

        else:
            optim_sig.step()
            sig_history.append(torch.exp(log_sig_est).item())
        if i % 500 == 0:
            print(
                f"[Adam] {i}/16000. Loss: {loss.item():.6e}. "
                f"L1(data): {loss1.item():.6e}. L2(bdry_left): {loss2.item():.6e}. "
                f"L3(bdry_right): {loss3.item():.6e}. L4(phys): {loss4.item():.6e}. "
                f"sig: {torch.exp(log_sig_est).item():.6e}"
            )
        if i % 2000 == 1820:
            sigma_phase = True
            print("Switch phase to: 'Training sigma'")
            for param in pinn.parameters():
                param.requires_grad_(not sigma_phase)
            log_sig_est.requires_grad_(sigma_phase)
        if i % 2000 == 0:
            sigma_phase = False
            print("Switch phase to: 'Training network'")
            for param in pinn.parameters():
                param.requires_grad_(not sigma_phase)
            log_sig_est.requires_grad_(sigma_phase)

    loss, loss1, loss2, loss3, loss4 = compute_loss()
    sig_est_final = torch.exp(log_sig_est).item()
    sig_est_final_scaled = sig_est_final / T_SCALE
    print(
        f"[FINAL] Loss: {loss.item():.6e}. "
        f"L1(data): {loss1.item():.6e}. L2(bdry_left): {loss2.item():.6e}. "
        f"L3(bdry_right): {loss3.item():.6e}. L4(phys): {loss4.item():.6e}. "
    )
    print(
        f"[FINAL] est. sigma:{sig_est_final_scaled:.6e} true sigma:{params.sigma:.6e}"
    )
    return pinn, sig_est_final_scaled, sig_history


def main():
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        print("GPU available. Running on GPU.")
    torch.set_default_device(device)

    parser = argparse.ArgumentParser(description="Plot PINN solution heatmap.")
    parser.add_argument(
        "-d", "--data", type=str, required=True, help="Path to training data"
    )
    parser.add_argument(
        "-s",
        "--saveto",
        type=str,
        default="models/pinn_backward_new.pt",
        help="Path to where the model should be saved",
    )
    args = parser.parse_args()

    # Load the domain/PDE params this run was solved with, plus the actual
    # solved field values as a tensor to train against.
    domain, params, data = read_black_scholes_run(
        args.data, spatial_resolution=20, time_steps=20
    )
    domain = modify_domain(domain, resolution=(100,), _t_res=50)
    pinn, sig_est_final, sig_hist = train(domain, params, data)
    plt.figure(figsize=(8, 5))
    plt.plot(sig_hist, linewidth=2)
    plt.xlabel("Epochs")
    plt.ylabel("Estimated sigma")
    plt.title("Sigma evolution during training")
    plt.grid(True)
    plt.savefig("sig_loss_hist_new_rate_low.png", dpi=150)
    plt.close()

    save_pinn(
        args.saveto,
        pinn,
        pinn.metadata,
        domain=domain.to_metadata(),
        pde=params.to_metadata(),
        extra={"sig_est": sig_est_final},
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
