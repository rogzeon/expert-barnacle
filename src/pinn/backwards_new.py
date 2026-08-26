import argparse
import math
import matplotlib.pyplot as plt

import torch
from model import FCN

from common.classes import BlackScholesPDE, Domain
from common.funcs import read_black_scholes_run, modify_domain
from pinn.pinn_io import save_pinn
from pinn.plot_heatmap import plot_single, predict_grid


# The following functions create the IC and/or boundary conditions.
def initial_condition(t0, x0, **kwargs):
    """
    Constructs the initial condition off of which to train.
    Modify this function in order to change the IC.
    """
    u0_target = torch.clamp(x0 - kwargs["STRIKE"], min=0)

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


def create_data_points(data, x_min, x_max, t_min, t_max, x_scale):
    """
    Build (x, t) input points and matching targets from a stored PDE run's
    solution tensor, in the same normalized [0, 1] x [0, 1] space used
    everywhere else in this module.

    Args:
        data: Tensor of shape (n_times, n_x), forward-chronological, as
            returned by ``read_black_scholes_run``.
        x_min, x_max, t_min, t_max: Normalized domain bounds (0.0 and 1.0
            in this module, since x/t are rescaled before this is called).
        x_scale: The physical range of the spatial domain (``X_SCALE`` in
            ``train``), used to put the raw PDE values into the same
            normalized units as ``u0_target``.

    Returns:
        A ``(data_inputs, u_real)`` tuple: ``data_inputs`` has shape
        (n_times * n_x, 2) with columns (x, t); ``u_real`` has shape
        (n_times * n_x, 1) and is normalized by ``x_scale`` to match the
        rest of the model's targets.
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


class KinkConstrainedFCN(torch.nn.Module):
    """FCN wrapped in a hard constraint that bakes the option payoff's kink
    into the architecture, instead of asking a smooth (SiLU) network to
    learn a non-differentiable function on its own.

    The ansatz is

        u(x, t) = w(t) * payoff(x) + (1 - w(t)) * base_net(x, t)

    where w(t_max) = 1 exactly, so at exercise the output is *exactly* the
    payoff (the initial condition is a hard constraint, not a soft loss
    term), and w decays smoothly (a Gaussian bump centered at t_max) away
    from expiry, handing off to the network everywhere else. The network
    never has to represent the kink itself -- it only ever needs to learn
    a blend/correction that's smooth everywhere the true (diffused)
    solution is smooth, i.e. everywhere t < t_max.

    Constructed from flat kwargs (rather than wrapping an already-built
    FCN instance) so it's reconstructable from a saved ``arch`` dict via
    ``model_cls(**arch)``, matching pinn_io.py's save_pinn/load_pinn
    contract -- see the note in train() about updating eval scripts to
    load with this class instead of plain FCN.
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim,
        num_layers,
        strike,
        t_max,
        kink_width,
    ):
        super().__init__()
        self.base_net = FCN(input_dim, output_dim, hidden_dim, num_layers)
        self.strike = strike
        self.t_max = t_max
        self.kink_width = kink_width
        self.metadata = {
            "input_dim": input_dim,
            "output_dim": output_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "strike": strike,
            "t_max": t_max,
            "kink_width": kink_width,
        }

    def payoff(self, x):
        return torch.clamp(x - self.strike, min=0.0)

    def weight(self, t):
        # Gaussian bump: exactly 1 at t=t_max, decays fast moving away from
        # it. kink_width controls how quickly the payoff hands off to the
        # network -- too wide and the network is still fighting the kink
        # over a large region; too narrow and the transition itself can
        # introduce sharp curvature in w(t) for the physics loss to fight
        # instead. Tune alongside NUM_PTS_T / the physics collocation density.
        return torch.exp(-(((self.t_max - t) / self.kink_width) ** 2))

    def forward(self, xt):
        x = xt[:, 0:1]
        t = xt[:, 1:2]
        w = self.weight(t)
        return w * self.payoff(x) + (1 - w) * self.base_net(xt)


def train(domain: Domain, params: BlackScholesPDE, data: torch.Tensor):
    torch.manual_seed(123)
    # PDE paramenters
    MIN_X = domain.mindim(0)
    MAX_X = domain.maxdim(0)
    NUM_PTS_X = domain.ndim(0)
    MIN_T = domain.t_min
    MAX_T = domain.t_max
    NUM_PTS_T = domain.nt

    # Physical (unnormalized) contract parameters. These are the single
    # source of truth for this run — the same `params` object gets saved
    # into the checkpoint (see main()) so eval_analytical.py can read them
    # back out instead of hardcoding its own copy that can drift out of sync.
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
    # KINK_WIDTH is in the same normalized [0,1] time units as everything
    # else here. 0.05 means the payoff dominates within roughly the last
    # 5-10% of the time-to-maturity window and the network handles the
    # rest -- tune this alongside how sharply your solve's kink actually
    # smooths out (governed by sigma and grid resolution).
    KINK_WIDTH = 0.05
    pinn = KinkConstrainedFCN(
        input_dim=2,
        output_dim=1,
        hidden_dim=64,
        num_layers=3,
        strike=STRIKE,
        t_max=MAX_T,
        kink_width=KINK_WIDTH,
    )

    # Learn log(sigma) rather than sigma directly. sig_est**2 has a zero
    # gradient exactly at 0, which makes 0 a spuriously stable fixed point
    # once the data/boundary terms dominate the loss (their gradient signal
    # swamps whatever weak, possibly-noisy physics gradient would otherwise
    # nudge sig_est away from 0). Parameterizing in log-space removes that
    # fixed point entirely — exp(log_sig_est) can get arbitrarily close to 0
    # but never reach it, and the gradient w.r.t. log_sig_est doesn't
    # degenerate there. Initialize from SIG (the actual, properly rescaled
    # volatility for this normalized domain) rather than an arbitrary
    # constant, so the starting point is already in the right ballpark.
    sig_init = 0.05
    sig_init_scaled = sig_init * (T_SCALE**0.5)
    print(T_SCALE)
    print(sig_init_scaled)
    print(SIG)
    print(params.sigma)
    log_sig_est = torch.nn.Parameter(
        torch.log(torch.tensor([sig_init_scaled])).view(-1, 1)
    )

    sig_history = []

    # Give log_sig_est its own learning rate: it's a single scalar competing
    # for gradient signal against a multi-thousand-parameter network, and
    # its gradient only ever comes from the (comparatively small) physics
    # loss, so the network's default lr is a poor fit for it.
    optim_net = torch.optim.Adam(
        [
            {"params": pinn.parameters(), "lr": 5e-4},
        ]
    )
    # optim_sig = torch.optim.Adam([log_sig_est], 0.9, (0.55, 0.99))
    optim_sig = torch.optim.SGD([log_sig_est], 1, 0.8)
    # lambda2 (physics loss weight) is raised relative to lambda1/lambda3:
    # the physics residual is naturally much smaller in scale than the data/
    # boundary losses, so at equal weighting it contributes almost nothing
    # to the total gradient — which means log_sig_est's only source of
    # signal barely factors into training at all. Tune further if needed.
    lambda1, lambda2, lambda3 = 1, 5, 2
    print(boundary_inputs.std(dim=0))
    print(u0_target.var().item())

    sigma_phase = False

    with torch.no_grad():
        test_x = torch.linspace(0, 1, 10).unsqueeze(1)
        test_t = torch.ones_like(test_x)
        test_in = torch.cat([test_x, test_t], dim=1)
        print(pinn(test_in).flatten())

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
            print(f"Switch phase to: {sigma_phase}")
            # optim_sig = torch.optim.Adam(
            #     [
            #         {"params": [log_sig_est], "lr": 5e-2},
            #     ]
            # )
            for param in pinn.parameters():
                param.requires_grad_(not sigma_phase)
            log_sig_est.requires_grad_(sigma_phase)
        if i % 2000 == 0:
            sigma_phase = False
            print(f"Switch phase to: {sigma_phase}")
            for param in pinn.parameters():
                param.requires_grad_(not sigma_phase)
            log_sig_est.requires_grad_(sigma_phase)

    # log_sig_est needs to be included here too -- LBFGS was previously only
    # given pinn.parameters(), so the entire second training phase silently
    # dropped the sigma estimate from optimization.
    # lbfgs = torch.optim.LBFGS(
    #     list(pinn.parameters()) + [log_sig_est],
    #     lr=1.0,
    #     max_iter=2000,
    #     tolerance_grad=1e-9,
    #     tolerance_change=1e-12,
    #     history_size=50,
    #     line_search_fn="strong_wolfe",
    # )
    #
    # lbfgs_steps = [0]
    #
    # def closure():
    #     lbfgs.zero_grad()
    #     loss, loss1, loss2, loss3, loss4 = compute_loss()
    #     loss.backward(retain_graph=True)
    #     if lbfgs_steps[0] % 200 == 0:
    #         print(
    #             f"[LBFGS] step {lbfgs_steps[0]}. Loss: {loss.item():.6e}. "
    #             f"L1(data): {loss1.item():.6e}. L2(bdry_left): {loss2.item():.6e}. "
    #             f"L3(bdry_right): {loss3.item():.6e}. L4(phys): {loss4.item():.6e}"
    #         )
    #     lbfgs_steps[0] += 1
    #     return loss
    #
    # lbfgs.step(closure)
    #
    loss, loss1, loss2, loss3, loss4 = compute_loss()
    sig_est_final = torch.exp(log_sig_est).item()
    print(
        f"[FINAL] Loss: {loss.item():.6e}. "
        f"L1(data): {loss1.item():.6e}. L2(bdry_left): {loss2.item():.6e}. "
        f"L3(bdry_right): {loss3.item():.6e}. L4(phys): {loss4.item():.6e}. "
    )
    print(f"[FINAL] est. sigma:{sig_est_final} true sigma:{SIG}")
    return pinn, sig_est_final, sig_history


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
