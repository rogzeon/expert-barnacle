"""
Helper functions shared by the forward-solving (main.py) and
sigma-calibration (backwards.py) PINN training scripts.

These build the initial condition, the left/right spatial boundaries, the
physics (PDE residual) training grid, and the autograd derivatives used by
both scripts' loss functions. Anything specific to a single script's
training loop (data loading, network architecture, optimiser schedule,
etc.) stays in that script.
"""

import torch


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
    # eps = 0.0025
    # k = x0 - kwargs["STRIKE"]
    # u0_target = (k + torch.sqrt(k**2 + eps)) / 2
    #   dudt0_target = torch.zeros_like(x0).view(-1, 1).requires_grad_(False)

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
    """
    Create input points and target values for the x = ``x_max`` boundary.

    Mirrors ``create_left_boundary()``: the boundary points pair the fixed
    ``x_max`` with each temporal coordinate in ``t_boundary_x_limits``, and the
    target values come from ``boundary_right()``.

    Args:
        t_boundary_x_limits:
            A 1D tensor of temporal coordinates at which to evaluate the boundary,
            typically a linspace from t_min to t_max. Requires gradients enabled.
        x_max:
            The spatial coordinate of the right boundary
        **kwargs:
            PDE parameters (ex. for diffusion, could contain
            the diffusion coefficient.) to be passed to ``boundary_right()``

    Returns:
        boundary_inputs_right:
            A 2D tensor of shape (N, 2) where each row is (x, t) for the boundary
            points, with x = ``x_max`` for all points.
        boundary_values_right:
            A 2D tensor of shape (N, 1) containing the corresponding target
            boundary values from ``boundary_right()``.
    """
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
