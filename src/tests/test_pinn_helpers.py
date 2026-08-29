"""
Tests for common.pinn_helpers: initial/boundary condition construction, the
physics-training-grid generator, and get_derivatives.
"""

import torch

from common.pinn_helpers import (
    boundary_left,
    boundary_right,
    create_left_boundary,
    create_right_boundary,
    gen_phys_training_pts,
    get_derivatives,
    initial_condition,
)

# ---------------------------------------------------------------------------
# Initial / boundary conditions
# ---------------------------------------------------------------------------


def test_initial_condition_is_call_payoff():
    x0 = torch.tensor([80.0, 100.0, 120.0])
    (u0,) = initial_condition(t0=0.0, x0=x0, STRIKE=100.0)
    torch.testing.assert_close(u0, torch.tensor([0.0, 0.0, 20.0]))


def test_boundary_left_is_always_zero():
    space_t_left = torch.linspace(0.0, 1.0, 5)
    out = boundary_left(space_t_left)
    torch.testing.assert_close(out, torch.zeros(5))


def test_boundary_right_matches_discounted_strike_formula():
    space_t_right = torch.tensor([0.0, 0.5, 1.0])
    kwargs = {"S_MAX": 200.0, "STRIKE": 100.0, "R": 0.05, "T_MAX": 1.0}
    out = boundary_right(space_t_right, **kwargs)
    expected = 200.0 - 100.0 * torch.exp(-0.05 * (1.0 - space_t_right))
    torch.testing.assert_close(out, expected)


def test_create_left_boundary_pins_x_to_x_min():
    t = torch.linspace(0.0, 1.0, 4).requires_grad_(True)
    inputs, values = create_left_boundary(t, x_min=0.0)
    assert inputs.shape == (4, 2)
    assert values.shape == (4, 1)
    # Every row's x-coordinate (column 0) must equal x_min.
    torch.testing.assert_close(inputs[:, 0], torch.zeros(4))


def test_create_right_boundary_pins_x_to_x_max_and_uses_formula():
    t = torch.linspace(0.0, 1.0, 4).requires_grad_(True)
    kwargs = {"S_MAX": 200.0, "STRIKE": 100.0, "R": 0.05, "T_MAX": 1.0}
    inputs, values = create_right_boundary(t, x_max=200.0, **kwargs)
    assert inputs.shape == (4, 2)
    torch.testing.assert_close(inputs[:, 0], torch.full((4,), 200.0))
    expected = (200.0 - 100.0 * torch.exp(-0.05 * (1.0 - t))).view(-1, 1)
    torch.testing.assert_close(values, expected)


# ---------------------------------------------------------------------------
# Physics training grid
# ---------------------------------------------------------------------------


def test_gen_phys_training_pts_shapes_and_column_order():
    t, x, phys_inputs = gen_phys_training_pts(
        min_t=0.0, max_t=1.0, min_x=0.0, max_x=10.0, num_pts_t=5, num_pts_x=7
    )
    n = 5 * 7
    assert t.shape == (n,)
    assert x.shape == (n,)
    assert phys_inputs.shape == (n, 2)
    # phys_inputs columns are (x, t), matching flattened_x_physics/flattened_t_physics.
    torch.testing.assert_close(phys_inputs[:, 0], x)
    torch.testing.assert_close(phys_inputs[:, 1], t)


def test_gen_phys_training_pts_requires_grad():
    t, x, phys_inputs = gen_phys_training_pts(
        min_t=0.0, max_t=1.0, min_x=0.0, max_x=1.0, num_pts_t=3, num_pts_x=3
    )
    # Gradients w.r.t. t and x must flow through, since the physics loss
    # differentiates the network output w.r.t. these inputs.
    assert t.requires_grad
    assert x.requires_grad


# ---------------------------------------------------------------------------
# get_derivatives
# ---------------------------------------------------------------------------


def test_get_derivatives_first_derivative_matches_analytical():
    # y = x^2  =>  dy/dx = 2x
    x = torch.linspace(-2.0, 2.0, 9).view(-1, 1).requires_grad_(True)
    y = x**2
    (dydx,) = get_derivatives(y, x, num=1)
    torch.testing.assert_close(dydx, 2 * x.detach())


def test_get_derivatives_second_derivative_matches_analytical():
    # y = x^3  =>  dy/dx = 3x^2, d2y/dx2 = 6x
    x = torch.linspace(-2.0, 2.0, 9).view(-1, 1).requires_grad_(True)
    y = x**3
    dydx, d2ydx2 = get_derivatives(y, x, num=2)
    torch.testing.assert_close(dydx, 3 * x.detach() ** 2)
    torch.testing.assert_close(d2ydx2, 6 * x.detach())


def test_get_derivatives_returns_requested_count():
    x = torch.linspace(0.0, 1.0, 5).view(-1, 1).requires_grad_(True)
    y = x**4
    derivs = get_derivatives(y, x, num=3)
    assert len(derivs) == 3
