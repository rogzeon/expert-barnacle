"""
Tests for common.classes: Option payoffs, Domain validation/round-tripping,
and BlackScholesPDE's derived grid + metadata round-tripping.
"""

import math

import numpy as np
import pytest

from common.classes import BlackScholesPDE, Domain, Option

# ---------------------------------------------------------------------------
# Option
# ---------------------------------------------------------------------------


def test_eurocall_payoff():
    x = np.array([80.0, 100.0, 120.0])
    payoff = Option.EUROCALL(x, k=100.0)
    np.testing.assert_allclose(payoff, [0.0, 0.0, 20.0])


def test_europut_payoff():
    x = np.array([80.0, 100.0, 120.0])
    payoff = Option.EUROPUT(x, k=100.0)
    np.testing.assert_allclose(payoff, [20.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# Domain construction & validation
# ---------------------------------------------------------------------------


def test_domain_valid_construction_shape():
    d = Domain(bounds=((0.0, 10.0),), resolution=(10.0,), t_max=1.0, _t_res=100.0)
    assert d.shape == (100,)
    assert d.nt == 100


def test_domain_mismatched_bounds_and_resolution_length():
    with pytest.raises(ValueError, match="bounds but"):
        Domain(bounds=((0.0, 10.0), (0.0, 5.0)), resolution=(10.0,), t_max=1.0)


def test_domain_rejects_hi_not_greater_than_lo():
    with pytest.raises(ValueError, match="hi > lo"):
        Domain(bounds=((10.0, 10.0),), resolution=(10.0,), t_max=1.0)


def test_domain_rejects_non_integer_grid_points():
    # resolution * (hi - lo) = 3 * 10 = 30 is fine; use a value that isn't whole.
    with pytest.raises(ValueError, match="grid points"):
        Domain(bounds=((0.0, 10.0),), resolution=(3.33,), t_max=1.0)


def test_domain_rejects_t_max_not_greater_than_t_min():
    with pytest.raises(ValueError, match="t_max > t_min"):
        Domain(bounds=((0.0, 10.0),), resolution=(10.0,), t_max=1.0, t_min=1.0)


def test_domain_rejects_non_integer_time_steps():
    with pytest.raises(ValueError, match="time steps"):
        Domain(bounds=((0.0, 10.0),), resolution=(10.0,), t_max=1.0, _t_res=3.33)


def test_domain_to_grid_matches_shape():
    d = Domain(bounds=((0.0, 10.0),), resolution=(10.0,), t_max=1.0, _t_res=100.0)
    grid = d.to_grid()
    assert tuple(grid.shape) == d.shape


def test_domain_metadata_round_trip():
    d = Domain(
        bounds=((0.0, 25.0),), resolution=(4.0,), t_max=2.0, t_min=0.5, _t_res=50.0
    )
    restored = Domain.from_metadata(d.to_metadata())
    assert restored == d


def test_domain_from_metadata_missing_field_raises_keyerror():
    incomplete = {"bounds": [[0.0, 10.0]], "resolution": [10.0]}
    with pytest.raises(KeyError):
        Domain.from_metadata(incomplete)


# ---------------------------------------------------------------------------
# BlackScholesPDE
# ---------------------------------------------------------------------------


def test_black_scholes_domain_starts_at_zero_and_covers_strike():
    bs = BlackScholesPDE(option=Option.EUROCALL, k=100.0, sigma=0.2, r=0.05, t_max=1.0)
    dom = bs.domain
    assert dom.bounds[0][0] == 0.0
    # Upper bound should comfortably exceed the strike for a 1-year, 20% vol option.
    assert dom.bounds[0][1] > bs.k


def test_black_scholes_from_domain_requires_1d():
    two_d_domain = Domain(
        bounds=((0.0, 10.0), (0.0, 5.0)), resolution=(10.0, 10.0), t_max=1.0
    )
    with pytest.raises(ValueError, match="1D"):
        BlackScholesPDE.from_domain(
            Option.EUROCALL, k=100.0, sigma=0.2, r=0.05, domain=two_d_domain
        )


def test_black_scholes_from_domain_requires_zero_lower_bound():
    domain = Domain(bounds=((5.0, 10.0),), resolution=(10.0,), t_max=1.0)
    with pytest.raises(ValueError, match="starting at 0.0"):
        BlackScholesPDE.from_domain(
            Option.EUROCALL, k=100.0, sigma=0.2, r=0.05, domain=domain
        )


def test_black_scholes_metadata_round_trip():
    bs = BlackScholesPDE(
        option=Option.EUROPUT, k=50.0, sigma=0.3, r=0.01, t_max=0.5, x_max=200.0
    )
    restored = BlackScholesPDE.from_metadata(bs.to_metadata())
    assert restored.option == bs.option
    assert math.isclose(restored.k, bs.k)
    assert math.isclose(restored.sigma, bs.sigma)
    assert math.isclose(restored.r, bs.r)
    assert math.isclose(restored.t_max, bs.t_max)
    assert restored.domain == bs.domain


def test_black_scholes_from_metadata_rejects_wrong_pde_type():
    with pytest.raises(ValueError, match="does not describe"):
        BlackScholesPDE.from_metadata({"pde_type": "diff1d"})


def test_black_scholes_initial_state_matches_payoff():
    bs = BlackScholesPDE(option=Option.EUROCALL, k=100.0, sigma=0.2, r=0.05, t_max=1.0)
    grid = bs.domain.to_grid()
    field = bs.initial_state(grid)
    x = grid.cell_coords[..., 0]
    expected = np.maximum(x - bs.k, 0)
    np.testing.assert_allclose(field.data, expected)
