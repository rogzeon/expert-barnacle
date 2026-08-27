"""PDE definitions and helpers for generating synthetic training data.

Every entry in :data:`pdes` maps a CLI-facing PDE name to a factory function
that solves the PDE and returns a :class:`pde.MemoryStorage` populated with
the resulting time series. Each factory embeds JSON-serializable metadata
describing itself in ``storage.info`` (PDE type, parameters, and the
:class:`Domain` it was solved on). For black_scholes runs specifically, that
metadata is enough to reconstruct the exact :class:`Domain` and
:class:`BlackScholesPDE` later, via :func:`read_black_scholes_run`.
"""

from __future__ import annotations

import numpy as np
import pde
from pde.pdes.allen_cahn import AllenCahnPDE
from pde.pdes.diffusion import DiffusionPDE

from common.classes import *
from common.globals import (
    PDE_TYPE_ALLEN_CAHN,
    PDE_TYPE_BLACK_SCHOLES,
    PDE_TYPE_DIFFUSION_1D,
    PDE_TYPE_DIFFUSION_2D,
    T_MIN,
)


def black_scholes(
    opt_name: str, k_str: str, sig_str: str, r_str: str, t_max_str: str
) -> pde.MemoryStorage:
    """Solve the Black-Scholes PDE for a European option.

    Args:
        opt_name: Name of an :class:`Option` member, e.g. ``"EUROCALL"``.
        k_str: Strike price.
        sig_str: Volatility.
        r_str: Risk-free interest rate.
        t_max_str: Time horizon to solve up to.
    """
    black_scholes_pde = BlackScholesPDE(
        option=Option[opt_name],
        k=float(k_str),
        sigma=float(sig_str),
        r=float(r_str),
        t_max=float(t_max_str),
        x_subsampling_factor=20,
    )
    t_max = black_scholes_pde.t_max
    grid = black_scholes_pde.domain.to_grid()
    eq = black_scholes_pde.to_pde()
    state = black_scholes_pde.initial_state(grid)
    N_TIME_STEPS = black_scholes_pde.domain.nt

    store = pde.MemoryStorage(info=black_scholes_pde.to_metadata())
    snapshot_times = np.linspace(black_scholes_pde.t_min, t_max, N_TIME_STEPS)
    eq.solve(
        state,
        t_range=t_max,
        dt=1e-3,
        adaptive=True,
        tracker=["progress", store.tracker(list(snapshot_times))],
    )

    # store.tracker() records fields in tau (time-to-maturity) order, with
    # the payoff at tau=0. Reversing fields (but not times) here remaps that
    # onto real calendar time, so the payoff ends up at t_max (exercise) and
    # tau=t_max ends up at real time 0
    items = list(store.items())
    times = [t for t, _ in items]
    fields_reversed = [f for _, f in reversed(items)]

    return pde.MemoryStorage.from_fields(
        times=times, fields=fields_reversed, info=store.info
    )


def diffusion1d(d_str: str, t_max_str: str) -> pde.MemoryStorage:
    """1D diffusion equation using the preset ``DiffusionPDE``."""
    d, t_max = float(d_str), float(t_max_str)
    domain = Domain(
        bounds=((-1.0, 1.0),),
        resolution=(50,),
        t_min=T_MIN,
        t_max=t_max,
        _t_res=100,
    )
    grid = domain.to_grid()
    x = grid.cell_coords[..., 0]
    u0 = np.exp(-(x**2) / (2 * 0.1**2))
    state = pde.ScalarField(grid, data=u0)
    eq = DiffusionPDE(d)
    N_TIME_STEPS = domain.nt

    info = {
        "pde_type": PDE_TYPE_DIFFUSION_1D,
        "diffusivity": d,
        "domain": domain.to_metadata(),
    }
    store = pde.MemoryStorage(info=info)
    times = np.linspace(T_MIN, t_max, N_TIME_STEPS)
    eq.solve(state, t_range=t_max, dt=1e-3, tracker=["progress", store.tracker(times)])
    return store


def diffusion2d(d_str: str, t_max_str: str) -> pde.MemoryStorage:
    """2D diffusion equation using the preset ``DiffusionPDE`` (works in any dimension)."""
    d, t_max = float(d_str), float(t_max_str)
    domain = Domain(
        bounds=((-1.0, 1.0), (-1.0, 1.0)),
        resolution=(25, 25),
        t_max=t_max,
    )
    grid = domain.to_grid()
    x, y = grid.cell_coords[..., 0], grid.cell_coords[..., 1]
    u0 = np.exp(-(x**2 + y**2) / (2 * 0.1**2))
    state = pde.ScalarField(grid, data=u0)
    eq = DiffusionPDE(d)
    N_TIME_STEPS = domain.nt

    info = {
        "pde_type": PDE_TYPE_DIFFUSION_2D,
        "diffusivity": d,
        "domain": domain.to_metadata(),
    }
    store = pde.MemoryStorage(info=info)
    times = np.linspace(T_MIN, t_max, N_TIME_STEPS)
    eq.solve(state, t_range=t_max, dt=1e-3, tracker=["progress", store.tracker(times)])
    return store


def allen_cahn(eps_str: str, t_max_str: str) -> pde.MemoryStorage:
    """
    Allen-Cahn equation (2D) using the preset ``AllenCahnPDE``.
    Mobility is not supported. (For now.)
    """
    eps, t_max = float(eps_str), float(t_max_str)
    domain = Domain(
        bounds=((-1.0, 1.0), (-1.0, 1.0)),
        resolution=(25, 25),
        t_max=t_max,
    )
    grid = domain.to_grid()
    np.random.seed(0)
    u0 = np.random.uniform(-1, 1, size=grid.shape)
    state = pde.ScalarField(grid, data=u0)
    eq = AllenCahnPDE(interface_width=eps**2)
    N_TIME_STEPS = domain.nt
    info = {
        "pde_type": PDE_TYPE_ALLEN_CAHN,
        "epsilon": eps,
        "domain": domain.to_metadata(),
    }
    store = pde.MemoryStorage(info=info)
    times = np.linspace(T_MIN, t_max, N_TIME_STEPS)
    eq.solve(state, t_range=t_max, dt=1e-3, tracker=["progress", store.tracker(times)])
    return store


def apply_noise(store: pde.MemoryStorage, sig: float) -> pde.MemoryStorage:
    """Add Gaussian noise with standard deviation ``sig`` to every stored field."""
    if sig == 0:
        return store
    for _, field in store.items():  # noqa: PERF102
        field.data += sig * np.random.normal(size=field.data.shape)
    return store


pdes = {
    PDE_TYPE_BLACK_SCHOLES: black_scholes,
    PDE_TYPE_DIFFUSION_1D: diffusion1d,
    PDE_TYPE_DIFFUSION_2D: diffusion2d,
    PDE_TYPE_ALLEN_CAHN: allen_cahn,
}
