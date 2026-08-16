import math
from enum import Enum

import numpy as np
import pde
from pde.pdes import AllenCahnPDE, DiffusionPDE


class Option(Enum):
    EUROCALL = 1
    EUROPUT = 2

    def __call__(self, x, K):
        if self is Option.EUROCALL:
            return np.maximum(x - K, 0)
        elif self is Option.EUROPUT:
            return np.maximum(K - x, 0)


def black_scholes(opt_name, K_str, sig_str, r_str, T_max_str):
    K, sig, r, T_max = float(K_str), float(sig_str), float(r_str), float(T_max_str)
    opt = Option[opt_name]

    grid = pde.CartesianGrid(
        [[0, K * (1 + 5 * sig * math.sqrt(T_max))]], [100]
    )  # domain sized relative to K, adjust as needed
    eq = pde.PDE(
        {"v": f"{(sig**2) / 2} * x**2 * d2_dx2(v) + {r} * x * d_dx(v) - {r} * v"}
    )

    x = grid.cell_coords[..., 0]
    state = pde.ScalarField(grid, data=opt(x, K))

    store = pde.MemoryStorage()
    eq.solve(
        state,
        t_range=T_max,
        dt=1e-3,
        tracker=["progress", store.tracker([T_max * i / 100 for i in range(100)])],
    )

    items = list(store.items())
    times = [t for t, _ in items]
    fields_reversed = [f for _, f in reversed(items)]

    return pde.MemoryStorage.from_fields(times=times, fields=fields_reversed)


def diffusion1d(d: float, t: float):
    """1D diffusion equation using the preset DiffusionPDE."""
    grid = pde.CartesianGrid([[-1, 1]], [100])
    x = grid.cell_coords[..., 0]
    u0 = np.exp(-(x**2) / (2 * 0.1**2))
    state = pde.ScalarField(grid, data=u0)
    eq = DiffusionPDE(d)
    store = pde.MemoryStorage()
    times = np.linspace(0, t, 100)
    eq.solve(state, t_range=t, dt=1e-3, tracker=["progress", store.tracker(times)])
    return store


def diffusion2d(d: float, t: float):
    """2D diffusion equation using the preset DiffusionPDE (works in any dimension)."""
    grid = pde.CartesianGrid([[-1, 1], [-1, 1]], [50, 50])
    x, y = grid.cell_coords[..., 0], grid.cell_coords[..., 1]
    u0 = np.exp(-(x**2 + y**2) / (2 * 0.1**2))
    state = pde.ScalarField(grid, data=u0)
    eq = DiffusionPDE(d)
    store = pde.MemoryStorage()
    times = np.linspace(0, t, 100)
    eq.solve(state, t_range=t, dt=1e-3, tracker=["progress", store.tracker(times)])
    return store


def allen_cahn(eps: float, t: float):
    """Allen–Cahn equation (2D) using the preset AllenCahnPDE."""
    grid = pde.CartesianGrid([[-1, 1], [-1, 1]], [50, 50])
    np.random.seed(0)  # reproducible initial condition
    u0 = np.random.uniform(-1, 1, size=grid.shape)
    state = pde.ScalarField(grid, data=u0)
    eq = AllenCahnPDE(epsilon=eps)  # preset PDE
    store = pde.MemoryStorage()
    times = np.linspace(0, t, 100)
    eq.solve(state, t_range=t, dt=1e-3, tracker=["progress", store.tracker(times)])
    return store


def apply_noise(store: pde.MemoryStorage, sig: float):
    """Add Gaussian noise with standard deviation sig to every stored field."""
    if sig == 0:
        return store
    for _, field in store.items():
        field.data += sig * np.random.normal(size=field.data.shape)
    return store


pdes = {
    "black_scholes": black_scholes,
    "diff1d": diffusion1d,
    "diff2d": diffusion2d,
    "allen_cahn": allen_cahn,
}
