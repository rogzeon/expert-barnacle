"""Domain and Black-Scholes PDE dataclasses, plus the metadata (de)serialization
used to embed them in and recover them from a stored PDE run.

Pulled out of ``funcs.py`` so this doesn't own the only definition of
``Domain``/``BlackScholesPDE`` in the project - wire this up against
``model.py``'s ``Domain``/``BlackScholesParams`` as needed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import pde

PDE_TYPE_BLACK_SCHOLES = "black_scholes"

# Every black_scholes solve starts at t=0 and records this many snapshots up
# to t_max. Stored in metadata (as "t_min" / "n_times") so a run's time
# resolution can be reconstructed later without relying on this staying
# unchanged.
T_MIN = 0.0
N_TIME_STEPS = 100


class Option(Enum):
    """Payoff function for a European option."""

    EUROCALL = 1
    EUROPUT = 2

    def __call__(self, x: np.ndarray, k: float) -> np.ndarray:
        if self is Option.EUROCALL:
            return np.maximum(x - k, 0)
        if self is Option.EUROPUT:
            return np.maximum(k - x, 0)
        raise ValueError(f"Unhandled option type: {self}")  # pragma: no cover


@dataclass(frozen=True)
class Domain:
    """Grid geometry needed to reconstruct a :class:`pde.CartesianGrid`."""

    bounds: tuple[tuple[float, float], ...]
    shape: tuple[int, ...]

    def to_grid(self) -> pde.CartesianGrid:
        """Rebuild the :class:`pde.CartesianGrid` this domain describes."""
        return pde.CartesianGrid(list(self.bounds), list(self.shape))

    @classmethod
    def from_grid(cls, grid: pde.CartesianGrid) -> Domain:
        """Extract a :class:`Domain` from an existing grid."""
        return cls(
            bounds=tuple((float(lo), float(hi)) for lo, hi in grid.axes_bounds),
            shape=tuple(int(n) for n in grid.shape),
        )

    def to_metadata(self) -> dict:
        """JSON-serializable representation, suitable for ``storage.info``."""
        return {"bounds": [[lo, hi] for lo, hi in self.bounds], "shape": list(self.shape)}

    @classmethod
    def from_metadata(cls, metadata: dict) -> Domain:
        """Inverse of :meth:`to_metadata`.

        Raises:
            KeyError: If ``metadata`` is missing ``bounds`` or ``shape``.
        """
        return cls(
            bounds=tuple((float(lo), float(hi)) for lo, hi in metadata["bounds"]),
            shape=tuple(int(n) for n in metadata["shape"]),
        )


@dataclass(frozen=True)
class BlackScholesPDE:
    """Parameters that fully specify a Black-Scholes PDE run.

    Bundles the coefficients of the pricing PDE together with the option
    payoff, and knows how to build the ``pde.PDE``, the initial condition, and
    the (JSON-serializable) metadata describing itself, as well as how to
    reconstruct itself from that metadata.
    """

    option: Option
    k: float
    sigma: float
    r: float
    t_max: float

    def domain(self) -> Domain:
        """The grid this PDE is solved on, sized relative to the strike."""
        upper = self.k * (1 + 5 * self.sigma * math.sqrt(self.t_max))
        return Domain(bounds=((0.0, upper),), shape=(100,))

    def to_pde(self) -> pde.PDE:
        """Build the underlying (backward) Black-Scholes ``pde.PDE``."""
        return pde.PDE(
            {
                "v": (
                    f"{(self.sigma**2) / 2} * x**2 * d2_dx2(v) "
                    f"+ {self.r} * x * d_dx(v) - {self.r} * v"
                )
            }
        )

    def initial_state(self, grid: pde.CartesianGrid) -> pde.ScalarField:
        """The option payoff at t=0 on ``grid``."""
        x = grid.cell_coords[..., 0]
        return pde.ScalarField(grid, data=self.option(x, self.k))

    def to_metadata(self) -> dict:
        """JSON-serializable representation, suitable for ``storage.info``."""
        return {
            "pde_type": PDE_TYPE_BLACK_SCHOLES,
            "option": self.option.name,
            "K": self.k,
            "sigma": self.sigma,
            "r": self.r,
            "t_min": T_MIN,
            "t_max": self.t_max,
            "n_times": N_TIME_STEPS,
            "domain": self.domain().to_metadata(),
        }

    @classmethod
    def from_metadata(cls, metadata: dict) -> BlackScholesPDE:
        """Inverse of :meth:`to_metadata`.

        Raises:
            ValueError: If ``metadata`` does not describe a black_scholes run,
                or is missing a required field.
        """
        pde_type = metadata.get("pde_type")
        if pde_type != PDE_TYPE_BLACK_SCHOLES:
            raise ValueError(
                f"Metadata does not describe a {PDE_TYPE_BLACK_SCHOLES} run "
                f"(pde_type={pde_type!r})"
            )
        try:
            return cls(
                option=Option[metadata["option"]],
                k=float(metadata["K"]),
                sigma=float(metadata["sigma"]),
                r=float(metadata["r"]),
                t_max=float(metadata["t_max"]),
            )
        except KeyError as e:
            raise ValueError(f"black_scholes metadata missing field: {e}") from e
        except (TypeError, ValueError) as e:
            raise ValueError(f"black_scholes metadata is malformed: {e}") from e


def read_black_scholes_run(path: str | Path) -> tuple[Domain, BlackScholesPDE]:
    """Read a stored PDE run and reconstruct the domain and PDE it came from.

    Args:
        path: Path to an HDF5 file previously written by ``data_gen.py``.

    Returns:
        A ``(domain, black_scholes_pde)`` tuple. ``domain.to_grid()`` rebuilds
        the grid the data lives on; ``black_scholes_pde.to_pde()`` and
        ``.initial_state(grid)`` rebuild the PDE and its t=0 condition.

    Raises:
        ValueError: If the file has no metadata, or the metadata does not
            describe a valid black_scholes run.
    """
    with pde.FileStorage(path, write_mode="readonly") as store:
        info = dict(store.info)

    if not info:
        raise ValueError(f"{path!r} has no stored metadata; cannot reconstruct the run")

    black_scholes_pde = BlackScholesPDE.from_metadata(info)
    try:
        domain = Domain.from_metadata(info["domain"])
    except KeyError as e:
        raise ValueError(f"black_scholes metadata missing field: {e}") from e

    return domain, black_scholes_pde
