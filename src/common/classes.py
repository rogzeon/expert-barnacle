import math
from dataclasses import dataclass
from enum import Enum
from typing import Self

import numpy as np
import pde

from .globals import *


class Option(Enum):
    """Payoff function for a European option."""

    EUROCALL = 1
    EUROPUT = 2

    def __call__(self, x: np.ndarray, k: float) -> np.ndarray:
        if self is Option.EUROCALL:
            return np.maximum(x - k, 0)
        if self is Option.EUROPUT:
            return np.maximum(k - x, 0)
        raise ValueError(f"Unhandled option type: {self}")


@dataclass(frozen=True)
class Domain:
    """Grid geometry needed to reconstruct a :class:`pde.CartesianGrid`."""

    bounds: tuple[tuple[float, float], ...]
    resolution: tuple[float, ...]
    t_max: float
    t_min: float = T_MIN
    _t_res: float = TIME_RES_DEFAULT

    def __post_init__(self):
        """Validate that this domain describes a whole number of grid points.

        ``shape``/``ndim``/``nt`` used to silently truncate via ``int(...)``,
        so e.g. a mismatched ``t_res``/``t_max`` could produce a domain with
        one fewer time step than intended without any error, only surfacing
        as a confusing shape mismatch somewhere downstream. This fails fast
        instead, at construction time.
        """
        if len(self.bounds) != len(self.resolution):
            raise ValueError(
                f"Domain has {len(self.bounds)} bounds but "
                f"{len(self.resolution)} resolution entries; they must have "
                "the same length"
            )
        for i, ((lo, hi), res) in enumerate(zip(self.bounds, self.resolution)):
            if hi <= lo:
                raise ValueError(f"Domain bounds[{i}] = ({lo}, {hi}) must have hi > lo")
            n = res * (hi - lo)
            if not math.isclose(n, round(n), rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError(
                    f"Domain bounds[{i}]=({lo}, {hi}) with resolution={res} "
                    f"gives {n} grid points, which isn't a whole number. "
                    "Adjust the resolution or bounds so "
                    "resolution * (hi - lo) is an integer."
                )

        if self.t_max <= self.t_min:
            raise ValueError(
                f"Domain requires t_max > t_min (got t_min={self.t_min}, "
                f"t_max={self.t_max})"
            )
        n_t = self._t_res * (self.t_max - self.t_min)
        if not math.isclose(n_t, round(n_t), rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(
                f"Domain t_res={self._t_res} with t_max - t_min = "
                f"{self.t_max - self.t_min} gives {n_t} time steps, which "
                "isn't a whole number. Adjust t_res, t_min, or t_max so "
                "t_res * (t_max - t_min) is an integer."
            )

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(
            round((self.bounds[i][1] - self.bounds[i][0]) * res)
            for i, res in enumerate(self.resolution)
        )

    def to_grid(self) -> pde.CartesianGrid:
        """Rebuild the :class:`pde.CartesianGrid` this domain describes."""
        return pde.CartesianGrid(list(self.bounds), list(self.shape))

    @classmethod
    def from_grid(
        cls, grid: pde.CartesianGrid, t_min: float, t_max: float, _t_res: int
    ) -> Self:
        """Extract a :class:`Domain` from an existing grid."""
        return cls(
            bounds=tuple((float(lo), float(hi)) for lo, hi in grid.axes_bounds),
            resolution=tuple(
                n / (grid.axes_bounds[i][1] - grid.axes_bounds[i][0])
                for i, n in enumerate(grid.shape)
            ),
            t_min=t_min,
            t_max=t_max,
            _t_res=_t_res,
        )

    def to_metadata(self) -> dict:
        """JSON-serializable representation, suitable for ``storage.info``."""
        return {
            "bounds": [[lo, hi] for lo, hi in self.bounds],
            "resolution": list(self.resolution),
            "t_min": self.t_min,
            "t_max": self.t_max,
            "_t_res": self._t_res,
        }

    @classmethod
    def from_metadata(cls, metadata: dict) -> Self:
        """Inverse of :meth:`to_metadata`.

        Raises:
            KeyError: If ``metadata`` is missing ``bounds``, ``resolution``,
                ``t_min``, ``t_max``, or ``_t_res``.
        """
        return cls(
            bounds=tuple((float(lo), float(hi)) for lo, hi in metadata["bounds"]),
            resolution=tuple(metadata["resolution"]),
            t_min=metadata["t_min"],
            t_max=metadata["t_max"],
            _t_res=metadata["_t_res"],
        )

    @property
    def nt(self) -> int:
        return round(self._t_res * (self.t_max - self.t_min))

    def ndim(self, dim) -> int:
        return round(self.resolution[dim] * (self.bounds[dim][1] - self.bounds[dim][0]))

    def mindim(self, dim):
        return self.bounds[dim][0]

    def maxdim(self, dim):
        return self.bounds[dim][1]


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
    t_min: float = T_MIN
    x_max: float | None = None
    x_res: float = SPACE_RES_DEFAULT
    t_res: float = TIME_RES_DEFAULT

    @property
    def domain(self) -> Domain:
        """The grid this PDE is solved on, sized relative to the strike."""
        if self.x_max is None:
            raw_upper = self.k * (1 + 5 * self.sigma * math.sqrt(self.t_max))
            # Domain requires x_res * upper to be an exact integer, but this
            # heuristic ceiling is generally irrational (it has a sqrt in
            # it). Round up to the nearest grid-aligned value instead of
            # failing Domain's guard: this only ever makes the price ceiling
            # slightly larger than the heuristic minimum, never smaller.
            n_x = math.ceil(round(raw_upper * self.x_res, 9))
            upper = n_x / self.x_res
        else:
            upper = self.x_max
        return Domain(
            bounds=((0.0, upper),),
            resolution=(self.x_res,),
            t_min=self.t_min,
            t_max=self.t_max,
            _t_res=self.t_res,
        )

    @classmethod
    def from_domain(
        cls, option: Option, k: float, sigma: float, r: float, domain: Domain
    ) -> Self:
        """Construct a BlackScholesPDE from an explicit domain.

        Assumes the domain is 1D in space, with bounds[0] = (0, x_max) —
        Black-Scholes' spatial domain always starts at spot price 0, so a
        domain with a nonzero lower bound can't be represented and is
        rejected rather than silently discarding that bound.
        """
        if len(domain.bounds) != 1:
            raise ValueError("BlackScholesPDE requires a 1D spatial domain")
        if len(domain.resolution) != 1:
            raise ValueError("BlackScholesPDE requires a 1D spatial resolution")
        if domain.bounds[0][0] != 0.0:
            raise ValueError(
                "BlackScholesPDE requires a spatial domain starting at 0.0 "
                f"(got lower bound {domain.bounds[0][0]!r})"
            )

        return cls(
            option=option,
            k=k,
            sigma=sigma,
            r=r,
            t_min=domain.t_min,
            t_max=domain.t_max,
            x_max=domain.bounds[0][1],
            x_res=domain.resolution[0],
            t_res=domain._t_res,
        )

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
            "t_max": self.t_max,
            "domain": self.domain.to_metadata(),
        }

    @classmethod
    def from_metadata(cls, metadata: dict) -> Self:
        """Inverse of :meth:`to_metadata`.

        ``x_max``, ``x_res``, ``t_res``, and ``t_min`` are recovered from the
        nested ``"domain"`` metadata when present, so a round-trip through
        ``to_metadata()`` / ``from_metadata()`` reconstructs the exact same
        grid the run was solved on rather than silently falling back to
        dataclass defaults (which could disagree with the original domain,
        e.g. if ``x_max`` was set explicitly or a non-default resolution was
        used).

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
        domain_metadata = metadata.get("domain") or {}
        try:
            x_max = (
                float(domain_metadata["bounds"][0][1])
                if "bounds" in domain_metadata
                else None
            )
            x_res = float(domain_metadata.get("resolution", [SPACE_RES_DEFAULT])[0])
            t_res = float(domain_metadata.get("_t_res", TIME_RES_DEFAULT))
            t_min = float(domain_metadata.get("t_min", T_MIN))

            return cls(
                option=Option[metadata["option"]],
                k=float(metadata["K"]),
                sigma=float(metadata["sigma"]),
                r=float(metadata["r"]),
                t_max=float(metadata["t_max"]),
                t_min=t_min,
                x_max=x_max,
                x_res=x_res,
                t_res=t_res,
            )
        except KeyError as e:
            raise ValueError(f"black_scholes metadata missing field: {e}") from e
        except (TypeError, ValueError, IndexError) as e:
            raise ValueError(f"black_scholes metadata is malformed: {e}") from e
