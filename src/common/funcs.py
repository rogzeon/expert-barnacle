# Every black_scholes solve starts at t=0 and records this many snapshots up
# to t_max. Stored in metadata (as "t_min" / "n_times") so a run's time
# resolution can be reconstructed later without relying on this staying
# unchanged.

from pathlib import Path

import numpy as np
import pde
import torch

from .classes import BlackScholesPDE, Domain


def read_black_scholes_run(
    path: str | Path,
) -> tuple[Domain, BlackScholesPDE, torch.Tensor]:
    """Read a stored PDE run and reconstruct the domain, PDE, and its data.

    Args:
        path: Path to an HDF5 file previously written by ``data_gen.py``.

    Returns:
        A ``(domain, black_scholes_pde, data)`` tuple:

        - ``domain.to_grid()`` rebuilds the grid the data lives on.
        - ``black_scholes_pde.to_pde()`` and ``.initial_state(grid)`` rebuild
          the PDE and its t=0 condition.
        - ``data`` is a ``torch.float32`` tensor of shape
          ``(n_times, n_x)`` holding every snapshot recorded during the
          solve, stacked in forward-chronological order (row 0 is the field
          at ``domain.t_min``, the last row is the field at ``domain.t_max``).
          This can be used directly as regression targets for a data-fitting
          loss when training a model (e.g. a PINN) against this run.

    Raises:
        ValueError: If the file has no metadata, or the metadata does not
            describe a valid black_scholes run.
    """
    with pde.FileStorage(path, write_mode="readonly") as store:
        info = dict(store.info)
        fields = [
            np.asarray(field.data, dtype=np.float32) for _, field in store.items()
        ]

    if not info:
        raise ValueError(f"{path!r} has no stored metadata; cannot reconstruct the run")

    black_scholes_pde = BlackScholesPDE.from_metadata(info)
    try:
        domain = Domain.from_metadata(info["domain"])
    except KeyError as e:
        raise ValueError(f"black_scholes metadata missing field: {e}") from e

    data = torch.tensor(np.stack(fields, axis=0), dtype=torch.float32)

    return domain, black_scholes_pde, data
