"""
Save/load a PINN checkpoint together with the metadata needed to reconstruct
it.

Typical use in train():

    from pinn_io import save_pinn
    ...
    pinn = FCN(2, 1, 32, 3)
    ... train ...
    save_pinn(
        "pinn.pt",
        pinn,
        arch={"input_dim": 2, "output_dim": 1, "hidden_dim": 32, "num_layers": 3},
        domain={"x_min": MIN_X, "x_max": MAX_X, "t_min": MIN_T, "t_max": MAX_T},
        pde={"D": D},
        extra={"lambda1": 1e-1, "lambda2": 1e-4, "epochs": 15001},
    )

Typical use in an eval/plot script:

    from pinn_io import load_pinn
    model, meta = load_pinn("pinn.pt", FCN, device=device)
    x_min, x_max = meta["domain"]["x_min"], meta["domain"]["x_max"]
    D = meta["pde"]["D"]
"""

from typing import Any

import torch
from torch import nn


def save_pinn(
    path: str,
    model: nn.Module,
    arch: dict[str, Any],
    domain: dict[str, Any] | None = None,
    pde: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """
    Save model weights plus the metadata needed to rebuild/interpret the model.

    arch:   kwargs to reconstruct the network, e.g.
            {"input_dim": 2, "output_dim": 1, "hidden_dim": 32, "num_layers": 3}
    domain: e.g. {"x_min": 0, "x_max": 10, "t_min": 0, "t_max": 3}
    """
    torch.save(
        {
            "state_dict": model.state_dict(),
            "arch": arch,
            "domain": domain or {},
        },
        path,
    )


def load_pinn(
    path: str,
    model_cls: type[nn.Module],
    device: str = "cpu",
) -> tuple[nn.Module, dict[str, Any]]:
    """
    Load a checkpoint saved with save_pinn(). Reconstructs the model from the
    stored `arch` metadata instead of requiring the caller to know the
    architecture in advance.

    Returns (model, metadata) where metadata = {"arch", "domain", "pde", "extra"}.
    """
    ckpt = torch.load(path, map_location=device)

    missing = {"state_dict", "arch"} - ckpt.keys()
    if missing:
        raise KeyError(
            f"Checkpoint at {path} is missing {missing}; was it saved with save_pinn()?"
        )

    model = model_cls(**ckpt["arch"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()

    metadata = {
        "arch": ckpt["arch"],
        "domain": ckpt.get("domain", {}),
        "pde": ckpt.get("pde", {}),
        "extra": ckpt.get("extra", {}),
    }
    return model, metadata
