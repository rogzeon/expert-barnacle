"""
Tests for pinn.pinn_io: save_pinn/load_pinn round-tripping, and
error handling for missing/corrupted checkpoints.
"""

import pytest
import torch

from pinn.model import FCN
from pinn.pinn_io import load_pinn, save_pinn


def test_save_then_load_round_trip(tmp_path):
    model = FCN(input_dim=2, output_dim=1, hidden_dim=8, num_layers=2)
    path = tmp_path / "checkpoint.pt"

    save_pinn(
        str(path),
        model,
        arch=model.metadata,
        domain={"x_min": 0, "x_max": 10, "t_min": 0, "t_max": 1},
        pde={"strike": 100.0, "sigma": 0.2, "r": 0.05},
        extra={"epochs": 100},
    )

    loaded_model, metadata = load_pinn(str(path), FCN)

    # Weights match exactly.
    for p_orig, p_loaded in zip(model.parameters(), loaded_model.parameters()):
        torch.testing.assert_close(p_orig, p_loaded)

    assert metadata["arch"] == model.metadata
    assert metadata["domain"] == {"x_min": 0, "x_max": 10, "t_min": 0, "t_max": 1}
    assert metadata["pde"] == {"strike": 100.0, "sigma": 0.2, "r": 0.05}
    assert metadata["extra"] == {"epochs": 100}


def test_save_pinn_creates_missing_parent_directory(tmp_path):
    model = FCN(input_dim=2, output_dim=1, hidden_dim=4, num_layers=1)
    path = tmp_path / "nested" / "dir" / "checkpoint.pt"
    assert not path.parent.exists()

    save_pinn(str(path), model, arch=model.metadata)

    assert path.exists()


def test_load_pinn_missing_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_pinn(str(tmp_path / "does_not_exist.pt"), FCN)


def test_load_pinn_corrupted_file_raises_runtime_error(tmp_path):
    path = tmp_path / "corrupted.pt"
    path.write_bytes(b"not a real torch checkpoint")

    with pytest.raises(RuntimeError, match="corrupted"):
        load_pinn(str(path), FCN)


def test_load_pinn_missing_required_keys_raises_key_error(tmp_path):
    path = tmp_path / "incomplete.pt"
    # Valid torch file, but missing the "arch"/"state_dict" keys save_pinn writes.
    torch.save({"something_else": 1}, path)

    with pytest.raises(KeyError):
        load_pinn(str(path), FCN)


def test_load_pinn_defaults_missing_optional_metadata(tmp_path):
    """
    domain/pde/extra are optional at save time; load_pinn should hand back
    empty dicts for them rather than raising, so callers don't need to
    special-case checkpoints saved without them.
    """
    model = FCN(input_dim=2, output_dim=1, hidden_dim=4, num_layers=1)
    path = tmp_path / "minimal.pt"
    save_pinn(str(path), model, arch=model.metadata)

    _, metadata = load_pinn(str(path), FCN)
    assert metadata["domain"] == {}
    assert metadata["pde"] == {}
    assert metadata["extra"] == {}
