"""
Tests for common.funcs: modify_domain's copy-with-overrides behaviour, and
the spatial/temporal subsampling logic in _apply_subsampling.
"""

import numpy as np
import pytest

from common.classes import Domain
from common.funcs import _apply_subsampling, modify_domain


def make_domain():
    return Domain(bounds=((0.0, 10.0),), resolution=(10.0,), t_max=1.0, _t_res=100.0)


def test_modify_domain_overrides_only_given_fields():
    d = make_domain()
    modified = modify_domain(d, t_max=2.0)
    assert modified.t_max == 2.0
    # Everything else carried over unchanged.
    assert modified.bounds == d.bounds
    assert modified.resolution == d.resolution
    assert modified.t_min == d.t_min
    assert modified._t_res == d._t_res


def test_modify_domain_with_no_overrides_is_equal():
    d = make_domain()
    assert modify_domain(d) == d


def test_apply_subsampling_spatial_requires_divisibility():
    d = make_domain()  # 100 spatial points
    fields = [np.zeros(100) for _ in range(5)]
    with pytest.raises(ValueError, match="divisible"):
        _apply_subsampling(fields, d, spatial_resolution=30, time_steps=None)


def test_apply_subsampling_spatial_slicing_picks_every_step_points():
    d = make_domain()  # 100 spatial points
    fields = [np.arange(100, dtype=float) for _ in range(3)]
    new_fields, new_domain = _apply_subsampling(
        fields, d, spatial_resolution=10, time_steps=None
    )
    assert all(len(f) == 10 for f in new_fields)
    # Slicing with step=10 keeps indices 0, 10, 20, ...
    np.testing.assert_allclose(new_fields[0], np.arange(0, 100, 10, dtype=float))
    assert new_domain.shape[0] == 10


def test_apply_subsampling_temporal_keeps_requested_snapshot_count():
    d = make_domain()
    fields = [np.full(100, i, dtype=float) for i in range(10)]
    new_fields, new_domain = _apply_subsampling(
        fields, d, spatial_resolution=None, time_steps=4
    )
    assert len(new_fields) == 4
    assert new_domain.nt == 4


def test_apply_subsampling_rejects_fewer_than_two_time_steps():
    d = make_domain()
    fields = [np.zeros(100) for _ in range(5)]
    with pytest.raises(ValueError, match="at least 2"):
        _apply_subsampling(fields, d, spatial_resolution=None, time_steps=1)
