"""Reusable runtime guards for conservative GBD moment recovery."""

from __future__ import annotations

import numpy as np
import pytest

from source.coupler.solver import _validate_gbd_moment_recovery


def _valid_recovery(**overrides):
    recovery = {
        "applied": True,
        "nonzero_node_count": 100,
        "retained_node_count": 90,
        "pruned_node_count": 10,
        "support_augmented_node_count": 2,
        "correction_fraction": 0.01,
        "normalized_vortex_strength_residual": 1.0e-8,
        "normalized_linear_impulse_residual": 2.0e-8,
        "normalized_angular_impulse_residual": 3.0e-8,
    }
    recovery.update(overrides)
    return recovery


def test_gbd_moment_recovery_accepts_a_closed_conservative_prune():
    _validate_gbd_moment_recovery(_valid_recovery(), 0.08)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"applied": False}, "without conservative moment recovery"),
        ({"correction_fraction": np.nan}, "is non-finite"),
        ({"normalized_linear_impulse_residual": np.inf}, "is non-finite"),
        ({"normalized_angular_impulse_residual": 1.1e-5}, "residual tolerance"),
        ({"correction_fraction": 0.081}, "excessive particle-strength correction"),
    ],
)
def test_gbd_moment_recovery_rejects_open_or_excessive_prunes(overrides, message):
    with pytest.raises(RuntimeError, match=message):
        _validate_gbd_moment_recovery(_valid_recovery(**overrides), 0.08)
