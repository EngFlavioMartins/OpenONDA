"""Analytic sanity checks for the active local vorticity transfer."""

from __future__ import annotations

import numpy as np
import pytest

from source.coupler.vorticity_transfer import (
    continuous_transfer,
    vortex_strength_from_velocity_trace,
)


def _lattice(n: int, h: float):
    axis = (np.arange(n) - (n - 1) / 2.0) * h
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
    return grid.reshape(-1, 3), (n, n, n)


def test_rung0_uniform_flow_produces_no_vorticity():
    """A uniform stream must hand off nothing. Catches sign and trace errors."""
    h = 0.1
    points, _ = _lattice(16, h)
    circulation = vortex_strength_from_velocity_trace(
        points, h, lambda q: np.tile([1.3, -0.4, 0.7], (len(np.atleast_2d(q)), 1))
    )
    assert np.abs(circulation).max() < 1e-15


def test_rung0_uniform_flow_transfer_creates_no_particles():
    box = np.array([-0.5, 0.5, -0.5, 0.5, -0.5, 0.5])
    h = 0.1
    result = continuous_transfer(
        np.zeros((0, 3)),
        np.zeros((0, 3)),
        box,
        h,
        vortex_strength_at_node=lambda q: vortex_strength_from_velocity_trace(
            q, h, lambda p: np.tile([1.0, 0.0, 0.0], (len(np.atleast_2d(p)), 1))
        ),
        authority_ramp_width=4 * h,
        transfer_buffer_length=2 * h,
        transfer_prune_threshold_abs=1e-18,
        freestream_velocity=[1.0, 0.0, 0.0],
    )
    assert result.n_total == 0
    assert result.transfer_out_of_band_fraction == pytest.approx(0.0, abs=1e-12)


def test_rung0_solid_body_rotation_is_reproduced_exactly():
    """u = omega x r has constant curl; the trace must return it to round-off."""
    h = 0.05
    points, _ = _lattice(12, h)
    omega = np.array([0.3, -0.2, 1.1])
    circulation = vortex_strength_from_velocity_trace(
        points, h, lambda q: 0.5 * np.cross(omega, np.asarray(q).reshape(-1, 3))
    )
    np.testing.assert_allclose(circulation, np.tile(omega * h**3, (len(points), 1)), atol=1e-16)


# ---------------------------------------------------------------------------
