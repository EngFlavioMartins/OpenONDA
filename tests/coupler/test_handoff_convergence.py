"""Rung 0 and rung 1 of the coupling verification ladder.

These exercise the FVM->VPM hand-off *alone*, against an analytic field, with no
FVM solve, no VPM time integration and no chaos.  Everything downstream of them
(the cylinder and cube benchmarks) is a physics validation and can only be
trusted once these pass.

Rung 0  uniform flow  -> the hand-off must produce no vorticity at all.
Rung 1  Lamb-Oseen tube -> the mollified particle field must reproduce the exact
        vorticity, and the error must fall as the core is resolved.

The rung-1 sweeps double as the sizing tool for a production case: they answer
"how many lattice cells per vortex core do I need for 1%?" and "does raising
sigma/h help?".  Run them directly for a table::

    pytest tests/coupler/test_handoff_convergence.py -s -m verification
"""

from __future__ import annotations

import numpy as np
import pytest

from source.coupler.core.helpers.continuous_overlap import (
    _gaussian_mollified_circulation,
    bandlimited_transfer,
    circulation_from_velocity_trace,
    continuous_handoff,
)

# ---------------------------------------------------------------------------
# Analytic Lamb-Oseen vortex tube along z (exact u and omega)
# ---------------------------------------------------------------------------
GAMMA = 1.0


def lamb_oseen_velocity(points: np.ndarray, core: float) -> np.ndarray:
    p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    r_sq = p[:, 0] ** 2 + p[:, 1] ** 2
    r_sq = np.maximum(r_sq, 1e-300)
    swirl = GAMMA * (1.0 - np.exp(-r_sq / core**2)) / (2.0 * np.pi * r_sq)
    out = np.zeros_like(p)
    out[:, 0] = -swirl * p[:, 1]
    out[:, 1] = swirl * p[:, 0]
    return out


def lamb_oseen_vorticity(points: np.ndarray, core: float) -> np.ndarray:
    p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    r_sq = p[:, 0] ** 2 + p[:, 1] ** 2
    out = np.zeros_like(p)
    out[:, 2] = GAMMA * np.exp(-r_sq / core**2) / (np.pi * core**2)
    return out


def _lattice(n: int, h: float):
    axis = (np.arange(n) - (n - 1) / 2.0) * h
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
    return grid.reshape(-1, 3), (n, n, n)


def _transfer_error(n: int, h: float, core: float, radius_ratio: float, cap: float):
    """Relative error of the mollified particle vorticity inside the core region."""
    points, shape = _lattice(n, h)
    target = circulation_from_velocity_trace(
        points, h, lambda q: lamb_oseen_velocity(q, core)
    )
    sigma = radius_ratio * h
    gamma, _, out_of_band = bandlimited_transfer(
        target, shape, h, sigma=sigma, amplification_cap=cap
    )
    mollified = _gaussian_mollified_circulation(gamma, shape, h, sigma=sigma).reshape(-1, 3)
    exact = lamb_oseen_vorticity(points, core) * h**3

    # Compare where the structure is AND where the lattice is interior.  The
    # Gaussian mollification uses zero padding and the transfer tapers the outer
    # lattice layers, exactly as in production, where the hand-off lattice always
    # carries a 2h M4' guard band plus the downstream buffer.  A z-invariant tube
    # touches both z faces, so a comparison that includes them measures the guard
    # band, not the transfer.
    guard = 8
    half = (n - 1) / 2.0
    interior = np.all(np.abs(points) <= (half - guard) * h + 1e-12, axis=1)
    radius = np.hypot(points[:, 0], points[:, 1])
    inside = interior & (radius <= 2.5 * core)
    error = np.linalg.norm(mollified[inside] - exact[inside]) / np.linalg.norm(exact[inside])
    peak = np.abs(mollified[interior, 2]).max() / np.abs(exact[interior, 2]).max()
    return error, peak, out_of_band


# ---------------------------------------------------------------------------
# Rung 0
# ---------------------------------------------------------------------------
def test_rung0_uniform_flow_produces_no_vorticity():
    """A uniform stream must hand off nothing. Catches sign and trace errors."""
    h = 0.1
    points, _ = _lattice(16, h)
    circulation = circulation_from_velocity_trace(
        points, h, lambda q: np.tile([1.3, -0.4, 0.7], (len(np.atleast_2d(q)), 1))
    )
    assert np.abs(circulation).max() < 1e-15


def test_rung0_uniform_flow_handoff_creates_no_particles():
    box = np.array([-0.5, 0.5, -0.5, 0.5, -0.5, 0.5])
    h = 0.1
    result = continuous_handoff(
        np.zeros((0, 3)),
        np.zeros((0, 3)),
        box,
        h,
        circulation_at_node=lambda q: circulation_from_velocity_trace(
            q, h, lambda p: np.tile([1.0, 0.0, 0.0], (len(np.atleast_2d(p)), 1))
        ),
        ramp_width=4 * h,
        buffer_length=2 * h,
        threshold_abs=1e-18,
        u_inf=[1.0, 0.0, 0.0],
    )
    assert result.n_total == 0
    assert result.transfer_out_of_band_fraction == pytest.approx(0.0, abs=1e-12)


def test_rung0_solid_body_rotation_is_reproduced_exactly():
    """u = omega x r has constant curl; the trace must return it to round-off."""
    h = 0.05
    points, _ = _lattice(12, h)
    omega = np.array([0.3, -0.2, 1.1])
    circulation = circulation_from_velocity_trace(
        points, h, lambda q: 0.5 * np.cross(omega, np.asarray(q).reshape(-1, 3))
    )
    np.testing.assert_allclose(circulation, np.tile(omega * h**3, (len(points), 1)), atol=1e-16)


# ---------------------------------------------------------------------------
# Rung 1
# ---------------------------------------------------------------------------
@pytest.mark.verification
def test_rung1_error_falls_as_the_core_is_resolved():
    n, h, cap = 48, 0.05, 2.0
    previous = np.inf
    for cells_per_core in (1.0, 2.0, 3.0, 4.0, 6.0):
        error, _, _ = _transfer_error(n, h, cells_per_core * h, radius_ratio=1.0, cap=cap)
        assert error < previous, f"error rose at r_c = {cells_per_core}h"
        previous = error
    assert previous < 0.01, f"6 cells per core should reach 1%, got {previous:.3%}"


@pytest.mark.verification
def test_rung1_peak_amplitude_is_recovered():
    n, h = 48, 0.05
    _, peak, _ = _transfer_error(n, h, 4.0 * h, radius_ratio=1.0, cap=2.0)
    assert 0.97 < peak < 1.03, f"peak vorticity ratio {peak:.3f}"


@pytest.mark.verification
def test_rung1_out_of_band_fraction_predicts_the_error():
    """The reported resolution diagnostic must actually bound the error."""
    n, h = 48, 0.05
    for cells_per_core in (1.0, 2.0, 4.0):
        error, _, out_of_band = _transfer_error(n, h, cells_per_core * h, 1.0, 2.0)
        assert out_of_band >= 0.2 * error, (
            f"out-of-band {out_of_band:.3e} does not track error {error:.3e} "
            f"at r_c = {cells_per_core}h"
        )


@pytest.mark.verification
@pytest.mark.slow
def test_rung1_sweep_table(capsys):
    """Print the sizing table: error vs cells-per-core and vs sigma/h."""
    n, h, cap = 48, 0.05, 2.0
    rows = []
    for radius_ratio in (0.8, 1.0, 1.25, 1.5, 2.0):
        for cells_per_core in (1.0, 2.0, 3.0, 4.0, 6.0, 8.0):
            error, peak, out_of_band = _transfer_error(
                n, h, cells_per_core * h, radius_ratio, cap
            )
            rows.append((radius_ratio, cells_per_core, error, peak, out_of_band))

    with capsys.disabled():
        print("\n  sigma/h  r_c/h   rel.error   peak ratio   out-of-band")
        for radius_ratio, cells, error, peak, out_of_band in rows:
            print(
                f"  {radius_ratio:7.2f} {cells:6.1f} {error:11.3%} "
                f"{peak:12.3f} {out_of_band:13.3%}"
            )

    # sigma/h = 1 must not be beaten by more than a factor of two by any other
    # value at the resolution that matters; if it is, the default is wrong.
    best_at_four = {
        radius_ratio: error
        for radius_ratio, cells, error, _, _ in rows
        if cells == 4.0
    }
    assert best_at_four[1.0] <= 2.0 * min(best_at_four.values())
