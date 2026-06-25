"""Vortex-ring tutorial diagnostic regressions."""

from pathlib import Path
import sys

import h5py
import numpy as np

_ASSETS = Path(__file__).resolve().parents[2] / "tutorials" / "VPM" / "vortexRing" / "assets"
sys.path.insert(0, str(_ASSETS))

from _common import load_length_integrated_strength, load_ring_circulation  # noqa: E402


def _write_ring_snapshot(path: Path, *, radius: float, gamma: float, time: float, n: int = 96) -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    pos = np.zeros((n, 3), dtype=np.float64)
    pos[:, 1] = radius * np.cos(theta)
    pos[:, 2] = radius * np.sin(theta)

    ds = 2.0 * np.pi * radius / n
    tangent = np.zeros_like(pos)
    tangent[:, 1] = -np.sin(theta)
    tangent[:, 2] = np.cos(theta)
    circ = gamma * ds * tangent

    with h5py.File(path, "w") as f:
        particles = f.create_group("particles")
        particles.create_dataset("position", data=pos)
        particles.create_dataset("circulation", data=circ)
        particles.create_dataset("group_id", data=np.zeros(n, dtype=np.int32))
        solver = f.create_group("solver")
        solver.attrs["flow_time"] = time


def test_ring_circulation_diagnostic_is_radius_independent(tmp_path):
    """Same tube circulation at larger radius changes Σ|alpha|, not Gamma_tube."""
    f0 = tmp_path / "vpm_ring_000000.h5"
    f1 = tmp_path / "vpm_ring_000001.h5"
    _write_ring_snapshot(f0, radius=1.0, gamma=np.pi, time=0.0)
    _write_ring_snapshot(f1, radius=1.2, gamma=np.pi, time=1.0)

    files = [str(f0), str(f1)]
    _, length_strength = load_length_integrated_strength(files)
    _, tube_circulation = load_ring_circulation(files)

    np.testing.assert_allclose(length_strength, [1.0, 1.2], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(tube_circulation, [1.0, 1.0], rtol=1e-12, atol=1e-12)
