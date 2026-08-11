"""Vortex-ring tutorial diagnostic regressions."""

from pathlib import Path
import sys
from types import SimpleNamespace

import h5py
import numpy as np

from source.solvers.VPM.io.sampler import SamplerExecutor

_ASSETS = Path(__file__).resolve().parents[2] / "tutorials" / "VPM" / "vortexRing" / "assets"
sys.path.insert(0, str(_ASSETS))

from _common import (  # noqa: E402
    load_length_integrated_strength,
    load_ring_circulation,
    load_sampled_ring_circulation,
    load_sampled_ring_speed,
)
from ring_diagnostics import RingDiagnosticsSampler  # noqa: E402


def _write_ring_snapshot(
    path: Path,
    *,
    radius: float,
    gamma: float,
    time: float,
    n: int = 96,
    rotation: np.ndarray | None = None,
) -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    pos = np.zeros((n, 3), dtype=np.float64)
    pos[:, 1] = radius * np.cos(theta)
    pos[:, 2] = radius * np.sin(theta)

    ds = 2.0 * np.pi * radius / n
    tangent = np.zeros_like(pos)
    tangent[:, 1] = -np.sin(theta)
    tangent[:, 2] = np.cos(theta)
    circ = gamma * ds * tangent
    if rotation is not None:
        pos = pos @ rotation.T
        circ = circ @ rotation.T

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


def test_ring_circulation_diagnostic_is_orientation_independent(tmp_path):
    """Tilting the same ring must not create a false circulation spike."""
    angle = np.deg2rad(80.0)
    rotation = np.array(
        [
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ]
    )
    f0 = tmp_path / "vpm_ring_000000.h5"
    f1 = tmp_path / "vpm_ring_000001.h5"
    _write_ring_snapshot(f0, radius=1.0, gamma=np.pi, time=0.0)
    _write_ring_snapshot(f1, radius=1.0, gamma=np.pi, time=1.0, rotation=rotation)

    _, tube_circulation = load_ring_circulation([str(f0), str(f1)])

    np.testing.assert_allclose(tube_circulation, [1.0, 1.0], rtol=1e-12, atol=1e-12)


def test_ring_sampler_writes_dense_diagnostics_beside_other_samples(tmp_path):
    theta = np.linspace(0.0, 2.0 * np.pi, 96, endpoint=False)
    positions = np.zeros((len(theta), 3))
    positions[:, 1] = np.cos(theta)
    positions[:, 2] = np.sin(theta)
    ds = 2.0 * np.pi / len(theta)
    circulation = np.zeros_like(positions)
    circulation[:, 1] = -np.pi * ds * np.sin(theta)
    circulation[:, 2] = np.pi * ds * np.cos(theta)

    solver = SimpleNamespace(
        config=SimpleNamespace(
            samplers=(RingDiagnosticsSampler(),),
            sample_subdirectory="DNS_direct",
        ),
        particles=SimpleNamespace(number_of_particles=len(theta)),
        particles_positions=positions,
        particles_circulation=circulation,
        particles_group_ids=np.zeros(len(theta), dtype=np.int32),
        backup_directory=str(tmp_path / "solution"),
        flow_time=0.1,
        time_step=5,
    )
    SamplerExecutor.execute(solver)
    solver.particles_positions[:, 0] += 0.02
    solver.flow_time = 0.2
    solver.time_step = 10
    SamplerExecutor.execute(solver)

    csv_path = tmp_path / "samples" / "DNS_direct" / "ring_diagnostics.csv"
    time, tube_circulation = load_sampled_ring_circulation(csv_path)
    speed_time, speed = load_sampled_ring_speed(csv_path)

    np.testing.assert_allclose(tube_circulation, [1.0, 1.0], rtol=1e-12, atol=1e-12)
    assert time.shape == speed_time.shape == speed.shape == (2,)
    assert np.isfinite(speed).all()
