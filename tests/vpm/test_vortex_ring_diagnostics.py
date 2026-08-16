"""Vortex-ring tutorial diagnostic regressions."""

from pathlib import Path
import sys
from types import SimpleNamespace

import h5py
import numpy as np

from source.solvers.VPM.initial_conditions import vortex_ring_centerline
from source.solvers.VPM.io.sampler import SamplerExecutor

_ASSETS = Path(__file__).resolve().parents[2] / "tutorials" / "VPM" / "vortexRing" / "assets"
sys.path.insert(0, str(_ASSETS))

from ring_diagnostics import RingDiagnosticsSampler, RingModeDiagnosticsSampler  # noqa: E402
from ring_initialization import initialize_single_mode_toroidal_ring  # noqa: E402
from ring_metrics import (  # noqa: E402
    load_length_integrated_strength,
    load_ring_circulation,
    load_sampled_ring_circulation,
    load_sampled_ring_speed,
)


def test_single_mode_toroidal_ring_recovers_prescribed_mode():
    from source.solvers.VPM import ParticleDistributor

    positions, volumes, radii = ParticleDistributor.toroidal_distribution(
        1.0, 0.12, 0.035, epsilon_w=0.0
    )
    positions, _, _, _, _, circulation, _ = initialize_single_mode_toroidal_ring(
        positions,
        volumes,
        radii,
        viscosity=np.pi / 3000.0,
        ring_radius=1.0,
        ring_strength=np.pi,
        ring_thickness=0.1,
        amplitude=0.05,
        mode=22,
    )
    sampler = RingModeDiagnosticsSampler(
        maximum_mode=30,
        azimuthal_bins=128,
        transverse_origin=(0.0, 0.0),
    )
    modes = np.asarray(sampler._sample_group(positions, circulation))

    assert abs(modes[21, 1] - 0.05) / 0.05 < 5.0e-3
    unseeded = np.delete(modes[:, 1], 21)
    assert np.sqrt(np.mean(unseeded**2)) / modes[21, 1] < 0.02


def test_ring_mode_quadrature_is_independent_of_diagnostic_bin_count():
    from source.solvers.VPM import ParticleDistributor

    positions, volumes, radii = ParticleDistributor.toroidal_distribution(
        1.0, 0.35, 0.15, epsilon_w=0.0
    )
    positions, _, _, _, _, circulation, _ = initialize_single_mode_toroidal_ring(
        positions,
        volumes,
        radii,
        viscosity=np.pi / 3000.0,
        ring_radius=1.0,
        ring_strength=np.pi,
        ring_thickness=0.4,
        amplitude=0.025,
        mode=6,
    )
    recovered = []
    for bins in (40, 64, 96):
        sampler = RingModeDiagnosticsSampler(
            maximum_mode=16,
            azimuthal_bins=bins,
            transverse_origin=(0.0, 0.0),
        )
        recovered.append(np.asarray(sampler._sample_group(positions, circulation)))

    np.testing.assert_allclose(recovered[0][:, 1:4], recovered[1][:, 1:4], atol=1.0e-13)
    np.testing.assert_allclose(recovered[0][:, 1:4], recovered[2][:, 1:4], atol=1.0e-13)
    seeded_coefficients = [rows[5, 1] * np.exp(1j * rows[5, 4]) for rows in recovered]
    np.testing.assert_allclose(seeded_coefficients, seeded_coefficients[0], atol=1.0e-13)


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


def test_ring_mode_sampler_recovers_known_radial_and_axial_bending_modes():
    theta = 2.0 * np.pi * (np.arange(512) + 0.5) / 512
    radial_amplitude = 0.05
    axial_amplitude = 0.02
    mode = 7
    radial_phase = 0.3
    axial_phase = -0.4
    radius = 1.0 + radial_amplitude * np.cos(mode * theta + radial_phase)
    axial = axial_amplitude * np.cos(mode * theta + axial_phase)
    positions = np.column_stack((axial, radius * np.cos(theta), radius * np.sin(theta)))
    circulation = np.column_stack(
        (
            np.zeros_like(theta),
            -np.sin(theta),
            np.cos(theta),
        )
    )

    sampler = RingModeDiagnosticsSampler(maximum_mode=12, azimuthal_bins=128)
    rows = sampler._sample_group(positions, circulation)
    measured = {int(row[0]): row for row in rows}

    np.testing.assert_allclose(measured[mode][1], radial_amplitude, rtol=2e-3)
    np.testing.assert_allclose(measured[mode][2], axial_amplitude, rtol=2e-3)
    assert measured[mode][7] == 1.0
    assert max(measured[index][1] for index in measured if index != mode) < 2.0e-4
    assert max(measured[index][2] for index in measured if index != mode) < 2.0e-4


def test_ring_mode_sampler_recovers_flat_broadband_seed_with_toroidal_jacobian():
    azimuth = 512
    theta = 2.0 * np.pi * (np.arange(azimuth) + 0.5) / azimuth
    epsilon = 0.05
    seeded_modes = 24
    centreline, slope = vortex_ring_centerline(
        theta,
        1.0,
        epsilon_w=epsilon,
        seed=42,
        max_modes=seeded_modes,
    )
    offsets = np.array((-0.04, 0.0, 0.04))
    rho = (centreline[None, :] + offsets[:, None]).reshape(-1)
    tiled_theta = np.tile(theta, len(offsets))
    positions = np.column_stack(
        (
            np.zeros_like(rho),
            rho * np.cos(tiled_theta),
            rho * np.sin(tiled_theta),
        )
    )
    tangent = np.column_stack(
        (
            np.zeros_like(rho),
            -np.sin(tiled_theta),
            np.cos(tiled_theta),
        )
    )
    radial = np.column_stack(
        (
            np.zeros_like(rho),
            np.cos(tiled_theta),
            np.sin(tiled_theta),
        )
    )
    circulation = rho[:, None] * (
        tangent + np.tile(slope, len(offsets))[:, None] * radial / rho[:, None]
    )

    sampler = RingModeDiagnosticsSampler(
        maximum_mode=40,
        azimuthal_bins=128,
        transverse_origin=(0.0, 0.0),
    )
    rows = np.asarray(sampler._sample_group(positions, circulation))
    expected = epsilon / np.sqrt(seeded_modes)
    relative_l2 = np.linalg.norm(rows[:seeded_modes, 1] - expected) / (
        np.sqrt(seeded_modes) * expected
    )

    assert relative_l2 < 3.0e-3
    assert np.sqrt(np.mean(rows[seeded_modes:, 1] ** 2)) < 2.0e-4
