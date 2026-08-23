"""Vortex-ring tutorial diagnostic regressions."""

from pathlib import Path
import sys
from types import SimpleNamespace

import h5py
import numpy as np

from source.solvers.vpm.initial_conditions import vortex_ring_centreline
from source.solvers.vpm.io.sampler import SamplerExecutor

_ASSETS = Path(__file__).resolve().parents[2] / "tutorials" / "VPM" / "vortex_ring" / "assets"
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
    from source.solvers.vpm import ParticleDistributor

    position, particle_volume, core_radius = ParticleDistributor.toroidal_distribution(
        1.0, 0.12, 0.035, widnall_amplitude=0.0
    )
    position, _, _, _, _, vortex_strength, _ = initialize_single_mode_toroidal_ring(
        position,
        particle_volume,
        core_radius,
        kinematic_viscosity=np.pi / 3000.0,
        ring_radius=1.0,
        tube_circulation=np.pi,
        ring_core_radius=0.1,
        amplitude=0.05,
        mode=22,
    )
    sampler = RingModeDiagnosticsSampler(
        max_mode=30,
        azimuthal_bins=128,
        transverse_origin=(0.0, 0.0),
    )
    modes = np.asarray(sampler._sample_group(position, vortex_strength))

    assert abs(modes[21, 1] - 0.05) / 0.05 < 5.0e-3
    unseeded = np.delete(modes[:, 1], 21)
    assert np.sqrt(np.mean(unseeded**2)) / modes[21, 1] < 0.02


def test_ring_mode_quadrature_is_independent_of_diagnostic_bin_count():
    from source.solvers.vpm import ParticleDistributor

    position, particle_volume, core_radius = ParticleDistributor.toroidal_distribution(
        1.0, 0.35, 0.15, widnall_amplitude=0.0
    )
    position, _, _, _, _, vortex_strength, _ = initialize_single_mode_toroidal_ring(
        position,
        particle_volume,
        core_radius,
        kinematic_viscosity=np.pi / 3000.0,
        ring_radius=1.0,
        tube_circulation=np.pi,
        ring_core_radius=0.4,
        amplitude=0.025,
        mode=6,
    )
    recovered = []
    for bins in (40, 64, 96):
        sampler = RingModeDiagnosticsSampler(
            max_mode=16,
            azimuthal_bins=bins,
            transverse_origin=(0.0, 0.0),
        )
        recovered.append(np.asarray(sampler._sample_group(position, vortex_strength)))

    np.testing.assert_allclose(recovered[0][:, 1:4], recovered[1][:, 1:4], atol=1.0e-13)
    np.testing.assert_allclose(recovered[0][:, 1:4], recovered[2][:, 1:4], atol=1.0e-13)
    seeded_coefficients = [rows[5, 1] * np.exp(1j * rows[5, 4]) for rows in recovered]
    np.testing.assert_allclose(seeded_coefficients, seeded_coefficients[0], atol=1.0e-13)


def _write_ring_snapshot(
    path: Path,
    *,
    ring_radius: float,
    tube_circulation: float,
    time: float,
    n: int = 96,
    rotation: np.ndarray | None = None,
) -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    pos = np.zeros((n, 3), dtype=np.float64)
    pos[:, 1] = ring_radius * np.cos(theta)
    pos[:, 2] = ring_radius * np.sin(theta)

    segment_length = 2.0 * np.pi * ring_radius / n
    tangent = np.zeros_like(pos)
    tangent[:, 1] = -np.sin(theta)
    tangent[:, 2] = np.cos(theta)
    vortex_strength = tube_circulation * segment_length * tangent
    if rotation is not None:
        pos = pos @ rotation.T
        vortex_strength = vortex_strength @ rotation.T

    with h5py.File(path, "w") as f:
        particles = f.create_group("particles")
        particles.create_dataset("position", data=pos)
        particles.create_dataset("vortex_strength", data=vortex_strength)
        particles.create_dataset("group_id", data=np.zeros(n, dtype=np.int32))
        solver = f.create_group("solver")
        solver.attrs["time"] = time


def test_ring_circulation_diagnostic_is_radius_independent(tmp_path):
    """Same tube circulation at larger radius changes Σ|alpha|, not Gamma_tube."""
    f0 = tmp_path / "vpm_ring_000000.h5"
    f1 = tmp_path / "vpm_ring_000001.h5"
    _write_ring_snapshot(f0, ring_radius=1.0, tube_circulation=np.pi, time=0.0)
    _write_ring_snapshot(f1, ring_radius=1.2, tube_circulation=np.pi, time=1.0)

    files = [str(f0), str(f1)]
    _, vortex_strength_magnitude_sum = load_length_integrated_strength(files)
    _, tube_circulation = load_ring_circulation(files)

    np.testing.assert_allclose(vortex_strength_magnitude_sum, [1.0, 1.2], rtol=1e-12, atol=1e-12)
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
    _write_ring_snapshot(f0, ring_radius=1.0, tube_circulation=np.pi, time=0.0)
    _write_ring_snapshot(f1, ring_radius=1.0, tube_circulation=np.pi, time=1.0, rotation=rotation)

    _, tube_circulation = load_ring_circulation([str(f0), str(f1)])

    np.testing.assert_allclose(tube_circulation, [1.0, 1.0], rtol=1e-12, atol=1e-12)


def test_ring_sampler_writes_dense_diagnostics_beside_other_samples(tmp_path):
    theta = np.linspace(0.0, 2.0 * np.pi, 96, endpoint=False)
    position = np.zeros((len(theta), 3))
    position[:, 1] = np.cos(theta)
    position[:, 2] = np.sin(theta)
    ds = 2.0 * np.pi / len(theta)
    vortex_strength = np.zeros_like(position)
    vortex_strength[:, 1] = -np.pi * ds * np.sin(theta)
    vortex_strength[:, 2] = np.pi * ds * np.cos(theta)

    solver = SimpleNamespace(
        setup=SimpleNamespace(
            samplers=(RingDiagnosticsSampler(),),
            sample_subdirectory="dns_direct",
        ),
        particles=SimpleNamespace(n_particles_total=len(theta)),
        particle_position=position,
        particle_vortex_strength=vortex_strength,
        particle_group_id=np.zeros(len(theta), dtype=np.int32),
        case_dir=tmp_path,
        checkpoint_directory=str(tmp_path / "solution"),
        time=0.1,
        step=5,
    )
    SamplerExecutor.execute(solver)
    solver.particle_position[:, 0] += 0.02
    solver.time = 0.2
    solver.step = 10
    SamplerExecutor.execute(solver)

    csv_path = tmp_path / "samples" / "dns_direct" / "ring_diagnostics.csv"
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
    position = np.column_stack((axial, radius * np.cos(theta), radius * np.sin(theta)))
    vortex_strength = np.column_stack(
        (
            np.zeros_like(theta),
            -np.sin(theta),
            np.cos(theta),
        )
    )

    sampler = RingModeDiagnosticsSampler(max_mode=12, azimuthal_bins=128)
    rows = sampler._sample_group(position, vortex_strength)
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
    centreline, slope = vortex_ring_centreline(
        theta,
        1.0,
        widnall_amplitude=epsilon,
        seed=42,
        n_widnall_modes=seeded_modes,
    )
    offsets = np.array((-0.04, 0.0, 0.04))
    rho = (centreline[None, :] + offsets[:, None]).reshape(-1)
    tiled_theta = np.tile(theta, len(offsets))
    position = np.column_stack(
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
    vortex_strength = rho[:, None] * (
        tangent + np.tile(slope, len(offsets))[:, None] * radial / rho[:, None]
    )

    sampler = RingModeDiagnosticsSampler(
        max_mode=40,
        azimuthal_bins=128,
        transverse_origin=(0.0, 0.0),
    )
    rows = np.asarray(sampler._sample_group(position, vortex_strength))
    expected = epsilon / np.sqrt(seeded_modes)
    relative_l2 = np.linalg.norm(rows[:seeded_modes, 1] - expected) / (
        np.sqrt(seeded_modes) * expected
    )

    assert relative_l2 < 3.0e-3
    assert np.sqrt(np.mean(rows[seeded_modes:, 1] ** 2)) < 2.0e-4
