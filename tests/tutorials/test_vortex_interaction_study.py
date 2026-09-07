"""Physical controls for the stabilization study launcher."""

from pathlib import Path

import numpy as np
import pytest

from tutorials.vpm.vortex_interactions.study import build_experiment, parser


def test_initial_gaussian_tail_does_not_amplify_the_core_peak():
    from source.solvers.vpm.diagnostics.axisymmetric_field import azimuthal_vorticity

    peaks = []
    for tail in (0.05, 1e-4):
        args = parser().parse_args(
            ["--spacing", ".04", "--amplitude", "0", "--initial-tail", str(tail)]
        )
        ring = (
            build_experiment(args, Path("/tmp/interaction-config-test"))
            .initial_conditions[1]
            .build()
        )
        peaks.append(
            azimuthal_vorticity(
                ring.position, ring.vortex_strength, ring.core_radius, [[0.5, 1.0]]
            )[0]
        )
    # Gamma/(pi*a^2) = 100 in the dimensional tutorial units.
    assert peaks[0] > 105
    assert peaks[1] == pytest.approx(100, rel=0.003)


def test_fixed_core_diffusion_control_preserves_the_physical_viscosity():
    from tutorials.vpm.vortex_interactions import setup

    args = parser().parse_args(["--diffusion", "GBD", "--core-ratio", "1", "--smagorinsky", "0"])
    case = build_experiment(args, Path("/tmp/interaction-config-test"))
    assert case.numerics.viscous.scheme == "GBD"
    assert case.numerics.turbulence.model == "DNS"
    cloud = case.initial_conditions[0].build()
    np.testing.assert_allclose(cloud.core_radius, args.spacing)
    np.testing.assert_allclose(cloud.kinematic_viscosity, setup.KINEMATIC_VISCOSITY)


def test_stretching_control_changes_the_restart_identity():
    from source.solvers.vpm.config.fingerprint import numerical_configuration

    identities = []
    for scheme in ("DIRECT", "TRANSPOSED"):
        args = parser().parse_args(["--stretching", scheme])
        case = build_experiment(args, Path("/tmp/interaction-config-test"))
        identities.append(numerical_configuration(case.numerics)["induction"])
    assert identities[0] != identities[1]


def test_collision_reverses_only_the_second_rings_circulation():
    cases = {}
    for scenario in ("leapfrog", "collision"):
        args = parser().parse_args(["--scenario", scenario])
        cases[scenario] = build_experiment(args, Path("/tmp/interaction-config-test"))
    for group in (0, 1):
        leap = cases["leapfrog"].initial_conditions[group].build()
        collide = cases["collision"].initial_conditions[group].build()
        np.testing.assert_array_equal(leap.position, collide.position)
        np.testing.assert_array_equal(leap.core_radius, collide.core_radius)
        np.testing.assert_array_equal(
            leap.vortex_strength * (1 if group == 0 else -1), collide.vortex_strength
        )
    impulse = sum(
        0.5 * np.cross(ring.build().position, ring.build().vortex_strength).sum(axis=0)
        for ring in cases["collision"].initial_conditions
    )
    np.testing.assert_allclose(impulse, 0, atol=1e-12)


def test_timestep_refinement_keeps_relaxation_frequency_fixed():
    factors = []
    for dt in (0.01, 0.005):
        args = parser().parse_args(
            ["--method", "p_relaxation", "--frequency", "2.0", "--dt", str(dt)]
        )
        case = build_experiment(args, Path("/tmp/interaction-config-test"))
        config = case.numerics.stabilization
        factors.append(config.pedrizzetti_relaxation_factor)
        assert not config.pedrizzetti_relaxation_preserve_vortex_strength
        assert not config.pedrizzetti_relaxation_preserve_moments
    assert factors[0] == pytest.approx(2 * factors[1])


def test_regeneration_does_not_silently_change_the_molecular_viscous_operator():
    args = parser().parse_args(["--method", "p_split_remesh"])
    case = build_experiment(args, Path("/tmp/interaction-config-test"))
    assert case.numerics.viscous.scheme == "CS"
    assert case.numerics.turbulence.model == "LES_SMAGORINSKY"
    assert case.numerics.stabilization.regularization_start_step < 450
    assert case.numerics.stabilization.filament_refinement.enabled


def test_disturbed_support_moves_geometry_and_uses_its_cylindrical_jacobian():
    from dataclasses import replace

    import openonda.vpm as vpm

    distribution = vpm.ToroidalDistribution(
        ring_radius=1.0, tube_radius=0.1, spacing=0.04, core_radius_ratio=2.0
    )
    original = distribution.build()
    disturbance = vpm.WidnallDisturbance.single_mode(amplitude=0.05, mode=8)
    moved = replace(distribution, disturbance=disturbance).build()
    radius = np.hypot(original.position[:, 1], original.position[:, 2])
    theta = np.arctan2(original.position[:, 2], original.position[:, 1])
    delta = disturbance.centreline(theta, 1.0)[0] - 1.0
    np.testing.assert_allclose(np.hypot(moved.position[:, 1], moved.position[:, 2]), radius + delta)
    np.testing.assert_allclose(
        moved.particle_volume, original.particle_volume * (radius + delta) / radius
    )
    np.testing.assert_array_equal(moved.position[:, 0], original.position[:, 0])
    np.testing.assert_array_equal(moved.core_radius, original.core_radius)
    assert moved.particle_volume.sum() == pytest.approx(original.particle_volume.sum())
