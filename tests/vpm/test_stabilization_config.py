from pathlib import Path
import runpy

import numpy as np
import pytest

from source.solvers.VPM import StabilizationConfig, TurbulenceConfig, VPMSetup


def test_equilibrium_smagorinsky_factory_recovers_ck():
    config = TurbulenceConfig.equilibrium_smagorinsky()

    recovered_ck = (config.cs**2 * config.ce**0.5) ** (2.0 / 3.0)
    assert config.flow_model == "LES"
    assert recovered_ck == pytest.approx(0.094)


def test_retention_domain_is_normalized_and_nested():
    retention = StabilizationConfig.bounded_domain([-1, 1, -2, 2, -3, 3])
    config = VPMSetup(stabilization=retention)

    assert retention.remove_particles_by_bounds == (-1.0, 1.0, -2.0, 2.0, -3.0, 3.0)
    assert config.stabilization is retention


def test_retention_domain_requires_six_coordinates():
    with pytest.raises(ValueError, match="must have 6 elements"):
        StabilizationConfig.bounded_domain([-1, 1])


def test_stretching_viscosity_factory_and_validation():
    stabilization = StabilizationConfig.stretching_viscosity(coefficient=0.6)

    assert stabilization.stretching_viscosity_coefficient == pytest.approx(0.6)
    assert stabilization.remove_particles_by_bounds is None
    with pytest.raises(ValueError, match="finite and non-negative"):
        StabilizationConfig.stretching_viscosity(coefficient=-0.1)


def test_conservative_filter_factory_round_trip_and_validation():
    stabilization = StabilizationConfig.conservative_filter(
        coefficient=0.4,
        frequency=25,
        start_step=100,
        grid_spacing=0.08,
        max_particles=4000,
        tail_budget=0.002,
        divergence_trigger=0.03,
        misalignment_trigger=15.0,
        capacity_fraction=0.8,
        capacity_grid_spacing=0.12,
        core_radius=0.2,
        capacity_core_radius=0.19,
    )
    restored = VPMSetup.from_dict(VPMSetup(stabilization=stabilization).to_dict())

    assert restored.stabilization == stabilization
    assert stabilization.regularization_capacity_fraction == pytest.approx(0.8)
    assert stabilization.regularization_capacity_grid_spacing == pytest.approx(0.12)
    assert stabilization.regularization_core_radius == pytest.approx(0.2)
    assert stabilization.regularization_capacity_core_radius == pytest.approx(0.19)
    with pytest.raises(ValueError, match="tail budget"):
        StabilizationConfig.conservative_filter(
            frequency=1,
            start_step=0,
            grid_spacing=0.08,
            max_particles=100,
            tail_budget=0.0,
        )
    with pytest.raises(ValueError, match="misalignment trigger"):
        StabilizationConfig.conservative_filter(
            frequency=1,
            start_step=0,
            grid_spacing=0.08,
            max_particles=100,
            misalignment_trigger=181.0,
        )
    with pytest.raises(ValueError, match="enstrophy-dissipation"):
        StabilizationConfig.conservative_filter(
            frequency=1,
            start_step=0,
            grid_spacing=0.08,
            max_particles=100,
            enstrophy_dissipation_limit=1.0,
        )
    with pytest.raises(ValueError, match="capacity fraction"):
        StabilizationConfig.conservative_filter(
            frequency=1,
            start_step=0,
            grid_spacing=0.08,
            max_particles=100,
            capacity_fraction=0.0,
        )
    for name in ("capacity_grid_spacing", "core_radius", "capacity_core_radius"):
        with pytest.raises(ValueError, match="positive"):
            StabilizationConfig.conservative_filter(
                frequency=1,
                start_step=0,
                grid_spacing=0.08,
                max_particles=100,
                **{name: np.nan},
            )


def test_retention_round_trip_is_nested():
    original = VPMSetup(stabilization=StabilizationConfig.bounded_domain((-1, 1, -2, 2, -3, 3)))

    restored = VPMSetup.from_dict(original.to_dict())

    assert restored.stabilization == original.stabilization


def test_solver_config_rejects_removed_top_level_stabilization_options():
    with pytest.raises(TypeError):
        VPMSetup(isr_enabled=True)


def _vortex_interactions_namespace():
    tutorial = Path(__file__).parents[2] / "tutorials/VPM/vortexInteractions/rings_setup.py"
    return runpy.run_path(tutorial)


def _rotor_flow_namespace():
    tutorial = Path(__file__).parents[2] / "tutorials/VPM/rotorFlow/rotor_setup.py"
    return runpy.run_path(tutorial)


def test_vortex_interactions_uses_one_hard_coded_six_case_matrix(tmp_path):
    namespace = _vortex_interactions_namespace()
    cases = namespace["CASES"]
    configs = {name: namespace["solver_setup"](name, tmp_path / name) for name in cases}

    assert tuple(cases) == (
        "leapfrog_dns",
        "leapfrog_les",
        "leapfrog_les_stabilized",
        "collide_dns",
        "collide_les",
        "collide_les_stabilized",
    )
    assert namespace["PARTICLE_SPACING"] == pytest.approx(0.04)
    assert namespace["TIME_STEP"] == pytest.approx(20.0 * 0.04**2 / np.pi)
    assert namespace["NUM_STEPS"] == 1140
    assert namespace["END_TIME"] == pytest.approx(1140 * namespace["TIME_STEP"])
    assert namespace["BASELINE_DIVERGENCE_LIMIT"] == pytest.approx(0.12)
    for config in configs.values():
        assert config.processing_unit == "CPU"
        assert config.precision == "f64"
        assert config.time_integration == "COUPLED"
        assert config.axisymmetric_no_swirl_axis is None
        assert config.velocity.method == "DIRECT"
        assert config.particles_kernel == "GAUSSIAN"
        assert config.advection.scheme == config.stretching.scheme == "RK2"
        assert config.stretching.mode == "TRANSPOSED"
        assert not config.stretching.use_treecode
        assert config.stretching.conserve_moments
        assert config.stretching.conserve_energy
        assert config.coupled_max_strain_increment == pytest.approx(0.15)
        assert config.coupled_max_advection_fraction == pytest.approx(0.5)
        assert config.viscous.viscosity == pytest.approx(namespace["KINEMATIC_VISCOSITY"])

    for name in ("leapfrog_dns", "leapfrog_les", "collide_dns", "collide_les"):
        assert configs[name].viscous.scheme == "CS"
        assert configs[name].viscous.characteristic_distance == pytest.approx(
            namespace["PARTICLE_SPACING"]
        )

    assert configs["leapfrog_dns"].turbulence.flow_model == "DNS"
    assert configs["leapfrog_les"].turbulence.cs == pytest.approx(
        namespace["LES_COEFFICIENT"]["leapfrog"]
    )
    assert configs["collide_les"].turbulence.cs == pytest.approx(
        namespace["LES_COEFFICIENT"]["collide"]
    )
    assert configs["leapfrog_les"].stabilization == StabilizationConfig.disabled()
    assert configs[
        "leapfrog_les_stabilized"
    ].stabilization.stretching_viscosity_coefficient == pytest.approx(
        namespace["STABILIZATION_COEFFICIENT"]
    )
    assert not configs["leapfrog_dns"].stabilization.filament_refinement.enabled
    assert not configs["leapfrog_les"].stabilization.filament_refinement.enabled
    assert not configs["leapfrog_les_stabilized"].stabilization.filament_refinement.enabled
    assert not configs["leapfrog_dns"].stabilization.divergence_relaxation.enabled
    assert not configs["leapfrog_les"].stabilization.divergence_relaxation.enabled
    assert not configs["leapfrog_les_stabilized"].stabilization.divergence_relaxation.enabled
    assert configs["leapfrog_les_stabilized"].viscous.scheme == "CS"
    assert (
        configs["leapfrog_les_stabilized"].stabilization.regularization_max_particles
        == namespace["STABILIZED_MAX_PARTICLES"]
    )
    assert (
        configs["leapfrog_les_stabilized"].stabilization.regularization_frequency
        == namespace["REGULARIZATION_FREQUENCY"]
    )
    assert configs[
        "leapfrog_les_stabilized"
    ].stabilization.regularization_tail_budget == pytest.approx(
        namespace["REGULARIZATION_TAIL_BUDGET"]
    )
    assert configs[
        "leapfrog_les_stabilized"
    ].stabilization.regularization_divergence_trigger == pytest.approx(
        namespace["REGULARIZATION_DIVERGENCE_TRIGGER"]
    )
    assert configs[
        "leapfrog_les_stabilized"
    ].stabilization.regularization_misalignment_trigger == pytest.approx(
        namespace["REGULARIZATION_MISALIGNMENT_TRIGGER"]
    )
    assert configs[
        "leapfrog_les_stabilized"
    ].stabilization.regularization_capacity_grid_spacing == pytest.approx(
        namespace["REGULARIZATION_CAPACITY_SPACING"]
    )
    assert configs[
        "leapfrog_les_stabilized"
    ].stabilization.regularization_core_radius == pytest.approx(
        namespace["REGULARIZATION_CORE_RADIUS"]["leapfrog"]
    )
    assert configs[
        "leapfrog_les_stabilized"
    ].stabilization.regularization_capacity_core_radius == pytest.approx(
        namespace["REGULARIZATION_CAPACITY_CORE_RADIUS"]
    )
    assert configs[
        "leapfrog_les_stabilized"
    ].stabilization.regularization_projection_trigger == pytest.approx(
        namespace["REGULARIZATION_PROJECTION_TRIGGER"]
    )
    assert configs[
        "leapfrog_les_stabilized"
    ].stabilization.regularization_projection_max_correction == pytest.approx(
        namespace["REGULARIZATION_PROJECTION_LIMIT"]["leapfrog"]
    )
    assert configs[
        "collide_les_stabilized"
    ].stabilization.regularization_projection_max_correction == pytest.approx(
        namespace["REGULARIZATION_PROJECTION_LIMIT"]["collide"]
    )


def test_vortex_interactions_initial_state_matches_ring_invariants():
    namespace = _vortex_interactions_namespace()
    position, _, radius, _, circulation = namespace["ring_particles"](
        0.0, namespace["RING_CIRCULATION"], namespace["RING_SEEDS"][0]
    )
    config = namespace["solver_setup"]("leapfrog_dns", Path("solution/test"))

    total = circulation.sum(axis=0)
    impulse_x = 0.5 * np.cross(position, circulation).sum(axis=0)[0]
    expected_impulse = np.pi * namespace["RING_CIRCULATION"] * namespace["RING_RADIUS"] ** 2

    assert 2 * len(position) <= config.max_particles
    assert np.linalg.norm(total) < 1.0e-12
    assert impulse_x == pytest.approx(expected_impulse, rel=5.0e-3)
    np.testing.assert_allclose(radius, namespace["PARTICLE_RADIUS"])
    assert config.axisymmetric_no_swirl_axis is None


def test_rotor_flow_cli_defaults_to_physics_preserving_policy():
    namespace = _rotor_flow_namespace()
    args = namespace["build_arg_parser"]().parse_args([])

    assert args.dt == pytest.approx(0.006)
    assert args.treecode_theta == pytest.approx(0.20)
    assert args.coupled_max_strain_increment == pytest.approx(0.08)
    assert args.coupled_max_advection_fraction == pytest.approx(0.25)
    assert args.coupled_max_substeps == 128
    assert args.guard_frequency == 20


def test_rotor_flow_uses_scalable_coupled_stabilization():
    namespace = _rotor_flow_namespace()
    args = namespace["build_arg_parser"]().parse_args([])
    config = namespace["build_solver_config"](args)

    assert config.time_integration == "COUPLED"
    assert config.advection.scheme == config.stretching.scheme == "RK2"
    assert config.stretching.mode == "TRANSPOSED"
    assert config.stretching.use_treecode is True
    assert config.stretching.treecode_theta == pytest.approx(args.treecode_theta)
    assert config.velocity.method == "TREECODE"
    assert config.velocity.theta == pytest.approx(args.treecode_theta)
    assert config.particles_kernel == "WINCKELMANS"
    assert config.viscous.scheme == "CS"
    assert config.viscous.viscosity == pytest.approx(namespace["KINEMATIC_VISCOSITY"])
    assert config.viscous.characteristic_distance == pytest.approx(
        namespace["nominal_wake_spacing"](args.dt)
    )
    assert config.stabilization.remove_particles_by_bounds is not None
