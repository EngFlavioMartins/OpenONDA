from pathlib import Path
import runpy

import pytest

from source.solvers.VPM import StabilizationConfig, VPMSetup


def test_retention_domain_is_normalized_and_nested():
    retention = StabilizationConfig.bounded_domain([-1, 1, -2, 2, -3, 3])
    config = VPMSetup(stabilization=retention)

    assert retention.remove_particles_by_bounds == (-1.0, 1.0, -2.0, 2.0, -3.0, 3.0)
    assert config.stabilization is retention


def test_retention_domain_requires_six_coordinates():
    with pytest.raises(ValueError, match="must have 6 elements"):
        StabilizationConfig.bounded_domain([-1, 1])


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


def test_vortex_interactions_cli_defaults_to_les_control():
    args = _vortex_interactions_namespace()["build_arg_parser"]().parse_args([])

    assert args.processing_unit == "AUTO"
    assert args.method == "les"
    assert args.dt == pytest.approx(20.0 * 0.02**2 / 3.141592653589793)
    assert args.particle_spacing == pytest.approx(0.02)
    assert args.epsilon_w == pytest.approx(0.025)
    assert args.guard_frequency == 1


def test_vortex_interactions_les_control_has_no_field_filter(tmp_path):
    namespace = _vortex_interactions_namespace()
    args = namespace["build_arg_parser"]().parse_args([])
    config = namespace["build_solver_config"](args, tmp_path, "leapfrog")

    assert config.time_integration == "FRACTIONAL"
    assert config.velocity.method == "TREECODE"
    assert config.particles_kernel == "GAUSSIAN"
    assert config.advection.scheme == config.stretching.scheme == "RK3"
    assert config.stretching.mode == "TRANSPOSED"
    assert config.turbulence.flow_model == "LES"
    assert config.turbulence.cs == pytest.approx(namespace["CONTROL_LES_CS"])
    assert config.stabilization == StabilizationConfig.disabled()
    assert config.viscous.viscosity == pytest.approx(namespace["KINEMATIC_VISCOSITY"])
    assert config.viscous.characteristic_distance == pytest.approx(args.particle_spacing)


def test_vortex_interactions_two_controls_are_distinct(tmp_path):
    namespace = _vortex_interactions_namespace()
    parser = namespace["build_arg_parser"]()

    baseline = namespace["build_solver_config"](
        parser.parse_args(["--method", "baseline"]), tmp_path, "leapfrog"
    )
    les = namespace["build_solver_config"](
        parser.parse_args(["--method", "les"]), tmp_path, "leapfrog"
    )
    assert baseline.turbulence.flow_model == "DNS"
    assert les.turbulence.flow_model == "LES"
    assert baseline.time_integration == les.time_integration == "FRACTIONAL"
    assert baseline.velocity.method == les.velocity.method == "TREECODE"


def test_vortex_interactions_stabilized_method_is_the_combined_candidate(
    tmp_path,
):
    namespace = _vortex_interactions_namespace()
    args = namespace["build_arg_parser"]().parse_args(
        [
            "--method",
            "les_stabilized",
            "--particle-spacing",
            "0.03",
            "--allow-underresolved",
        ]
    )

    config = namespace["build_solver_config"](
        args,
        tmp_path,
        "collide",
    )

    assert config.turbulence.flow_model == "LES"
    assert config.filament_refinement.enabled
    assert config.filament_refinement.frequency == 1
    assert config.divergence_relaxation.enabled
    assert config.divergence_relaxation.frequency == 10
    assert config.divergence_relaxation.start_step == 50
    assert config.divergence_relaxation.grid_spacing == pytest.approx(0.045)
    assert config.divergence_relaxation.max_correction_norm == pytest.approx(0.02)
    assert config.divergence_relaxation.max_residual_ratio == pytest.approx(0.9)
    assert config.divergence_relaxation.spectral_convergence_fraction == pytest.approx(0.1)


def test_vortex_interactions_scripts_require_the_six_case_matrix():
    tutorial = Path(__file__).parents[2] / "tutorials/VPM/vortexInteractions"
    allrun = (tutorial / "allrun.sh").read_text(encoding="utf-8")
    validator = (tutorial / "assets/validate_plot_inputs.py").read_text(encoding="utf-8")

    assert 'DEFAULT_METHODS="baseline les les_stabilized"' in allrun
    assert 'METHODS = ("baseline", "les", "les_stabilized")' in validator
    assert "complete six-case matrix" in validator
    assert 'name.endswith("_les_stabilized")' in validator
    assert 'status != "completed"' in validator


def test_vortex_interactions_reference_contract_rejects_coarse_spacing():
    namespace = _vortex_interactions_namespace()
    args = namespace["build_arg_parser"]().parse_args(["--particle-spacing", "0.03"])

    with pytest.raises(ValueError, match="h/a0=.* > 0.2"):
        namespace["validate_resolution"](args)


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
