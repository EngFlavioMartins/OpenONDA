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


def test_vortex_interactions_cli_defaults_to_stabilized_reference():
    args = _vortex_interactions_namespace()["build_arg_parser"]().parse_args([])

    assert args.processing_unit == "AUTO"
    assert args.method == "stabilized"
    assert args.dt == pytest.approx(20.0 * 0.02**2 / 3.141592653589793)
    assert args.particle_spacing == pytest.approx(0.02)
    assert args.epsilon_w == pytest.approx(0.025)
    assert args.guard_frequency == 1


def test_vortex_interactions_stabilized_config_has_no_field_filter(tmp_path):
    namespace = _vortex_interactions_namespace()
    args = namespace["build_arg_parser"]().parse_args([])
    config = namespace["build_solver_config"](args, tmp_path, "leapfrog")

    assert config.time_integration == "COUPLED"
    assert config.velocity.method == "DIRECT"
    assert config.particles_kernel == "WINCKELMANS"
    assert config.advection.scheme == config.stretching.scheme == "RK2"
    assert config.stretching.mode == "CONSERVATIVE"
    assert config.stabilization == StabilizationConfig.disabled()


def test_vortex_interactions_stabilized_contract_rejects_coarse_spacing():
    namespace = _vortex_interactions_namespace()
    args = namespace["build_arg_parser"]().parse_args(["--particle-spacing", "0.03"])

    with pytest.raises(ValueError, match="h/a0=.* > 0.2"):
        namespace["validate_resolution"](args)
