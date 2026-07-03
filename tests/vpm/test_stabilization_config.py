import pytest

from source.solvers.VPM import SolverConfig, StabilizationConfig
from source.solvers.VPM.config.types import RVPM_DEFAULT_F, RVPM_DEFAULT_G


def test_strength_relaxation_factory_builds_nested_config():
    stabilization = StabilizationConfig.strength_relaxation(
        mode="pedrizzetti",
        gate="constant",
        factor=0.9,
    )
    config = SolverConfig(stabilization=stabilization)

    assert config.stabilization is stabilization
    assert stabilization.relaxation_enabled
    assert stabilization.relaxation_mode == "pedrizzetti"
    assert stabilization.relaxation_factor == 0.9


def test_stabilization_supports_combined_mechanisms():
    stabilization = StabilizationConfig(
        max_core_radius=0.2,
        remove_particles_by_bounds=[-1, 1, -1, 1, -1, 1],
        relaxation_enabled=True,
    )

    assert stabilization.max_core_radius == 0.2
    assert stabilization.relaxation_enabled


def test_solver_config_rejects_removed_top_level_stabilization_options():
    with pytest.raises(TypeError):
        SolverConfig(isr_enabled=True)


def test_stabilization_round_trip_is_nested():
    original = SolverConfig(
        stabilization=StabilizationConfig.strength_relaxation(rate=1.5, deconv=2)
    )

    restored = SolverConfig.from_dict(original.to_dict())

    assert restored.stabilization == original.stabilization
    assert restored.stabilization.relaxation_rate == 1.5


def test_parallel_strain_factory_uses_flowvpm_rvpm_defaults():
    stabilization = StabilizationConfig.parallel_strain_relaxation()

    assert stabilization.parallel_strain_enabled
    assert stabilization.parallel_strain_f == pytest.approx(RVPM_DEFAULT_F)
    assert stabilization.parallel_strain_g == pytest.approx(RVPM_DEFAULT_G)


def test_vortex_interactions_cli_uses_flowvpm_rvpm_default():
    from tutorials.VPM.vortexInteractions.rings_setup import build_arg_parser

    args = build_arg_parser().parse_args([])

    assert args.parallel_strain_f == pytest.approx(RVPM_DEFAULT_F)
    assert args.parallel_strain_g == pytest.approx(RVPM_DEFAULT_G)
