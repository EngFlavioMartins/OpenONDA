import pytest

from source.solvers.VPM import SolverConfig, StabilizationConfig


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
