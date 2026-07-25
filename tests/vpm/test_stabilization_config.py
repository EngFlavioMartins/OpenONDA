from pathlib import Path
import runpy

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
    tutorial = Path(__file__).parents[2] / "tutorials/VPM/vortexInteractions/rings_setup.py"
    build_arg_parser = runpy.run_path(tutorial)["build_arg_parser"]

    args = build_arg_parser().parse_args([])

    assert args.parallel_strain_f == pytest.approx(RVPM_DEFAULT_F)
    assert args.parallel_strain_g == pytest.approx(RVPM_DEFAULT_G)
    assert args.processing_unit == "GPU"
    assert args.perturbation_model == "solenoidal"
    assert args.stabilization == "adaptive"
    assert args.dt == pytest.approx(0.01)
    assert args.particle_spacing == pytest.approx(0.045)
    assert args.remesh_frequency == 100
    assert args.remesh_relative_threshold == pytest.approx(0.005)
    assert args.remesh_max_particles == 30000
    assert args.remesh_max_particle_growth == pytest.approx(1.5)
    assert args.split_strength == pytest.approx(0.5)
    assert not args.disable_stage_safety


@pytest.mark.parametrize(
    "variant",
    ["les", "rvpm", "relax", "remesh", "projection", "split", "energy", "adaptive"],
)
def test_vortex_interactions_variants_share_stage_safety(variant):
    tutorial = Path(__file__).parents[2] / "tutorials/VPM/vortexInteractions/rings_setup.py"
    namespace = runpy.run_path(tutorial)
    args = namespace["build_arg_parser"]().parse_args(["--stabilization", variant])

    stabilization = namespace["build_stabilization_config"](args, args.particle_spacing)

    assert stabilization.stretching_limiter_enabled
    assert stabilization.stretching_limiter_project_step_invariants
    assert stabilization.stretching_limiter_project_step_angular_impulse
    if variant in {"les", "rvpm", "relax", "remesh", "projection", "split"}:
        assert stabilization.energy_budget_enabled
        if variant in {"remesh", "projection"}:
            assert stabilization.energy_budget_tolerance == pytest.approx(0.2)
            assert stabilization.energy_budget_r_max == pytest.approx(0.3)
            assert stabilization.relaxation_deconv == 0
            assert stabilization.remeshing_conserve_energy
            assert stabilization.remeshing_preserve_radius_profile
            assert stabilization.remeshing_max_particles == 30000
            assert stabilization.remeshing_max_particle_growth == pytest.approx(1.5)
        else:
            assert stabilization.energy_budget_tolerance == pytest.approx(0.5)
            assert stabilization.energy_budget_r_max == pytest.approx(0.05)
    if variant == "relax":
        assert stabilization.relaxation_mode == "blend"
        assert stabilization.relaxation_deconv == 1
        assert stabilization.relaxation_factor == pytest.approx(0.01)
    if variant == "split":
        assert stabilization.max_particle_strength == pytest.approx(0.5)


def test_adaptive_rvpm_combines_limiter_rvpm_and_budget_control():
    stabilization = StabilizationConfig.adaptive_rvpm()

    assert stabilization.stretching_limiter_enabled
    assert stabilization.parallel_strain_enabled
    assert stabilization.energy_budget_enabled
    assert stabilization.relaxation_enabled
    assert stabilization.stretching_limiter_project_step_invariants
    assert stabilization.stretching_limiter_project_step_angular_impulse
    assert stabilization.energy_budget_r_max == pytest.approx(0.02)


def test_angular_projection_requires_complete_step_projection():
    with pytest.raises(ValueError, match="angular-impulse projection requires"):
        StabilizationConfig(
            stretching_limiter_enabled=True,
            stretching_limiter_project_step_angular_impulse=True,
        )
