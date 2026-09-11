"""Physics and launcher checks for the vortex-interactions tutorial."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SETUP_PATH = (
    Path(__file__).resolve().parents[2] / "tutorials" / "vpm" / "vortex_interactions" / "setup.py"
)
CASE_DIR = SETUP_PATH.parent


def _load_setup():
    spec = importlib.util.spec_from_file_location("vortex_interactions_tutorial", SETUP_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ring_pair_is_a_translated_symmetric_toroidal_cloud():
    setup = _load_setup()
    left_model = setup.create_ring(-0.5 * setup.RING_SEPARATION, 0)
    right_model = setup.create_ring(0.5 * setup.RING_SEPARATION, 1)
    left = left_model.build()
    right = right_model.build()

    np.testing.assert_allclose(
        right.position,
        left.position + np.array([setup.RING_SEPARATION, 0.0, 0.0]),
        atol=2.0e-15,
    )
    np.testing.assert_allclose(right.vortex_strength, left.vortex_strength, atol=2.0e-15)
    np.testing.assert_array_equal(left.particle_volume, right.particle_volume)
    np.testing.assert_array_equal(left.core_radius, right.core_radius)
    np.testing.assert_array_equal(left.group_id, np.zeros(len(left), dtype=np.int32))
    np.testing.assert_array_equal(right.group_id, np.ones(len(right), dtype=np.int32))
    assert left_model.distribution.spacing == setup.PARTICLE_SPACING == 0.035
    assert left_model.distribution.core_radius_ratio == 2.0
    np.testing.assert_allclose(left.core_radius, setup.PARTICLE_CORE_RADIUS)


def test_all_cases_share_the_transposed_les_rk3_baseline():
    setup = _load_setup()
    for case_name in setup.CASES:
        case = setup.build_case(case_name)
        numerics = case.numerics
        assert numerics.integrator.name == "SSPRK3"
        assert numerics.integrator.order == 3
        assert numerics.induction.method == "TREECODE"
        assert numerics.induction.stretching_scheme == "TRANSPOSED"
        assert numerics.turbulence.model == "LES_SMAGORINSKY"
        assert numerics.turbulence.smagorinsky_coefficient == 0.20
        assert numerics.viscous.scheme == "CS"
        assert numerics.particle_kernel == "GAUSSIAN"
        assert case.run.health_limit_action == "STOP"
        assert not case.run.final_backup
        assert numerics.health_limits.lagrangian_cfl.maximum == 1.0
        assert numerics.health_limits.divergence.maximum == 0.12
        assert numerics.health_limits.misalignment.maximum_degrees == 25.0


def test_each_case_enables_only_its_named_stabilization_method():
    setup = _load_setup()
    expected = {
        "baseline": (),
        "stretching_viscosity": ("residual stretching viscosity",),
        "pedrizzetti": ("Pedrizzetti relaxation",),
        "splitting": ("filament refinement",),
        "divergence_relaxation": ("divergence relaxation",),
        "remeshing": ("conservative regularization",),
    }
    assert tuple(expected) == setup.CASES

    for case_name, mechanisms in expected.items():
        config = setup.stabilization(case_name)
        active = []
        if config.stretching_viscosity_coefficient > 0.0:
            active.append("residual stretching viscosity")
        if config.pedrizzetti_relaxation_enabled:
            active.append("Pedrizzetti relaxation")
        if config.filament_refinement.enabled:
            active.append("filament refinement")
        if config.divergence_relaxation.enabled:
            active.append("divergence relaxation")
        if config.regularization_interval_steps > 0:
            active.append("conservative regularization")
        assert tuple(active) == mechanisms


def test_core_growth_remeshing_time_is_derived_from_core_spreading():
    setup = _load_setup()
    elapsed = setup.REMESH_INTERVAL_STEPS * setup.TIME_STEP_SIZE
    expected = 3.0 * setup.PARTICLE_CORE_RADIUS**2 / (4.0 * setup.KINEMATIC_VISCOSITY)
    assert setup.REMESH_INTERVAL_STEPS == 450
    assert elapsed == pytest.approx(expected)


@pytest.mark.parametrize("variant", ["baseline", "stretching_viscosity", "p_moments", "splitting"])
def test_official_les_cases_preserve_the_studied_physics_and_native_outputs(variant):
    from types import SimpleNamespace

    from openonda.tutorial_runner import load_case_module
    from source.solvers.vpm.io.manifest import _case_configuration

    study = load_case_module(CASE_DIR, "assets.study")
    setup = load_case_module(CASE_DIR, "setup_les")
    args = study.parser().parse_args(
        [
            "--method",
            variant,
            "--steps",
            "1200",
            "--dt",
            ".0075",
            "--wall-minutes",
            "24",
            "--spacing",
            ".06",
            "--core-ratio",
            "1",
            "--amplitude",
            "0",
            "--smagorinsky",
            ".20",
            "--initial-tail",
            ".0001",
            "--tree-theta",
            ".5",
            "--tree-order",
            "3",
            "--capacity",
            "1000000",
            "--field-interval",
            ".15",
            "--frequency",
            ".384684814725",
            "--diffusion",
            "CS",
        ]
    )
    candidate = setup.build_case(variant)
    reference = study.build_experiment(args, candidate.directory)
    actual = _case_configuration(SimpleNamespace(case=candidate))
    expected = _case_configuration(SimpleNamespace(case=reference))
    actual_numerics = dict(actual["numerics"])
    expected_numerics = dict(expected["numerics"])
    assert actual_numerics.pop("domain_bounds") == [-2.0, 12.0, -3.0, 3.0, -3.0, 3.0]
    assert expected_numerics.pop("domain_bounds") is None
    assert actual_numerics == expected_numerics
    assert actual["initial_conditions"] == expected["initial_conditions"]
    assert candidate.name == f"cs_{variant}"
    assert candidate.directory == CASE_DIR
    assert candidate.backup.directory == f"solution/cs_{variant}"
    assert candidate.backup.interval_steps == 100
    assert candidate.run.final_backup
    assert candidate.samplers.directory == f"cs_{variant}"
    assert candidate.run.steps == 1200
    assert candidate.run.wall_time_limit_seconds == 1440
    assert {sampler.file_name for sampler in candidate.samplers.samples} == {
        "flow_integrals",
        "ring_diagnostics",
        "core_section",
        "cross_section",
    }


def test_official_launcher_runs_baseline_before_stabilizers():
    commands = [
        line
        for line in (CASE_DIR / "allrun.sh").read_text().splitlines()
        if line.startswith("python ")
    ]
    assert commands == [
        f"python setup_les.py --variant {variant}"
        for variant in ("baseline", "stretching_viscosity", "p_moments", "splitting")
    ]
