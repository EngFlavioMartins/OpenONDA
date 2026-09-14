"""Physics and launcher checks for the vortex-interactions tutorial."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from openonda.tutorial_runner import load_case_module

CASE_DIR = (
    Path(__file__).resolve().parents[2] / "tutorials" / "vpm" / "03_vortex_interactions_PENDING"
)


def _load_setup():
    return load_case_module(CASE_DIR)


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
    assert left_model.distribution.spacing == setup.PARTICLE_SPACING == 0.05
    assert left_model.distribution.core_radius_ratio == 1.0
    assert left_model.disturbance.amplitude == 0.0
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
        assert case.run.final_backup
        assert case.run.wall_time_limit_seconds is None
        assert numerics.time_step_size == 0.00375
        assert numerics.turbulence.filter_width == pytest.approx(setup.LES_FILTER_WIDTH)
        assert numerics.health_limits.lagrangian_cfl.maximum == 1.0
        assert numerics.health_limits.divergence.maximum == 0.12
        assert numerics.health_limits.misalignment.maximum_degrees == 25.0


def test_each_case_enables_only_its_named_stabilization_method():
    setup = _load_setup()
    expected = {
        "baseline": ("conservative regularization",),
        "stretching_viscosity": ("residual stretching viscosity", "conservative regularization"),
        "p_moments": ("Pedrizzetti relaxation", "conservative regularization"),
    }
    assert tuple(expected) == setup.CASES

    for case_name, mechanisms in expected.items():
        config = setup.build_case(case_name).numerics.stabilization
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
        assert config.regularization_transfer_only


def test_frozen_representation_controls_are_retained():
    setup = _load_setup()
    config = setup.build_case("baseline").numerics.stabilization
    assert config.regularization_interval_steps == 20
    assert config.regularization_grid_spacing == config.regularization_core_radius == 0.05
    assert config.regularization_core_radius_trigger == 0.1
    assert config.regularization_tail_budget == 0.003
    assert config.regularization_max_particles == 120000
    assert config.regularization_total_kinetic_energy_dissipation_limit == 0.01
    assert config.regularization_total_enstrophy_dissipation_limit == 0.01


@pytest.mark.parametrize("variant", ["baseline", "stretching_viscosity", "p_moments", "splitting"])
def test_official_les_cases_preserve_the_studied_physics_and_native_outputs(variant):
    from types import SimpleNamespace

    from openonda.tutorial_runner import load_case_module
    from source.solvers.vpm.io.manifest import _case_configuration

    study = load_case_module(CASE_DIR, "assets.study")
    setup = load_case_module(CASE_DIR, "assets.legacy_les")
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


def test_run_and_plot_launchers_use_the_same_cases(tmp_path):
    import json
    import os
    import shutil
    import subprocess
    import sys

    setup = _load_setup()
    log = tmp_path / "commands.jsonl"
    stub = tmp_path / "python"
    stub.write_text(
        f"#!{sys.executable}\nimport json,os,sys\n"
        "with open(os.environ['COMMAND_LOG'],'a') as f: f.write(json.dumps(sys.argv[1:])+'\\n')\n"
        "sys.exit(1 if 'baseline' in sys.argv or 'divergence_relaxation' in sys.argv else 0)\n"
    )
    stub.chmod(0o755)
    for script in ("allrun.sh", "allplot.sh"):
        shutil.copy2(CASE_DIR / script, tmp_path / script)
    clean = tmp_path / "allclean.sh"
    clean.write_text("#!/bin/bash\nexit 0\n")
    clean.chmod(0o755)
    env = dict(os.environ, PATH=f"{tmp_path}{os.pathsep}{os.environ['PATH']}", COMMAND_LOG=str(log))
    subprocess.run(["bash", "allrun.sh"], cwd=tmp_path, env=env, check=True)
    subprocess.run(["bash", "allplot.sh"], cwd=tmp_path, env=env, check=True)
    commands = [json.loads(line) for line in log.read_text().splitlines()]
    assert [c[1] for c in commands[: len(setup.CASES)]] == list(setup.CASES)
    assert all(len(c) == 2 for c in commands[: len(setup.CASES)])
    sections, assessment = commands[-2:]
    expected = [f"fig5_{case}" for case in setup.CASES]
    assert sections[sections.index("--runs") + 1 : sections.index("--times")] == expected
    assert assessment[1 : assessment.index("--peak-merge-bridge")] == expected
