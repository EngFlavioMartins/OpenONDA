"""Physics and launcher checks for the vortex-interactions tutorial."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from openonda.tutorial_runner import load_case_module
from tests._tutorial_helpers import load_tutorial_module

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
    assert left_model.disturbance is None
    assert left_model.distribution.disturbance is None
    np.testing.assert_allclose(left.core_radius, setup.PARTICLE_CORE_RADIUS)


def test_nonzero_validation_seed_shifts_geometry_in_the_papers_axial_direction(monkeypatch):
    setup = _load_setup()
    unperturbed = setup.create_ring(0.0, 0).distribution.build().position
    monkeypatch.setattr(setup, "DISTURBANCE_AMPLITUDE", 0.05)
    ring = setup.create_ring(0.0, 0)
    perturbed = ring.distribution.build().position

    assert ring.disturbance is ring.distribution.disturbance
    assert ring.disturbance.direction == "axial"
    assert ring.disturbance.mode == 8
    np.testing.assert_allclose(perturbed[:, 1:], unperturbed[:, 1:])
    assert np.max(np.abs(perturbed[:, 0] - unperturbed[:, 0])) == pytest.approx(0.05)


def test_all_cases_share_the_transposed_dns_rk3_baseline():
    setup = _load_setup()
    for case_name in setup.CASES:
        case = setup.build_case(case_name)
        numerics = case.numerics
        assert numerics.integrator.name == "SSPRK3"
        assert numerics.integrator.order == 3
        assert numerics.induction.method == "TREECODE"
        assert numerics.induction.stretching_scheme == "TRANSPOSED"
        assert numerics.turbulence.model == "DNS"
        assert numerics.domain_bounds is None
        assert numerics.viscous.scheme == "CS"
        assert numerics.particle_kernel == "GAUSSIAN"
        assert case.run.health_limit_action == "STOP"
        assert case.run.final_backup
        assert case.run.wall_time_limit_seconds is None
        assert numerics.time_step_size == 0.00375
        assert numerics.viscous.kinematic_viscosity == pytest.approx(np.pi / 3000)
        assert numerics.max_n_particles == setup.MAX_N_PARTICLES
        assert case.run.resource_limits.max_particles == setup.MAX_N_PARTICLES
        assert not any(
            isinstance(s, setup.vpm.RingDiagnosticsSampler) for s in case.samplers.samples
        )
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
    assert config.regularization_max_particles == setup.MAX_N_PARTICLES
    assert config.regularization_total_kinetic_energy_dissipation_limit == 0.01
    assert config.regularization_total_enstrophy_dissipation_limit == 0.01


def test_dns_metadata_does_not_report_an_active_smagorinsky_closure():
    postprocess = load_tutorial_module("vpm/vortex_interactions", "assets.postprocess")
    settings = postprocess.metadata_settings(
        {
            "configuration": {
                "initial_conditions": [{"circulation": np.pi, "kinematic_viscosity": np.pi / 3000}],
                "numerics": {
                    "time_step_size": 0.00375,
                    "turbulence": {"model": "DNS", "smagorinsky_coefficient": 0.2},
                },
            }
        }
    )
    assert settings["flow_model"] == "DNS"
    assert settings["smagorinsky"] == 0.0


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
        "sys.exit(1 if sys.argv[1:3] == ['setup.py', 'baseline'] else 0)\n"
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
    expected = list(setup.CASES)
    assert sections[sections.index("--runs") + 1 : sections.index("--times")] == expected
    assert assessment[1 : assessment.index("--peak-merge-bridge")] == expected
