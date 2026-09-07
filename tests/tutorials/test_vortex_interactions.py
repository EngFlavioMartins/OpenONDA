"""Physics and launcher checks for the vortex-interactions tutorial."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess

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


def test_shell_launchers_parse():
    for script in ("allrun.sh", "allplot.sh", "allclean.sh"):
        subprocess.run(["bash", "-n", str(CASE_DIR / script)], check=True)


def test_allrun_launches_every_case_and_continues_after_one_failure(tmp_path):
    setup = _load_setup()
    log = tmp_path / "calls.txt"
    python = tmp_path / "python"
    python.write_text(
        "#!/usr/bin/env bash\n"
        'printf \'%s\\n\' "$*" >> "$VORTEX_INTERACTIONS_CALLS"\n'
        "[[ \"$*\" == *'.setup pedrizzetti '* ]] && exit 7\n"
        "exit 0\n",
        encoding="utf-8",
    )
    python.chmod(0o755)
    environment = os.environ.copy()
    environment["OPENONDA_PYTHON"] = str(python)
    environment["VORTEX_INTERACTIONS_CALLS"] = str(log)

    result = subprocess.run(
        [str(CASE_DIR / "allrun.sh"), "--campaign", "legacy", "--steps", "1"],
        cwd=CASE_DIR.parents[2],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    calls = log.read_text(encoding="utf-8").splitlines()
    launched = [line.split(".setup ", 1)[1].split()[0] for line in calls if ".setup " in line]
    assert launched == list(setup.CASES)
    assert "Pedrizzetti relaxation | exit 7; continuing" in result.stderr


def test_strategy_launcher_runs_matched_scenarios_and_retains_failed_comparisons(tmp_path):
    import shlex

    log = tmp_path / "calls.txt"
    python = tmp_path / "python"
    python.write_text(
        "#!/usr/bin/env bash\n"
        'printf \'%s\\n\' "$*" >> "$VORTEX_INTERACTIONS_CALLS"\n'
        '[[ "$*" == *"--method splitting"* ]] && exit 7\nexit 0\n'
    )
    python.chmod(0o755)
    environment = {
        **os.environ,
        "OPENONDA_PYTHON": str(python),
        "VORTEX_INTERACTIONS_CALLS": str(log),
    }
    result = subprocess.run(
        [str(CASE_DIR / "allrun.sh"), "--campaign", "strategies", "--quick"],
        env=environment,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    calls = [shlex.split(line) for line in log.read_text().splitlines()]
    simulations = [line for line in calls if "--method" in line]
    assert len(simulations) == 8
    for scenario in ("leapfrog", "collision"):
        matched = [line for line in simulations if line[line.index("--scenario") + 1] == scenario]
        assert [line[line.index("--method") + 1] for line in matched] == [
            "baseline",
            "splitting",
            "remeshing",
            "p_remesh",
        ]
        for line in matched:
            assert line[line.index("--spacing") + 1] == ".04"
            assert line[line.index("--support") + 1] == "disturbed"
            assert line[line.index("--steps") + 1] == "400"
    assert result.stderr.count("exit 7; continuing") == 2
    assert "--runs" in calls[-1]
