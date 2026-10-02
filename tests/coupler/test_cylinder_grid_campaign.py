"""Grid-family contract for the cylinder reference flow."""

import importlib.util
import os
from pathlib import Path
import shlex
import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
CASE = ROOT / "tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow/reference_flow"


def load_setup():
    spec = importlib.util.spec_from_file_location("cylinder_reference_setup", CASE / "setup.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_setup_uses_h_as_the_realized_wall_spacing(monkeypatch):
    module = load_setup()
    captured = {}

    def capture(config, **kwargs):
        captured.update(config=config, **kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(module.fvm, "create_fvm_solver", capture)
    module.create_solver("grid_h004", 0.04)

    mesh = captured["mesh"]
    source = mesh.source
    assert source.cell_size_anchor is None
    assert source.max_cell_size == pytest.approx(0.64)
    assert source.patch_refinements[0].cell_size == pytest.approx(0.05333333333333334)
    assert [refinement.cell_size for refinement in source.refinements] == pytest.approx(
        [0.05333333333333334, 0.10666666666666667, 0.21333333333333335]
    )
    assert source.effective_cell_size(source.patch_refinements[0].cell_size) == pytest.approx(0.04)
    assert [
        source.effective_cell_size(refinement.cell_size) for refinement in source.refinements
    ] == pytest.approx([0.04, 0.08, 0.16])
    assert mesh.domain.bounds == module.DOMAIN
    assert len(mesh.levels) - 1 == 24
    assert np.diff(mesh.levels) == pytest.approx(0.96 / 24.0)
    assert np.diff(mesh.levels).max() <= 0.04 + 1.0e-12
    assert captured["solution_dir"] == CASE / "solution"
    assert captured["samples_dir"] == CASE / "samples"

    config = captured["config"]
    assert config.cores == 6
    assert config.time.end_time == 100.0
    assert config.time.adjustment is None
    assert config.time.time_step_size == 0.008
    assert config.pimple.algorithm == "PIMPLE"
    assert config.pimple.n_outer_correctors == 2
    samplers = {sampler.file_name: sampler for sampler in config.samplers}
    assert set(samplers) == {
        "forces_history",
        "centreline",
        "span_lower",
        "span_middle",
        "span_upper",
        "phase_near_upper",
        "phase_near_lower",
        "phase_wake_upper",
        "phase_wake_lower",
        "transverse_x2",
        "transverse_x4",
        "midspan",
    }
    assert samplers["forces_history"].schedule.every_n_steps * config.time.time_step_size == 0.04


def test_allrun_dispatches_single_grid_from_an_isolated_copy(tmp_path):
    # Never execute a tutorial launcher in the checkout during tests.
    case_copy = tmp_path / "case"
    case_copy.mkdir()
    shutil.copy2(CASE / "allrun.sh", case_copy / "allrun.sh")
    cleanup = case_copy / "allclean.sh"
    cleanup.write_text('#!/bin/bash\nprintf "clean\\n" >> "$SHELL_TEST_LOG"\n')
    cleanup.chmod(0o755)
    stub = tmp_path / "python"
    stub.write_text('#!/bin/bash\nprintf "%s\\n" "$*" >> "$SHELL_TEST_LOG"\n')
    stub.chmod(0o755)
    environment = dict(
        os.environ,
        PATH=str(tmp_path) + os.pathsep + os.environ["PATH"],
        SHELL_TEST_LOG=str(tmp_path / "calls"),
    )
    subprocess.run(["/bin/bash", str(case_copy / "allrun.sh")], env=environment, check=True)

    calls = [shlex.split(line) for line in (tmp_path / "calls").read_text().splitlines()]
    assert calls == [["setup.py", "-h", "0.04"]]


def test_setup_exposes_only_name_and_spacing():
    source = (CASE / "setup.py").read_text()
    assert source.count("parser.add_argument") == 2
    for removed_option in (
        "--campaign",
        "--output-root",
        "--restart-from",
        "--lean",
        "--span-layers",
        "--maximum-time-step",
    ):
        assert removed_option not in source
