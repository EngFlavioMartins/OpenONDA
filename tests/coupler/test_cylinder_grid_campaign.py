"""Grid-family contract for the cylinder reference flow."""

import importlib.util
import math
import os
from pathlib import Path
import shlex
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
    assert len(mesh.levels) - 1 == 7
    assert np.diff(mesh.levels) == pytest.approx(0.25 / 7.0)
    assert np.diff(mesh.levels).max() <= 0.04
    assert captured["solution_dir"] == CASE / "solution/grid_h004"
    assert captured["samples_dir"] == CASE / "samples/grid_h004"

    config = captured["config"]
    assert config.cores == 6
    assert config.time.end_time == 80.0
    assert config.time.adjustment.maximum == 0.7
    assert config.time.adjustment.maximum_time_step_size == 0.01
    assert config.pimple.algorithm == "PISO"
    assert config.pimple.n_outer_correctors == 1
    samplers = {sampler.file_name: sampler for sampler in config.samplers}
    assert set(samplers) == {"forces_history", "centreline"}
    assert samplers["forces_history"].schedule.every_time == 0.04


def test_allrun_lists_only_the_geometric_grid_family(tmp_path):
    stub = tmp_path / "python"
    stub.write_text('#!/bin/bash\nprintf "%s\\n" "$*" >> "$SHELL_TEST_LOG"\n')
    stub.chmod(0o755)
    environment = dict(
        os.environ,
        PATH=str(tmp_path) + os.pathsep + os.environ["PATH"],
        SHELL_TEST_LOG=str(tmp_path / "calls"),
    )
    subprocess.run(["/bin/bash", str(CASE / "allrun.sh")], env=environment, check=True)

    calls = [shlex.split(line) for line in (tmp_path / "calls").read_text().splitlines()]
    assert all(arguments[0] == "setup.py" for arguments in calls)
    assert [arguments[2] for arguments in calls] == [
        "grid_h008",
        "grid_h00565685",
        "grid_h004",
        "grid_h00282843",
    ]
    spacings = np.array([float(arguments[4]) for arguments in calls])
    assert spacings[:-1] / spacings[1:] == pytest.approx(np.sqrt(2.0))
    assert [max(4, math.ceil(0.25 / spacing)) for spacing in spacings] == [4, 5, 7, 9]
    assert all(arguments[1::2] == ["--name", "-h"] for arguments in calls)


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
