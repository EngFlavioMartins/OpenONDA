"""Declarative cube reference case and launcher contract."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np

import openonda.fvm.mesher as msh

ROOT = Path(__file__).resolve().parents[2]
CASE_DIR = ROOT / "tutorials/coupled_fvm_vpm/cube_flow/reference_flow"


def load_setup():
    spec = spec_from_file_location("cube_reference_setup", CASE_DIR / "setup.py")
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reference_source_is_the_four_file_interface():
    assert {path.name for path in CASE_DIR.iterdir()} - {"__pycache__"} == {
        "README.md",
        "allclean.sh",
        "allrun.sh",
        "setup.py",
    }


def test_creator_uses_current_public_mesher(monkeypatch):
    setup = load_setup()
    captured = {}

    def create(config, **kwargs):
        captured.update(config=config, **kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(setup.fvm, "create_fvm_solver", create)
    setup.create_solver("fine", 0.03125)
    mesh = captured["mesh"]
    assert isinstance(mesh, msh.CartesianMesher)
    assert mesh.surfaces[0].path == (CASE_DIR.parent / "assets/cube.stl").resolve()
    assert mesh.domain.bounds == (-5.0, 10.0, -5.0, 5.0, -5.0, 5.0)
    assert mesh.boundary_cell_size == 0.5
    assert mesh.patch_refinements == (msh.PatchRefinement("cube", 0.03125),)
    np.testing.assert_allclose(
        [mesh.effective_cell_size(item.cell_size) for item in mesh.refinements],
        [0.0625, 0.125],
    )
    assert captured["solution_dir"] == CASE_DIR / "solution/fine"
    assert captured["samples_dir"] == CASE_DIR / "samples/fine"


def test_launcher_runs_the_complete_study(tmp_path):
    python = tmp_path / "python"
    python.write_text("#!/bin/sh\nprintf '%s\\n' \"$@\"\n")
    python.chmod(0o755)
    result = subprocess.run(
        [str(CASE_DIR / "allrun.sh")],
        env={"PYTHON": str(python)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert result.stdout.splitlines() == [
        "-u",
        "setup.py",
        "--name",
        "coarse",
        "--dx",
        "0.125",
        "-u",
        "setup.py",
        "--name",
        "medium",
        "--dx",
        "0.0625",
        "-u",
        "setup.py",
        "--name",
        "fine",
        "--dx",
        "0.03125",
    ]
