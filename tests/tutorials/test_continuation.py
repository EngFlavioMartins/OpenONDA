"""Tutorial launchers retain their declared fresh/continue contracts."""

import ast
from importlib.util import find_spec
import os
from pathlib import Path
import shlex
import subprocess
import sys

import numpy as np
import pytest

from openonda.tutorials import TUTORIALS, materialize_tutorial

ROOT = Path(__file__).resolve().parents[2] / "tutorials"
CASES = [tutorial.relative_path for tutorial in TUTORIALS]
PRESERVING_RUNS = {
    "coupled_fvm_vpm/01_cylinder_shedding_flow",
    "coupled_fvm_vpm/01_cylinder_shedding_flow/reference_flow",
    "coupled_fvm_vpm/02_cube_flow",
    "coupled_fvm_vpm/02_cube_flow/reference_flow",
}


def _variants():
    for case in CASES:
        for line in (ROOT / case / "allcontinue.sh").read_text().splitlines():
            command = shlex.split(line)
            if command[:5] == ["python", "-m", "openonda.tutorial_runner", ".", "setup"]:
                yield case, [arg for arg in command[5:] if arg != "$@"]


@pytest.mark.parametrize("relative,arguments", list(_variants()))
def test_every_launcher_variant_reaches_the_latest_start_policy(tmp_path, relative, arguments):

    case = tmp_path / "case"
    # Materialize via the public catalog to exclude retained simulation data.
    public = next((item for item in TUTORIALS if item.relative_path == relative), None)
    assert public is not None
    case = materialize_tutorial(public.name, tmp_path)
    probe = tmp_path / "probe.py"
    probe.write_text("""
import sys
import numpy as np
from openonda import fvm, vpm, coupler
from pathlib import Path
from types import SimpleNamespace
import source.restart
from openonda.tutorial_runner import run_case
class Done(Exception):
    pass
def selected(value):
    assert value == "latest", value
    print("LATEST_SELECTED")
    raise Done
class Driver:
    geo_data = {"cell_centre": np.zeros((1, 3))}
    mesh_data = {"n_cells": 1}
    fvm_solver = None
    _comm, _is_master = None, True
    step = 0
    solution_dir = Path(sys.argv[1]).parent / "solution"
    parallel = SimpleNamespace(comm=None, is_root=True)
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def run(self, *args, start_from=None, **kwargs): selected(start_from)
    def start_from(self, value): selected(value)
    def initialize(self): pass
    def write_csv(self, *args, **kwargs): pass
    def set_initial_velocity(self, *args): pass
    def evaluate(self, *args): return 1.0, 1.0
def fvm_driver(setup, *args, **kwargs):
    driver = Driver()
    driver.setup = driver._resolved_setup = setup
    return driver
fvm.create_fvm_solver = fvm_driver
vpm.VPMSolver = lambda *args, **kwargs: Driver()
coupler.create_coupler = lambda *args, **kwargs: Driver()
source.restart.select_backup = lambda selection, **kwargs: selected(selection)
sys.argv = sys.argv[1:]
try:
    run_case(Path(sys.argv[0]).parent, "setup", sys.argv[1:])
except Done:
    pass
else:
    raise AssertionError("No start policy selected")
""")
    result = subprocess.run(
        [sys.executable, str(probe), str(case / "setup.py"), *arguments],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-4000:]
    assert "LATEST_SELECTED" in result.stdout


@pytest.mark.parametrize("relative", CASES, ids=str)
def test_launchers_clean_only_for_fresh_runs_and_setups_select_latest(relative):
    directory = ROOT / relative

    def commands(path):
        return [
            line
            for line in path.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]

    fresh = commands(directory / "allrun.sh")
    continuing = commands(directory / "allcontinue.sh")
    assert fresh[:2] == continuing[:2] == ["set -e", 'cd "$(dirname "$0")"']
    if relative.as_posix() in PRESERVING_RUNS:
        assert "./allclean.sh" not in fresh
        fresh_runs = [shlex.split(command) for command in fresh[2:]]
    else:
        assert fresh[2] == "./allclean.sh"
        fresh_runs = [shlex.split(command) for command in fresh[3:]]
    continuing_runs = [shlex.split(command) for command in continuing[2:]]
    assert len(fresh_runs) == len(continuing_runs) > 0
    for fresh_run, continuing_run in zip(fresh_runs, continuing_runs, strict=True):
        assert fresh_run == continuing_run
        assert continuing_run[:4] == ["python", "-m", "openonda.tutorial_runner", "."]
        assert continuing_run[4] in {"setup", "assets.rwm_ensemble", "assets.postprocess"}
    assert os.access(directory / "allcontinue.sh", os.X_OK)
    tree = ast.parse((directory / "setup.py").read_text())
    assert any(
        isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Constant)
        and node.value.value == "latest"
        and any(isinstance(t, ast.Name) and t.id == "START_FROM" for t in node.targets)
        for node in tree.body
    )


@pytest.mark.parametrize("relative", CASES, ids=str)
def test_cleanup_removes_materialized_restart_data_only(tmp_path, relative):
    """Exercise cleanup only in a throw-away tutorial copy, never in the repository."""
    public = next((item for item in TUTORIALS if item.relative_path == relative), None)
    assert public is not None
    case = materialize_tutorial(public.name, tmp_path)

    backup = case / "solution" / "backup" / "restart.bin"
    backup.parent.mkdir(parents=True)
    backup.write_bytes(b"temporary restart state")
    for output in (case / "samples", case / "figures"):
        output.mkdir()
        (output / "generated.txt").write_text("temporary output")

    result = subprocess.run(
        [str(case / "allclean.sh")], cwd=case, capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert not (case / "solution").exists()
    assert not (case / "samples").exists()
    assert not (case / "figures").exists()


@pytest.mark.parametrize("name", ["taylor_green", "step_profile"])
@pytest.mark.parametrize("cores", [1, 2])
def test_custom_history_continues_and_completed_invocation_is_idle(tmp_path, name, cores):
    if cores > 1 and (find_spec("mpi4py") is None or find_spec("petsc4py") is None):
        pytest.skip("MPI and PETSc required")
    case = materialize_tutorial(f"fvm/{name}", tmp_path)
    wrapper = tmp_path / "small_case.py"
    wrapper.write_text("""
import sys
from openonda.tutorial_runner import load_case_module
case = load_case_module(sys.argv[1])
case.NUMBER_OF_CELLS = 8
case.N_UPSTREAM, case.N_DOWNSTREAM, case.N_HEIGHT = 4, 8, 4
case.FINAL_TIME = float(sys.argv[3])
case.TIME_STEP_SIZE = case.MAX_TIME_STEP_SIZE = .01
factory = case.fvm.FVMSetup
def configured(*args, **kwargs):
    return factory(*args, **kwargs, cores=int(sys.argv[2]))
case.fvm.FVMSetup = configured
case.main()
""")
    history = (
        case
        / "solution"
        / ("history.csv" if name == "taylor_green" else "reattachment_history.csv")
    )
    for end in (0.02, 0.04, 0.04):
        before = history.read_bytes() if history.exists() else b""
        result = subprocess.run(
            [sys.executable, str(wrapper), str(case), str(cores), str(end)],
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-4000:]
        records = np.genfromtxt(history, delimiter=",", names=True)
        assert records["time"][-1] == pytest.approx(end)
        assert len(np.unique(records["time"])) == len(records)
        if before and b"0.04" in before:
            assert history.read_bytes() == before
    assert (case / "solution/backup").exists()
