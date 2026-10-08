"""allrun.sh cleans then runs; allcontinue.sh resumes the latest native backup."""

import ast
from importlib.util import find_spec
import os
import shlex
import subprocess
import sys

import numpy as np
import pytest

from tests._tutorial_helpers import TUTORIALS, copy_tutorial, tutorial_cases

CASES = [case.relative_to(TUTORIALS) for case in tutorial_cases()]


def _variants():
    for case in CASES:
        for line in (TUTORIALS / case / "allcontinue.sh").read_text().splitlines():
            command = shlex.split(line)
            if command[:2] == ["python", "setup.py"]:
                yield case, [arg for arg in command[2:] if arg != "$@"]


@pytest.mark.parametrize("relative,arguments", list(_variants()))
def test_every_launcher_variant_reaches_the_latest_restart_selection(tmp_path, relative, arguments):

    case = copy_tutorial(TUTORIALS / relative, tmp_path / "case")
    probe = tmp_path / "probe.py"
    probe.write_text("""
import sys
import numpy as np
from openonda import fvm, vpm, coupler
from pathlib import Path
from types import SimpleNamespace
import runpy
import openonda.core.restart
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
openonda.core.restart.select_backup = lambda selection, **kwargs: selected(selection)
sys.argv = sys.argv[1:]
sys.path.insert(0, str(Path(sys.argv[0]).parent))
try:
    runpy.run_path(sys.argv[0], run_name="__main__")
except Done:
    pass
else:
    raise AssertionError("No restart mode selected")
""")
    result = subprocess.run(
        [sys.executable, str(probe), str(case / "setup.py"), *arguments],
        cwd=case,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-4000:]
    assert "LATEST_SELECTED" in result.stdout


@pytest.mark.parametrize("relative", CASES, ids=str)
def test_fresh_runs_clean_first_and_setups_select_latest(relative):
    directory = TUTORIALS / relative

    def commands(path):
        return [
            line
            for line in path.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]

    fresh = commands(directory / "allrun.sh")
    continuing = commands(directory / "allcontinue.sh")
    assert fresh[:3] == ["set -e", 'cd "$(dirname "$0")"', "./allclean.sh"]
    assert continuing[:2] == fresh[:2]
    assert fresh[3:] == continuing[2:] and continuing[2:]
    for command in continuing[2:]:
        words = shlex.split(command)
        assert words[:2] == ["python", "setup.py"] or words[:2] == ["python", "-m"]
    assert os.access(directory / "allcontinue.sh", os.X_OK)
    tree = ast.parse((directory / "setup.py").read_text())
    assert any(
        isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Constant)
        and node.value.value == "latest"
        and any(isinstance(t, ast.Name) and t.id == "START_FROM" for t in node.targets)
        or isinstance(node, ast.keyword)
        and node.arg == "start_from"
        and isinstance(node.value, ast.Constant)
        and node.value.value == "latest"
        for node in ast.walk(tree)
    )


@pytest.mark.parametrize("name", ["taylor_green", "step_profile"])
@pytest.mark.parametrize("cores", [1, 2])
def test_custom_history_continues_and_completed_invocation_is_idle(tmp_path, name, cores):
    if cores > 1 and (find_spec("mpi4py") is None or find_spec("petsc4py") is None):
        pytest.skip("MPI and PETSc required")
    case = copy_tutorial(f"fvm/{name}", tmp_path / name)
    wrapper = tmp_path / "small_case.py"
    wrapper.write_text("""
import sys
sys.path.insert(0, sys.argv[1])
import setup as case
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
