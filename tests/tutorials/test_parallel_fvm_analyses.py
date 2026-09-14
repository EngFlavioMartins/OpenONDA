"""Custom tutorial CSVs agree between serial and MPI on tiny meshes."""

from importlib.util import find_spec
import os
import subprocess
import sys

import numpy as np
import pytest

from openonda.tutorials import materialize_tutorial


@pytest.mark.integration
@pytest.mark.parametrize("name", ["taylor_green", "step_profile"])
def test_custom_analyses_use_the_complete_field_in_serial_and_mpi(tmp_path, name):
    if find_spec("mpi4py") is None or find_spec("petsc4py") is None:
        pytest.skip("MPI and PETSc required")
    wrapper = tmp_path / "small_case.py"
    wrapper.write_text("""
import sys
from dataclasses import replace
from openonda.tutorial_runner import load_case_module
case = load_case_module(sys.argv[1])
case.NUMBER_OF_CELLS = 8
case.N_UPSTREAM, case.N_DOWNSTREAM, case.N_HEIGHT = 4, 8, 4
case.FINAL_TIME = .02
case.TIME_STEP_SIZE = case.MAX_TIME_STEP_SIZE = .01
make_setup = case.fvm.FVMSetup
def configured(*args, **kwargs):
    # Compare the analyses at tight linear convergence, independently of the
    # default serial/MPI iterative solvers' different stopping histories.
    kwargs['linear'] = replace(kwargs['linear'], momentum_tolerance=1e-10,
        pressure_tolerance=1e-10, momentum_relative_tolerance=0.0,
        pressure_relative_tolerance=0.0, momentum_final_relative_tolerance=0.0,
        pressure_final_relative_tolerance=0.0)
    return make_setup(*args, **kwargs, cores=int(sys.argv[2]))
case.fvm.FVMSetup = configured
case.main()
""")
    outputs = []
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    for cores in (1, 2):
        case = materialize_tutorial(f"fvm/{name}", tmp_path / f"{cores} cores")
        result = subprocess.run(
            [sys.executable, str(wrapper), str(case), str(cores)],
            cwd=tmp_path,
            env=env,
            text=True,
            capture_output=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-4000:]
        filenames = (
            ("history.csv",)
            if name == "taylor_green"
            else ("fields.csv", "reattachment_history.csv")
        )
        outputs.append(
            {
                file: np.genfromtxt(case / "solution" / file, delimiter=",", names=True)
                for file in filenames
            }
        )
    for filename in filenames:
        serial, parallel = outputs[0][filename], outputs[1][filename]
        assert serial.shape == parallel.shape
        for column in serial.dtype.names:
            np.testing.assert_allclose(
                parallel[column],
                serial[column],
                atol=1e-7,
                rtol=1e-6,
                err_msg=f"{filename}: {column}",
            )
