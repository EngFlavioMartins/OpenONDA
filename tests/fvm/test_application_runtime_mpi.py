"""Tiny real MPI applications own mesh generation, analyses and output centrally."""

from importlib.util import find_spec
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

import openonda


@pytest.mark.integration
@pytest.mark.parametrize(
    "mode",
    [
        "partitioned",
        "replicated",
        "mesh_failure",
        "analysis_failure",
        "four_ranks",
        "freestream",
    ],
)
def test_plain_python_application_owns_output_and_propagates_failures(tmp_path, mode):
    if find_spec("mpi4py") is None or find_spec("petsc4py") is None:
        pytest.skip("MPI and PETSc required")
    if not (Path(sys.executable).with_name("mpiexec").is_file() or shutil.which("mpiexec")):
        pytest.skip("mpiexec required")
    script = tmp_path / "setup.py"
    script.write_text("""
import json
from pathlib import Path
import sys
import numpy as np
from openonda import fvm
from openonda.fvm.mesher import coupling_box_mesh

directory = Path(__file__).parent
mode = sys.argv[1]
cores = 4 if mode == "four_ranks" else 2
def build_mesh():
    print("MESH_BUILT_ONCE", flush=True)
    if mode == "mesh_failure":
        raise ValueError("injected meshing failure")
    return coupling_box_mesh((-.5,.5,-.5,.5,-.5,.5), .25)

velocity = (1.,0.,0.)
boundaries = (fvm.BoundaryConfig(name="numericalBoundary",
    velocity_type="fixed_value", velocity_value=velocity,
    pressure_type="fixed_flux_pressure"),)
if mode == "freestream":
    boundaries = (fvm.BoundaryConfig.freestream("numericalBoundary", velocity),)
execution = fvm.ComputeConfig(linear_backend="petsc", parallel_mode="petsc_replicated") if mode == "replicated" else fvm.ComputeConfig()
def analyse(fields):
    print("ANALYSIS_WRITTEN_ONCE", flush=True)
    assert fields.velocity.shape == (64, 3)
    assert len(fields.boundaries["numericalBoundary"].face_centre) == 96
    assert abs(np.sum(fields.cell_volume) - 1.) < 1e-12
    np.testing.assert_allclose(fields.velocity, np.tile(velocity, (64,1)), atol=1e-8)
    if mode == "analysis_failure":
        raise ValueError("injected analysis failure")
    return [fields.time, len(fields.velocity), float(np.sum(fields.cell_volume))]

try:
    solver = fvm.FVMSolver(fvm.FVMCase(
        name="runtime", directory=directory,
        numerics=fvm.Numerics(time_step_size=.01, execution=execution),
        initial_conditions=fvm.InitialFields(velocity=velocity),
        run=fvm.RunPlan(end_time=.01),
        output=fvm.Output(schedule=fvm.EverySteps(1)),
        backup=fvm.Backup(schedule=fvm.EverySteps(1)),
        mesh=build_mesh, boundaries=boundaries, cores=cores,
    ))
    assert solver.parallel.size == cores
    with solver:
        row = solver.evaluate(analyse)
        solver.write_csv("analysis.csv", [row], columns=("time","cells","volume"))
        solver.run()
        row = solver.evaluate(analyse)
        solver.write_csv("analysis.csv", [row], columns=("time","cells","volume"), append=True)
    status = "completed"
except (RuntimeError, ValueError) as error:
    expected = "injected meshing failure" if mode == "mesh_failure" else "injected analysis failure"
    if mode not in ("mesh_failure", "analysis_failure") or expected not in str(error):
        raise
    status = "expected_failure"

from mpi4py import MPI
rank = MPI.COMM_WORLD.Get_rank()
(directory / f"rank-{rank}.json").write_text(json.dumps({"rank":rank,"status":status}))
""")
    env = os.environ.copy()
    for key in (
        "PYTHONPATH",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "TI_CPU_MAX_NUM_THREADS",
        "_OPENONDA_MPI_CHILD",
    ):
        env.pop(key, None)
    # Run the checkout under test, not whichever checkout is installed.
    env["PYTHONPATH"] = str(Path(openonda.__file__).resolve().parents[1])
    if mode == "four_ranks":
        # A resource reservation is not an MPI world, either before launch
        # or in the children using four of its available tasks.
        env["SLURM_NTASKS"] = "64"
    # Exercise the same plain invocation used by tutorials.
    completed = subprocess.run(
        [sys.executable, "setup.py", mode],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=90,
    )
    assert completed.returncode == 0, completed.stdout[-5000:] + completed.stderr[-5000:]
    assert completed.stdout.count("MESH_BUILT_ONCE") == 1
    expected = "expected_failure" if mode.endswith("failure") else "completed"
    for rank in range(4 if mode == "four_ranks" else 2):
        assert json.loads((tmp_path / f"rank-{rank}.json").read_text())["status"] == expected
    if expected == "completed":
        assert completed.stdout.count("ANALYSIS_WRITTEN_ONCE") == 2
        assert completed.stdout.lower().count("restart backup written:") == 1
        assert len((tmp_path / "solution/analysis.csv").read_text().splitlines()) == 3
        assert len(list((tmp_path / "solution").glob("*.pvd"))) == 1
        assert len(list((tmp_path / "solution").glob("fvm.log"))) == 1
