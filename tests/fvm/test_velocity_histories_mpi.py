"""Two-rank physical field admission and native history continuation."""

from importlib.util import find_spec
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


@pytest.mark.integration
@pytest.mark.parametrize("mode", ["valid", "callback_failure", "shape_failure", "continuation"])
def test_module_case_admits_initial_fields_collectively_and_continues(tmp_path, mode):
    if find_spec("mpi4py") is None or find_spec("petsc4py") is None:
        pytest.skip("MPI and PETSc required")
    if not (Path(sys.executable).with_name("mpiexec").is_file() or shutil.which("mpiexec")):
        pytest.skip("mpiexec required")
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "field.py").write_text("""
import numpy as np

def initial_velocity(points):
    return np.column_stack((.5 + .01 * points[:, 1], np.full(len(points), .1), np.zeros(len(points))))
""")
    (tmp_path / "setup.py").write_text("""
import json
from pathlib import Path
import sys
import numpy as np
from openonda import fvm
from openonda.fvm.mesher import coupling_box_mesh
from .assets.field import initial_velocity

directory = Path(__file__).parent
mode = sys.argv[1]
ramp = fvm.VelocityRamp((.5, .1, 0), (.5, 0, 0), .01, .03)
calls = 0

def setup(end):
    return fvm.FVMSetup(
        case_name="velocity_history", cores=2,
        time=fvm.TimeConfig(time_step_size=.01, end_time=end,
            output_schedule=fvm.RunSchedule(every_n_steps=100)),
        schemes=fvm.DiscretizationConfig(convection_scheme="upwind", time_scheme="backward"),
        transport=fvm.TransportConfig(density=1., kinematic_viscosity=.02),
        boundaries=[fvm.BoundaryConfig(name="numericalBoundary", velocity_type="fixedValue",
            velocity_value=list(ramp.initial), pressure_type="fixedFluxPressure")],
        velocity_boundaries=(fvm.VelocityBoundary(("numericalBoundary",), ramp),),
        initial_velocity=list(ramp.initial),
        backup=fvm.BackupConfig(write_at_end=True),
    )

def solver(path, end):
    flow = fvm.create_fvm_solver(setup(end), case_dir=path,
        mesh=lambda: coupling_box_mesh((-.5,.5,-.5,.5,-.5,.5), .25))
    flow.auto_write = False
    return flow

def seed(points):
    global calls
    calls += 1
    from mpi4py import MPI
    if MPI.COMM_WORLD.Get_rank() == 1:
        if mode == "callback_failure":
            raise ValueError("rank-local physical field failure")
        if mode == "shape_failure":
            return np.zeros((len(points), 2))
    return initial_velocity(points)

def state(flow):
    return {name: getattr(flow, name).copy() for name in (
        "velocity", "velocity_old", "velocity_older", "kinematic_pressure",
        "volumetric_face_flux", "volumetric_face_flux_old", "volumetric_face_flux_older")}

try:
    with solver(directory / "reference", .04) as flow:
        assert flow.parallel.size == 2
        flow.run(start_from="initial", initial_velocity=seed)
        assert flow.step == 4 and abs(flow.time - .04) < 1e-14
        reference = state(flow)
        for boundary in flow.boundaries:
            np.testing.assert_allclose(boundary["velocity_value_field"],
                np.tile(ramp.final, (boundary["n_faces"], 1)), atol=1e-14)
    if mode == "continuation":
        with solver(directory / "interrupted", .02) as flow:
            flow.run(start_from="initial", initial_velocity=seed)
        with solver(directory / "interrupted", .04) as flow:
            flow.run(start_from="latest", initial_velocity=seed)
            assert flow.step == 4 and abs(flow.time - .04) < 1e-14
            for name, expected in reference.items():
                np.testing.assert_allclose(getattr(flow, name), expected, atol=1e-12, rtol=1e-12)
        assert calls == 2
    else:
        assert calls == 1
    status = "completed"
except (RuntimeError, ValueError) as error:
    if mode not in ("callback_failure", "shape_failure"):
        raise
    if isinstance(error, RuntimeError):
        assert "initial velocity field admission" in str(error)
    expected = "rank-local physical field failure" if mode == "callback_failure" else "finite with shape"
    assert expected in str(error)
    status = "expected_failure"

from mpi4py import MPI
rank = MPI.COMM_WORLD.Get_rank()
(directory / f"rank-{rank}.json").write_text(json.dumps({"status":status,"calls":calls}))
""")
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    for name in ("PYTHONPATH", "_OPENONDA_MPI_CHILD"):
        environment.pop(name, None)
    completed = subprocess.run(
        [sys.executable, "-m", "openonda.tutorial_runner", str(tmp_path), "setup", mode],
        cwd=tmp_path.parent,
        env=environment,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert completed.returncode == 0, completed.stdout[-5000:] + completed.stderr[-5000:]
    expected = "expected_failure" if mode.endswith("failure") else "completed"
    for rank in range(2):
        report = json.loads((tmp_path / f"rank-{rank}.json").read_text())
        assert report == {"status": expected, "calls": 2 if mode == "continuation" else 1}
    assert not list(tmp_path.rglob("__pycache__"))
    assert not list(tmp_path.rglob("*startup*"))
