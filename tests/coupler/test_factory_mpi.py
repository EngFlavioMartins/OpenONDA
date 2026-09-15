"""Plain Python launches the coupled runtime; no tutorial rank/environment logic."""

from importlib.util import find_spec
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


@pytest.mark.integration
def test_partitioned_coupled_restart_keeps_boundary_collectives_in_order(tmp_path):
    if find_spec("mpi4py") is None or find_spec("petsc4py") is None:
        pytest.skip("MPI and PETSc are required")
    if not (Path(sys.executable).with_name("mpiexec").is_file() or shutil.which("mpiexec")):
        pytest.skip("mpiexec is required")
    script = tmp_path / "restart.py"
    script.write_text("""
from pathlib import Path
import json
import sys
import numpy as np
from openonda import coupler, fvm, vpm
from source.solvers.fvm.mesh.rectilinear import coupling_box_mesh

directory = Path(sys.argv[1])
flow = fvm.FVMSetup(
    case_name="restart", cores=2,
    time=fvm.TimeConfig(time_step_size=.01, end_time=.02),
    transport=fvm.TransportConfig(kinematic_viscosity=.01),
    boundaries=[fvm.BoundaryConfig(name="numericalBoundary",
        velocity_type="fixedValue", velocity_value=[1., 0., 0.],
        pressure_type="fixedFluxPressure")],
    initial_velocity=[1., 0., 0.],
)
particles = vpm.VPMCase(
    directory=directory,
    numerics=vpm.Numerics(
        time_step_size=.01, compute_device="CPU", max_n_particles=1000,
        domain_bounds=(-1.,1.,-1.,1.,-1.,1.),
        freestream_velocity=[1.,0.,0.],
        viscous=vpm.ViscousConfig.cs(kinematic_viscosity=.01, particle_spacing=.25),
    ),
    backup=vpm.Backup(interval_steps=0),
)
policy = coupler.CouplerSetup(eta_blend_width=0., backup_interval_steps=0,
    boundary_condition_mode="vorticity_mixed")
mesh = lambda: coupling_box_mesh((-.5,.5,-.5,.5,-.5,.5),.25)
with coupler.create_coupler(flow, particles, policy, mesh=mesh,
        case_dir=directory / "first") as first:
    assert first.run(max_coupling_steps=1, backup_at_stop=True) == 1
with coupler.create_coupler(flow, particles, policy, mesh=mesh,
        case_dir=directory / "resumed") as resumed:
    assert resumed.run(restart_from=directory / "first/solution/backups") == 2
    assert resumed.fvm_solver.time == .02
    assert np.allclose(resumed.fvm_solver.get_velocity_field(), [1.,0.,0.], atol=1e-8)
    rank = resumed.fvm_solver.parallel.rank
    (directory / f"rank-{rank}.json").write_text(json.dumps({"step": 2, "time": .02}))
""")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    environment["TI_OFFLINE_CACHE_FILE_PATH"] = str(tmp_path / "taichi-cache")
    result = subprocess.run(
        [sys.executable, str(script), str(tmp_path / "outputs")],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    for rank in range(2):
        assert json.loads((tmp_path / f"outputs/rank-{rank}.json").read_text()) == {
            "step": 2,
            "time": 0.02,
        }


@pytest.mark.integration
@pytest.mark.parametrize(
    "mode", ["run", "construction_failure", "health_failure", "application_failure"]
)
def test_coupled_factory_launches_and_closes_two_ranks(tmp_path, mode):
    if find_spec("mpi4py") is None or find_spec("petsc4py") is None:
        pytest.skip("MPI and PETSc are required")
    if not (Path(sys.executable).with_name("mpiexec").is_file() or shutil.which("mpiexec")):
        pytest.skip("mpiexec is required")
    script = tmp_path / "case.py"
    script.write_text("""
from pathlib import Path
import json
import sys
import numpy as np
from openonda import coupler, fvm, vpm
from openonda.runtime import worker_thread_count
from source.solvers.fvm.mesh.rectilinear import coupling_box_mesh
import source.solvers.vpm as vpm_api

directory = Path(sys.argv[1])
mode = sys.argv[2]
fvm_setup = fvm.FVMSetup(
    case_name="runtime_smoke", cores=2,
    time=fvm.TimeConfig(time_step_size=.01, end_time=.01),
    transport=fvm.TransportConfig(kinematic_viscosity=.01),
    boundaries=[fvm.BoundaryConfig(name="numericalBoundary",
        velocity_type="fixedValue", velocity_value=[1., 0., 0.],
        pressure_type="fixedFluxPressure")],
    initial_velocity=[1., 0., 0.],
)
particle_case = vpm.VPMCase(
    directory=directory,
    numerics=vpm.Numerics(
        time_step_size=.01, compute_device="CPU", max_n_particles=1000,
        domain_bounds=(-1.,1.,-1.,1.,-1.,1.),
        freestream_velocity=[1.,0.,0.],
        viscous=vpm.ViscousConfig.cs(kinematic_viscosity=.01,particle_spacing=.25),
    ),
    backup=vpm.Backup(interval_steps=0),
)
policy = coupler.CouplerSetup(eta_blend_width=0., backup_interval_steps=0)
if mode == "construction_failure":
    def fail(case):
        raise ValueError("injected construction failure")
    vpm_api.VPMSolver = fail
driver = None
try:
    with coupler.create_coupler(fvm_setup, particle_case, policy,
        mesh=lambda: coupling_box_mesh((-.5,.5,-.5,.5,-.5,.5),.25)) as driver:
        native = driver._injected_fvm
        assert native.parallel.size == 2
        assert (driver._injected_vpm is not None) == native.parallel.is_root
        assert worker_thread_count() == 2
        def instrument(particle_solver):
            print("INSTRUMENTED_ONCE", flush=True)
            if mode == "application_failure":
                raise ValueError("injected application failure")
            return particle_solver is driver._injected_vpm
        assert driver.apply_vpm(instrument)
        if mode == "health_failure" and driver._injected_vpm is not None:
            def fail():
                raise vpm_api.HealthError("injected accepted-state health failure")
            driver._injected_vpm.execute_scheduled_samplers = fail
        assert driver.run() == 1
        assert np.allclose(native.get_velocity_field(), [1.,0.,0.], atol=1e-8)
        assert mode == "run"
    status = "completed"
except (RuntimeError, ValueError) as error:
    expected = {"construction_failure": "injected construction failure",
                "health_failure": "injected accepted-state health failure",
                "application_failure": "injected application failure"}.get(mode, "")
    if mode == "run" or expected not in str(error):
        raise
    status = "expected_failure"
from mpi4py import MPI
rank = MPI.COMM_WORLD.Get_rank()
if driver is not None:
    assert driver._closed
    assert driver._injected_fvm._closed
    assert not driver._injected_fvm.algorithm._partitioned_linear_workspaces
(directory / f"rank-{rank}.json").write_text(json.dumps({"rank":rank,"status":status}))
""")
    environment = os.environ.copy()
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "TI_CPU_MAX_NUM_THREADS",
        "FVM_PETSC_WORKSPACE_POLICY",
        "_OPENONDA_MPI_CHILD",
    ):
        environment.pop(key, None)
    # Locate the code under test in CI, including an uninstalled checkout.
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    result = subprocess.run(
        [sys.executable, str(script), str(tmp_path / "outputs"), mode],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout[-6000:] + result.stderr[-6000:]
    assert "[Taichi] version" not in result.stdout
    if mode != "construction_failure":
        assert result.stdout.count(" OPENONDA VPM\n") == 1
        assert result.stdout.count("INSTRUMENTED_ONCE") == 1
    expected = "completed" if mode == "run" else "expected_failure"
    for rank in range(2):
        assert json.loads((tmp_path / "outputs" / f"rank-{rank}.json").read_text()) == {
            "rank": rank,
            "status": expected,
        }
