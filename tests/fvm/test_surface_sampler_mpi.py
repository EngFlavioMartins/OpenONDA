"""MPI SurfaceSampler: collective fields with root-only output.

Blocker #4: in a partitioned run the SurfaceSampler must gather owned data to
root and only root may write the ``.vts`` / ``.pvd`` files.  Non-root ranks must
never create ``samples/`` or touch the filesystem.

Run collectively, e.g. ``mpiexec -n 2 python -m pytest tests/fvm/test_surface_sampler_mpi.py``.
"""

from __future__ import annotations

import contextlib
import io
from pathlib import Path

import pytest

mpi4py = pytest.importorskip("mpi4py", reason="parallel FVM test requires mpi4py")
pytest.importorskip("petsc4py", reason="parallel FVM test requires petsc4py")

from source.solvers.FVM import (  # noqa: E402
    BoundaryConfig,
    ExecutionConfig,
    FVMSetup,
    LinearSolverConfig,
    PimpleControl,
    SchemesConfig,
    Solver,
    TimeConfig,
    TransportConfig,
)
from source.solvers.FVM.core.parallel import ParallelContext  # noqa: E402
from source.solvers.FVM.sampling.base import SamplingSchedule  # noqa: E402
from source.solvers.FVM.sampling.fields import SurfaceSampler  # noqa: E402

from ._structured_mesh import structured_box  # noqa: E402

pytestmark = pytest.mark.mpi


def _config():
    return FVMSetup(
        case_name="surface-mpi",
        execution=ExecutionConfig.petsc_partitioned(),
        time=TimeConfig.transient(dt=0.01, duration=0.01, write_interval=100),
        schemes=SchemesConfig(convection_scheme="upwind", gradient_scheme="gauss"),
        linear=LinearSolverConfig(
            momentum_solver="bicgstab",
            pressure_solver="cg",
            momentum_tol=1e-6,
            pressure_tol=1e-6,
        ),
        pimple=PimpleControl(n_correctors=2),
        transport=TransportConfig(density=1.0, nu=0.02),
        boundaries=[
            BoundaryConfig.inlet("xmin", [1.0, 0.0, 0.0]),
            BoundaryConfig.outlet("xmax", 0.0),
            BoundaryConfig.wall("ymin"),
            BoundaryConfig.wall("ymax"),
            BoundaryConfig.wall("zmin"),
            BoundaryConfig.wall("zmax"),
        ],
        samplers=[
            SurfaceSampler(
                point=[0.5, 0.5, 0.0],
                normal=[0, 0, 1],
                bounds=[0, 1, 0, 1],
                spacing=0.5,
                file_name="slice_z0",
                schedule=SamplingSchedule(every_n_steps=1),
            )
        ],
        initial_U=[1.0, 0.0, 0.0],
        initial_p=0.0,
    )


def test_surface_sampler_files_are_root_owned(tmp_path):
    context = ParallelContext.create(ExecutionConfig.petsc_replicated())
    mesh = structured_box(2, 2, 2)
    case_dir = Path(
        context.bcast(
            str(tmp_path / "surface-root") if context.is_root else None,
            root=0,
        )
    )

    with contextlib.redirect_stdout(io.StringIO()):
        solver = Solver(
            _config(),
            str(case_dir),
            mesh_data=mesh if context.is_root else None,
        )
        solver.auto_write = False
        solver.evolve(0.01)
        solver.close()

    context.barrier()
    if context.is_root:
        samples = case_dir / "samples"
        assert samples.exists()
        vts_files = list(samples.glob("slice_z0*.vts"))
        assert len(vts_files) == 1
        pvd = samples / "slice_z0.pvd"
        assert pvd.exists()
        assert "slice_z0" in pvd.read_text()
