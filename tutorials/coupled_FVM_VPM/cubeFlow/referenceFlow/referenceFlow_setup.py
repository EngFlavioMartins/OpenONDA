"""Fully meshed FVM reference for flow past a cube at Re = 1000.

The mesh is built beforehand by ``assets/create_mesh.py`` and read from
``constant/polyMesh/``; this file holds only the case physics and run loop.
"""

from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "openonda_bootstrap.py").is_file()
)
sys.path.insert(0, str(REPO_ROOT))
from openonda_bootstrap import activate  # noqa: E402

activate(__file__)

import numpy as np  # noqa: E402

from source.solvers.FVM import (  # noqa: E402
    BoundaryConfig,
    ForcesConfig,
    FVMSetup,
    LinearSolverConfig,
    OutputSetup,
    PimpleControl,
    SchemesConfig,
    TimeConfig,
    TransportConfig,
    setup_fvm_solver,
)


CASE_DIR = Path(__file__).resolve().parent
MESH = str(CASE_DIR / "assets" / "mesh.msh")  # built by assets/create_mesh.py

# Physical problem
CUBE_SIDE = 1.0
U_INF = (1.0, 0.0, 0.0)
RHO = 1.0
REYNOLDS = 1000.0
NU = np.linalg.norm(U_INF) * CUBE_SIDE / REYNOLDS
INITIAL_U = (1.0, 0.0, 0.0)
DT_FVM = 0.0125
T_END = 20.0
WRITE_INTERVAL = 0.15
PERTURBATION = 1.0e-3


FVM_SETUP = FVMSetup(
    case_name="referenceFlow",
    cores=4,
    output=OutputSetup(
        format="vtk_xml",
        data_location="cell",
        encoding="appended",
        compression="lz4",
        precision="float32",
        asynchronous=True,
        ghost_layers=1,
    ),
    time=TimeConfig(
        delta_t=DT_FVM,
        start_time=0.0,
        end_time=T_END,
        write_interval=10**9,
        write_interval_time=WRITE_INTERVAL,
        adjust_timestep=False,
    ),
    schemes=SchemesConfig(
        convection_scheme="lust",
        gradient_scheme="lsq",
        time_scheme="backward",
    ),
    linear=LinearSolverConfig(
        linear_solver="bicgstab",
        pressure_solver="amg",
        pressure_tol=1e-8,
        momentum_maxiter=2000,
        ilu_drop_tol=1e-4,
        ilu_fill_factor=10.0,
        ilu_reuse_tol=0.05,
    ),
    pimple=PimpleControl(n_correctors=2, n_outer_correctors=2),
    forces=ForcesConfig(
        force_patches=["cube"],
        ref_velocity=np.linalg.norm(U_INF),
        ref_area=CUBE_SIDE**2,
        ref_length=CUBE_SIDE,
        moment_centre=[0.0, 0.0, 0.0],
        force_log_interval=1,
    ),
    transport=TransportConfig(density=RHO, nu=NU),
    turbulence=None,
    boundaries=[
        BoundaryConfig.inlet("inlet", list(U_INF)),
        BoundaryConfig.outlet("outlet", p=0.0),
        BoundaryConfig.slip("ymin"),
        BoundaryConfig.slip("ymax"),
        BoundaryConfig.slip("zmin"),
        BoundaryConfig.slip("zmax"),
        BoundaryConfig.wall("cube"),
    ],
    initial_U=list(INITIAL_U),
    initial_p=0.0,
)


def _break_symmetry(solver) -> None:
    """Seed a small transverse kick in the near wake."""
    centroids = solver.geo_data["element_centroids"]
    n_cells = solver.mesh_data["n_elements"]
    x, y, z = centroids[:n_cells, 0], centroids[:n_cells, 1], centroids[:n_cells, 2]

    near_wake = (x > 0.5) & (x < 2.5) & (np.abs(y) < 1.0) & (np.abs(z) < 1.0)
    kick = PERTURBATION * np.linalg.norm(U_INF) * np.sign(z + 1e-12) * np.exp(-((x - 1.0) ** 2))
    solver.U[:n_cells, 1] += np.where(near_wake, kick, 0.0)


def main() -> None:
    solver = setup_fvm_solver(FVM_SETUP, case_dir=CASE_DIR, mesh=MESH)
    _break_symmetry(solver)
    solver.write_vtk()
    while solver.flow_time < FVM_SETUP.time.end_time:
        solver.evolve()


if __name__ == "__main__":
    main()
