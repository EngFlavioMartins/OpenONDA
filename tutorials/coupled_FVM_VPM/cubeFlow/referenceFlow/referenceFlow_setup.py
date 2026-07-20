"""Fully meshed FVM reference for flow past a cube at Re = 1000."""

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
from source.solvers.FVM.mesh.rectilinear import (  # noqa: E402
    box_mesh_3d,
    stretched,
    wall_refined_axis,
)


CASE_DIR = Path(__file__).resolve().parent

# Physical problem
CUBE_SIDE = 1.0
CUBE_BOX = (-0.5, 0.5, -0.5, 0.5, -0.5, 0.5)
U_INF = (1.0, 0.0, 0.0)
RHO = 1.0
REYNOLDS = 1000.0
NU = np.linalg.norm(U_INF) * CUBE_SIDE / REYNOLDS
INITIAL_U = (1.0, 0.02, 0.0)

# Common-region discretisation: identical to the hybrid FVM mesh
CORE_BOX = (-1.5, 1.5, -1.5, 1.5, -1.5, 1.5)
H_CORE = 0.05
WALL_REFINE = 0.025
WALL_RATIO = 1.25
DT_FVM = 0.0125
T_END = 7.5
WRITE_INTERVAL = 0.15

# Fully meshed far field
X_BOUNDS = (-4.75, 11.0)
YZ_BOUNDS = (-4.75, 4.75)
WAKE_END = 5.5
WAKE_H_MAX = 0.14
STRETCH_RATIO = 1.18
H_MAX = 0.5


def reference_mesh(
    h: float = H_CORE,
    wall_refinement: float = WALL_REFINE,
) -> tuple[dict, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Build the far-field mesh around the common body-fitted core."""
    core_axis = wall_refined_axis(
        CORE_BOX[0],
        CORE_BOX[1],
        CUBE_BOX[0],
        CUBE_BOX[1],
        wall_refinement,
        h,
        WALL_RATIO,
    )

    def extend(nodes, lower, upper, *, wake=False):
        left = stretched(CORE_BOX[0], lower, h, STRETCH_RATIO, H_MAX)[::-1]
        if wake:
            plateau = stretched(CORE_BOX[1], WAKE_END, h, 1.06, WAKE_H_MAX)
            outer_h = float(plateau[-1] - plateau[-2])
            right = np.concatenate(
                [
                    plateau,
                    stretched(float(plateau[-1]), upper, outer_h, STRETCH_RATIO, H_MAX),
                ]
            )
        else:
            right = stretched(CORE_BOX[1], upper, h, STRETCH_RATIO, H_MAX)
        return np.concatenate([left, nodes, right])

    xs = extend(core_axis, X_BOUNDS[0], X_BOUNDS[1], wake=True)
    ys = extend(core_axis, YZ_BOUNDS[0], YZ_BOUNDS[1])
    zs = ys.copy()
    mesh = box_mesh_3d(
        xs,
        ys,
        zs,
        hole_box=CUBE_BOX,
        wall_patch_name="cube",
    )
    return mesh, (xs, ys, zs)


def production_mesh() -> dict:
    mesh, (xs, ys, zs) = reference_mesh()
    print(
        f"referenceFlow mesh: {mesh['n_elements']} cells "
        f"({len(xs) - 1} x {len(ys) - 1} x {len(zs) - 1} minus cube), "
        f"x=[{xs[0]:.2f}, {xs[-1]:.2f}], y,z=[{ys[0]:.2f}, {ys[-1]:.2f}]"
    )
    return mesh


FVM_SETUP = FVMSetup(
    case_name="referenceFlow",
    cores=4,
    output=OutputSetup(
        format="vtk_xml",
        data_location="cell",
        encoding="appended",
        compression="lz4",
        precision="float64",
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
        convection_scheme="limitedLinear",
        gradient_scheme="lsq",
        time_scheme="euler_implicit",
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


def main() -> None:
    solver = setup_fvm_solver(FVM_SETUP, case_dir=CASE_DIR, mesh=production_mesh)
    solver.write_vtk()
    while solver.flow_time < FVM_SETUP.time.end_time:
        solver.evolve()


if __name__ == "__main__":
    main()
