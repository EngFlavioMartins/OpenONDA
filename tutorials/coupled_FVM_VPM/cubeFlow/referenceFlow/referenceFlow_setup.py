"""Fully meshed LES FVM reference for flow past a cube at Re = 1000.

The mesh is generated directly as solver-native data by OpenONDA's
cfMesh-inspired adaptive Cartesian mesher.  It matches the corresponding OFW
case's requested 0.2 far field, 0.05 wake, and 0.0125 cube sizing.
The SGS model and coefficients match OpenFOAM's equilibrium Smagorinsky LES.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from openonda.fvm import (
    AdaptiveCartesianMesher,
    BoundaryConfig,
    BoxRefinement,
    ForcesConfig,
    FVMSetup,
    LinearSolverConfig,
    OutputSetup,
    PimpleControl,
    SchemesConfig,
    TimeConfig,
    TransportConfig,
    TurbulenceConfig,
    setup_fvm_solver,
)


CASE_DIR = Path(__file__).resolve().parent
CUBE_STL = CASE_DIR / "assets" / "cube.stl"

# Physical problem
CUBE_SIDE = 1.0
U_INF = (1.0, 0.0, 0.0)
RHO = 1.0
REYNOLDS = 1000.0
NU = np.linalg.norm(U_INF) * CUBE_SIDE / REYNOLDS
SMAGORINSKY_CK = 0.094
SMAGORINSKY_CE = 1.048
INITIAL_U = (1.0, 0.0, 0.0)
# Match the stable OpenFOAM reference numerics.  This mesh reaches a local
# Courant number above two during startup, so its PIMPLE equation relaxation
# is part of the discretisation rather than an optional run-time control.
DT_FVM = 0.01
T_END = 20.0
FVM_CORES = 4
WRITE_INTERVAL = 0.15
PERTURBATION = 1.0e-3
PERTURBATION_CENTRE = (1.0, 0.0, 0.0)
PERTURBATION_RADIUS = 0.75

FVM_DOMAIN = (-5.0, 15.0, -5.0, 5.0, -5.0, 5.0)
FVM_MESH = AdaptiveCartesianMesher(
    FVM_DOMAIN,
    0.2,
    surface_file=CUBE_STL,
    wall_patch_name="cube",
    surface_cell_size=0.0125,
    refinements=(BoxRefinement((-1.0, 5.0, -1.5, 1.5, -1.5, 1.5), 0.05, "wakeBox"),),
)


FVM_SETUP = FVMSetup(
    case_name="referenceFlow",
    cores=FVM_CORES,
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
        convection_scheme="LUST",
        gradient_scheme="lsq",
        time_scheme="backward",
    ),
    linear=LinearSolverConfig(
        linear_solver="bicgstab",
        pressure_solver="amg",
        pressure_tol=1e-8,
        momentum_tol=1e-6,
        momentum_maxiter=2000,
        ilu_drop_tol=1e-4,
        ilu_fill_factor=10.0,
        ilu_reuse_tol=0.05,
    ),
    pimple=PimpleControl(
        n_correctors=2,
        n_outer_correctors=2,
        alpha_u=0.7,
        alpha_p=0.3,
    ),
    forces=ForcesConfig(
        force_patches=["cube"],
        ref_velocity=np.linalg.norm(U_INF),
        ref_area=CUBE_SIDE**2,
        ref_length=CUBE_SIDE,
        moment_centre=[0.0, 0.0, 0.0],
        force_log_interval=1,
    ),
    transport=TransportConfig(density=RHO, nu=NU),
    turbulence=TurbulenceConfig.openfoam_smagorinsky(
        Ck=SMAGORINSKY_CK,
        Ce=SMAGORINSKY_CE,
    ),
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


def _wake_perturbation(centroids: np.ndarray) -> np.ndarray:
    """Return a smooth, divergence-free streamwise wake-vortex mode.

    The field is the curl of a streamwise Gaussian vector potential. Its peak
    speed is ``PERTURBATION * |U_inf|`` and it selects one of the otherwise
    exactly degenerate asymmetric modes of the Cartesian cube wake without
    imposing net lateral momentum.
    """
    x, y, z = centroids.T
    x0, y0, z0 = PERTURBATION_CENTRE
    radius = PERTURBATION_RADIUS
    xi = (x - x0) / radius
    eta = (y - y0) / radius
    zeta = (z - z0) / radius
    envelope = np.exp(-(xi * xi + eta * eta + zeta * zeta))

    # A_x = psi gives delta U = curl(psi, 0, 0).  The normalisation makes
    # max(|delta U|) = PERTURBATION * |U_inf| in the x=x0 plane.
    scale = PERTURBATION * np.linalg.norm(U_INF) * np.sqrt(2.0 * np.e)
    delta = np.zeros_like(centroids, dtype=np.float64)
    delta[:, 1] = -scale * zeta * envelope
    delta[:, 2] = scale * eta * envelope
    return delta


def _break_symmetry(solver) -> None:
    """Seed the wake while keeping BDF histories, halos, and flux consistent."""
    n_cells = solver.mesh_data["n_elements"]
    centroids = solver.geo_data["element_centroids"][:n_cells]
    initial_velocity = solver.U[:n_cells].copy()
    initial_velocity += _wake_perturbation(centroids)
    solver.set_initial_velocity(initial_velocity)


def main() -> None:
    solver = setup_fvm_solver(FVM_SETUP, case_dir=CASE_DIR, mesh=FVM_MESH)
    _break_symmetry(solver)
    solver.write_vtk()
    while solver.flow_time < FVM_SETUP.time.end_time:
        solver.evolve()


if __name__ == "__main__":
    main()
