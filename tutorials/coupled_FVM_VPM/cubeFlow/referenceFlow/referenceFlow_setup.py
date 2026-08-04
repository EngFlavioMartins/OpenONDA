"""Fully meshed LES FVM reference for flow past a cube at Re = 1000.

The mesh is generated as solver-native data by OpenONDA's cfMesh-inspired
adaptive Cartesian mesher, sized to reproduce the mesh the OFW case's
``cartesianMesh`` run actually delivered — 0.2 far field, 0.025 wake, 0.0125
cube — rather than the sizes its ``meshDict`` requests (see below).  The SGS
model and coefficients match OpenFOAM's equilibrium Smagorinsky LES, and the
PIMPLE controls match its ``fvSolution``.
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
DT_FVM = 0.01
T_END = 20.0
# Four ranks keep the per-rank partition practical without paying the Python
# and PETSc fixed-memory cost eight times. The solver now partitions its large
# arrays, so increasing this is a speed/per-rank-memory choice, not a
# requirement for fitting the reference mesh.
FVM_CORES = 4
WRITE_INTERVAL = 0.15
PERTURBATION = 1.0e-3
PERTURBATION_CENTRE = (1.0, 0.0, 0.0)
PERTURBATION_RADIUS = 0.75

FVM_DOMAIN = (-5.0, 15.0, -5.0, 5.0, -5.0, 5.0)
WAKE_BOX = (-1.0, 5.0, -1.5, 1.5, -1.5, 1.5)

# The wake is meshed at 0.025, not the 0.05 the OFW meshDict asks for: cfMesh
# rounds an objectRefinements cell size up to an octree level, and its root box
# (25.6000061 m) puts 25.6000061/0.05 = 512.0001 just past the exact 0.05
# subdivision, so it refines one level further.  3.38 M of the reference mesh's
# 3.93 M cells really sit in this box at h = 0.025 (measured from
# constant/polyMesh).  At 0.05 the separated shear layers, which decide whether
# the wake destabilises, are resolved twice as coarsely as the reference and
# carry ~2.3x its Smagorinsky eddy viscosity (nu_sgs ~ Delta^2; near-wake mean
# 1.01e-4 against OFW's 4.47e-5).
#
FVM_MESH = AdaptiveCartesianMesher(
    FVM_DOMAIN,
    0.2,
    surface_file=CUBE_STL,
    wall_patch_name="cube",
    surface_cell_size=0.0125,
    refinements=(BoxRefinement(WAKE_BOX, 0.025, "wakeBox"),),
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
        # Owned-cell output avoids retaining a second, visualization-only
        # halo mesh on every MPI rank. ParaView still reads the .pvtu pieces
        # as one global data set; use ghost_layers=1 only when smooth parallel
        # cell-to-point interpolation at rank interfaces is required.
        ghost_layers=0,
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
        # Match referenceFlow/system/fvSchemes exactly.
        convection_scheme="linearUpwind",
        gradient_scheme="gauss",
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
        n_orthogonal_correctors=1,
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
    """Return a smooth, divergence-free lateral wake-displacement mode.

    ``curl(0, 0, psi)`` with ``psi = -A * xi * exp(-r^2/R^2)``: divergence-free
    by construction, peak speed ``PERTURBATION * |U_inf|``.

    The mode matters as much as the amplitude.  A cube wake first loses
    stability to a plane-symmetric mode — the wake tilts to one side and the
    body picks up lift — whose lateral velocity is *even* in ``y``, which is
    what this field seeds (on the wake axis it is a clean push
    ``delta U = (0, A, 0)``).  A streamwise swirl centred on the axis,
    ``curl(psi, 0, 0)``, changes sign under a reflection about ``y = 0``, so it
    has no projection onto that mode and cannot select a side however long the
    case runs.
    """
    x, y, z = centroids.T
    x0, y0, z0 = PERTURBATION_CENTRE
    radius = PERTURBATION_RADIUS
    xi = (x - x0) / radius
    eta = (y - y0) / radius
    zeta = (z - z0) / radius
    envelope = np.exp(-(xi * xi + eta * eta + zeta * zeta))

    amplitude = PERTURBATION * np.linalg.norm(U_INF)
    delta = np.zeros_like(centroids, dtype=np.float64)
    delta[:, 0] = 2.0 * amplitude * xi * eta * envelope
    delta[:, 1] = amplitude * (1.0 - 2.0 * xi * xi) * envelope
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
