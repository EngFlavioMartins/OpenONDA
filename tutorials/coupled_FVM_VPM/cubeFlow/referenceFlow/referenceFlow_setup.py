"""Fully meshed LES FVM reference for flow past a cube at Re = 1000.

The mesh is generated as solver-native data by OpenONDA's cfMesh-inspired
adaptive Cartesian mesher, sized to reproduce the original reference case's
``cartesianMesh`` run actually delivered — 0.2 far field, 0.025 wake, 0.0125
cube — rather than the sizes its ``meshDict`` requests (see below).  The SGS
model and coefficients use the equilibrium Smagorinsky LES formulation, and the
PIMPLE controls match the coupled case.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from openonda.fvm import (
    AdaptiveCartesianMesher,
    BoundaryConfig,
    BoxRefinement,
    ExecutionConfig,
    ForceSampler,
    FVMSetup,
    LinearSolverConfig,
    LineSampler,
    OutputSetup,
    PimpleControl,
    SamplingSchedule,
    SchemesConfig,
    SurfaceSampler,
    TimeConfig,
    TransportConfig,
    TurbulenceConfig,
    create_fvm_solver,
)


CASE_DIR = Path(__file__).resolve().parent
CUBE_STL = CASE_DIR / "assets" / "cube.stl"

# Physical problem
CUBE_SIDE = 1.0
FREESTREAM_VELOCITY = (1.0, 0.0, 0.0)
RHO = 1.0
REYNOLDS = 1000.0
NU = float(np.linalg.norm(FREESTREAM_VELOCITY)) * CUBE_SIDE / REYNOLDS
SMAGORINSKY_CK = 0.094
SMAGORINSKY_CE = 1.048
INITIAL_VELOCITY = (1.0, 0.0, 0.0)
FVM_TIME_STEP_SIZE = 0.01
T_END = 20.0
FVM_CORES = 4
FVM_DOMAIN = (-5.0, 10.0, -5.0, 5.0, -5.0, 5.0)
WAKE_BOX = (-1.25, 4.25, -1.25, 1.25, -1.25, 1.25)
DOWNSTREAM_WAKE_BOX = (-1.5, 10.0, -1.5, 1.5, -1.5, 1.5)
SURFACE_CELL_SIZE = 0.015625
SAMPLE_SPACING = 0.04
OFFAXIS_Y = 0.75 * CUBE_SIDE
WAKE_SLICE_BOUNDS = (0.0, 5.0, -1.5, 1.5)

SAMPLE_INTERVAL = 0.05  # forces, line probes
FACE_INTERVAL = FVM_TIME_STEP_SIZE
SLICE_INTERVAL = 0.10  # full-domain field slices
VOLUME_INTERVAL = 1.00  # complete .pvtu volume archive

SAMPLE_SCHEDULE = SamplingSchedule(every_time=SAMPLE_INTERVAL)
SLICE_SCHEDULE = SamplingSchedule(every_time=SLICE_INTERVAL)
FACE_SCHEDULE = SamplingSchedule(every_time=FACE_INTERVAL)

SAMPLERS = (
    ForceSampler(
        patch_names=["cube"],
        ref_velocity=float(np.linalg.norm(FREESTREAM_VELOCITY)),
        ref_area=CUBE_SIDE**2,
        ref_length=CUBE_SIDE,
        moment_centre=[0.0, 0.0, 0.0],
        schedule=SAMPLE_SCHEDULE,
    ),
    LineSampler(
        start=[FVM_DOMAIN[0], 0.0, 0.0],
        end=[FVM_DOMAIN[1], 0.0, 0.0],
        spacing=SAMPLE_SPACING,
        file_name="centerline",
        schedule=SAMPLE_SCHEDULE,
    ),
    LineSampler(
        start=[FVM_DOMAIN[0], OFFAXIS_Y, 0.0],
        end=[FVM_DOMAIN[1], OFFAXIS_Y, 0.0],
        spacing=SAMPLE_SPACING,
        file_name="offaxis_y075",
        schedule=SAMPLE_SCHEDULE,
    ),
    SurfaceSampler(
        point=[0.0, 0.0, 0.0],
        normal=[0, 0, 1],
        bounds=[FVM_DOMAIN[0], FVM_DOMAIN[1], FVM_DOMAIN[2], FVM_DOMAIN[3]],
        spacing=SAMPLE_SPACING,
        schedule=SLICE_SCHEDULE,
        file_name="slice_z0",
    ),
    SurfaceSampler(
        point=[0.0, 0.0, 0.0],
        normal=[0, 0, 1],
        bounds=WAKE_SLICE_BOUNDS,
        spacing=SAMPLE_SPACING,
        schedule=SLICE_SCHEDULE,
        file_name="wake_slice_z0",
    ),
)

FVM_MESH = AdaptiveCartesianMesher(
    domain=FVM_DOMAIN,
    max_cell_size=SURFACE_CELL_SIZE * 32,
    surface_file=CUBE_STL,
    wall_patch_name="cube",
    surface_cell_size=SURFACE_CELL_SIZE,
    refinements=(
        BoxRefinement(WAKE_BOX, SURFACE_CELL_SIZE * 2, "wakeBox"),
        BoxRefinement(DOWNSTREAM_WAKE_BOX, SURFACE_CELL_SIZE * 4, "downstreamWakeBox"),
    ),
)

FVM_SETUP = FVMSetup(
    case_name="referenceFlow",
    cores=FVM_CORES,
    execution=ExecutionConfig(operator_backend="numba"),
    output=OutputSetup(
        format="vtk_xml",
        data_location="cell",
        encoding="appended",
        compression="lz4",
        precision="float32",
        asynchronous=True,
        ghost_layers=0,
    ),
    time=TimeConfig(
        time_step_size=FVM_TIME_STEP_SIZE,
        start_time=0.0,
        end_time=T_END,
        write_interval=10**9,
        write_interval_time=VOLUME_INTERVAL,
        adjust_timestep=False,
    ),
    schemes=SchemesConfig(
        # Match the coupled reference-flow discretisation exactly.
        convection_scheme="linearUpwind",
        gradient_scheme="gauss",
        time_scheme="backward",
    ),
    linear=LinearSolverConfig(
        linear_solver="bicgstab",
        pressure_solver="amg",
        pressure_tol=1e-6,
        pressure_rel_tol=0.01,
        pressure_final_rel_tol=0.0,
        momentum_tol=1e-6,
        momentum_rel_tol=0.1,
        momentum_final_rel_tol=0.0,
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
    samplers=SAMPLERS,
    transport=TransportConfig(density=RHO, nu=NU),
    turbulence=TurbulenceConfig.equilibrium_smagorinsky(
        Ck=SMAGORINSKY_CK,
        Ce=SMAGORINSKY_CE,
    ),
    boundaries=[
        BoundaryConfig.inlet("inlet", list(FREESTREAM_VELOCITY)),
        BoundaryConfig.outlet("outlet", p=0.0),
        BoundaryConfig.slip("ymin"),
        BoundaryConfig.slip("ymax"),
        BoundaryConfig.slip("zmin"),
        BoundaryConfig.slip("zmax"),
        BoundaryConfig.wall("cube"),
    ],
    initial_velocity=list(INITIAL_VELOCITY),
    initial_p=0.0,
)


def main() -> None:
    solver = create_fvm_solver(FVM_SETUP, case_dir=CASE_DIR, mesh=FVM_MESH)
    solver.write_vtk()

    while solver.time < FVM_SETUP.time.end_time:
        solver.advance()

    solver.close()


if __name__ == "__main__":
    main()
