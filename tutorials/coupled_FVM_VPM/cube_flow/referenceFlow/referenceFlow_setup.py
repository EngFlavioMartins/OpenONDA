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

import openonda.fvm as fvm


CASE_DIR = Path(__file__).resolve().parent
CUBE_STL = CASE_DIR / "assets" / "cube.stl"

# Physical problem
CUBE_SIDE = 1.0
FREESTREAM_VELOCITY = (1.0, 0.0, 0.0)
DENSITY = 1.0
REYNOLDS = 1000.0
KINEMATIC_VISCOSITY = float(np.linalg.norm(FREESTREAM_VELOCITY)) * CUBE_SIDE / REYNOLDS
SMAGORINSKY_CK = 0.094
SMAGORINSKY_CE = 1.048
INITIAL_VELOCITY = (1.0, 0.0, 0.0)
FVM_TIME_STEP_SIZE = 0.01
END_TIME = 20.0
FVM_CORES = 4
FVM_DOMAIN = (-5.0, 10.0, -5.0, 5.0, -5.0, 5.0)
WAKE_BOX = (-1.25, 4.25, -1.25, 1.25, -1.25, 1.25)
DOWNSTREAM_WAKE_BOX = (-1.5, 10.0, -1.5, 1.5, -1.5, 1.5)
SURFACE_CELL_SIZE = 0.015625
SAMPLE_SPACING = 0.04
OFFAXIS_Y = 0.75 * CUBE_SIDE
WAKE_SLICE_BOUNDS = (0.0, 5.0, -1.5, 1.5)

SAMPLE_INTERVAL_TIME = 0.05  # forces, line probes
FACE_INTERVAL_TIME = FVM_TIME_STEP_SIZE
SLICE_INTERVAL_TIME = 0.10  # full-domain field slices
VOLUME_INTERVAL_TIME = 1.00  # complete .pvtu volume archive

SAMPLE_SCHEDULE = fvm.SamplingSchedule(every_time=SAMPLE_INTERVAL_TIME)
SLICE_SCHEDULE = fvm.SamplingSchedule(every_time=SLICE_INTERVAL_TIME)
FACE_SCHEDULE = fvm.SamplingSchedule(every_time=FACE_INTERVAL_TIME)

SAMPLERS = (
    fvm.ForceSampler(
        patch_names=["cube"],
        ref_velocity=float(np.linalg.norm(FREESTREAM_VELOCITY)),
        ref_area=CUBE_SIDE**2,
        ref_length=CUBE_SIDE,
        moment_centre=[0.0, 0.0, 0.0],
        schedule=SAMPLE_SCHEDULE,
    ),
    fvm.LineSampler(
        start=[FVM_DOMAIN[0], 0.0, 0.0],
        end=[FVM_DOMAIN[1], 0.0, 0.0],
        spacing=SAMPLE_SPACING,
        file_name="centerline",
        schedule=SAMPLE_SCHEDULE,
    ),
    fvm.LineSampler(
        start=[FVM_DOMAIN[0], OFFAXIS_Y, 0.0],
        end=[FVM_DOMAIN[1], OFFAXIS_Y, 0.0],
        spacing=SAMPLE_SPACING,
        file_name="offaxis_y075",
        schedule=SAMPLE_SCHEDULE,
    ),
    fvm.SurfaceSampler(
        point=[0.0, 0.0, 0.0],
        normal=[0, 0, 1],
        bounds=[FVM_DOMAIN[0], FVM_DOMAIN[1], FVM_DOMAIN[2], FVM_DOMAIN[3]],
        spacing=SAMPLE_SPACING,
        schedule=SLICE_SCHEDULE,
        file_name="slice_z0",
    ),
    fvm.SurfaceSampler(
        point=[0.0, 0.0, 0.0],
        normal=[0, 0, 1],
        bounds=WAKE_SLICE_BOUNDS,
        spacing=SAMPLE_SPACING,
        schedule=SLICE_SCHEDULE,
        file_name="wake_slice_z0",
    ),
)

FVM_MESH = fvm.AdaptiveCartesianMesher(
    domain=FVM_DOMAIN,
    max_cell_size=SURFACE_CELL_SIZE * 32,
    surface_file=CUBE_STL,
    wall_patch_name="cube",
    surface_cell_size=SURFACE_CELL_SIZE,
    refinements=(
        fvm.BoxRefinement(WAKE_BOX, SURFACE_CELL_SIZE * 2, "wakeBox"),
        fvm.BoxRefinement(DOWNSTREAM_WAKE_BOX, SURFACE_CELL_SIZE * 4, "downstreamWakeBox"),
    ),
)

FVM_SETUP = fvm.FVMSetup(
    case_name="referenceFlow",
    cores=FVM_CORES,
    execution=fvm.ComputeConfig(operator_backend="numba"),
    output=fvm.OutputConfig(
        format="vtk_xml",
        data_location="cell",
        encoding="appended",
        compression="lz4",
        precision="float32",
        asynchronous=True,
        ghost_layers=0,
    ),
    time=fvm.TimeConfig(
        time_step_size=FVM_TIME_STEP_SIZE,
        start_time=0.0,
        end_time=END_TIME,
        output_interval_steps=10**9,
        output_interval_time=VOLUME_INTERVAL_TIME,
        adjust_time_step=False,
    ),
    schemes=fvm.DiscretizationConfig(
        # Match the coupled reference-flow discretisation exactly.
        convection_scheme="linearUpwind",
        gradient_scheme="gauss",
        time_scheme="backward",
    ),
    linear=fvm.LinearSolverConfig(
        linear_solver="bicgstab",
        pressure_solver="amg",
        pressure_tolerance=1e-6,
        pressure_relative_tolerance=0.01,
        pressure_final_relative_tolerance=0.0,
        momentum_tolerance=1e-6,
        momentum_relative_tolerance=0.1,
        momentum_final_relative_tolerance=0.0,
        momentum_max_iterations=2000,
        ilu_drop_tolerance=1e-4,
        ilu_fill_factor=10.0,
        ilu_reuse_tolerance=0.05,
    ),
    pimple=fvm.PimpleControl(
        n_correctors=2,
        n_outer_correctors=2,
        n_orthogonal_correctors=1,
        velocity_relaxation=0.7,
        pressure_relaxation=0.3,
    ),
    samplers=SAMPLERS,
    transport=fvm.TransportConfig(density=DENSITY, kinematic_viscosity=KINEMATIC_VISCOSITY),
    turbulence=fvm.TurbulenceConfig.equilibrium_smagorinsky(
        c_k=SMAGORINSKY_CK,
        c_e=SMAGORINSKY_CE,
    ),
    boundaries=[
        fvm.BoundaryConfig.inlet("inlet", list(FREESTREAM_VELOCITY)),
        fvm.BoundaryConfig.outlet("outlet", kinematic_pressure=0.0),
        fvm.BoundaryConfig.slip("ymin"),
        fvm.BoundaryConfig.slip("ymax"),
        fvm.BoundaryConfig.slip("zmin"),
        fvm.BoundaryConfig.slip("zmax"),
        fvm.BoundaryConfig.wall("cube"),
    ],
    initial_velocity=list(INITIAL_VELOCITY),
    initial_kinematic_pressure=0.0,
)


def main() -> None:
    fvm_solver = fvm.create_fvm_solver(FVM_SETUP, case_dir=CASE_DIR, mesh=FVM_MESH)
    fvm_solver.write_vtk()

    while fvm_solver.time < FVM_SETUP.time.end_time:
        fvm_solver.advance()

    fvm_solver.close()


if __name__ == "__main__":
    main()
