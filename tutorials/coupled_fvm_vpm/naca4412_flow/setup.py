"""Finite-span NACA 4412 at 10 degrees with native OpenONDA FVM-VPM.

The airfoil is generated analytically, represented by the FVM immersed-
boundary method, and coupled to VPM on a solver-native Cartesian mesh.  The
case has no external solver, mesher, or repository-path dependency.

Run with ``python setup.py``. Edit the physical and numerical constants below
to study a different NACA 4412 case.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

import openonda.fvm as fvm
import openonda.fvm.mesher as msh
import openonda.coupler as coupling
import openonda.vpm as vpm
from openonda.vpm import Backup, Samplers

CASE_DIR = Path(__file__).resolve().parent
# Physical parameters -----------------------------------------------------
NACA_CODE = "4412"
ALPHA_DEG = 10.0
CHORD = 1.0
SPAN = 5.0
DENSITY = 1.0
REYNOLDS = 1000.0
SPACING = 0.04
FVM_TIME_STEP_SIZE = 0.01
VPM_TIME_STEP_SIZE = 0.04
END_TIME = 12.0
ANGLE = math.radians(ALPHA_DEG)
FREESTREAM_VELOCITY = (math.cos(ANGLE), math.sin(ANGLE), 0.0)
KINEMATIC_VISCOSITY = np.linalg.norm(FREESTREAM_VELOCITY) * CHORD / REYNOLDS
FVM_BOX = (-1.2, 1.4, -0.8, 0.8, -3.3, 3.3)
VPM_DOMAIN = (-2.5, 10.0, -2.0, 2.0, -4.0, 4.0)
MAX_N_PARTICLES = 1_500_000
VPM_CORE_RADIUS_RATIO = 1.5
IBM_MARKER_RATIO = 2.5
WRITE_INTERVAL_TIME = 0.8
SAMPLE_INTERVAL_TIME = min(WRITE_INTERVAL_TIME, END_TIME)
FVM_LOGGING_INTERVAL_STEPS = max(1, int(round(SAMPLE_INTERVAL_TIME / FVM_TIME_STEP_SIZE)))
VPM_LOGGING_INTERVAL_STEPS = max(1, int(round(SAMPLE_INTERVAL_TIME / VPM_TIME_STEP_SIZE)))


def naca4_vertices(code: str, n_chord: int = 161) -> np.ndarray:
    """Return a closed clockwise polygon for a four-digit NACA section."""
    if len(code) != 4 or not code.isdigit():
        raise ValueError("NACA code must contain four digits")
    m = int(code[0]) / 100.0
    max_camber_position = int(code[1]) / 10.0
    thickness = int(code[2:]) / 100.0
    beta = np.linspace(0.0, np.pi, n_chord)
    x = 0.5 * (1.0 - np.cos(beta))
    yt = (
        5.0
        * thickness
        * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1036 * x**4)
    )
    yc = np.where(
        x < max_camber_position,
        m / max_camber_position**2 * (2.0 * max_camber_position * x - x**2),
        m
        / (1.0 - max_camber_position) ** 2
        * ((1.0 - 2.0 * max_camber_position) + 2.0 * max_camber_position * x - x**2),
    )
    slope = np.where(
        x < max_camber_position,
        2.0 * m / max_camber_position**2 * (max_camber_position - x),
        2.0 * m / (1.0 - max_camber_position) ** 2 * (max_camber_position - x),
    )
    theta = np.arctan(slope)
    upper = np.column_stack((x - yt * np.sin(theta), yc + yt * np.cos(theta)))
    lower = np.column_stack((x + yt * np.sin(theta), yc - yt * np.cos(theta)))
    section = np.vstack((upper[::-1], lower[1:-1]))
    section[:, 0] = CHORD * (section[:, 0] - 0.5)
    section[:, 1] *= CHORD
    return section


AIRFOIL_VERTICES = naca4_vertices(NACA_CODE)
FVM_MESH = msh.coupling_box_mesh(FVM_BOX, SPACING, patch_name="numericalBoundary")
AIRFOIL = fvm.ImmersedBody.extruded_polygon_z(
    AIRFOIL_VERTICES,
    z_bounds=[-0.5 * SPAN, 0.5 * SPAN],
    grid_spacing=SPACING,
    marker_spacing_ratio=IBM_MARKER_RATIO,
    name="airfoil",
    caps=True,
)

FVM_SAMPLERS = (
    fvm.IBMForceSampler(
        reference_velocity=float(np.linalg.norm(FREESTREAM_VELOCITY)),
        reference_area=CHORD * SPAN,
        schedule=fvm.RunSchedule(every_n_steps=FVM_LOGGING_INTERVAL_STEPS),
    ),
    fvm.LineSampler(
        start=[FVM_BOX[0], 0.0, 0.0],
        end=[FVM_BOX[1], 0.0, 0.0],
        spacing=SPACING,
        file_name="fvm_centreline",
        schedule=fvm.RunSchedule(every_n_steps=FVM_LOGGING_INTERVAL_STEPS),
    ),
    fvm.SurfaceSampler(
        point=[0.0, 0.0, 0.0],
        normal=[0.0, 0.0, 1.0],
        bounds=[FVM_BOX[0], FVM_BOX[1], FVM_BOX[2], FVM_BOX[3]],
        spacing=SPACING,
        file_name="fvm_slice_z0",
        schedule=fvm.RunSchedule(every_n_steps=FVM_LOGGING_INTERVAL_STEPS),
    ),
)

VPM_SAMPLERS = (
    vpm.LineSampler(
        start=[VPM_DOMAIN[0], 0.0, 0.0],
        end=[VPM_DOMAIN[1], 0.0, 0.0],
        spacing=SPACING,
        file_name="vpm_centreline",
        schedule=vpm.EverySteps(VPM_LOGGING_INTERVAL_STEPS),
    ),
    vpm.SurfaceSampler(
        point=[0.0, 0.0, 0.0],
        normal=[0.0, 0.0, 1.0],
        bounds=[VPM_DOMAIN[0], VPM_DOMAIN[1], VPM_DOMAIN[2], VPM_DOMAIN[3]],
        spacing=SPACING,
        file_name="vpm_slice_z0",
        schedule=vpm.EverySteps(VPM_LOGGING_INTERVAL_STEPS),
    ),
)

FVM_SETUP = fvm.FVMSetup(
    case_name="naca4412_flow",
    cores=1,
    execution=fvm.ComputeConfig(operator_backend="numba"),
    output=fvm.OutputConfig(
        compression="lz4",
        precision="f32",
        asynchronous=False,
        ghost_layers=0,
    ),
    time=fvm.TimeConfig(
        time_step_size=FVM_TIME_STEP_SIZE,
        end_time=END_TIME,
        output_schedule=fvm.RunSchedule(every_time=WRITE_INTERVAL_TIME),
    ),
    schemes=fvm.DiscretizationConfig(
        convection_scheme="limitedLinear",
        gradient_scheme="gauss",
        time_scheme="backward",
    ),
    linear=fvm.LinearSolverConfig(
        pressure_solver="amg",
        pressure_tolerance=1e-6,
        pressure_relative_tolerance=0.01,
        momentum_tolerance=1e-6,
        momentum_relative_tolerance=0.1,
        momentum_max_iterations=2000,
    ),
    pimple=fvm.PimpleControl(
        n_correctors=2,
        n_outer_correctors=2,
        velocity_relaxation=0.7,
        pressure_relaxation=0.3,
        ibm_forcing_loops=2,
    ),
    samplers=FVM_SAMPLERS,
    transport=fvm.TransportConfig(density=DENSITY, kinematic_viscosity=KINEMATIC_VISCOSITY),
    turbulence=fvm.TurbulenceConfig.smagorinsky(smagorinsky_coefficient=0.17),
    boundaries=[
        fvm.BoundaryConfig(
            name="numericalBoundary",
            velocity_type="fixedValue",
            velocity_value=list(FREESTREAM_VELOCITY),
            pressure_type="fixedFluxPressure",
        )
    ],
    initial_velocity=list(FREESTREAM_VELOCITY),
    initial_kinematic_pressure=0.0,
)

VPM_CASE = vpm.VPMCase(
    numerics=vpm.Numerics(
        time_step_size=VPM_TIME_STEP_SIZE,
        freestream_velocity=list(FREESTREAM_VELOCITY),
        viscous=vpm.ViscousConfig.cs(
            kinematic_viscosity=KINEMATIC_VISCOSITY, particle_spacing=SPACING
        ),
        integrator=vpm.RK2(),
        turbulence=vpm.TurbulenceConfig.les_smagorinsky(smagorinsky_coefficient=0.17),
        induction=vpm.TreecodeInduction(theta=0.3),
        stabilization=vpm.StabilizationConfig.bounded_domain(VPM_DOMAIN),
        particle_kernel="GAUSSIAN",
        precision="f32",
        compute_device="AUTO",
        max_n_particles=MAX_N_PARTICLES,
        max_evaluation_points=MAX_N_PARTICLES,
        domain_bounds=list(VPM_DOMAIN),
    ),
    # Coupled restart state is written atomically by COUPLER_SETUP.
    backup=Backup(interval_steps=0, directory="solution", log_directory="solution"),
    samplers=Samplers(samples=VPM_SAMPLERS),
    run=vpm.RunPlan(steps=round(END_TIME / VPM_TIME_STEP_SIZE)),
    directory=CASE_DIR,
)

COUPLER_SETUP = coupling.CouplerSetup(
    freestream_velocity=list(FREESTREAM_VELOCITY),
    eta_blend_width=0.0,
    backup_interval_steps=VPM_LOGGING_INTERVAL_STEPS,
)


def main() -> None:
    print("\n===== SIMULATION =====")
    print(
        f"  FVM time_step_size={FVM_TIME_STEP_SIZE}s / "
        f"VPM time_step_size={VPM_TIME_STEP_SIZE}s, "
        f"spacing={SPACING}, particles<={MAX_N_PARTICLES}"
    )
    fvm_solver = fvm.create_fvm_solver(FVM_SETUP, case_dir=CASE_DIR, mesh=FVM_MESH)
    fvm_solver.set_immersed_bodies(AIRFOIL, grid_spacing=SPACING)
    fvm_solver.write_vtk()
    vpm_solver = vpm.VPMSolver(VPM_CASE)
    coupled_solver = coupling.create_coupler(fvm_solver, vpm_solver, COUPLER_SETUP)
    coupled_solver.run()


if __name__ == "__main__":
    main()
