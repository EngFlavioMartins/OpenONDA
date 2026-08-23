"""Finite-span NACA 4412 at 10 degrees with native OpenONDA FVM-VPM.

The airfoil is generated analytically, represented by the FVM immersed-
boundary method, and coupled to VPM on a solver-native Cartesian mesh.  The
case has no external solver, mesher, or repository-path dependency.

The case can be configured through the OPENONDA_* environment variables or
through the convenience command-line flags below, which override them.

Usage:
    python naca4412_setup.py
    python naca4412_setup.py --smoke --end-time 0.4 --spacing 0.2
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
from pathlib import Path
import sys

import numpy as np

import openonda.fvm as fvm
import openonda.coupler as coupling
import openonda.vpm as vpm

CASE_DIR = Path(__file__).resolve().parent
SMOKE = os.environ.get("OPENONDA_SMOKE", "0") == "1"

NACA_CODE = "4412"
ALPHA_DEG = 10.0
CHORD = 1.0
SPAN = 5.0
DENSITY = 1.0
REYNOLDS = 1000.0
SPACING = float(os.environ.get("OPENONDA_SPACING", "0.20" if SMOKE else "0.04"))
FVM_TIME_STEP_SIZE = float(os.environ.get("OPENONDA_FVM_DT", "0.01"))
VPM_TIME_STEP_SIZE = float(os.environ.get("OPENONDA_VPM_DT", "0.04"))
END_TIME = float(os.environ.get("OPENONDA_T_END", "0.12" if SMOKE else "12.0"))
ANGLE = math.radians(ALPHA_DEG)
FREESTREAM_VELOCITY = (math.cos(ANGLE), math.sin(ANGLE), 0.0)
KINEMATIC_VISCOSITY = np.linalg.norm(FREESTREAM_VELOCITY) * CHORD / REYNOLDS
FVM_BOX = (-1.2, 1.4, -0.8, 0.8, -3.3, 3.3)
VPM_DOMAIN = (-2.5, 10.0, -2.0, 2.0, -4.0, 4.0)
MAX_N_PARTICLES = int(os.environ.get("OPENONDA_MAX_PARTICLES", "100000" if SMOKE else "1500000"))
VPM_CORE_RADIUS_RATIO = 1.5
IBM_MARKER_RATIO = float(os.environ.get("OPENONDA_IBM_MARKER_RATIO", "2.5"))
WRITE_INTERVAL_TIME = VPM_TIME_STEP_SIZE if SMOKE else 0.8
SAMPLE_INTERVAL_TIME = min(WRITE_INTERVAL_TIME, END_TIME)
FVM_LOGGING_INTERVAL_STEPS = max(1, int(round(SAMPLE_INTERVAL_TIME / FVM_TIME_STEP_SIZE)))
VPM_LOGGING_INTERVAL_STEPS = max(1, int(round(SAMPLE_INTERVAL_TIME / VPM_TIME_STEP_SIZE)))


def naca4_vertices(code: str, n_chord: int = 161) -> np.ndarray:
    """Return a closed clockwise polygon for a four-digit NACA section."""
    if len(code) != 4 or not code.isdigit():
        raise ValueError("NACA code must contain four digits")
    m = int(code[0]) / 100.0
    p = int(code[1]) / 10.0
    thickness = int(code[2:]) / 100.0
    beta = np.linspace(0.0, np.pi, n_chord)
    x = 0.5 * (1.0 - np.cos(beta))
    yt = (
        5.0
        * thickness
        * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1036 * x**4)
    )
    yc = np.where(
        x < p,
        m / p**2 * (2.0 * p * x - x**2),
        m / (1.0 - p) ** 2 * ((1.0 - 2.0 * p) + 2.0 * p * x - x**2),
    )
    slope = np.where(
        x < p,
        2.0 * m / p**2 * (p - x),
        2.0 * m / (1.0 - p) ** 2 * (p - x),
    )
    theta = np.arctan(slope)
    upper = np.column_stack((x - yt * np.sin(theta), yc + yt * np.cos(theta)))
    lower = np.column_stack((x + yt * np.sin(theta), yc - yt * np.cos(theta)))
    section = np.vstack((upper[::-1], lower[1:-1]))
    section[:, 0] = CHORD * (section[:, 0] - 0.5)
    section[:, 1] *= CHORD
    return section


AIRFOIL_VERTICES = naca4_vertices(NACA_CODE)
FVM_MESH = fvm.coupling_box_mesh(FVM_BOX, SPACING, patch_name="numericalBoundary")
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
        schedule=fvm.SamplingSchedule(every_n_steps=FVM_LOGGING_INTERVAL_STEPS),
    ),
    fvm.LineSampler(
        start=[FVM_BOX[0], 0.0, 0.0],
        end=[FVM_BOX[1], 0.0, 0.0],
        spacing=SPACING,
        file_name="fvm_centreline",
        schedule=fvm.SamplingSchedule(every_n_steps=FVM_LOGGING_INTERVAL_STEPS),
    ),
    fvm.SurfaceSampler(
        point=[0.0, 0.0, 0.0],
        normal=[0.0, 0.0, 1.0],
        bounds=[FVM_BOX[0], FVM_BOX[1], FVM_BOX[2], FVM_BOX[3]],
        spacing=SPACING,
        file_name="fvm_slice_z0",
        schedule=fvm.SamplingSchedule(every_n_steps=FVM_LOGGING_INTERVAL_STEPS),
    ),
)

VPM_SAMPLERS = (
    vpm.LineSampler(
        start=[VPM_DOMAIN[0], 0.0, 0.0],
        end=[VPM_DOMAIN[1], 0.0, 0.0],
        spacing=SPACING,
        file_name="vpm_centreline",
    ),
    vpm.SurfaceSampler(
        point=[0.0, 0.0, 0.0],
        normal=[0.0, 0.0, 1.0],
        bounds=[VPM_DOMAIN[0], VPM_DOMAIN[1], VPM_DOMAIN[2], VPM_DOMAIN[3]],
        spacing=SPACING,
        file_name="vpm_slice_z0",
    ),
)

FVM_SETUP = fvm.FVMSetup(
    case_name="naca4412_flow",
    cores=1,
    execution=fvm.ComputeConfig(operator_backend="numba"),
    output=fvm.OutputConfig(
        compression="lz4",
        precision="float32",
        asynchronous=False,
        ghost_layers=0,
    ),
    time=fvm.TimeConfig(
        time_step_size=FVM_TIME_STEP_SIZE,
        end_time=END_TIME,
        output_interval_steps=10**9,
        output_interval_time=WRITE_INTERVAL_TIME,
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

VPM_SETUP = vpm.VPMSetup(
    time_step_size=VPM_TIME_STEP_SIZE,
    freestream_velocity=list(FREESTREAM_VELOCITY),
    viscous=vpm.ViscousConfig.cs(kinematic_viscosity=KINEMATIC_VISCOSITY, particle_spacing=SPACING),
    stretching=vpm.StretchingConfig.transposed(scheme="RK2"),
    advection=vpm.AdvectionConfig(scheme="RK2"),
    turbulence=vpm.TurbulenceConfig.les_smagorinsky(smagorinsky_coefficient=0.17),
    velocity=vpm.VelocityConfig.treecode(theta=0.3, multipole_order=2),
    stabilization=vpm.StabilizationConfig.bounded_domain(VPM_DOMAIN),
    particle_kernel="GAUSSIAN",
    precision="f32",
    compute_device="AUTO",
    max_n_particles=MAX_N_PARTICLES,
    max_evaluation_points=MAX_N_PARTICLES,
    domain_bounds=list(VPM_DOMAIN),
    log_mode="file",
    logging_interval_steps=VPM_LOGGING_INTERVAL_STEPS,
    checkpoint_interval_steps=VPM_LOGGING_INTERVAL_STEPS,
    checkpoint_directory=str(CASE_DIR / "solution"),
    samplers=VPM_SAMPLERS,
)

COUPLER_SETUP = coupling.CouplerSetup(
    freestream_velocity=list(FREESTREAM_VELOCITY),
    vpm_particle_spacing=SPACING,
    authority_ramp_width=6 * SPACING,
    vpm_only_width=3.0 * SPACING,
    vpm_core_radius_ratio=VPM_CORE_RADIUS_RATIO,
    checkpoint_interval_steps=VPM_LOGGING_INTERVAL_STEPS,
)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    """Build the command-line overrides for the OPENONDA_* defaults."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="fast smoke-test settings")
    parser.add_argument("--end-time", type=float, help="simulation end time [s]")
    parser.add_argument("--spacing", type=float, help="core grid spacing [m]")
    parser.add_argument("--fvm-dt", type=float, help="FVM time step [s]")
    parser.add_argument("--vpm-dt", type=float, help="VPM time step [s]")
    parser.add_argument("--max-particles", type=int, help="particle budget")
    parser.add_argument("--marker-ratio", type=float, help="IBM marker spacing / grid spacing")
    return parser.parse_args(argv)


def _run_with_overrides(argv: list[str]) -> int:
    """Run the case with any command-line overrides applied.

    The module-level constants already read the OPENONDA_* environment
    variables (so ``OPENONDA_SMOKE=1 ./allrun.sh`` keeps working). When the
    user passes flags, they are translated back into those environment
    variables and the module is loaded once more so every dependent setting
    follows.
    """
    args = _parse_args(argv)

    overrides: dict[str, str] = {}
    if args.smoke:
        overrides["OPENONDA_SMOKE"] = "1"
    for flag, key in (
        ("end_time", "OPENONDA_T_END"),
        ("spacing", "OPENONDA_SPACING"),
        ("fvm_dt", "OPENONDA_FVM_DT"),
        ("vpm_dt", "OPENONDA_VPM_DT"),
        ("max_n_particles", "OPENONDA_MAX_PARTICLES"),
        ("marker_ratio", "OPENONDA_IBM_MARKER_RATIO"),
    ):
        if getattr(args, flag) is not None:
            overrides[key] = str(getattr(args, flag))

    if not overrides:
        main()
        return 0

    os.environ.update(overrides)
    spec = importlib.util.spec_from_file_location("naca4412_setup", __file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()
    return 0


def main() -> None:
    print("\n===== SIMULATION =====")
    print(
        f"  FVM dt={FVM_TIME_STEP_SIZE}s / VPM dt={VPM_TIME_STEP_SIZE}s, "
        f"spacing={SPACING}, particles<={MAX_N_PARTICLES}"
    )
    fvm_solver = fvm.create_fvm_solver(FVM_SETUP, case_dir=CASE_DIR, mesh=FVM_MESH)
    fvm_solver.set_immersed_bodies(AIRFOIL, grid_spacing=SPACING)
    fvm_solver.write_vtk()
    vpm_solver = vpm.create_vpm_solver(VPM_SETUP, case_dir=CASE_DIR)
    coupled_solver = coupling.create_coupler(fvm_solver, vpm_solver, COUPLER_SETUP)
    coupled_solver.run()


if __name__ == "__main__":
    raise SystemExit(_run_with_overrides(sys.argv[1:]))
