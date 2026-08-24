"""Laminar cylinder shedding at Re = 150 with hybrid FVM–VPM coupling.

The infinite immersed cylinder is spanwise invariant. An optional divergence-
free seed lets this case and the FVM reference compare instability growth and
saturated shedding frequency from identical initial disturbances.

Usage:
    python cylinder_shedding_flow_setup.py
    OPENONDA_SMOKE=1 python cylinder_shedding_flow_setup.py
    OPENONDA_SEED_AMPLITUDE=1e-4 python cylinder_shedding_flow_setup.py
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

import openonda.fvm as fvm
import openonda.coupler as coupling
import openonda.vpm as vpm
from seed_perturbation import build_seed_velocity

CASE_DIR = Path(__file__).resolve().parent
SMOKE = os.environ.get("OPENONDA_SMOKE", "0") == "1"

# Physical problem
DIAMETER = 1.0
FREESTREAM_VELOCITY = (1.0, 0.0, 0.0)
DENSITY = 1.0
REYNOLDS = 150.0
FREESTREAM_SPEED = float(np.linalg.norm(FREESTREAM_VELOCITY))
KINEMATIC_VISCOSITY = FREESTREAM_SPEED * DIAMETER / REYNOLDS

INITIAL_VELOCITY = FREESTREAM_VELOCITY
# The unseeded reference needs 100 convective units to reach a saturated window.
END_TIME = 1.0 if SMOKE else 100.0

# FVM domain, mesh, and numerics
# Body, wake, and background resolutions are 16, 8, and 4 cells per diameter.
FVM_CORES = int(os.environ.get("OPENONDA_FVM_CORES", "1"))
if SMOKE:
    FVM_CORES = 1
FVM_BOX = (-2.0, 3.0, -2.5, 2.5, -6.5, 6.5)
TRANSFER_REGION_BOX = FVM_BOX

FVM_CELL_SIZE = 0.5 if SMOKE else 0.25
FVM_WAKE_CELL_SIZE = 0.25 if SMOKE else 0.125
FVM_BODY_CELL_SIZE = 0.125 if SMOKE else 0.0625

# The uncapped cylinder removes tip effects from the Karman instability.
CYLINDER_Z_BOUNDS = (FVM_BOX[4], FVM_BOX[5])

# Force coefficients are normalized per unit span.
CYLINDER_LENGTH = FVM_BOX[5] - FVM_BOX[4]

# The body and wake boxes resolve the boundary layer and initial shedding region.
BODY_BOX = (-0.65, 0.65, -0.65, 0.65, FVM_BOX[4], FVM_BOX[5])
WAKE_BOX = (-0.75, 3.0, -1.25, 1.25, -5.5, 5.5)

# Backward 2nd-order; CFL ~ 0.32 in the body region, 0.16 wake, 0.08 far field.
FVM_TIME_STEP_SIZE = 0.02
PIMPLE_N_CORRECTORS = 2
PIMPLE_N_OUTER_CORRECTORS = 1
PIMPLE_N_ORTHOGONAL_CORRECTORS = 0
IBM_FORCING_LOOPS = 2

if FVM_CORES > 1:
    # Direct-forcing IBM is not partition-aware, so parallel runs use the
    # replicated PETSc path (the factory's designated IBM parallel mode).
    FVM_EXECUTION = fvm.ComputeConfig(
        operator_backend="numba",
        linear_backend="petsc",
        parallel_mode="petsc_replicated",
    )
else:
    FVM_EXECUTION = fvm.ComputeConfig(operator_backend="numba")

FVM_VOLUME_INTERVAL_TIME = 2.0

# VPM domain and resolution
VPM_DOMAIN = (-5.0, 15.0, -5.0, 5.0, -8.0, 8.0)
VPM_PARTICLE_SPACING = 0.25 if SMOKE else 0.125
VPM_CORE_RADIUS_RATIO = 1.0
VPM_TIME_STEP_SIZE = 0.10
VPM_SCHEME = "RK2"
PARTICLE_LIMIT = 200_000 if SMOKE else 1_000_000

# GBD diffusion has alpha = nu*dt/h^2 ~ 0.043, below the 1/6 stability limit.
GBD_VORTICITY_FLOOR = 0.01

# Coupling
BOUNDARY_CONDITION_MODE = "vorticity_mixed"
TRANSFER_DIAGNOSTIC_INTERVAL_STEPS = 12
COUPLER_CHECKPOINT_INTERVAL_STEPS = 20

# Output and diagnostics cadence
FORCE_INTERVAL_TIME = 0.05
LINE_INTERVAL_TIME = 0.5
SLICE_INTERVAL_TIME = 1.0
VPM_CHECKPOINT_INTERVAL_TIME = 1.0
VPM_LOGGING_INTERVAL_STEPS = 12
SAMPLE_SPACING = VPM_PARTICLE_SPACING
PROBE_X = 1.5  # x/D of the primary instability observable q(t) = u_y/U_inf
TRANSVERSE_HALF = 1.5
SPAN_HALF = 5.5
WAKE_SLICE_BOUNDS = (0.0, 8.0, -2.5, 2.5)
MIDSPAN_SLICE_BOUNDS = (FVM_BOX[0], FVM_BOX[1], FVM_BOX[2], FVM_BOX[3])

# Optional controlled seed shared with the reference.
SEED_AMPLITUDE = float((os.environ.get("OPENONDA_SEED_AMPLITUDE") or "0").strip())

# Case objects
CYLINDER = fvm.ImmersedBody.extruded_cylinder_z(
    centre=[0.0, 0.0, 0.0],
    diameter=DIAMETER,
    z_bounds=CYLINDER_Z_BOUNDS,
    grid_spacing=FVM_BODY_CELL_SIZE,
    name="cylinder",
    caps=False,
)

FVM_MESH = fvm.AdaptiveCartesianMesher(
    domain=FVM_BOX,
    max_cell_size=FVM_CELL_SIZE,
    refinements=(
        fvm.BoxRefinement(BODY_BOX, FVM_BODY_CELL_SIZE, "bodyBox"),
        fvm.BoxRefinement(WAKE_BOX, FVM_WAKE_CELL_SIZE, "wakeBox"),
    ),
    merge_outer_patch="numericalBoundary",
)

FVM_SAMPLERS = (
    fvm.IBMForceSampler(
        reference_velocity=FREESTREAM_SPEED,
        reference_area=DIAMETER * CYLINDER_LENGTH,
        file_name="fvm_ibm_forces_history",
        schedule=fvm.SamplingSchedule(every_time=FORCE_INTERVAL_TIME),
    ),
    fvm.LineSampler(
        start=[PROBE_X, 0.0, 0.0],
        end=[PROBE_X, 0.0, 0.0],
        n_points=1,
        file_name="fvm_midspan_probe",
        schedule=fvm.SamplingSchedule(every_time=FORCE_INTERVAL_TIME),
    ),
    fvm.LineSampler(
        start=[PROBE_X, -TRANSVERSE_HALF, 0.0],
        end=[PROBE_X, TRANSVERSE_HALF, 0.0],
        spacing=SAMPLE_SPACING,
        file_name="fvm_transverse_line",
        schedule=fvm.SamplingSchedule(every_time=LINE_INTERVAL_TIME),
    ),
    fvm.LineSampler(
        start=[PROBE_X, 0.0, -SPAN_HALF],
        end=[PROBE_X, 0.0, SPAN_HALF],
        spacing=SAMPLE_SPACING,
        file_name="fvm_spanwise_line",
        schedule=fvm.SamplingSchedule(every_time=LINE_INTERVAL_TIME),
    ),
    fvm.SurfaceSampler(
        point=[0.0, 0.0, 0.0],
        normal=[0.0, 0.0, 1.0],
        bounds=MIDSPAN_SLICE_BOUNDS,
        spacing=SAMPLE_SPACING,
        file_name="fvm_slice_z0",
        schedule=fvm.SamplingSchedule(every_time=SLICE_INTERVAL_TIME),
    ),
)

FVM_SETUP = fvm.FVMSetup(
    case_name="cylinder_shedding_flow",
    cores=FVM_CORES,
    execution=FVM_EXECUTION,
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
        output_interval_time=FVM_VOLUME_INTERVAL_TIME,
        adjust_time_step=False,
    ),
    schemes=fvm.DiscretizationConfig(
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
        momentum_tolerance=1e-4,
        momentum_relative_tolerance=0.1,
        momentum_final_relative_tolerance=0.0,
        momentum_max_iterations=2000,
        ilu_drop_tolerance=1e-4,
        ilu_fill_factor=10.0,
        ilu_reuse_tolerance=0.05,
    ),
    pimple=fvm.PimpleControl(
        n_correctors=PIMPLE_N_CORRECTORS,
        n_outer_correctors=PIMPLE_N_OUTER_CORRECTORS,
        n_orthogonal_correctors=PIMPLE_N_ORTHOGONAL_CORRECTORS,
        velocity_relaxation=0.7,
        pressure_relaxation=0.3,
        ibm_forcing_loops=IBM_FORCING_LOOPS,
    ),
    samplers=FVM_SAMPLERS,
    transport=fvm.TransportConfig(density=DENSITY, kinematic_viscosity=KINEMATIC_VISCOSITY),
    turbulence=fvm.TurbulenceConfig.none(),
    boundaries=[
        fvm.BoundaryConfig(
            name="numericalBoundary",
            velocity_type="fixedValue",
            velocity_value=list(FREESTREAM_VELOCITY),
            pressure_type="fixedFluxPressure",
        )
    ],
    initial_velocity=list(INITIAL_VELOCITY),
    initial_kinematic_pressure=0.0,
)

COUPLER_SETUP = coupling.CouplerSetup(
    freestream_velocity=list(FREESTREAM_VELOCITY),
    transfer_region_bounds=TRANSFER_REGION_BOX,
    is_boundary_condition_resynchronized_after_transfer=True,
    is_pressure_anchored_to_freestream=False,
    checkpoint_interval_steps=COUPLER_CHECKPOINT_INTERVAL_STEPS,
    boundary_condition_mode=BOUNDARY_CONDITION_MODE,
    vpm_particle_spacing=VPM_PARTICLE_SPACING,
    vpm_core_radius_ratio=VPM_CORE_RADIUS_RATIO,
    eta_blend_width=0.0,
    transfer_diagnostic_interval_steps=TRANSFER_DIAGNOSTIC_INTERVAL_STEPS,
)


def _apply_seed(fvm_solver) -> None:
    """Impose the controlled, divergence-free perturbation on the initial field."""
    n_cells = fvm_solver.mesh_data["n_cells"]
    cell_centre = np.asarray(fvm_solver.geo_data["cell_centre"][:n_cells], dtype=np.float64)
    if cell_centre.shape[0] != n_cells:
        raise RuntimeError("Seed requires the full cell-centroid array on every rank")
    velocity = build_seed_velocity(
        cell_centre,
        base_velocity=INITIAL_VELOCITY,
        epsilon=SEED_AMPLITUDE,
        freestream_speed=FREESTREAM_SPEED,
        diameter=DIAMETER,
    )
    fvm_solver.set_initial_velocity(velocity)
    peak = float(np.linalg.norm(velocity - np.asarray(INITIAL_VELOCITY), axis=1).max())
    print(
        f"  Applied controlled seed: eps={SEED_AMPLITUDE:.3e}, max|u'|/Uinf={peak / FREESTREAM_SPEED:.3e}"
    )


VPM_SAMPLERS = (
    vpm.LineSampler(
        start=[VPM_DOMAIN[0], 0.0, 0.0],
        end=[VPM_DOMAIN[1], 0.0, 0.0],
        spacing=SAMPLE_SPACING,
        file_name="vpm_centreline",
    ),
    vpm.SurfaceSampler(
        point=[0.0, 0.0, 0.0],
        normal=[0, 0, 1],
        bounds=MIDSPAN_SLICE_BOUNDS,
        spacing=SAMPLE_SPACING,
        file_name="vpm_slice_z0",
        include_derivatives=False,
    ),
    vpm.SurfaceSampler(
        point=[0.0, 0.0, 0.0],
        normal=[0, 0, 1],
        bounds=WAKE_SLICE_BOUNDS,
        spacing=SAMPLE_SPACING,
        file_name="vpm_wake_slice_z0",
        include_derivatives=False,
    ),
)
# No panel solver: the FVM owns the body (IBM cylinder) entirely, and this
# is an injected-VPM wake solver, so the VPM needs no boundary-element
# surface.  Omitting it keeps the VPM boundary condition at the FVM boundary equal to
# freestream + wake-particle induction, matching the reference far-field.
VPM_SETUP = vpm.VPMSetup(
    time_step_size=VPM_TIME_STEP_SIZE,
    freestream_velocity=list(FREESTREAM_VELOCITY),
    viscous=vpm.ViscousConfig.gbd(
        particle_spacing=VPM_PARTICLE_SPACING,
        padding=3.0,
        kinematic_viscosity=KINEMATIC_VISCOSITY,
        threshold_mode="absolute",
        threshold=GBD_VORTICITY_FLOOR * VPM_PARTICLE_SPACING**3,
        max_nodes=PARTICLE_LIMIT,
        core_radius_ratio=VPM_CORE_RADIUS_RATIO,
    ),
    stretching=vpm.StretchingConfig.transposed(scheme=VPM_SCHEME),
    advection=vpm.AdvectionConfig(scheme=VPM_SCHEME),
    turbulence=vpm.TurbulenceConfig.inviscid(),
    velocity=vpm.VelocityConfig.treecode(theta=0.3, multipole_order=2),
    stabilization=vpm.StabilizationConfig.bounded_domain(VPM_DOMAIN),
    particle_kernel="GAUSSIAN",
    precision="f32",
    compute_device="AUTO",
    max_n_particles=PARTICLE_LIMIT,
    max_evaluation_points=PARTICLE_LIMIT,
    domain_bounds=list(VPM_DOMAIN),
    log_mode="file",
    logging_interval_steps=VPM_LOGGING_INTERVAL_STEPS,
    timing_interval_steps=VPM_LOGGING_INTERVAL_STEPS,
    checkpoint_interval_steps=int(VPM_CHECKPOINT_INTERVAL_TIME / VPM_TIME_STEP_SIZE),
    checkpoint_directory=str(CASE_DIR / "solution"),
    export_flow_integrals=False,
    samplers=VPM_SAMPLERS,
)


def main() -> None:
    print("\n===== SIMULATION (hybrid) =====")
    print(
        f"  Re={REYNOLDS}, infinite cylinder D={DIAMETER}, "
        f"FVM time_step_size={FVM_TIME_STEP_SIZE}s / "
        f"VPM time_step_size={VPM_TIME_STEP_SIZE}s, "
        f"body/wake cell size={FVM_BODY_CELL_SIZE}/{FVM_WAKE_CELL_SIZE}, "
        f"particles<={PARTICLE_LIMIT}, seed={SEED_AMPLITUDE:g}"
    )
    fvm_solver = fvm.create_fvm_solver(FVM_SETUP, case_dir=CASE_DIR, mesh=FVM_MESH)
    fvm_solver.set_immersed_bodies(CYLINDER, grid_spacing=FVM_BODY_CELL_SIZE)
    if SEED_AMPLITUDE > 0.0:
        _apply_seed(fvm_solver)
    fvm_solver.write_vtk()
    vpm_solver = vpm.create_vpm_solver(VPM_SETUP, case_dir=CASE_DIR)
    coupled_solver = coupling.create_coupler(fvm_solver, vpm_solver, COUPLER_SETUP)
    max_n_steps = int(os.environ.get("OPENONDA_MAX_STEPS", "0"))
    if max_n_steps > 0:
        capped_t = min(END_TIME, max_n_steps * VPM_TIME_STEP_SIZE)
        FVM_SETUP.time.end_time = float(capped_t)
        print(f"  [probe] capping run to {max_n_steps} VPM steps (t={capped_t:g})")
    coupled_solver.run()


if __name__ == "__main__":
    main()
