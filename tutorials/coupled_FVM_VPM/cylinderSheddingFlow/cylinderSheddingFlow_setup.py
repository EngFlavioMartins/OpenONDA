"""Controlled vortex-shedding instability experiment at Re = 150 (hybrid FVM–VPM).

This is a validation experiment, not a benchmark.  The hypothesis under test is
that the coupled FVM–VPM calculation and an FVM-only reference share the same
linear growth rate and saturated shedding frequency, but that the coupling
injects a larger initial antisymmetric disturbance.  The consequence would be
that the hybrid reaches nonlinear saturation earlier, producing an apparent
phase lag at equal physical times.

The FVM mesh is generated directly as solver-native data by OpenONDA's adaptive
Cartesian mesher; the cylinder is an infinite immersed boundary passing through
the FVM slab (``ImmersedBody.extruded_cylinder_z(..., caps=False)``).  No external
solver case is used.  The SGS model is disabled in both solvers (laminar Re=150
validation).

The single validation-only control is ``OPENONDA_SEED_AMPLITUDE`` (default 0).
When nonzero, the same analytic, divergence-free streamfunction perturbation is
added to the initial FVM velocity field of both this case and the reference
before time integration (see ``seed_perturbation.py``).

Usage:
    python cylinderSheddingFlow_setup.py
    OPENONDA_SMOKE=1 python cylinderSheddingFlow_setup.py
    OPENONDA_SEED_AMPLITUDE=1e-4 python cylinderSheddingFlow_setup.py
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from openonda.coupler import CouplerSetup, FVMVPMCoupler, setup_coupler
from openonda.fvm import (
    AdaptiveCartesianMesher,
    BoundaryConfig,
    BoxRefinement,
    ExecutionConfig,
    FVMSetup,
    IBMForceSampler,
    ImmersedBody,
    LinearSolverConfig,
    LineSampler as FVMLineSampler,
    OutputSetup,
    PimpleControl,
    SamplingSchedule,
    SchemesConfig,
    SurfaceSampler as FVMSurfaceSampler,
    TimeConfig,
    TransportConfig,
    TurbulenceConfig as FVMTurbulenceConfig,
    setup_fvm_solver,
)
from seed_perturbation import build_seed_velocity

CASE_DIR = Path(__file__).resolve().parent
SMOKE = os.environ.get("OPENONDA_SMOKE", "0") == "1"

# --------------------------------------------------------------------------- #
# Physical problem (laminar, primary Karman instability only)
# --------------------------------------------------------------------------- #
DIAMETER = 1.0
FREESTREAM_VELOCITY = (1.0, 0.0, 0.0)
RHO = 1.0
REYNOLDS = 150.0
U_INF = float(np.linalg.norm(FREESTREAM_VELOCITY))
NU = U_INF * DIAMETER / REYNOLDS

INITIAL_VELOCITY = FREESTREAM_VELOCITY
# Production horizon: 100 convective units leaves a robust (~30 s) saturated
# window for the unseeded reference, whose slow growth (A0 ~ 1e-5) pushes its
# onset past t~65.  A short 80-unit run leaves too little post-onset record for
# a reliable Welch/FFT saturated frequency.
T_END = 1.0 if SMOKE else 100.0

# --------------------------------------------------------------------------- #
# FVM domain, mesh, and numerics
# --------------------------------------------------------------------------- #
# RAM-safe production sizing (the machine cannot host the original 0.03125 D
# body box spanning the full z extent): body 16 cells/D, wake 8 cells/D,
# background 4 cells/D => ~157k FVM cells here.  The reference shares these
# spacings, so every comparative metric (sigma, St, A0, onset shift) stays
# consistent.  The 2x refinement ratio also improves the momentum-solver
# conditioning.
# Execution is serial by default: on this 17 GB machine the PETSc-replicated
# path (FVM_CORES>1) holds a full copy of the mesh + ILU factors on every
# rank and exhausts RAM even at ~366k cells (measured swap-thrash at 4 ranks).
# OPENONDA_FVM_CORES can override for machines with more RAM; smoke forces 1.
FVM_CORES = int(os.environ.get("OPENONDA_FVM_CORES", "1"))
if SMOKE:
    FVM_CORES = 1
FVM_BOX = (-2.0, 3.0, -2.5, 2.5, -6.5, 6.5)
TRANSFER_REGION_BOX = FVM_BOX

FVM_CELL_SIZE = 0.5 if SMOKE else 0.25
FVM_WAKE_CELL_SIZE = 0.25 if SMOKE else 0.125
FVM_BODY_CELL_SIZE = 0.125 if SMOKE else 0.0625

# Infinite cylinder passing through the FVM slab (caps=False).  A spanwise-
# invariant cylinder is the clean model for the von-Karman instability: it
# produces 2-D shedding at every z (no tip/end contamination of the z=0 probe)
# and, unlike a capped body, keeps the IBM marker cloud well-conditioned (the
# cap-rim markers of a capped cylinder sit closer than h to the lateral
# surface, which makes the Pinelli quadrature singular).
CYLINDER_Z_BOUNDS = (FVM_BOX[4], FVM_BOX[5])

# Force coefficients are normalised per unit span (the force acts over the
# cylinder span that is meshed in this case).
CYLINDER_LENGTH = FVM_BOX[5] - FVM_BOX[4]

# Body and wake refinement boxes (task prescription; each box is snapped to the
# finest Cartesian lattice by the mesher).  The body box encloses the cylinder
# surface plus a thin buffer; the wake box resolves the initial shedding region.
BODY_BOX = (-0.65, 0.65, -0.65, 0.65, FVM_BOX[4], FVM_BOX[5])
WAKE_BOX = (-0.75, 3.0, -1.25, 1.25, -5.5, 5.5)

# FVM time integration / discretisation (cubeFlow numerics, laminar).
# Backward 2nd-order; CFL ~ 0.32 in the body region, 0.16 wake, 0.08 far field.
DT_FVM = 0.02
PIMPLE_N_CORRECTORS = 2
PIMPLE_N_OUTER_CORRECTORS = 1
PIMPLE_N_ORTHOGONAL_CORRECTORS = 0
IBM_FORCING_LOOPS = 2

if FVM_CORES > 1:
    # Direct-forcing IBM is not partition-aware, so parallel runs use the
    # replicated PETSc path (the factory's designated IBM parallel mode).
    FVM_EXECUTION = ExecutionConfig(
        operator_backend="numba",
        linear_backend="petsc",
        parallel_mode="petsc_replicated",
    )
else:
    FVM_EXECUTION = ExecutionConfig(operator_backend="numba")

FVM_VOLUME_INTERVAL = 2.0

# --------------------------------------------------------------------------- #
# VPM domain and resolution
# --------------------------------------------------------------------------- #
VPM_DOMAIN = (-5.0, 15.0, -5.0, 5.0, -8.0, 8.0)
# Coarsened with the FVM wake (8 cells/D) to keep particle memory bounded.
VPM_PARTICLE_SPACING = 0.25 if SMOKE else 0.125
VPM_CORE_RADIUS_RATIO = 1.0
DT_VPM = 0.10
VPM_SCHEME = "RK2"
PARTICLE_LIMIT = 200_000 if SMOKE else 1_000_000

# GBD molecular diffusion with an absolute physical-vorticity floor scaled by
# h^3, as in cubeFlow.  alpha = nu*dt/h^2 = (1/150)*0.10/0.125^2 ~ 0.043 < 1/6.
GBD_VORTICITY_FLOOR = 0.01
TRANSFER_PRUNE_VORTICITY_MIN = 0.01
TRANSFER_BOUNDARY_PRUNE_MULTIPLIER = 10.0

# --------------------------------------------------------------------------- #
# Coupling
# --------------------------------------------------------------------------- #
VPM_BC_MODE = "vorticity_mixed"
TRANSFER_AMPLIFICATION_CAP = 1.8
OVERLAP_ZONE_RAMP_WIDTH = 6.0 * VPM_PARTICLE_SPACING
TRANSFER_DIAGNOSTIC_INTERVAL = 12
COUPLER_BACKUP_PERIOD = 20

# --------------------------------------------------------------------------- #
# Output and diagnostics cadence
# --------------------------------------------------------------------------- #
FORCE_INTERVAL = 0.05
LINE_INTERVAL = 0.5
SLICE_INTERVAL = 1.0
CHECKPOINT_INTERVAL = 1.0
VPM_LOG_PERIOD = 12
SAMPLE_SPACING = VPM_PARTICLE_SPACING
PROBE_X = 1.5  # x/D of the primary instability observable q(t) = u_y/U_inf
TRANSVERSE_HALF = 1.5
SPAN_HALF = 5.5
WAKE_SLICE_BOUNDS = (0.0, 8.0, -2.5, 2.5)
MIDSPAN_SLICE_BOUNDS = (FVM_BOX[0], FVM_BOX[1], FVM_BOX[2], FVM_BOX[3])

# Validation-only controlled seed (shared with the reference).
SEED_AMPLITUDE = float((os.environ.get("OPENONDA_SEED_AMPLITUDE") or "0").strip())

# --------------------------------------------------------------------------- #
# Immersed-boundary cylinder (infinite, spans the FVM slab)
# --------------------------------------------------------------------------- #
CYLINDER = ImmersedBody.extruded_cylinder_z(
    centre=[0.0, 0.0, 0.0],
    diameter=DIAMETER,
    z_bounds=CYLINDER_Z_BOUNDS,
    h=FVM_BODY_CELL_SIZE,
    name="cylinder",
    caps=False,
)

# --------------------------------------------------------------------------- #
# FVM mesh
# --------------------------------------------------------------------------- #
FVM_MESH = AdaptiveCartesianMesher(
    domain=FVM_BOX,
    max_cell_size=FVM_CELL_SIZE,
    refinements=(
        BoxRefinement(BODY_BOX, FVM_BODY_CELL_SIZE, "bodyBox"),
        BoxRefinement(WAKE_BOX, FVM_WAKE_CELL_SIZE, "wakeBox"),
    ),
    merge_outer_patch="numericalBoundary",
)

# --------------------------------------------------------------------------- #
# Samplers
# --------------------------------------------------------------------------- #
FVM_SAMPLERS = (
    IBMForceSampler(
        ref_velocity=U_INF,
        ref_area=DIAMETER * CYLINDER_LENGTH,
        file_name="fvm_ibm_forces_history",
        schedule=SamplingSchedule(every_time=FORCE_INTERVAL),
    ),
    FVMLineSampler(
        start=[PROBE_X, 0.0, 0.0],
        end=[PROBE_X, 0.0, 0.0],
        n_points=1,
        file_name="fvm_midspan_probe",
        schedule=SamplingSchedule(every_time=FORCE_INTERVAL),
    ),
    FVMLineSampler(
        start=[PROBE_X, -TRANSVERSE_HALF, 0.0],
        end=[PROBE_X, TRANSVERSE_HALF, 0.0],
        spacing=SAMPLE_SPACING,
        file_name="fvm_transverse_line",
        schedule=SamplingSchedule(every_time=LINE_INTERVAL),
    ),
    FVMLineSampler(
        start=[PROBE_X, 0.0, -SPAN_HALF],
        end=[PROBE_X, 0.0, SPAN_HALF],
        spacing=SAMPLE_SPACING,
        file_name="fvm_spanwise_line",
        schedule=SamplingSchedule(every_time=LINE_INTERVAL),
    ),
    FVMSurfaceSampler(
        point=[0.0, 0.0, 0.0],
        normal=[0.0, 0.0, 1.0],
        bounds=MIDSPAN_SLICE_BOUNDS,
        spacing=SAMPLE_SPACING,
        file_name="fvm_slice_z0",
        schedule=SamplingSchedule(every_time=SLICE_INTERVAL),
    ),
)

# --------------------------------------------------------------------------- #
# FVM solver configuration
# --------------------------------------------------------------------------- #
FVM_SETUP = FVMSetup(
    case_name="cylinderSheddingFlow",
    cores=FVM_CORES,
    execution=FVM_EXECUTION,
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
        delta_t=DT_FVM,
        start_time=0.0,
        end_time=T_END,
        write_interval=10**9,
        write_interval_time=FVM_VOLUME_INTERVAL,
        adjust_timestep=False,
    ),
    schemes=SchemesConfig(
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
        momentum_tol=1e-4,
        momentum_rel_tol=0.1,
        momentum_final_rel_tol=0.0,
        momentum_maxiter=2000,
        ilu_drop_tol=1e-4,
        ilu_fill_factor=10.0,
        ilu_reuse_tol=0.05,
    ),
    pimple=PimpleControl(
        n_correctors=PIMPLE_N_CORRECTORS,
        n_outer_correctors=PIMPLE_N_OUTER_CORRECTORS,
        n_orthogonal_correctors=PIMPLE_N_ORTHOGONAL_CORRECTORS,
        alpha_u=0.7,
        alpha_p=0.3,
        ibm_forcing_loops=IBM_FORCING_LOOPS,
    ),
    samplers=FVM_SAMPLERS,
    transport=TransportConfig(density=RHO, nu=NU),
    turbulence=FVMTurbulenceConfig.none(),
    boundaries=[
        BoundaryConfig(
            name="numericalBoundary",
            type_velocity="fixedValue",
            value_velocity=list(FREESTREAM_VELOCITY),
            type_p="fixedFluxPressure",
        )
    ],
    initial_velocity=list(INITIAL_VELOCITY),
    initial_p=0.0,
)

# --------------------------------------------------------------------------- #
# Coupler configuration
# --------------------------------------------------------------------------- #
COUPLER_SETUP = CouplerSetup(
    freestream_velocity=list(FREESTREAM_VELOCITY),
    transfer_region_box=TRANSFER_REGION_BOX,
    bc_resync_after_transfer=True,
    pressure_anchor_to_freestream=False,
    coupler_backup_period=COUPLER_BACKUP_PERIOD,
    vpm_bc_mode=VPM_BC_MODE,
    vpm_particle_spacing=VPM_PARTICLE_SPACING,
    vpm_core_radius_ratio=VPM_CORE_RADIUS_RATIO,
    overlap_zone_ramp_width=OVERLAP_ZONE_RAMP_WIDTH,
    overlap_zone_dead_zone_width=0.0,
    transfer_prune_vorticity_min=TRANSFER_PRUNE_VORTICITY_MIN,
    transfer_boundary_prune_multiplier=TRANSFER_BOUNDARY_PRUNE_MULTIPLIER,
    transfer_max_particles=PARTICLE_LIMIT,
    transfer_amplification_cap=TRANSFER_AMPLIFICATION_CAP,
    transfer_diagnostic_interval=TRANSFER_DIAGNOSTIC_INTERVAL,
)


def _apply_seed(solver) -> None:
    """Impose the controlled, divergence-free perturbation on the initial field."""
    n_elements = solver.mesh_data["n_elements"]
    centroids = np.asarray(solver.geo_data["element_centroids"][:n_elements], dtype=np.float64)
    if centroids.shape[0] != n_elements:
        raise RuntimeError("Seed requires the full cell-centroid array on every rank")
    velocity = build_seed_velocity(
        centroids,
        base_velocity=INITIAL_VELOCITY,
        epsilon=SEED_AMPLITUDE,
        u_inf=U_INF,
        diameter=DIAMETER,
    )
    solver.set_initial_velocity(velocity)
    peak = float(np.linalg.norm(velocity - np.asarray(INITIAL_VELOCITY), axis=1).max())
    print(f"  Applied controlled seed: eps={SEED_AMPLITUDE:.3e}, max|u'|/Uinf={peak / U_INF:.3e}")


def make_vpm_setup():
    """Build the VPM setup on the MPI rank that owns the VPM solver."""
    from openonda.vpm import (
        AdvectionConfig,
        LineSampler,
        StabilizationConfig,
        StretchingConfig,
        SurfaceSampler,
        TurbulenceConfig,
        VelocityConfig,
        ViscousConfig,
        VPMSetup,
    )

    # Note: the VPM samplers live in the same ``samples/`` directory as the
    # FVM samplers, so they carry their own ``vpm_`` prefix in ``file_name``
    # (the post-processing "_plotutil" adds no extra prefix for the vpm source).
    samplers = (
        LineSampler(
            start=[VPM_DOMAIN[0], 0.0, 0.0],
            end=[VPM_DOMAIN[1], 0.0, 0.0],
            spacing=SAMPLE_SPACING,
            file_name="vpm_centerline",
        ),
        SurfaceSampler(
            point=[0.0, 0.0, 0.0],
            normal=[0, 0, 1],
            bounds=MIDSPAN_SLICE_BOUNDS,
            spacing=SAMPLE_SPACING,
            file_name="vpm_slice_z0",
            include_derivatives=False,
        ),
        SurfaceSampler(
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
    # surface.  Omitting it keeps the VPM-BC at the FVM boundary equal to
    # freestream + wake-particle induction, matching the reference far-field.
    return VPMSetup(
        time_step_size=DT_VPM,
        freestream_velocity=list(FREESTREAM_VELOCITY),
        viscous=ViscousConfig.gbd(
            particle_spacing=VPM_PARTICLE_SPACING,
            padding=3.0,
            viscosity=NU,
            threshold_mode="absolute",
            threshold=GBD_VORTICITY_FLOOR * VPM_PARTICLE_SPACING**3,
            max_nodes=PARTICLE_LIMIT,
            cap_abs_fraction=0.95,
            core_radius_ratio=VPM_CORE_RADIUS_RATIO,
        ),
        stretching=StretchingConfig.transposed(scheme=VPM_SCHEME),
        advection=AdvectionConfig(scheme=VPM_SCHEME),
        turbulence=TurbulenceConfig.inviscid(),
        velocity=VelocityConfig.treecode(theta=0.3, multipole_order=2),
        stabilization=StabilizationConfig.bounded_domain(VPM_DOMAIN),
        particles_kernel="GAUSSIAN",
        precision="f32",
        processing_unit="AUTO",
        max_particles=PARTICLE_LIMIT,
        max_targets=PARTICLE_LIMIT,
        vpm_domain_bounds=list(VPM_DOMAIN),
        log_mode="file",
        logging_frequency=VPM_LOG_PERIOD,
        timing_frequency=VPM_LOG_PERIOD,
        backup_frequency=int(CHECKPOINT_INTERVAL / DT_VPM),
        backup_directory=str(CASE_DIR / "solution"),
        export_flow_integrals=False,
        samplers=samplers,
    )


def main() -> None:
    print("\n===== SIMULATION (hybrid) =====")
    print(
        f"  Re={REYNOLDS}, infinite cylinder D={DIAMETER}, "
        f"FVM dt={DT_FVM}s / VPM dt={DT_VPM}s, "
        f"body/wake cell size={FVM_BODY_CELL_SIZE}/{FVM_WAKE_CELL_SIZE}, "
        f"particles<={PARTICLE_LIMIT}, seed={SEED_AMPLITUDE:g}"
    )
    fvm_solver = setup_fvm_solver(FVM_SETUP, case_dir=CASE_DIR, mesh=FVM_MESH)
    fvm_solver.set_immersed_bodies(CYLINDER, h=FVM_BODY_CELL_SIZE)
    if SEED_AMPLITUDE > 0.0:
        _apply_seed(fvm_solver)
    fvm_solver.write_vtk()

    is_master = FVMVPMCoupler.is_master_rank()
    vpm_solver = None
    if is_master:
        from openonda.vpm import setup_vpm_solver

        vpm_solver = setup_vpm_solver(make_vpm_setup())

    coupled_solver = setup_coupler(vpm_solver, fvm_solver, COUPLER_SETUP)
    max_steps = int(os.environ.get("OPENONDA_MAX_STEPS", "0"))
    if max_steps > 0:
        # The coupler derives its horizon from the FVM config's end_time during
        # initialize(), so cap that (the coupling step count follows).
        capped_t = min(T_END, max_steps * DT_VPM)
        FVM_SETUP.time.end_time = float(capped_t)
        print(f"  [probe] capping run to {max_steps} VPM steps (t={capped_t:g})")
    coupled_solver.run()
    if is_master:
        print("\n===== DONE =====")
        print("Simulation completed. Run ./allvalidate.sh for the instability analysis.")


if __name__ == "__main__":
    main()
