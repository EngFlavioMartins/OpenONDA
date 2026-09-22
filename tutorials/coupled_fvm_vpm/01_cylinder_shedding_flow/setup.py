#!/usr/bin/env python3
"""Coupled FVM–VPM flow past a spanwise cylinder section at Re = 150.

The body-fitted FVM resolves the cylinder and a compact near-body box. The
VPM carries vorticity through the outer domain. The FVM force is normalized
by the resolved span.

Usage:
    ./allrun.sh
"""

from pathlib import Path
from dataclasses import replace
import math

import numpy as np

import openonda.coupler as coupling
from openonda.cylinder_case import (
    DEFAULT_CYLINDER_CASE,
    normalize_coupled_overrides,
    validate_coupled_geometry,
    validate_cylinder_authority,
)
import openonda.fvm as fvm
import openonda.fvm.mesher as msh
import openonda.vpm as vpm
from openonda.vpm import Backup, Samplers

# Physical problem
START_FROM = "latest"  # Resume the latest backup; ./allrun.sh cleans first.

CASE_NAME = "coupled_cylinder_flow"
DIAMETER = DEFAULT_CYLINDER_CASE.diameter
FREESTREAM_VELOCITY = (1.0, 0.0, 0.0)
DENSITY = 1.0
REYNOLDS_NUMBER = DEFAULT_CYLINDER_CASE.reynolds_number
KINEMATIC_VISCOSITY = np.linalg.norm(FREESTREAM_VELOCITY) * DIAMETER / REYNOLDS_NUMBER

# FVM domain and mesh
FVM_CORES = 4
PIMPLE_CORRECTORS = 2
CELL_SIZE = DEFAULT_CYLINDER_CASE.coupled_wall_spacing
FVM_RESOLVED_SPAN = DEFAULT_CYLINDER_CASE.resolved_span
REFERENCE_AREA = DIAMETER * FVM_RESOLVED_SPAN
FVM_HALF_SPAN = 0.5 * FVM_RESOLVED_SPAN
FVM_BOX = (
    -1.60,
    1.60,
    -1.60,
    1.60,
    -FVM_HALF_SPAN,
    FVM_HALF_SPAN,
)
TRANSFER_REGION_BOX = (
    -1.25,
    1.25,
    -1.25,
    1.25,
    -FVM_HALF_SPAN,
    FVM_HALF_SPAN,
)

# VPM domain and resolution. Choose a spanwise lattice spacing that closes the
# physical slab exactly; the slab induction wrapper supplies the free-slip
# images at its two span boundaries.
VPM_DOMAIN = (*(-5.0, 15.0, -5.0, 5.0), -FVM_HALF_SPAN, FVM_HALF_SPAN)
SPANWISE_LAYERS = math.ceil(FVM_RESOLVED_SPAN / 0.05)
VPM_PARTICLE_SPACING = FVM_RESOLVED_SPAN / SPANWISE_LAYERS
# Keep renewal and grid-based diffusion on one VPM lattice.
GBD_GRID_SPACING = VPM_PARTICLE_SPACING
SAMPLE_SPACING = 2.0 * CELL_SIZE
PARTICLE_LIMIT = 1_000_000

# Coupling
BOUNDARY_CONDITION_MODE = "vorticity_mixed"
TRANSFER_METHOD = "buffered_m4_renewal"
TRANSFER_VORTICITY_CUTOFF = 0.05
TRANSFER_AMPLIFICATION_CAP = 1.8
FVM_CONSISTENCY_WIDTH = 0.0
INTERFACE_ITERATIONS = 3
INTERFACE_TOLERANCE = 1.0e-5

# Time and output
FVM_TIME_STEP_SIZE = 0.004
VPM_TIME_STEP_MULTIPLIER = 10
VPM_TIME_STEP_SIZE = VPM_TIME_STEP_MULTIPLIER * FVM_TIME_STEP_SIZE
END_TIME = DEFAULT_CYLINDER_CASE.coupled_end_time
SAMPLING_INTERVAL_TIME = 0.2
SLICE_INTERVAL_TIME = 0.24
OUTPUT_INTERVAL_TIME = 0.24
# Every schedule must land on a 0.04 s accepted coupling boundary while
# interface iteration is active.
BACKUP_INTERVAL_TIME = 0.24
GBD_VORTICITY_FLOOR = 0.01

FVM_OUTPUT_INTERVAL_STEPS = round(OUTPUT_INTERVAL_TIME / FVM_TIME_STEP_SIZE)
COUPLED_BACKUP_INTERVAL_STEPS = round(BACKUP_INTERVAL_TIME / VPM_TIME_STEP_SIZE)
FVM_SAMPLING_INTERVAL_STEPS = round(SAMPLING_INTERVAL_TIME / FVM_TIME_STEP_SIZE)
VPM_SAMPLING_INTERVAL_STEPS = round(SAMPLING_INTERVAL_TIME / VPM_TIME_STEP_SIZE)
FVM_SLICE_INTERVAL_STEPS = round(SLICE_INTERVAL_TIME / FVM_TIME_STEP_SIZE)
VPM_SLICE_INTERVAL_STEPS = round(SLICE_INTERVAL_TIME / VPM_TIME_STEP_SIZE)
TRANSFER_DIAGNOSTIC_INTERVAL_STEPS = round(2.4 / VPM_TIME_STEP_SIZE)

# Case files and derived sampling data
CASE_DIR = Path(__file__).resolve().parent
CYLINDER_STL = CASE_DIR / "assets" / "cylinder_long.stl"

FVM_PATCHES = msh.BoxPatches(
    xmin="numericalBoundary",
    xmax="numericalBoundary",
    ymin="numericalBoundary",
    ymax="numericalBoundary",
    zmin="zmin",
    zmax="zmax",
)
FVM_MESH = msh.CartesianMesher(
    domain=msh.BoxDomain(bounds=FVM_BOX, patches=FVM_PATCHES),
    surfaces=(msh.STLSurface(CYLINDER_STL, patch="cylinder"),),
    max_cell_size=CELL_SIZE,
    cell_size_anchor=CELL_SIZE,
    surface_may_cross_domain_boundary=True,
)

FVM_SAMPLING_SCHEDULE = fvm.RunSchedule(every_n_steps=FVM_SAMPLING_INTERVAL_STEPS)
FVM_SLICE_SCHEDULE = fvm.RunSchedule(every_n_steps=FVM_SLICE_INTERVAL_STEPS)
VPM_SAMPLING_SCHEDULE = vpm.EverySteps(VPM_SAMPLING_INTERVAL_STEPS)
VPM_SLICE_SCHEDULE = vpm.EverySteps(VPM_SLICE_INTERVAL_STEPS)

FVM_SAMPLERS = (
    fvm.ForceSampler(
        patch_names=["cylinder"],
        reference_velocity=np.linalg.norm(FREESTREAM_VELOCITY),
        reference_area=REFERENCE_AREA,
        reference_length=DIAMETER,
        moment_centre=[0.0, 0.0, 0.0],
        file_name="forces_history",
        schedule=FVM_SAMPLING_SCHEDULE,
    ),
    fvm.LineSampler(
        start=[FVM_BOX[0], 0.0, 0.0],
        end=[FVM_BOX[1], 0.0, 0.0],
        spacing=SAMPLE_SPACING,
        file_name="fvm_centreline",
        schedule=FVM_SAMPLING_SCHEDULE,
    ),
    fvm.LineSampler(
        start=[1.0, FVM_BOX[2], 0.0],
        end=[1.0, FVM_BOX[3], 0.0],
        spacing=SAMPLE_SPACING,
        file_name="fvm_transverse_x1",
        schedule=FVM_SAMPLING_SCHEDULE,
    ),
    fvm.SurfaceSampler(
        point=[0.0, 0.0, 0.0],
        normal=[0.0, 0.0, 1.0],
        bounds=list(FVM_BOX[:4]),
        spacing=SAMPLE_SPACING,
        file_name="fvm_midspan",
        schedule=FVM_SLICE_SCHEDULE,
        body_bounds=[-0.5, 0.5, -0.5, 0.5, -6.0, 6.0],
        body_geometry="cylinder_z",
    ),
)

FVM_SETUP = fvm.FVMSetup(
    case_name=CASE_NAME,
    cores=FVM_CORES,
    execution=fvm.ComputeConfig(operator_backend="numba"),
    output=fvm.OutputConfig(compression="lz4", asynchronous=True, ghost_layers=0),
    time=fvm.TimeConfig(
        time_step_size=FVM_TIME_STEP_SIZE,
        end_time=END_TIME,
        # Retained fields and the coupled checkpoint use the same accepted times.
        output_schedule=fvm.RunSchedule(every_n_steps=FVM_OUTPUT_INTERVAL_STEPS),
    ),
    schemes=fvm.DiscretizationConfig(
        convection_scheme="limitedLinear",
        gradient_scheme="lsq",
        time_scheme="euler_implicit",
    ),
    linear=fvm.LinearSolverConfig(
        pressure_solver="amg",
        pressure_tolerance=1.0e-7,
        pressure_relative_tolerance=0.005,
        momentum_tolerance=1.0e-6,
        momentum_relative_tolerance=0.05,
    ),
    pimple=fvm.PimpleControl(
        n_correctors=PIMPLE_CORRECTORS,
        n_outer_correctors=2,
        velocity_relaxation=0.7,
        pressure_relaxation=0.3,
    ),
    samplers=FVM_SAMPLERS,
    transport=fvm.TransportConfig(
        density=DENSITY,
        kinematic_viscosity=KINEMATIC_VISCOSITY,
    ),
    turbulence=fvm.TurbulenceConfig.none(),
    boundaries=[
        fvm.BoundaryConfig(
            name="numericalBoundary",
            velocity_type="fixedValue",
            velocity_value=list(FREESTREAM_VELOCITY),
            pressure_type="fixedFluxPressure",
        ),
        fvm.BoundaryConfig.slip("zmin"),
        fvm.BoundaryConfig.slip("zmax"),
        fvm.BoundaryConfig.wall("cylinder"),
    ],
    initial_velocity=list(FREESTREAM_VELOCITY),
    initial_kinematic_pressure=0.0,
)

COUPLER_SETUP = coupling.CouplerSetup(
    freestream_velocity=list(FREESTREAM_VELOCITY),
    transfer_method=TRANSFER_METHOD,
    transfer_region_bounds=TRANSFER_REGION_BOX,
    backup_interval_steps=COUPLED_BACKUP_INTERVAL_STEPS,
    boundary_condition_mode=BOUNDARY_CONDITION_MODE,
    fvm_consistency_width=FVM_CONSISTENCY_WIDTH,
    interface_iterations=INTERFACE_ITERATIONS,
    interface_normal_tolerance=INTERFACE_TOLERANCE,
    interface_gradient_tolerance=INTERFACE_TOLERANCE,
    eta_blend_width=6.0 * VPM_PARTICLE_SPACING,
    vpm_only_width=2.0 * VPM_PARTICLE_SPACING,
    transfer_vorticity_cutoff=TRANSFER_VORTICITY_CUTOFF,
    transfer_amplification_cap=TRANSFER_AMPLIFICATION_CAP,
    transfer_diagnostic_interval_steps=TRANSFER_DIAGNOSTIC_INTERVAL_STEPS,
)

VPM_SAMPLERS = (
    # Keep VPM diagnostics on exterior transverse lines; FVM owns the body
    # centreline.
    vpm.LineSampler(
        start=[1.0, -3.0, 0.0],
        end=[1.0, 3.0, 0.0],
        spacing=SAMPLE_SPACING,
        file_name="vpm_transverse_x1",
        schedule=VPM_SAMPLING_SCHEDULE,
    ),
    vpm.LineSampler(
        start=[2.0, -3.0, 0.0],
        end=[2.0, 3.0, 0.0],
        spacing=SAMPLE_SPACING,
        file_name="vpm_transverse_x2",
        schedule=VPM_SAMPLING_SCHEDULE,
    ),
    vpm.LineSampler(
        start=[4.0, -3.0, 0.0],
        end=[4.0, 3.0, 0.0],
        spacing=SAMPLE_SPACING,
        file_name="vpm_transverse_x4",
        schedule=VPM_SAMPLING_SCHEDULE,
    ),
)

VPM_CASE = vpm.VPMCase(
    name=CASE_NAME,
    numerics=vpm.Numerics(
        time_step_size=VPM_TIME_STEP_SIZE,
        freestream_velocity=FREESTREAM_VELOCITY,
        viscous=vpm.ViscousConfig.gbd(
            particle_spacing=VPM_PARTICLE_SPACING,
            gbd_grid_spacing=GBD_GRID_SPACING,
            padding=5.0,
            kinematic_viscosity=KINEMATIC_VISCOSITY,
            threshold_mode="absolute",
            threshold=GBD_VORTICITY_FLOOR * VPM_PARTICLE_SPACING**3,
            max_nodes=PARTICLE_LIMIT,
            core_radius_ratio=1.0,
        ),
        integrator=vpm.RK2(),
        turbulence=vpm.TurbulenceConfig.inviscid(),
        induction=vpm.SlipSlabInduction(
            vpm.FMMInduction(),
            z_min=-FVM_HALF_SPAN,
            z_max=FVM_HALF_SPAN,
            tail_tolerance=1.0e-4,
            max_shells=129,
        ),
        stabilization=vpm.StabilizationConfig.bounded_domain(VPM_DOMAIN),
        particle_kernel="GAUSSIAN",
        precision="f32",
        compute_device="AUTO",
        max_n_particles=PARTICLE_LIMIT,
        max_evaluation_points=min(32_768, PARTICLE_LIMIT),
        domain_bounds=VPM_DOMAIN,
        write_precision="f32",
    ),
    # Coupler backups contain the FVM state, VPM state, and boundary history
    # atomically; independent native backups would not be restart-consistent.
    backup=Backup(interval_steps=0, directory="solution", log_directory="solution"),
    samplers=Samplers(samples=VPM_SAMPLERS),
    run=vpm.RunPlan(steps=round(END_TIME / VPM_TIME_STEP_SIZE)),
    directory=CASE_DIR,
)


def build_case(
    *,
    end_time: float | None = None,
    overrides: dict[str, object] | None = None,
):
    """Construct one resolved coupled case without creating solver state."""
    values = normalize_coupled_overrides(overrides)
    hxy = float(values.pop("hxy", CELL_SIZE))
    span = float(values.pop("span", FVM_RESOLVED_SPAN))
    dz_target = float(values.pop("dz", hxy))
    hp_ratio = float(values.pop("particle_spacing_ratio", 1.25))
    core_ratio = float(values.pop("core_radius_ratio", 1.0))
    blend_ratio = float(values.pop("blend_width_ratio", 6.0))
    release_ratio = float(values.pop("release_width_ratio", 2.0))
    exchange_dt = float(values.pop("exchange_dt", VPM_TIME_STEP_SIZE))
    cores = int(values.pop("cores", FVM_CORES))
    compute_device = str(values.pop("compute_device", "AUTO"))
    particle_limit = int(values.pop("particle_limit", PARTICLE_LIMIT))
    physical_end = END_TIME if end_time is None else float(end_time)
    validate_coupled_geometry(
        hxy=hxy,
        span=span,
        dz=dz_target,
        particle_spacing_ratio=hp_ratio,
        core_radius_ratio=core_ratio,
        blend_width_ratio=blend_ratio,
        release_width_ratio=release_ratio,
        exchange_dt=exchange_dt,
        cores=cores,
        particle_limit=particle_limit,
        end_time=physical_end,
        fvm_time_step=FVM_TIME_STEP_SIZE,
    )
    total_exchanges = round(physical_end / exchange_dt)
    half_span = span / 2.0
    fvm_box = (*FVM_BOX[:4], -half_span, half_span)
    transfer_box = (*TRANSFER_REGION_BOX[:4], -half_span, half_span)
    # The slip-slab GBD stencil needs six physical axial intervals. Refine the
    # requested spacing when a short span would otherwise under-resolve it.
    particle_layers = max(6, math.ceil(span / (hxy * hp_ratio)))
    particle_spacing = span / particle_layers
    validate_cylinder_authority(
        transfer_edge_x=transfer_box[1],
        radius=DIAMETER / 2.0,
        blend_width=blend_ratio * particle_spacing,
    )
    axial_layers = max(4, math.ceil(span / dz_target))
    realized_dz = span / axial_layers
    vpm_domain = (*VPM_DOMAIN[:4], -half_span, half_span)

    mesh_changed = any(name in (overrides or {}) for name in ("hxy", "span", "dz"))
    if not mesh_changed:
        mesh = FVM_MESH
    else:
        # CartesianMesher quantizes box planes to its root lattice; even when
        # dz == hxy, that can move the physical slip planes (e.g. ±0.48 to
        # ±0.512 at hxy=0.064). Extrusion keeps the requested span exact.
        # Its x/y planes must match the *resolved* source section perimeter;
        # projecting a quantized outer strip inward can fold adjacent cells.
        source_half_span = 16.0 * hxy
        source_mesh = msh.CartesianMesher(
            domain=msh.BoxDomain(
                bounds=(*fvm_box[:4], -source_half_span, source_half_span),
                patches=FVM_PATCHES,
            ),
            surfaces=(msh.STLSurface(CYLINDER_STL, patch="cylinder"),),
            max_cell_size=hxy,
            cell_size_anchor=hxy,
            patch_refinements=(msh.PatchRefinement("cylinder", hxy),),
            surface_may_cross_domain_boundary=True,
        )
        mesh = msh.ExtrudedCartesianMesher(
            source=source_mesh,
            domain=msh.BoxDomain(
                bounds=(*source_mesh.domain.bounds[:4], -half_span, half_span),
                patches=FVM_PATCHES,
            ),
            levels=tuple(-half_span + layer * realized_dz for layer in range(axial_layers + 1)),
        )

    desired_sample_steps = SAMPLING_INTERVAL_TIME / exchange_dt
    sample_steps = min(
        (step for step in range(1, total_exchanges + 1) if total_exchanges % step == 0),
        key=lambda step: (abs(step - desired_sample_steps), step),
    )
    sample_every_fvm = sample_steps * round(exchange_dt / FVM_TIME_STEP_SIZE)
    slice_steps = max(1, math.ceil(SLICE_INTERVAL_TIME / exchange_dt))
    slice_every_fvm = slice_steps * round(exchange_dt / FVM_TIME_STEP_SIZE)
    output_steps = max(1, math.ceil(OUTPUT_INTERVAL_TIME / exchange_dt))
    output_every_fvm = output_steps * round(exchange_dt / FVM_TIME_STEP_SIZE)
    force_schedule = fvm.RunSchedule(every_n_steps=sample_every_fvm)
    slice_schedule = fvm.RunSchedule(every_n_steps=slice_every_fvm)
    force_sampler = fvm.ForceSampler(
        patch_names=["cylinder"],
        reference_velocity=np.linalg.norm(FREESTREAM_VELOCITY),
        reference_area=DIAMETER * span,
        reference_length=DIAMETER,
        moment_centre=[0.0, 0.0, 0.0],
        file_name="forces_history",
        schedule=force_schedule,
    )
    span_samplers = tuple(
        fvm.LineSampler(
            start=[1.0, -1.0, fraction * span],
            end=[1.0, 1.0, fraction * span],
            spacing=2.0 * hxy,
            file_name=f"span_{label}",
            schedule=force_schedule,
        )
        for label, fraction in (("lower", -0.25), ("middle", 0.0), ("upper", 0.25))
    )
    fvm_samplers = (
        force_sampler,
        fvm.LineSampler(
            start=[fvm_box[0], 0.0, 0.0],
            end=[fvm_box[1], 0.0, 0.0],
            spacing=2.0 * hxy,
            file_name="fvm_centreline",
            schedule=force_schedule,
        ),
        fvm.LineSampler(
            start=[1.0, fvm_box[2], 0.0],
            end=[1.0, fvm_box[3], 0.0],
            spacing=2.0 * hxy,
            file_name="fvm_transverse_x1",
            schedule=force_schedule,
        ),
        fvm.SurfaceSampler(
            point=[0.0, 0.0, 0.0],
            normal=[0.0, 0.0, 1.0],
            bounds=list(fvm_box[:4]),
            spacing=2.0 * hxy,
            file_name="fvm_midspan",
            schedule=slice_schedule,
            body_bounds=[-0.5, 0.5, -0.5, 0.5, -6.0, 6.0],
            body_geometry="cylinder_z",
        ),
        *span_samplers,
    )
    fvm_setup = replace(
        FVM_SETUP,
        cores=cores,
        samplers=fvm_samplers,
        time=replace(
            FVM_SETUP.time,
            end_time=physical_end,
            output_schedule=fvm.RunSchedule(every_n_steps=output_every_fvm),
        ),
    )
    viscous = replace(
        VPM_CASE.numerics.viscous,
        particle_spacing=particle_spacing,
        gbd_grid_spacing=particle_spacing,
        gbd_threshold=GBD_VORTICITY_FLOOR * particle_spacing**3,
        gbd_max_nodes=particle_limit,
        core_radius_ratio=core_ratio,
    )
    numerics = replace(
        VPM_CASE.numerics,
        time_step_size=exchange_dt,
        viscous=viscous,
        induction=vpm.SlipSlabInduction(
            vpm.FMMInduction(),
            z_min=-half_span,
            z_max=half_span,
            tail_tolerance=1.0e-4,
            max_shells=129,
        ),
        stabilization=vpm.StabilizationConfig.bounded_domain(vpm_domain),
        compute_device=compute_device,
        max_n_particles=particle_limit,
        max_evaluation_points=min(32_768, particle_limit),
        domain_bounds=vpm_domain,
    )
    vpm_case = replace(
        VPM_CASE,
        numerics=numerics,
        run=vpm.RunPlan(steps=round(physical_end / exchange_dt)),
        samplers=Samplers(
            samples=tuple(
                vpm.LineSampler(
                    start=sample.start,
                    end=sample.end,
                    spacing=2.0 * hxy,
                    file_name=sample.file_name,
                    schedule=vpm.EverySteps(sample_steps),
                )
                for sample in VPM_SAMPLERS
            )
        ),
    )
    coupler_setup = replace(
        COUPLER_SETUP,
        transfer_region_bounds=transfer_box,
        eta_blend_width=blend_ratio * particle_spacing,
        vpm_only_width=release_ratio * particle_spacing,
        backup_interval_steps=max(1, round(BACKUP_INTERVAL_TIME / exchange_dt)),
        transfer_diagnostic_interval_steps=max(1, round(2.4 / exchange_dt)),
    )
    if "transfer_region_scale" in values:
        scale = values.pop("transfer_region_scale")
        coupler_setup = replace(
            coupler_setup,
            transfer_region_bounds=tuple(value * scale for value in transfer_box),
        )
    if values:
        coupler_setup = replace(coupler_setup, **values)
    return fvm_setup, vpm_case, coupler_setup, mesh


def create_solver(
    *,
    output_root: Path | None = None,
    end_time: float | None = None,
    restart_from: Path | None = None,
    max_coupling_steps: int | None = None,
    overrides: dict[str, object] | None = None,
) -> int:
    """Run the coupled case in an optional isolated campaign directory."""
    fvm_setup, vpm_case, coupler_setup, mesh = build_case(end_time=end_time, overrides=overrides)
    with coupling.create_coupler(
        fvm_setup,
        vpm_case,
        coupler_setup,
        mesh=mesh,
        case_dir=output_root,
    ) as solver:
        if restart_from is None:
            from openonda.cylinder_campaign import initialize_cylinder_perturbation

            solver.initialize()
            initialize_cylinder_perturbation(
                solver.fvm_solver,
                vpm_case.numerics.induction.z_max - vpm_case.numerics.induction.z_min,
            )
        return solver.run(
            restart_from=restart_from,
            start_from=START_FROM if restart_from is None else None,
            max_coupling_steps=max_coupling_steps,
            backup_at_stop=max_coupling_steps is not None,
        )


def main() -> int:
    create_solver()
    return 0


if __name__ == "__main__":
    main()
