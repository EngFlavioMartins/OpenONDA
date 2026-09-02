#!/usr/bin/env python3
"""Body-fitted Re=150 cylinder reference used by the grid study.

Example:
    python -u setup.py --dx 0.0417 \
        --case-name coarse
"""

from __future__ import annotations

import argparse
from pathlib import Path

import openonda.fvm as fvm

CASE_DIR = Path(__file__).resolve().parent
CYLINDER_STL = CASE_DIR.parent / "assets" / "cylinder_spanwise.stl"

# ---- Physics -------------------------------------------------------------
DIAMETER = 1.0
REYNOLDS_NUMBER = 150.0
FREESTREAM_VELOCITY = [1.0, 0.0, 0.0]
KINEMATIC_VISCOSITY = 1.0 / REYNOLDS_NUMBER
DOMAIN = (-8.0, 20.0, -8.0, 8.0, -0.6, 0.6)
# The checked-in surface deliberately extends beyond the finite computational
# span, so the side wall crosses both z boundaries without STL end caps.
CYLINDER_LENGTH = DOMAIN[5] - DOMAIN[4]

# ---- Time and output -----------------------------------------------------
TIME_STEP_SIZE = 0.001
MAX_TIME_STEP_SIZE = 4.0 * TIME_STEP_SIZE
MAX_COURANT_NUMBER = 0.9
# Discard the start-up transient and analyse the final 30 convective units.
TOTAL_TIME = 60.0
FORCE_INTERVAL_TIME = 0.02
LINE_INTERVAL_TIME = 0.1
SLICE_INTERVAL_TIME = 0.5
FIELD_INTERVAL_TIME = 2.5
NUMBER_OF_CORES = 6


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dx", required=True, type=float, help="cylinder-wall cell size in D")
    parser.add_argument("--case-name", required=True, help="solution/ and samples/ subdirectory")
    return parser.parse_args()


def grid_mesh(dx: float) -> fvm.CartesianMesher:
    """Return a declarative grid-study mesh at requested wall size ``dx``."""
    if dx <= 0.0:
        raise ValueError("--dx must be positive")
    return fvm.CartesianMesher(
        domain=fvm.BoxDomain(
            bounds=DOMAIN,
            patches=fvm.BoxPatches("inlet", "outlet", "ymin", "ymax", "zmin", "zmax"),
        ),
        surfaces=(fvm.STLSurface(CYLINDER_STL, patch="cylinder"),),
        # Keep the background lattice fixed across the grid study. The two
        # declarative refinement zones and wall request vary with ``dx``.
        max_cell_size=0.5,
        boundary_cell_size=dx,
        min_cell_size=dx,
        refinements=(
            fvm.BoxRefinement(
                name="near_body",
                bounds=(-2.0, 6.0, -2.0, 2.0, -0.5, 0.5),
                cell_size=2.0 * dx,
            ),
            fvm.BoxRefinement(
                name="wake",
                bounds=(-4.0, 12.0, -4.0, 4.0, -0.5, 0.5),
                cell_size=4.0 * dx,
            ),
        ),
        boundary_layers=(
            fvm.BoundaryLayers(
                patches=("cylinder",),
                layers=10,
                first_cell_height=dx / 16.0,
                growth_ratio=1.18,
            ),
        ),
        # The body is deliberately longer than the finite reference span, so
        # the generic surface/domain intersection keeps it continuous through
        # both spanwise boundaries.
        surface_may_cross_domain_boundary=True,
    )


def samplers(dx: float) -> tuple:
    force_schedule = fvm.RunSchedule(every_time=FORCE_INTERVAL_TIME)
    line_schedule = fvm.RunSchedule(every_time=LINE_INTERVAL_TIME)
    slice_schedule = fvm.RunSchedule(every_time=SLICE_INTERVAL_TIME)
    sample_spacing = min(0.125, 2.0 * dx)
    return (
        fvm.ForceSampler(
            patch_names=["cylinder"],
            reference_velocity=1.0,
            reference_area=DIAMETER * CYLINDER_LENGTH,
            reference_length=DIAMETER,
            moment_centre=[0.0, 0.0, 0.0],
            file_name="forces_history",
            schedule=force_schedule,
        ),
        fvm.LineSampler(
            start=[1.5, 0.0, 0.0],
            end=[1.5, 0.0, 0.0],
            n_points=1,
            k=12,
            reconstruction="affine",
            file_name="midspan_probe",
            schedule=force_schedule,
        ),
        fvm.LineSampler(
            start=[-2.0, 0.0, 0.0],
            end=[12.0, 0.0, 0.0],
            spacing=sample_spacing,
            k=12,
            reconstruction="affine",
            file_name="centreline",
            schedule=line_schedule,
        ),
        fvm.LineSampler(
            start=[1.0, -3.0, 0.0],
            end=[1.0, 3.0, 0.0],
            spacing=sample_spacing,
            k=12,
            reconstruction="affine",
            file_name="transverse_x1",
            schedule=line_schedule,
        ),
        fvm.LineSampler(
            start=[2.0, -3.0, 0.0],
            end=[2.0, 3.0, 0.0],
            spacing=sample_spacing,
            k=12,
            reconstruction="affine",
            file_name="transverse_x2",
            schedule=line_schedule,
        ),
        fvm.LineSampler(
            start=[4.0, -3.0, 0.0],
            end=[4.0, 3.0, 0.0],
            spacing=sample_spacing,
            k=12,
            reconstruction="affine",
            file_name="transverse_x4",
            schedule=line_schedule,
        ),
        fvm.SurfaceSampler(
            point=[0.0, 0.0, 0.0],
            normal=[0.0, 0.0, 1.0],
            bounds=[DOMAIN[0], DOMAIN[1], DOMAIN[2], DOMAIN[3]],
            spacing=sample_spacing,
            k=12,
            reconstruction="affine",
            file_name="midspan",
            schedule=slice_schedule,
            body_bounds=[-0.5, 0.5, -0.5, 0.5, -6.0, 6.0],
            body_geometry="cylinder_z",
        ),
    )


def solver_setup(case_name: str, dx: float) -> fvm.FVMSetup:
    return fvm.FVMSetup(
        case_name=case_name,
        cores=NUMBER_OF_CORES,
        mesh=fvm.MeshQualityConfig(
            # The native patch-normal layer collar is intentionally graded
            # independently of the Cartesian core. These limits describe the
            # bounded reference mesh contract; the full report remains
            # available in mesh_generation.cartesian_report.
            max_non_orthogonality_deg=90.0,
            max_skewness=8.0,
            max_lsq_condition=5.5,
        ),
        execution=fvm.ComputeConfig(operator_backend="numba"),
        output=fvm.OutputConfig(
            format="vtk_xml",
            data_location="cell",
            encoding="appended",
            compression="lz4",
            precision="f32",
            asynchronous=False,
            ghost_layers=0,
        ),
        logging=fvm.LoggingConfig(schedule=fvm.RunSchedule(every_time=0.1)),
        acceptance=fvm.RunAcceptanceLimits(
            sustained_steps=1,
            max_continuity_error_warning=1.0e-4,
            max_continuity_error_abort=1.0e-2,
            max_equation_residual_warning=1.0e-4,
            max_equation_residual_abort=1.0e-2,
            max_courant_number_warning=MAX_COURANT_NUMBER,
            max_courant_number_abort=1.5,
            max_velocity_magnitude_warning=3.0,
            max_velocity_magnitude_abort=5.0,
        ),
        backup=fvm.BackupConfig(
            schedule=fvm.RunSchedule(every_time=FIELD_INTERVAL_TIME),
            write_at_end=True,
        ),
        time=fvm.TimeConfig(
            time_step_size=TIME_STEP_SIZE,
            start_time=0.0,
            end_time=TOTAL_TIME,
            output_schedule=fvm.RunSchedule(every_time=FIELD_INTERVAL_TIME),
            adjustment=fvm.MaximumCourantTimeStep(
                maximum=MAX_COURANT_NUMBER,
                maximum_time_step_size=MAX_TIME_STEP_SIZE,
            ),
        ),
        schemes=fvm.DiscretizationConfig(
            convection_scheme="limitedLinear",
            gradient_scheme="lsq",
            time_scheme="euler",
        ),
        linear=fvm.LinearSolverConfig(
            linear_solver="bicgstab",
            pressure_solver="amg",
            pressure_tolerance=1.0e-7,
            pressure_relative_tolerance=0.005,
            momentum_tolerance=1.0e-6,
            momentum_relative_tolerance=0.05,
        ),
        pimple=fvm.PimpleControl(
            n_correctors=2,
            n_outer_correctors=2,
            # The Rhie--Chow assembly already includes the explicit
            # non-orthogonal pressure flux.
            n_orthogonal_correctors=0,
            velocity_relaxation=0.7,
            pressure_relaxation=0.3,
        ),
        samplers=samplers(dx),
        transport=fvm.TransportConfig(
            density=1.0,
            kinematic_viscosity=KINEMATIC_VISCOSITY,
        ),
        turbulence=fvm.TurbulenceConfig.none(),
        boundaries=[
            fvm.BoundaryConfig.inlet("inlet", FREESTREAM_VELOCITY),
            fvm.BoundaryConfig.outlet("outlet", kinematic_pressure=0.0),
            fvm.BoundaryConfig.slip("ymin"),
            fvm.BoundaryConfig.slip("ymax"),
            fvm.BoundaryConfig.slip("zmin"),
            fvm.BoundaryConfig.slip("zmax"),
            fvm.BoundaryConfig.wall("cylinder"),
            # The finite-span surface has a sharp rim.  The generic layer
            # block exposes its rim closure as a separate physical wall patch
            # so it is explicit in the solver boundary contract.
            fvm.BoundaryConfig.wall("layer_termination"),
        ],
        initial_velocity=FREESTREAM_VELOCITY,
        initial_kinematic_pressure=0.0,
    )


def main() -> None:
    arguments = parse_arguments()
    solution_dir = CASE_DIR / "solution" / arguments.case_name
    samples_dir = CASE_DIR / "samples" / arguments.case_name
    solver = fvm.create_fvm_solver(
        solver_setup(arguments.case_name, arguments.dx),
        case_dir=CASE_DIR,
        solution_dir=solution_dir,
        samples_dir=samples_dir,
        mesh=grid_mesh(arguments.dx),
    )
    try:
        solver.write_run_manifest()
        solver.run()
    finally:
        solver.close()


if __name__ == "__main__":
    main()
