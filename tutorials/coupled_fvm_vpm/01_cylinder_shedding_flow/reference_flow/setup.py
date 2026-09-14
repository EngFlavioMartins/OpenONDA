#!/usr/bin/env python3
"""Body-fitted, quasi-2D cylinder flow at Re=150.

Usage:
    python -u setup.py --name DIRECTORY_NAME --dx WALL_CELL_SIZE

Example:
    python -u setup.py --name coarse --dx 0.125
"""

import argparse
import math
from pathlib import Path

import numpy as np

import openonda.fvm as fvm
import openonda.fvm.mesher as msh


def create_solver(directory_name: str, dx: float):
    case_dir = Path(__file__).resolve().parent
    domain = (-8.0, 24.0, -10.0, 10.0, -0.5, 0.5)
    velocity = [1.0, 0.0, 0.0]
    background_size = 8.0 * dx

    source_domain = (*domain[:4], -16.0 * dx, 16.0 * dx)
    span_levels = tuple(np.linspace(-0.5, 0.5, math.ceil(1.0 / (4.0 * dx)) + 1))
    patches = msh.BoxPatches(
        xmin="inlet",
        xmax="outlet",
        ymin="ymin",
        ymax="ymax",
        zmin="zmin",
        zmax="zmax",
    )
    mesh = msh.ExtrudedCartesianMesher(
        source=msh.CartesianMesher(
            domain=msh.BoxDomain(bounds=source_domain, patches=patches),
            surfaces=(msh.STLSurface(case_dir / "assets/cylinder_long.stl", patch="cylinder"),),
            max_cell_size=background_size,
            refinements=(
                # cfMesh treats box cell sizes as strict upper bounds.
                msh.BoxRefinement(
                    name="nearBody",
                    bounds=(-1.0, 2.0, -1.0, 1.0, source_domain[4], source_domain[5]),
                    cell_size=3.0 * dx,
                ),
                msh.BoxRefinement(
                    name="nearWake",
                    bounds=(0.0, 6.0, -1.0, 1.0, source_domain[4], source_domain[5]),
                    cell_size=3.0 * dx,
                ),
                msh.BoxRefinement(
                    name="wake",
                    bounds=(0.0, 12.0, -1.5, 1.5, source_domain[4], source_domain[5]),
                    cell_size=6.0 * dx,
                ),
            ),
            patch_refinements=(msh.PatchRefinement("cylinder", dx),),
            surface_may_cross_domain_boundary=True,
        ),
        domain=msh.BoxDomain(bounds=domain, patches=patches),
        levels=span_levels,
    )

    force_schedule = fvm.RunSchedule(every_time=0.02)
    line_schedule = fvm.RunSchedule(every_time=0.1)
    slice_schedule = fvm.RunSchedule(every_time=0.5)
    sample_spacing = min(0.125, 2.0 * dx)
    solver_setup = fvm.FVMSetup(
        case_name=directory_name,
        cores=6,
        mesh=fvm.MeshQualityConfig(
            max_non_orthogonality_deg=70.0,
            max_skewness=1.0,
            max_lsq_condition=9.0,
        ),
        execution=fvm.ComputeConfig(operator_backend="numba"),
        output=fvm.OutputConfig(
            compression="lz4",
            asynchronous=False,
            ghost_layers=0,
        ),
        logging=fvm.LoggingConfig(schedule=fvm.RunSchedule(every_time=0.1)),
        acceptance=fvm.RunAcceptanceLimits(
            max_continuity_error_warning=1.0e-4,
            max_continuity_error_abort=1.0e-2,
            max_equation_residual_warning=1.0e-4,
            max_equation_residual_abort=1.0e-2,
            max_courant_number_warning=0.9,
            max_courant_number_abort=1.5,
            max_velocity_magnitude_warning=3.0,
            max_velocity_magnitude_abort=5.0,
        ),
        backup=fvm.BackupConfig(
            schedule=fvm.RunSchedule(every_time=2.5),
            write_at_end=True,
        ),
        time=fvm.TimeConfig(
            time_step_size=0.001,
            end_time=60.0,
            output_schedule=fvm.RunSchedule(every_time=2.5),
            adjustment=fvm.MaximumCourantTimeStep(
                maximum=0.9,
                maximum_time_step_size=0.004,
            ),
        ),
        linear=fvm.LinearSolverConfig(
            pressure_solver="amg",
            pressure_tolerance=1.0e-7,
            pressure_relative_tolerance=0.005,
            momentum_tolerance=1.0e-6,
            momentum_relative_tolerance=0.05,
        ),
        pimple=fvm.PimpleControl(
            n_outer_correctors=2,
            velocity_relaxation=0.7,
            pressure_relaxation=0.3,
        ),
        samplers=(
            fvm.ForceSampler(
                patch_names=["cylinder"],
                reference_velocity=1.0,
                reference_area=domain[5] - domain[4],
                reference_length=1.0,
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
            *(
                fvm.LineSampler(
                    start=[x, -3.0, 0.0],
                    end=[x, 3.0, 0.0],
                    spacing=sample_spacing,
                    k=12,
                    reconstruction="affine",
                    file_name=f"transverse_x{x:g}",
                    schedule=line_schedule,
                )
                for x in (1.0, 2.0, 4.0)
            ),
            fvm.SurfaceSampler(
                point=[0.0, 0.0, 0.0],
                normal=[0.0, 0.0, 1.0],
                bounds=[domain[0], domain[1], domain[2], domain[3]],
                spacing=sample_spacing,
                k=12,
                reconstruction="affine",
                file_name="midspan",
                schedule=slice_schedule,
                body_bounds=[-0.5, 0.5, -0.5, 0.5, -6.0, 6.0],
                body_geometry="cylinder_z",
            ),
        ),
        transport=fvm.TransportConfig(
            density=1.0,
            kinematic_viscosity=1.0 / 150.0,
        ),
        turbulence=fvm.TurbulenceConfig.none(),
        boundaries=[
            fvm.BoundaryConfig.inlet("inlet", velocity),
            fvm.BoundaryConfig.outlet("outlet", kinematic_pressure=0.0),
            fvm.BoundaryConfig.slip("ymin"),
            fvm.BoundaryConfig.slip("ymax"),
            fvm.BoundaryConfig.slip("zmin"),
            fvm.BoundaryConfig.slip("zmax"),
            fvm.BoundaryConfig.wall("cylinder"),
        ],
        initial_velocity=velocity,
    )

    return fvm.create_fvm_solver(
        solver_setup,
        case_dir=case_dir,
        solution_dir=case_dir / "solution" / directory_name,
        samples_dir=case_dir / "samples" / directory_name,
        mesh=mesh,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="medium")
    parser.add_argument("--dx", default=0.04, type=float)
    arguments = parser.parse_args()
    with create_solver(arguments.name, arguments.dx) as solver:
        solver.run()
        fvm.update_grid_study(solver, arguments.dx, profiles=("centreline",))
