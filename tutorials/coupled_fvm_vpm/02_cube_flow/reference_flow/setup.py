#!/usr/bin/env python3
"""Body-fitted cube flow at Re=1000.

Usage:
    python -u setup.py --name DIRECTORY_NAME --dx WALL_SIZE_TARGET

Example:
    python -u setup.py --name coarse --dx 0.125
"""

import argparse
from pathlib import Path

import openonda.fvm as fvm
import openonda.fvm.mesher as msh


def create_solver(directory_name: str, dx: float):
    case_dir = Path(__file__).resolve().parent
    domain = (-6.5, 13.0, -6.5, 6.5, -6.5, 6.5)
    velocity = [1.0, 0.0, 0.0]
    patches = msh.BoxPatches(
        xmin="inlet",
        xmax="outlet",
        ymin="ymin",
        ymax="ymax",
        zmin="zmin",
        zmax="zmax",
    )
    mesh = msh.CartesianMesher(
        domain=msh.BoxDomain(bounds=domain, patches=patches),
        surfaces=(msh.STLSurface(case_dir / "assets/cube.stl", patch="cube"),),
        max_cell_size=12 * dx,
        refinements=(
            # cfMesh treats box cell sizes as strict upper bounds.
            msh.BoxRefinement(
                "nearBody",
                (-1.5, 3.0, -1.5, 1.5, -1.5, 1.5),
                dx,
            ),
            msh.BoxRefinement(
                "wake",
                (-2.0, 8.0, -2.0, 2.0, -2.0, 2.0),
                2.0 * dx,
            ),
        ),
        patch_refinements=(msh.PatchRefinement("cube", dx),),
    )

    force_schedule = fvm.RunSchedule(every_time=0.05)
    line_schedule = fvm.RunSchedule(every_time=0.25)
    sample_spacing = min(0.125, 2.0 * dx)
    solver_setup = fvm.FVMSetup(
        case_name=directory_name,
        cores=4,
        mesh=fvm.MeshQualityConfig(
            max_non_orthogonality_deg=70.0,
            max_skewness=1.0,
        ),
        execution=fvm.ComputeConfig(operator_backend="numba"),
        output=fvm.OutputConfig(
            compression="lz4",
            asynchronous=False,
            ghost_layers=0,
        ),
        logging=fvm.LoggingConfig(schedule=fvm.RunSchedule(every_time=0.25)),
        acceptance=fvm.RunAcceptanceLimits(
            max_continuity_error_warning=1.0e-4,
            max_continuity_error_abort=1.0e-2,
            max_equation_residual_warning=1.0e-4,
            max_equation_residual_abort=1.0e-2,
            max_courant_number_warning=0.9,
            max_courant_number_abort=1.5,
            max_velocity_magnitude_warning=4.0,
            max_velocity_magnitude_abort=6.0,
        ),
        backup=fvm.BackupConfig(
            schedule=fvm.RunSchedule(every_time=1.0),
            write_at_end=True,
        ),
        time=fvm.TimeConfig(
            time_step_size=0.01,
            end_time=30.0,
            output_schedule=fvm.RunSchedule(every_time=1.0),
            adjustment=fvm.MaximumCourantTimeStep(
                maximum=0.9,
                maximum_time_step_size=0.04,
            ),
        ),
        schemes=fvm.DiscretizationConfig(
            convection_scheme="linearUpwind",
            gradient_scheme="gauss",
            time_scheme="backward",
        ),
        linear=fvm.LinearSolverConfig(
            pressure_solver="amg",
            pressure_tolerance=1.0e-6,
            pressure_relative_tolerance=0.01,
            momentum_tolerance=1.0e-6,
            momentum_relative_tolerance=0.1,
            momentum_max_iterations=2000,
        ),
        pimple=fvm.PimpleControl(
            n_outer_correctors=2,
            n_nonorthogonal_correctors=1,
            velocity_relaxation=0.7,
            pressure_relaxation=0.3,
        ),
        samplers=(
            fvm.ForceSampler(
                patch_names=["cube"],
                reference_velocity=1.0,
                reference_area=1.0,
                reference_length=1.0,
                schedule=force_schedule,
            ),
            fvm.LineSampler(
                start=[domain[0], 0.0, 0.0],
                end=[domain[1], 0.0, 0.0],
                spacing=sample_spacing,
                k=12,
                reconstruction="affine",
                file_name="centreline",
                schedule=line_schedule,
            ),
            fvm.LineSampler(
                start=[domain[0], 0.75, 0.0],
                end=[domain[1], 0.75, 0.0],
                spacing=sample_spacing,
                k=12,
                reconstruction="affine",
                file_name="offaxis_y075",
                schedule=line_schedule,
            ),
        ),
        transport=fvm.TransportConfig(
            density=1.0,
            kinematic_viscosity=1.0 / 1000.0,
        ),
        turbulence=fvm.TurbulenceConfig.equilibrium_smagorinsky(
            subgrid_kinetic_energy_coefficient=0.094,
            subgrid_dissipation_coefficient=1.048,
        ),
        boundaries=[
            fvm.BoundaryConfig.inlet("inlet", velocity),
            fvm.BoundaryConfig.outlet("outlet", kinematic_pressure=0.0),
            fvm.BoundaryConfig.slip("ymin"),
            fvm.BoundaryConfig.slip("ymax"),
            fvm.BoundaryConfig.slip("zmin"),
            fvm.BoundaryConfig.slip("zmax"),
            fvm.BoundaryConfig.wall("cube"),
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
    parser.add_argument("--name", default="fine")
    parser.add_argument("--dx", default=0.06, type=float)
    arguments = parser.parse_args()
    with create_solver(arguments.name, arguments.dx) as solver:
        solver.run()
        fvm.update_grid_study(
            solver,
            arguments.dx,
            profiles=("centreline", "offaxis_y075"),
        )
