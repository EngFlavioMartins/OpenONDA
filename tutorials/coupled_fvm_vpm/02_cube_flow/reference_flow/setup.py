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


def create_solver(
    directory_name: str,
    dx: float,
    *,
    campaign: bool = False,
    output_root: Path | None = None,
    cores: int = 4,
    end_time: float = 30.0,
    max_dt: float = 0.04,
    courant: float = 0.9,
    lean: bool = False,
    output_interval: float | None = None,
    backup_interval: float | None = None,
    restart_from: Path | None = None,
    mesh: Path | None = None,
) -> fvm.FVMSolver:
    """Construct the Re=1000 cube with wall spacing ``dx`` in metres.

    ``campaign`` selects the fixed-domain geometric family. ``max_dt`` is the
    timestep ceiling in seconds and ``courant`` is its CFL limit. ``lean`` saves
    initial/final fields and five-second checkpoints. ``output_interval`` and
    ``backup_interval`` independently override those schedules in seconds.
    Visualization frames remain a series; checkpoints retain the two latest
    committed states. Force and profile sampling keep their own schedules.
    An explicit native mesh
    avoids remeshing; restart always uses the case's own saved mesh. The FVM
    factory owns mesh construction, parallel execution and output validation.
    """
    case_dir = Path(__file__).resolve().parent
    domain = (
        (-6.48, 12.96, -6.48, 6.48, -6.48, 6.48) if campaign else (-6.5, 13.0, -6.5, 6.5, -6.5, 6.5)
    )
    output_root = case_dir if output_root is None else Path(output_root).resolve()
    native_mesh = (
        output_root / "solution" / directory_name / "fvm" / "mesh.npz"
        if restart_from is not None
        else mesh
    )
    velocity = [1.0, 0.0, 0.0]
    patches = msh.BoxPatches(
        xmin="inlet",
        xmax="outlet",
        ymin="ymin",
        ymax="ymax",
        zmin="zmin",
        zmax="zmax",
    )
    generated_mesh = msh.CartesianMesher(
        domain=msh.BoxDomain(bounds=domain, patches=patches),
        surfaces=(msh.STLSurface(case_dir / "assets/cube.stl", patch="cube"),),
        max_cell_size=(8 if campaign else 12) * dx,
        cell_size_anchor=dx,
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
    solution_schedule = (
        fvm.RunSchedule(final_only=True) if lean else fvm.RunSchedule(every_time=0.5)
    )
    if output_interval is not None:
        solution_schedule = fvm.RunSchedule(every_time=output_interval)
    backup_schedule = fvm.RunSchedule(
        every_time=backup_interval if backup_interval is not None else (5.0 if lean else 0.5)
    )
    sample_spacing = 0.06 if campaign else min(0.125, 2.0 * dx)
    solver_setup = fvm.FVMSetup(
        case_name=directory_name,
        cores=cores,
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
            schedule=backup_schedule,
            write_at_end=True,
        ),
        time=fvm.TimeConfig(
            time_step_size=min(0.01, max_dt),
            end_time=end_time,
            output_schedule=solution_schedule,
            adjustment=fvm.MaximumCourantTimeStep(
                maximum=courant,
                maximum_time_step_size=max_dt,
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
        solution_dir=output_root / "solution" / directory_name,
        samples_dir=output_root / "samples" / directory_name,
        mesh=native_mesh if native_mesh is not None else generated_mesh,
        require_empty_output=restart_from is None,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", dest="directory_name", default="fine")
    parser.add_argument("--dx", default=0.06, type=float)
    parser.add_argument(
        "--campaign", action="store_true", help="Use the fixed-domain geometric family"
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--end-time", type=float, default=30.0)
    parser.add_argument("--max-dt", type=float, default=0.04)
    parser.add_argument("--courant", type=float, default=0.9)
    parser.add_argument(
        "--lean", action="store_true", help="Initial/final fields and rolling 5 s backups"
    )
    parser.add_argument(
        "--output-interval", type=float, help="Visualization interval in seconds; overrides --lean"
    )
    parser.add_argument(
        "--backup-interval", type=float, help="Rolling checkpoint interval in seconds"
    )
    parser.add_argument("--restart-from", type=Path)
    parser.add_argument("--mesh", type=Path, help="Reuse this exact native mesh for a fresh run")
    arguments = vars(parser.parse_args())
    if arguments["campaign"]:
        from studies.panel_removal.cube_reference_campaign import run_campaign_level

        run_campaign_level(create_solver, case_dir=Path(__file__).resolve().parent, **arguments)
    else:
        with create_solver(**arguments) as solver:
            if arguments["restart_from"]:
                solver.load_state(arguments["restart_from"])
            solver.run()
            fvm.update_grid_study(solver, arguments["dx"], profiles=("centreline", "offaxis_y075"))
