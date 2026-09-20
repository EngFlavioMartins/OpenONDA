#!/usr/bin/env python3
"""Body-fitted, quasi-2D cylinder flow at Re=150.

The diameter is 1 m, freestream speed is 1 m/s and viscosity is 1/150 m²/s.
Requested XY spacing, uniform span layers and adaptive time-step cap are
independent physical-resolution controls.

Example:
    python setup.py --name xy_fine --dx 0.04 --span-layers 4 --end-time 100

To continue an existing run from its latest backup, add
``--restart-from solution/DIRECTORY_NAME/backup`` and set a later ``--end-time``.
"""

import argparse
from pathlib import Path

import numpy as np

import openonda.fvm as fvm
import openonda.fvm.mesher as msh

DIAMETER = 1.0  # m
FREESTREAM_SPEED = 1.0  # m/s
REYNOLDS_NUMBER = 150.0
KINEMATIC_VISCOSITY = FREESTREAM_SPEED * DIAMETER / REYNOLDS_NUMBER  # m^2/s


def create_solver(
    directory_name: str,
    dx: float,
    end_time: float = 100.0,
    restart_from: Path | None = None,
    *,
    output_root: Path | None = None,
    cores: int = 6,
    span: float = 1.0,
    span_layers: int | None = None,
    maximum_time_step: float = 0.004,
    lean: bool = False,
    output_interval: float | None = None,
    backup_interval: float | None = None,
) -> fvm.FVMSolver:
    """Configure a body-fitted cylinder with a slip-bounded span.

    Parameters
    ----------
    directory_name : str
        Case name below the solution and samples directories.
    dx : float
        Requested near-body mesh spacing in m; the nominal lattice is 0.75*dx.
    end_time : float
        Final physical time in s.
    restart_from : pathlib.Path or None
        Saved FVM checkpoint; reuse its cached native mesh when available.
    output_root : pathlib.Path or None
        Separate campaign output root; None uses this tutorial directory.
    cores : int
        Number of solver processes.
    span : float
        Extruded span in m, with slip end planes and no physical endcaps.
    span_layers : int or None
        Number of uniform span layers; None uses ceil(span/(4*dx)).
    maximum_time_step : float
        Maximum adaptive time step in s, with maximum Courant number 0.9.
    lean : bool
        Retain force/line samples and 10 s checkpoints; omit surface series and
        volume output unless output_interval is explicitly supplied.
    output_interval : float or None
        Volume visualization interval in s. An explicit interval enables volume
        snapshots even with lean output. None retains the usual 2.5 s schedule.
    backup_interval : float or None
        Restart interval in s. None retains 10 s for lean output, otherwise 2.5 s.

    Returns
    -------
    openonda.fvm.FVMSolver
        Constructed solver. Its context manager owns output closure.
    """
    case_dir = Path(__file__).resolve().parent
    if span_layers is None:
        span_layers = int(np.ceil(span / (4.0 * dx)))
    destination = case_dir if output_root is None else Path(output_root).resolve()
    domain = (-8.0, 24.0, -10.0, 10.0, -0.5 * span, 0.5 * span)
    velocity = [FREESTREAM_SPEED, 0.0, 0.0]
    background_size = 12.0 * dx

    source_domain = (*domain[:4], -16.0 * dx, 16.0 * dx)
    span_levels = tuple(np.linspace(domain[4], domain[5], span_layers + 1))
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
                    bounds=(-1.5, 1.5, -1.5, 1.5, source_domain[4], source_domain[5]),
                    cell_size=dx,
                ),
                msh.BoxRefinement(
                    name="nearWake",
                    bounds=(-2.0, 6.0, -2.0, 2.0, source_domain[4], source_domain[5]),
                    cell_size=2.0 * dx,
                ),
                msh.BoxRefinement(
                    name="wake",
                    bounds=(-2.5, 12.0, -2.5, 2.5, source_domain[4], source_domain[5]),
                    cell_size=4.0 * dx,
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
    visualization_interval = 2.5 if output_interval is None else output_interval
    checkpoint_interval = (10.0 if lean else 2.5) if backup_interval is None else backup_interval
    # A common profile sampling lattice prevents sampling changes masquerading as grid error.
    sample_spacing = 0.08 if lean else min(0.125, 2.0 * dx)
    solver_setup = fvm.FVMSetup(
        case_name=directory_name,
        cores=cores,
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
            schedule=fvm.RunSchedule(every_time=checkpoint_interval),
            write_at_end=True,
        ),
        time=fvm.TimeConfig(
            time_step_size=min(0.001, maximum_time_step),
            end_time=end_time,
            output_schedule=fvm.RunSchedule(every_time=visualization_interval),
            adjustment=fvm.MaximumCourantTimeStep(
                maximum=0.9,
                maximum_time_step_size=maximum_time_step,
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
            fvm.LineSampler(
                start=[1.5, 0.0, -0.45 * span],
                end=[1.5, 0.0, 0.45 * span],
                n_points=9,
                # Observe raw cells in an XY stack: affine XY support changes
                # with z and can manufacture a spanwise variation.
                k=1,
                reconstruction="idw",
                file_name="span_probe",
                schedule=line_schedule,
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
            kinematic_viscosity=KINEMATIC_VISCOSITY,
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

    if lean:
        solver_setup.samplers = tuple(
            sampler
            for sampler in solver_setup.samplers
            if not isinstance(sampler, fvm.SurfaceSampler)
        )
    mesh_source = mesh
    cached_mesh = destination / "solution" / directory_name / "fvm" / "mesh.npz"
    if restart_from is not None and cached_mesh.is_file():
        mesh_source = cached_mesh

    solver = fvm.create_fvm_solver(
        solver_setup,
        case_dir=case_dir,
        solution_dir=destination / "solution" / directory_name,
        samples_dir=destination / "samples" / directory_name,
        mesh=mesh_source,
        require_empty_output=output_root is not None and restart_from is None,
    )
    if lean:
        solver.auto_write = output_interval is not None
    return solver


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="medium")
    parser.add_argument("--dx", default=0.04, type=float)
    parser.add_argument("--end-time", default=100.0, type=float)
    parser.add_argument("--restart-from", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--cores", type=int, default=6)
    parser.add_argument("--span", type=float, default=1.0)
    parser.add_argument("--span-layers", type=int)
    parser.add_argument("--maximum-time-step", type=float, default=0.004)
    parser.add_argument(
        "--output-interval",
        type=float,
        help="Volume snapshot interval [s]; enables volume output with --lean",
    )
    parser.add_argument("--backup-interval", type=float, help="Restart checkpoint interval [s]")
    parser.add_argument(
        "--lean",
        action="store_true",
        help="Keep forces/profiles/checkpoints; omit surfaces and volumes unless --output-interval is set",
    )
    arguments = parser.parse_args()
    with create_solver(
        arguments.name,
        arguments.dx,
        end_time=arguments.end_time,
        restart_from=arguments.restart_from,
        output_root=arguments.output_root,
        cores=arguments.cores,
        span=arguments.span,
        span_layers=arguments.span_layers,
        maximum_time_step=arguments.maximum_time_step,
        lean=arguments.lean,
        output_interval=arguments.output_interval,
        backup_interval=arguments.backup_interval,
    ) as solver:
        if arguments.restart_from is not None:
            solver.load_state(arguments.restart_from)
        solver.run()
        fvm.update_grid_study(solver, arguments.dx, profiles=("centreline",))
