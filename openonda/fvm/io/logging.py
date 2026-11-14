"""Log records of the FVM solver: the quantities of a step, the reports of a run.

The solver writes through a :class:`~openonda.core.log_style.StepLog`. The
functions here turn the diagnostics of the FVM into its sections.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

from openonda.core import log_style
from openonda.core.profiling import Profiler

if TYPE_CHECKING:
    from ..core.solver import FVMSolver

_MIB = 1024.0 * 1024.0

_RESIDUAL_LABELS = {
    "velocity": "residual, velocity",
    "velocity_increment": "residual, velocity increment",
    "velocity_x": "residual, velocity x",
    "velocity_y": "residual, velocity y",
    "velocity_z": "residual, velocity z",
    "kinematic_pressure": "residual, pressure",
    "initial_kinematic_pressure": "residual, pressure initial",
}
_RESIDUAL_ORDER = (
    "velocity",
    "velocity_increment",
    "velocity_x",
    "velocity_y",
    "velocity_z",
    "kinematic_pressure",
    "initial_kinematic_pressure",
)


def standalone_report(topic: str, *rows: log_style.Row, warning: bool = False) -> None:
    """Print a mesh or linear-algebra report made outside a solver.

    Solver code reports through its :class:`~openonda.core.log_style.StepLog`;
    this function gives library operations without a solver the same layout.
    """
    print(
        log_style.block_section(topic, rows), file=sys.stderr if warning else sys.stdout, flush=True
    )


def _record(log: log_style.StepLog, title: str, rows: list[log_style.Row]) -> None:
    """Add a section to the open step; outside a step only the debug mode writes it."""
    if rows and log.enabled and (log.in_step or log.debug_enabled):
        log.section(title, *rows)


def record_convergence(log: log_style.StepLog, residuals: dict[str, float]) -> None:
    """Record the nonlinear and linear residuals; the simple layout shows velocity and pressure."""
    if log.debug_enabled:
        keys = [key for key in _RESIDUAL_ORDER if key in residuals]
        keys.extend(key for key in residuals if key not in keys)
        rows = [(_RESIDUAL_LABELS.get(key, key), f"{float(residuals[key]):.3e}") for key in keys]
    else:
        rows = [
            (_RESIDUAL_LABELS[key], f"{float(residuals[key]):.2e}")
            for key in ("velocity", "kinematic_pressure")
            if key in residuals
        ]
    _record(log, "convergence", rows)


def record_continuity(log: log_style.StepLog, maximum: float, total: float) -> None:
    """Record the global mass-conservation errors."""
    if log.debug_enabled:
        rows: list[log_style.Row] = [
            ("continuity error, max", f"{maximum:.3e}", "1/s"),
            ("boundary imbalance", f"{total:.3e}", "m^3/s"),
        ]
    else:
        rows = [("continuity error, max", f"{maximum:.2e}", "1/s")]
    _record(log, "conservation", rows)


def record_courant(
    log: log_style.StepLog, time_step_size: float, maximum: float, target: float | None
) -> None:
    """Record the time step in s and the global Courant number."""
    detailed = log.debug_enabled
    rows: list[log_style.Row] = [
        ("time step", f"{time_step_size:.3e}", "s"),
        ("courant, max", f"{maximum:.3e}" if detailed else f"{maximum:.3f}"),
    ]
    if detailed and target is not None:
        rows.append(("courant target", f"{target:.3e}"))
    _record(log, "time control", rows)


def record_y_plus(log: log_style.StepLog, statistics: dict[str, dict[str, float]]) -> None:
    """Record the wall resolution of every wall patch; debug mode only."""
    if not log.debug_enabled:
        return
    rows: list[log_style.Row] = []
    for name, stats in statistics.items():
        rows.extend(
            (
                (f"{name}, y+ min", f"{stats['min']:.3e}"),
                (f"{name}, y+ mean", f"{stats['avg']:.3e}"),
                (f"{name}, y+ max", f"{stats['max']:.3e}"),
            )
        )
    _record(log, "wall resolution", rows)


def record_forces(log: log_style.StepLog, forces: dict[str, Any]) -> None:
    """Record the force coefficients of every patch; the simple layout shows the drag."""
    rows: list[log_style.Row] = []
    for patch, values in forces.items():
        coefficients = values.get("coeffs", {})
        rows.append((f"{patch}, drag", f"{float(coefficients.get('drag_coefficient', 0.0)):.4f}"))
        if log.debug_enabled:
            rows.extend(
                (
                    (f"{patch}, lift", f"{float(coefficients.get('lift_coefficient', 0.0)):.4f}"),
                    (
                        f"{patch}, pitching moment",
                        f"{float(coefficients.get('pitching_moment_coefficient', 0.0)):.4f}",
                    ),
                )
            )
    _record(log, "aerodynamic loads", rows)


def record_turbulence(
    log: log_style.StepLog,
    minimum: float,
    maximum: float,
    mean: float,
    kinematic_viscosity: float,
) -> None:
    """Record the eddy viscosity in m^2/s next to the molecular viscosity in m^2/s."""
    if not log.debug_enabled:
        rows: list[log_style.Row] = [
            ("eddy viscosity, min", float(minimum), "m^2/s"),
            ("eddy viscosity, mean", float(mean), "m^2/s"),
            ("eddy viscosity, max", float(maximum), "m^2/s"),
        ]
    else:
        ratio_min = minimum / kinematic_viscosity if kinematic_viscosity > 0 else float("inf")
        ratio_max = maximum / kinematic_viscosity if kinematic_viscosity > 0 else float("inf")
        rows = [
            ("eddy viscosity, min", f"{minimum:.3e}", "m^2/s"),
            ("eddy viscosity, mean", f"{mean:.3e}", "m^2/s"),
            ("eddy viscosity, max", f"{maximum:.3e}", "m^2/s"),
            ("viscosity ratio, min", f"{ratio_min:.3e}"),
            ("viscosity ratio, max", f"{ratio_max:.3e}"),
        ]
    _record(log, "turbulence", rows)


def end_step(log: log_style.StepLog, elapsed: float, *, accepted: bool) -> bool:
    """Add the step timing, close the step and publish it when it is due.

    Returns
    -------
    bool
        Whether the step was published.
    """
    if accepted and log.in_step:
        rows: list[log_style.Row] = [("step wall time", f"{elapsed:.2f}", "s")]
        if log.debug_enabled:
            rows.append(("cumulative wall time", f"{log.wall_time + elapsed:.3e}", "s"))
        log.section("timing", *rows)
    return log.end_step(elapsed, accepted=accepted)


def record_profile(log: log_style.StepLog, record: dict[str, Any]) -> None:
    """Record the section times of a profiled, published step; debug mode only."""
    if not log.debug_enabled:
        return
    rows: list[log_style.Row] = [("sections, max over ranks:", "")]
    for name, values in record["sections"].items():
        seconds = values["seconds"]
        rows.append((f"  {name}", f"{seconds['max']:.3f}", "s"))
        if "fraction_of_step" in values:
            rows.append((f"  {name}, share", f"{100.0 * values['fraction_of_step']:.1f}", "%"))
    for name, values in record["details"].items():
        rows.append((f"  {name}, max", f"{values['max']:.0f}"))
    memory = record["memory"]
    rows.extend(
        (
            ("memory:", ""),
            (
                "  resident, all ranks",
                f"{memory['total_resident_end_bytes'] / _MIB:.1f}",
                "MiB",
            ),
            ("  slowest rank", f"{record['step_seconds']['max']:.3f}", "s"),
        )
    )
    log.section("performance profile", *rows)


def close_log(
    log: log_style.StepLog,
    *,
    status: str,
    failure: BaseException | None = None,
    profiler: Profiler | None = None,
) -> None:
    """Write the run summary and the runtime profile, then close the log. Idempotent."""
    if log.closed:
        return
    log.abandon_step()
    if log.enabled:
        mean = log.wall_time / log.n_steps if log.n_steps else 0.0
        rows: list[log_style.Row] = [
            ("status", str(status)),
            ("steps", f"{log.n_steps:,}"),
            ("wall time, total", f"{log.wall_time:.3e}", "s"),
            ("wall time, mean per step", f"{mean:.3f}", "s"),
        ]
        if failure is not None:
            rows.append(("error", f"{type(failure).__name__}: {failure}"))
        log.section(f"run {status}", *rows)
        if profiler is not None and profiler.enabled:
            log.report(profiler.report("FVM runtime profile"))
    log.close()


def configuration_report(
    solver: FVMSolver, initialization_seconds: float
) -> log_style.FormattedText:
    """Return the solver configuration report written after construction.

    Parameters
    ----------
    solver : FVMSolver
        The constructed solver.
    initialization_seconds : float
        Wall time of the construction, in s.
    """
    config = solver._configuration
    numerics = config.numerics
    mesh = solver.mesh_data
    parallel = solver.parallel
    partition = getattr(parallel, "partition", None)

    sections: list[tuple[str, list[log_style.Row]]] = [
        (
            "RUN",
            [
                ("case", str(config.case_name)),
                ("execution mode", str(numerics.execution.parallel_mode)),
                ("mpi ranks", str(parallel.size)),
                ("operator backend", str(numerics.execution.operator_backend)),
                ("linear backend", str(numerics.execution.linear_backend)),
            ],
        )
    ]

    if partition is None:
        mesh_items: list[log_style.Row] = [
            ("cells", f"{int(mesh['n_cells']):,}"),
            ("faces", f"{int(mesh['n_faces']):,}"),
            ("faces, interior", f"{int(mesh['n_interior_faces']):,}"),
            ("boundary patches", str(len(mesh["boundary"]))),
        ]
    else:
        mesh_items = [
            ("cells, global", f"{int(partition.n_global_cells):,}"),
            ("cells, rank 0 owned", f"{int(parallel.n_owned):,}"),
            ("faces, rank 0 local", f"{int(mesh['n_faces']):,}"),
            ("patches, configured", str(len(config.boundaries))),
        ]
    sections.append(("MESH", mesh_items))

    linear_method = numerics.linear.linear_solver
    pressure_method = numerics.linear.pressure_solver or linear_method
    authored = solver.case.numerics.linear
    replaced = [
        equation
        for equation, method, authored_method in (
            ("momentum", linear_method, authored.linear_solver),
            ("pressure", pressure_method, authored.pressure_solver or authored.linear_solver),
        )
        if authored_method != method
    ]
    sections.append(
        (
            "NUMERICS",
            [
                ("time step", f"{numerics.time_step_size:.3e}", "s"),
                ("end time", f"{config.run.end_time:.3e}", "s"),
                ("time scheme", str(numerics.schemes.time_scheme)),
                ("convection scheme", str(numerics.schemes.convection_scheme)),
                ("gradient scheme", str(numerics.schemes.gradient_scheme)),
                ("momentum solver", str(linear_method)),
                ("pressure solver", str(pressure_method)),
                *(
                    [
                        (
                            "direct solves",
                            f"{' and '.join(replaced)} by BiCGSTAB or AMG to 1e-10 under MPI",
                        )
                    ]
                    if replaced
                    else []
                ),
                ("correctors", str(numerics.pimple.n_correctors)),
                ("outer correctors", str(numerics.pimple.n_outer_correctors)),
            ],
        )
    )
    turbulence_name = numerics.turbulence.model
    sections.append(
        (
            "PHYSICS",
            [
                ("density", f"{numerics.transport.density:.6g}", "kg/m^3"),
                (
                    "kinematic viscosity",
                    f"{numerics.transport.kinematic_viscosity:.6e}",
                    "m^2/s",
                ),
                ("turbulence model", turbulence_name),
            ],
        )
    )
    boundary_items: list[log_style.Row] = []
    for boundary in config.boundaries:
        boundary_items.extend(
            (
                (f"{boundary.name}, velocity", str(boundary.velocity_type)),
                (f"{boundary.name}, pressure", str(boundary.pressure_type)),
                (f"{boundary.name}, value", str(boundary.velocity_value)),
            )
        )
    sections.append(("BOUNDARY CONDITIONS", boundary_items))
    sections.append(
        (
            "MONITORING AND OUTPUT",
            [
                ("solution directory", str(solver.solution_dir)),
                ("log file", str(solver.logger.path)),
                ("log mode", config.logging.mode),
                ("log interval", config.logging.schedule.describe()),
                ("visualization", "VTK XML, cell-centred, appended binary"),
                ("output compression", str(config.output.compression).upper()),
                (
                    "output scheduling",
                    "asynchronous"
                    if config.output.asynchronous and not parallel.is_partitioned
                    else "synchronous",
                ),
                (
                    "visualization cadence",
                    (
                        config.output.schedule.describe()
                        if config.output.schedule is not None
                        else "initial and final"
                    ),
                ),
                (
                    "restart cadence",
                    (
                        config.backup.schedule.describe()
                        if config.backup.schedule is not None
                        else "disabled"
                    ),
                ),
                ("visualization ghost layers", str(config.output.ghost_layers)),
                ("initialization time", f"{initialization_seconds:.3e}", "s"),
            ],
        )
    )
    return log_style.block_report("FVM SOLVER CONFIGURATION", sections)
