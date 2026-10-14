"""Log records of the FVM solver, written through the ``openonda.fvm`` logger."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path
import sys
from typing import Any

from openonda.core import log_style

LOGGER_NAME = "openonda.fvm"
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

    Solver code reports through its :class:`Logging` instance; this function
    gives library operations without a solver the same layout.
    """
    print(
        log_style.block_section(topic, rows), file=sys.stderr if warning else sys.stdout, flush=True
    )


@dataclass
class _StepRecord:
    """Raw per-step diagnostics, rendered by the simple or debug layout."""

    step: int
    time: float
    time_step_size: float
    residuals: dict[str, float] = field(default_factory=dict)
    max_continuity_error: float | None = None
    sum_absolute_continuity_error: float | None = None
    courant: float | None = None
    courant_target: float | None = None
    y_plus: dict[str, dict[str, float]] = field(default_factory=dict)
    forces: dict[str, Any] = field(default_factory=dict)
    turbulence: tuple[float, float, float] | None = None
    kinematic_viscosity: float = 0.0
    warnings: tuple[str, ...] = ()
    events: list[tuple[str, tuple[log_style.Row, ...]]] = field(default_factory=list)
    sections: list[tuple[str, list[log_style.Row]]] = field(default_factory=list)
    wall_time_at_start: float = 0.0
    elapsed: float = 0.0

    def drag(self) -> float | None:
        """Total drag coefficient over all patches."""
        total = 0.0
        found = False
        for values in self.forces.values():
            total += float(values.get("coeffs", {}).get("drag_coefficient", 0.0))
            found = True
        return total if found else None


def _sections(record: _StepRecord) -> list[tuple[str, list[log_style.Row]]]:
    """Return the populated diagnostic groups of one step record."""
    sections: list[tuple[str, list[log_style.Row]]] = []

    if record.residuals:
        keys = [key for key in _RESIDUAL_ORDER if key in record.residuals]
        keys.extend(key for key in record.residuals if key not in keys)
        sections.append(
            (
                "convergence",
                [
                    (_RESIDUAL_LABELS.get(key, key), f"{float(record.residuals[key]):.3e}")
                    for key in keys
                ],
            )
        )

    if record.max_continuity_error is not None:
        sections.append(
            (
                "conservation",
                [
                    ("continuity error, max", f"{record.max_continuity_error:.3e}", "1/s"),
                    (
                        "boundary imbalance",
                        f"{record.sum_absolute_continuity_error:.3e}",
                        "m^3/s",
                    ),
                ],
            )
        )

    if record.courant is not None:
        rows: list[log_style.Row] = [("courant, max", f"{record.courant:.3e}")]
        if record.courant_target is not None:
            rows.append(("courant target", f"{record.courant_target:.3e}"))
        sections.append(("time control", rows))

    if record.y_plus:
        wall_rows: list[log_style.Row] = []
        for name, stats in record.y_plus.items():
            wall_rows.extend(
                (
                    (f"{name}, y+ min", f"{stats['min']:.3e}"),
                    (f"{name}, y+ mean", f"{stats['avg']:.3e}"),
                    (f"{name}, y+ max", f"{stats['max']:.3e}"),
                )
            )
        sections.append(("wall resolution", wall_rows))

    loads: list[log_style.Row] = []
    for patch, values in record.forces.items():
        coefficients = values.get("coeffs", {})
        loads.extend(
            (
                (f"{patch}, drag", f"{float(coefficients.get('drag_coefficient', 0.0)):.4f}"),
                (f"{patch}, lift", f"{float(coefficients.get('lift_coefficient', 0.0)):.4f}"),
                (
                    f"{patch}, pitching moment",
                    f"{float(coefficients.get('pitching_moment_coefficient', 0.0)):.4f}",
                ),
            )
        )
    if loads:
        sections.append(("aerodynamic loads", loads))

    if record.turbulence is not None:
        minimum, maximum, mean = record.turbulence
        kinematic_viscosity = record.kinematic_viscosity
        ratio_min = minimum / kinematic_viscosity if kinematic_viscosity > 0 else float("inf")
        ratio_max = maximum / kinematic_viscosity if kinematic_viscosity > 0 else float("inf")
        sections.append(
            (
                "turbulence",
                [
                    ("eddy viscosity, min", f"{minimum:.3e}", "m^2/s"),
                    ("eddy viscosity, mean", f"{mean:.3e}", "m^2/s"),
                    ("eddy viscosity, max", f"{maximum:.3e}", "m^2/s"),
                    ("viscosity ratio, min", f"{ratio_min:.3e}"),
                    ("viscosity ratio, max", f"{ratio_max:.3e}"),
                ],
            )
        )

    if record.warnings:
        sections.append(
            (
                "warnings",
                [(f"({index + 1})", text) for index, text in enumerate(record.warnings)],
            )
        )

    return sections


def _event_rows(record: _StepRecord) -> list[log_style.Row]:
    """Flatten event records into rows for the current block's event section."""
    rows: list[log_style.Row] = []
    for topic, details in record.events:
        rows.append((topic, ""))
        for detail in details:
            label = f"  {detail[0]}"
            if len(detail) > 2:
                rows.append((label, detail[1], detail[2]))
            else:
                rows.append((label, detail[1]))
    return rows


class Logging:
    """Step records and reports of one FVM solver.

    Rank 0 writes through the standard-library logger ``openonda.fvm`` to
    ``solution/fvm.log`` and, optionally, the console; other ranks hold a
    disabled instance, so callers need no rank checks. ``simple`` mode writes
    a compact block per logged step; ``debug`` mode the full diagnostics.

    Parameters
    ----------
    path : str or pathlib.Path
        Log file, opened for appending.
    config : LoggingConfig
        Mode, schedule and console switch.
    enabled : bool, default=True
        Write records; ``False`` on the ranks other than 0.
    """

    def __init__(self, path: str | Path, config: Any, *, enabled: bool = True) -> None:
        self.enabled = bool(enabled)
        self.mode = str(config.mode)
        self.schedule = config.schedule
        self.path = Path(path)
        self._closed = False
        self._step: _StepRecord | None = None
        self._step_reported = True
        self._collect_routine = True
        self._step_wall_time = 0.0
        self._steps = 0
        self._count_label = "steps"
        self._logger = logging.getLogger(LOGGER_NAME)
        if self.enabled:
            log_style.configure_logging(LOGGER_NAME, self.path, console=bool(config.console))

    @property
    def debug_mode(self) -> bool:
        """Whether the full step diagnostics are logged."""
        return self.mode == "debug"

    def _emit(self, text: str) -> None:
        if self.enabled and not self._closed:
            self._logger.info(text)

    def info(self, text: str) -> None:
        """Log an event; during a step it joins the step record."""
        self.event(text)

    def warning(self, text: str) -> None:
        """Log a warning immediately, also on steps that are not logged."""
        if not self.enabled or self._closed:
            return
        rows: list[log_style.Row] = [(text, "")]
        if self._step is not None:
            rows.insert(0, ("FVM step", self._step.step))
        self._logger.warning(log_style.block_section("warnings", rows))

    def event(self, topic: str, *rows: log_style.Row) -> None:
        """Log an event with optional detail rows, collected into a running step."""
        if not self.enabled:
            return
        if self._step is not None:
            if self._collect_routine:
                self._step.events.append((topic, rows))
            return
        self._emit(log_style.block_section(topic if rows else "events", rows or [(topic, "")]))

    def debug(self, text: str, *args: object) -> None:
        """Log a debug-mode note; ``text % args`` is formatted only when logged."""
        if self.enabled and self.debug_mode and self._collect_routine:
            self.info(text % args if args else text)

    def run_header(self, precision: str) -> None:
        """Log the report that opens the run."""
        self._emit(log_style.run_header("FVM", precision))

    def section(self, title: str, *rows: log_style.Row) -> None:
        """Log one titled section, collected into a running step."""
        if not self.enabled:
            return
        if self._step is not None:
            if self._collect_routine:
                self._step.sections.append((title, list(rows)))
            return
        self._emit(log_style.block_section(title, rows))

    # Step records.

    def begin_step(self, step: int, time: float, time_step_size: float) -> None:
        """Open the record of a candidate step; an unfinished record is dropped."""
        self._step = None
        self._collect_routine = self.schedule.is_due(step, time, time_step_size)
        self._step = _StepRecord(
            step=int(step),
            time=float(time),
            time_step_size=float(time_step_size),
            wall_time_at_start=self._step_wall_time,
        )

    def _record(self, **values: Any) -> None:
        if not self.enabled or not self._collect_routine:
            return
        if self._step is not None:
            for name, value in values.items():
                setattr(self._step, name, value)
            return
        if not self.debug_mode:
            return
        orphan = _StepRecord(step=0, time=0.0, time_step_size=0.0)
        for name, value in values.items():
            setattr(orphan, name, value)
        for title, items in _sections(orphan):
            self.section(title, *items)

    def convergence(self, residuals: dict[str, float] | None) -> None:
        """Record the nonlinear and linear convergence state."""
        if self.enabled and self._collect_routine and residuals:
            self._record(residuals={key: float(value) for key, value in residuals.items()})

    def continuity(self, maximum: float, total: float) -> None:
        """Record global mass-conservation diagnostics."""
        self._record(
            max_continuity_error=float(maximum), sum_absolute_continuity_error=float(total)
        )

    def courant_number(self, maximum: float, target: float | None = None) -> None:
        """Record the global Courant-number state."""
        self._record(
            courant=float(maximum),
            courant_target=None if target is None else float(target),
        )

    def y_plus(self, yplus_stats: dict[str, dict[str, float]] | None) -> None:
        """Record wall-resolution diagnostics."""
        if self.enabled and self._collect_routine and yplus_stats:
            self._record(y_plus=dict(yplus_stats))

    def forces(self, forces: dict[str, Any]) -> None:
        """Record force coefficients for every configured patch."""
        if self.enabled and self._collect_routine and forces:
            self._record(forces=dict(forces))

    def turbulence(
        self, minimum: float, maximum: float, mean: float, kinematic_viscosity: float
    ) -> None:
        """Record the eddy viscosity in m²/s and the molecular viscosity in m²/s."""
        if self.enabled and self._collect_routine:
            self._record(
                turbulence=(float(minimum), float(maximum), float(mean)),
                kinematic_viscosity=float(kinematic_viscosity),
            )

    def warnings(self, warnings: tuple[str, ...]) -> None:
        """Record acceptance-settings warnings raised by the step."""
        for warning in warnings:
            self.warning(warning)

    def end_step(self, elapsed: float, *, accepted: bool = True) -> None:
        """Close a trial and publish only accepted state, including its wall duration."""
        self._step_wall_time += float(elapsed)
        if not accepted:
            self._step = None
            return
        self._steps += 1
        if self._step is not None:
            self._step.elapsed = float(elapsed)
        self._flush_step()

    def steady_iterations(self, iterations: int) -> None:
        """Record steady SIMPLE iterations without pretending they are time steps."""
        if isinstance(iterations, bool) or int(iterations) < 0:
            raise ValueError("steady iteration count must be a non-negative integer")
        self._steps = int(iterations)
        self._count_label = "iterations"

    def _reportable(self, record: _StepRecord) -> bool:
        if record.warnings:
            return True
        return self.schedule.is_due(record.step, record.time, record.time_step_size)

    def _flush_step(self) -> None:
        record, self._step = self._step, None
        if record is None or not self.enabled:
            return
        self._step_reported = self._reportable(record)
        if not self._step_reported:
            return
        if self.debug_mode:
            self._emit(self._debug_block(record))
        else:
            self._emit(self._step_block(record))

    def _debug_block(self, record: _StepRecord) -> str:
        """Return the full step report, one detail row per quantity."""
        return self._render_step_block(record, detailed=True)

    @staticmethod
    def _core_rows(record: _StepRecord) -> list[log_style.Row]:
        """Return the time-control quantities reported for every step."""
        return [("time step", f"{record.time_step_size:.3e}", "s")]

    @staticmethod
    def _render_sections(record: _StepRecord, *, detailed: bool) -> list[str]:
        """Return the current step's populated sections in display order."""
        sections: list[tuple[str, list[log_style.Row]]] = []
        events = _event_rows(record)
        if events:
            sections.append(("events", events))
        sections.extend(record.sections)

        time_rows = Logging._core_rows(record)
        if record.courant is not None:
            courant_value = f"{record.courant:.3e}" if detailed else f"{record.courant:.3f}"
            time_rows.append(("courant, max", courant_value))
        if detailed and record.courant_target is not None:
            time_rows.append(("courant target", f"{record.courant_target:.3e}"))
        sections.append(("time control", time_rows))

        if detailed:
            sections.extend(
                (title, items) for title, items in _sections(record) if title != "time control"
            )
        else:
            residuals = record.residuals
            convergence: list[log_style.Row] = []
            if "velocity" in residuals:
                convergence.append(("residual, velocity", f"{residuals['velocity']:.2e}"))
            if "kinematic_pressure" in residuals:
                convergence.append(("residual, pressure", f"{residuals['kinematic_pressure']:.2e}"))
            if convergence:
                sections.append(("convergence", convergence))

            if record.max_continuity_error is not None:
                sections.append(
                    (
                        "conservation",
                        [("continuity error, max", f"{record.max_continuity_error:.2e}", "1/s")],
                    )
                )

            drag = record.drag()
            if drag is not None:
                sections.append(("aerodynamic loads", [("drag coefficient", f"{drag:.4f}")]))

        if not detailed and record.warnings:
            sections.append(
                (
                    "warnings",
                    [(f"({index + 1})", text) for index, text in enumerate(record.warnings)],
                )
            )

        if not detailed and record.turbulence is not None:
            minimum, maximum, mean = record.turbulence
            sections.append(
                (
                    "turbulence",
                    [
                        ("eddy viscosity, min", minimum, "m^2/s"),
                        ("eddy viscosity, mean", mean, "m^2/s"),
                        ("eddy viscosity, max", maximum, "m^2/s"),
                    ],
                )
            )

        timing_rows: list[log_style.Row] = [("step wall time", f"{record.elapsed:.2f}", "s")]
        if detailed:
            timing_rows.append(
                (
                    "cumulative wall time",
                    f"{record.wall_time_at_start + record.elapsed:.3e}",
                    "s",
                )
            )
        sections.append(("timing", timing_rows))
        return [
            log_style.block_section(title, items)
            for title, items in sorted(
                sections, key=lambda item: log_style.step_section_order(item[0])
            )
        ]

    def _render_step_block(self, record: _StepRecord, *, detailed: bool) -> str:
        """Render one complete FVM step in the shared block layout."""
        parts: list[str] = [
            log_style.step_header(
                record.step,
                record.time,
                record.wall_time_at_start + record.elapsed,
                scope="FVM",
            )
        ]
        parts.extend(self._render_sections(record, detailed=detailed))
        return "\n".join(parts)

    def _step_block(self, record: _StepRecord) -> str:
        """Return the routine step report: the quantities watched every step."""
        return self._render_step_block(record, detailed=False)

    # Startup and shutdown reports.

    def solver_configuration(self, solver: Any, initialization_seconds: float) -> None:
        """Log the solver configuration report after construction."""
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
                    ("algorithm", numerics.pimple.algorithm),
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
        momentum_method = numerics.linear.momentum_solver or linear_method
        sections.append(
            (
                "NUMERICS",
                [
                    ("time step", f"{numerics.time_step_size:.3e}", "s"),
                    ("end time", f"{config.run.end_time:.3e}", "s"),
                    ("time scheme", str(numerics.schemes.time_scheme)),
                    ("convection scheme", str(numerics.schemes.convection_scheme)),
                    ("gradient scheme", str(numerics.schemes.gradient_scheme)),
                    ("momentum solver", str(momentum_method)),
                    ("pressure solver", str(pressure_method)),
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
                    ("log file", str(self.path)),
                    ("log mode", self.mode),
                    ("log interval", self.schedule.describe()),
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
        self._emit(log_style.block_report("FVM SOLVER CONFIGURATION", sections))

    def output(self, text: str) -> None:
        """Log a visualization-frame or backup event immediately."""
        if self.enabled:
            self._emit(log_style.block_section("events", [(text, "")]))

    def profile(self, record: dict[str, Any]) -> None:
        """Log the section times of a profiled step (debug mode, logged steps)."""
        if not self.debug_mode or not self._step_reported:
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
        self.section("performance profile", *rows)

    def run_summary(self, *, status: str, failure: BaseException | None = None) -> None:
        """Log the closing wall-time summary with the run status."""
        mean = self._step_wall_time / self._steps if self._steps else 0.0
        rows: list[log_style.Row] = [
            ("status", str(status)),
            (self._count_label, f"{self._steps:,}"),
            ("wall time, total", f"{self._step_wall_time:.3e}", "s"),
            ("wall time, mean per step", f"{mean:.3f}", "s"),
        ]
        if failure is not None:
            rows.append(("error", f"{type(failure).__name__}: {failure}"))
        self.section(f"run {status}", *rows)

    def flush(self) -> None:
        """Flush the log handlers."""
        if self.enabled and not self._closed:
            for handler in self._logger.handlers:
                handler.flush()

    def close(
        self, *, status: str, failure: BaseException | None = None, profiler: Any | None = None
    ) -> None:
        """Log the run summary and the timing report, then close the log file.

        Idempotent.
        """
        if self._closed:
            return
        self._step = None
        if self.enabled:
            self.run_summary(status=status, failure=failure)
            if profiler is not None and profiler.enabled:
                self._emit(profiler.report("FVM runtime profile"))
            log_style.close_logging(LOGGER_NAME)
        self._closed = True
