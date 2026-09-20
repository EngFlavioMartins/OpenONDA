"""Rank-aware console and file logging for the native FVM solver."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, TextIO

import numpy as np

from source import log_style
from source.version import __version__

_CONSOLE_STDOUT = sys.stdout
_WIDTH = log_style.WIDTH
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

_MEMORY_LABELS = {
    "mesh_topology": "mesh topology",
    "geometry": "geometry",
    "solution_fields_history": "fields and history",
    "turbulence_model": "turbulence model",
    "linear_algorithm_workspaces": "linear workspaces",
    "derived_diagnostics": "derived diagnostics",
    "output_buffers": "output buffers",
    "numpy_unique_total": "numpy total",
    "native_python_petsc_rss_remainder": "non-numpy, python and petsc",
}


def resolve_mode(default: str = "simple") -> str:
    """Return the log mode, letting ``FVM_LOG`` override the configured one."""
    mode = os.environ.get("FVM_LOG", default)
    if mode not in {"simple", "debug"}:
        raise ValueError(f"FVM_LOG must be 'simple' or 'debug'; got {mode!r}")
    return mode


def format_openonda_header(precision: str | None = "f64") -> log_style.FormattedText:
    """Return the shared run-identity report for the FVM owner."""
    return log_style.block_report(
        "OpenONDA FVM",
        [
            (
                "run",
                [
                    ("version", __version__),
                    ("precision", precision),
                    ("started", f"{datetime.now():%Y-%m-%d %H:%M:%S}"),
                    ("platform", f"{platform.system()} {platform.machine()}"),
                    ("Python", platform.python_version()),
                ],
            )
        ],
    )


def print_openonda_header(precision: str | None = "f64") -> None:
    """Print the FVM banner to the current console."""
    print(format_openonda_header(precision), flush=True)


def emit_standalone_report(topic: str, *rows: log_style.Row, warning: bool = False) -> None:
    """Report a standalone mesh/linear operation before a solver sink exists.

    Normal solver calls use their instance logger. This entry point provides the
    same layout for standalone library operations, without opening a case file.
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
    ibm_forces: dict[str, tuple[float, float]] = field(default_factory=dict)
    ibm_slip: float | None = None
    turbulence: tuple[float, float, float] | None = None
    kinematic_viscosity: float = 0.0
    warnings: tuple[str, ...] = ()
    events: list[tuple[str, tuple[log_style.Row, ...]]] = field(default_factory=list)
    sections: list[tuple[str, list[log_style.Row]]] = field(default_factory=list)
    wall_time_at_start: float = 0.0
    elapsed: float = 0.0

    def drag(self) -> float | None:
        """Total drag coefficient over all patches and immersed bodies."""
        total = 0.0
        found = False
        for values in self.forces.values():
            total += float(values.get("coeffs", {}).get("drag_coefficient", 0.0))
            found = True
        for drag, _lift in self.ibm_forces.values():
            total += drag
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
    for name, (drag, lift) in record.ibm_forces.items():
        loads.extend(
            (
                (f"{name}, drag, immersed", f"{drag:.4f}"),
                (f"{name}, lift, immersed", f"{lift:.4f}"),
            )
        )
    if record.ibm_slip is not None:
        loads.append(("marker slip", f"{record.ibm_slip:.3e}", "m/s"))
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
    """Single FVM output sink and formatter.

    Rank zero tees messages to the live console and ``solution/<filename>``;
    worker ranks use a disabled instance, so call sites need no MPI guard.

    ``simple`` mode prints a compact block for each reported step; ``debug``
    mode prints the full per-step block and the performance profile.
    """

    def __init__(
        self,
        case_dir: str | Path,
        *,
        solution_dir: str | Path | None = None,
        config: Any | None = None,
        enabled: bool = True,
        filename: str | None = None,
        console: bool | None = None,
    ) -> None:
        """Open the FVM console/file sink for one case.

        Parameters
        ----------
        case_dir : str or pathlib.Path
            Case root used to resolve the default ``solution`` directory.
        solution_dir : str or pathlib.Path or None, default=None
            Explicit log directory; otherwise ``case_dir/solution``.
        config : LoggingConfig or None, default=None
            Formatting, filename, console, and cadence policy.
        enabled : bool, default=True
            Disable all output while retaining a compatible object for worker
            ranks.
        filename : str or None, default=None
            Log filename; the configured/default name is used when omitted.
        console : bool or None, default=None
            Whether to mirror records to stdout. ``None`` follows ``config``.

        Side Effects
        ------------
        Creates the configured output directory/file on the enabled owner
        rank. The instance owns the opened stream until :meth:`close`.
        """
        if console is None:
            console = True if config is None else bool(config.console)
        if filename is None:
            filename = "fvm.log" if config is None else str(config.filename)

        self.enabled = bool(enabled)
        self.console = bool(console)
        self.mode = resolve_mode("simple" if config is None else str(config.mode))
        from ..config.scheduling import RunSchedule

        self.schedule = RunSchedule(every_n_steps=1) if config is None else config.schedule
        self.profiler: Any | None = None
        self.log_file_path: Path | None = None

        self._file: TextIO | None = None
        self._closed = False
        self._step: _StepRecord | None = None
        self._step_reported = True
        self._collect_routine = True
        self._step_wall_time = 0.0
        self._steps = 0
        self._count_label = "steps"
        self._reported_steps = 0
        # VPM file logging may replace process-global ``sys.stdout`` after the
        # FVM is constructed.  Keep the FVM console sink stable so its records
        # cannot migrate into ``vpm.log`` midway through a coupled run.
        self._console_stream = _CONSOLE_STDOUT

        if self.enabled:
            output_directory = (
                Path(solution_dir).resolve()
                if solution_dir is not None
                else Path(case_dir).resolve() / "solution"
            )
            output_directory.mkdir(parents=True, exist_ok=True)
            self.log_file_path = output_directory / filename
            had_previous_log = self.log_file_path.exists() and self.log_file_path.stat().st_size > 0
            self._file = self.log_file_path.open("a", buffering=1, encoding="utf-8")
            if had_previous_log:
                self._file.write(
                    log_style.block_section(
                        "FVM log session", [("started", f"{datetime.now():%Y-%m-%d %H:%M:%S}")]
                    )
                    + "\n"
                )
                self._file.flush()

    @property
    def debug(self) -> bool:
        """True when the full per-step diagnostics are being printed."""
        return self.mode == "debug"

    def _emit(self, text: str, *, flush: bool = False) -> None:
        if not self.enabled or self._closed:
            return
        if self.console:
            print(text, file=self._console_stream, flush=flush)
        if self._file is not None:
            print(text, file=self._file, flush=True)

    def message(self, text: str = "", *, flush: bool = False) -> None:
        """Emit a shared report, or submit plain text as a module event."""
        if not self.enabled:
            return
        if isinstance(text, log_style.FormattedText):
            self._emit(text, flush=flush)
        else:
            self.info(text, flush=flush)

    def info(self, text: str, *, flush: bool = False) -> None:
        """Submit an event without formatting it on unreported steps."""
        self.record(text, flush=flush)

    def warning(self, text: str, *, flush: bool = True) -> None:
        """Emit immediately, including during a long step or suppressed interval."""
        if not self.enabled:
            return
        rows: list[log_style.Row] = [(text, "")]
        if self._step is not None:
            rows.insert(0, ("FVM step", self._step.step))
        self._emit(log_style.block_section("warnings", rows), flush=True)

    def record(self, topic: str, *rows: log_style.Row, flush: bool = False) -> None:
        """Collect host values for a report, or flush an explicit output event."""
        if not self.enabled:
            return
        if self._step is not None and not flush:
            if self._collect_routine:
                self._step.events.append((topic, rows))
            return
        self._emit(
            log_style.block_section(topic if rows else "events", rows or [(topic, "")]), flush=flush
        )

    def debug_message(self, text: str, *args, flush: bool = False) -> None:
        """Interpolate optional details only when this owner will report them."""
        if self.enabled and self.debug and self._collect_routine:
            self.info(text % args if args else text, flush=flush)

    def header(self, precision: str | None = "f64") -> None:
        """Emit the OpenONDA FVM startup banner."""
        self.message(format_openonda_header(precision), flush=True)

    @staticmethod
    def _section(title: str, items: list[log_style.Row]) -> list[str]:
        return log_style.section(title, items).split("\n")

    def section(
        self,
        title: str,
        items: list[log_style.Row],
        *,
        flush: bool = False,
    ) -> None:
        """Record or emit one consistently formatted information section."""
        if not self.enabled:
            return
        if self._step is not None:
            if self._collect_routine:
                self._step.sections.append((title, items))
            return
        self._emit(log_style.block_section(title, items), flush=flush)

    # Per-step diagnostics.

    def step_begin(self, step: int, time: float, time_step_size: float) -> None:
        """Open a trial record; discard an interrupted predecessor without publishing it."""
        self._step = None
        self._collect_routine = self.should_report(step, time, time_step_size)
        self._step = _StepRecord(
            step=int(step),
            time=float(time),
            time_step_size=float(time_step_size),
            wall_time_at_start=self._step_wall_time,
        )

    def should_report(self, step: int, flow_time: float, time_step_size: float) -> bool:
        """Return the rank-independent sampling decision for routine log metrics.

        Every MPI rank must use this same schedule before optional collective
        reductions. The disabled worker sink is deliberately not consulted here.
        """
        return self.schedule.is_due(step, flow_time, time_step_size)

    def _record(self, **values: Any) -> None:
        if not self.enabled or not self._collect_routine:
            return
        if self._step is not None:
            for name, value in values.items():
                setattr(self._step, name, value)
            return
        if not self.debug:
            return
        orphan = _StepRecord(step=0, time=0.0, time_step_size=0.0)
        for name, value in values.items():
            setattr(orphan, name, value)
        for title, items in _sections(orphan):
            self.section(title, items)

    def convergence_info(self, residuals: dict[str, float] | None) -> None:
        """Record the nonlinear and linear convergence state."""
        if self.enabled and self._collect_routine and residuals:
            self._record(residuals={key: float(value) for key, value in residuals.items()})

    def continuity_info(self, maximum: float, total: float) -> None:
        """Record global mass-conservation diagnostics."""
        self._record(
            max_continuity_error=float(maximum), sum_absolute_continuity_error=float(total)
        )

    def courant_info(self, maximum: float, target: float | None = None) -> None:
        """Record the global Courant-number state."""
        self._record(
            courant=float(maximum),
            courant_target=None if target is None else float(target),
        )

    def yplus_info(self, yplus_stats: dict[str, dict[str, float]] | None) -> None:
        """Record wall-resolution diagnostics."""
        if self.enabled and self._collect_routine and yplus_stats:
            self._record(y_plus=dict(yplus_stats))

    def force_info(self, forces: dict[str, Any]) -> None:
        """Record force coefficients for every configured patch."""
        if self.enabled and self._collect_routine and forces:
            self._record(forces=dict(forces))

    def ibm_force_info(self, forces: dict[str, tuple[float, float]], slip: float) -> None:
        """Record immersed-body force coefficients and marker slip."""
        if self.enabled and self._collect_routine and forces:
            self._record(ibm_forces=dict(forces), ibm_slip=float(slip))

    def turbulence_info(
        self,
        eddy_viscosity: np.ndarray | None,
        kinematic_viscosity: float,
        *,
        statistics: tuple[float, float, float] | None = None,
    ) -> None:
        """Record turbulent-viscosity diagnostics."""
        if not self.enabled or not self._collect_routine:
            return
        if statistics is None:
            if eddy_viscosity is None:
                return
            values = np.asarray(eddy_viscosity)
            if values.size == 0:
                return
            statistics = (
                float(np.min(values)),
                float(np.max(values)),
                float(np.mean(values)),
            )
        self._record(
            turbulence=tuple(float(value) for value in statistics),
            kinematic_viscosity=float(kinematic_viscosity),
        )

    def warnings_info(self, warnings: tuple[str, ...]) -> None:
        """Record acceptance-policy warnings raised by the step."""
        for warning in warnings:
            self.warning(warning)

    def step_end(self, elapsed: float, *, accepted: bool = True) -> None:
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
        self._reported_steps += 1
        if self.debug:
            self._emit(self._debug_block(record), flush=True)
        else:
            self._emit(self._step_block(record), flush=True)

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

    @staticmethod
    def solver_info(solver: Any, initialization_time: float) -> str:
        """Return a comprehensive FVM initialization report."""
        config = solver._resolved_setup
        mesh = solver.mesh_data
        parallel = solver.parallel
        partition = getattr(parallel, "partition", None)

        sections: list[tuple[str, list[log_style.Row]]] = [
            (
                "RUN",
                [
                    ("case", str(config.case_name)),
                    ("algorithm", str(config.pimple.algorithm).upper()),
                    ("execution mode", str(config.execution.parallel_mode)),
                    ("mpi ranks", str(parallel.size)),
                    ("operator backend", str(config.execution.operator_backend)),
                    ("linear backend", str(config.execution.linear_backend)),
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

        linear_method = config.linear.linear_solver
        pressure_method = config.linear.pressure_solver or linear_method
        momentum_method = config.linear.momentum_solver or linear_method
        sections.append(
            (
                "NUMERICS",
                [
                    ("time step", f"{config.time.time_step_size:.3e}", "s"),
                    ("end time", f"{config.time.end_time:.3e}", "s"),
                    ("time scheme", str(config.schemes.time_scheme)),
                    ("convection scheme", str(config.schemes.convection_scheme)),
                    ("gradient scheme", str(config.schemes.gradient_scheme)),
                    ("momentum solver", str(momentum_method)),
                    ("pressure solver", str(pressure_method)),
                    ("correctors", str(config.pimple.n_correctors)),
                    ("outer correctors", str(config.pimple.n_outer_correctors)),
                ],
            )
        )
        turbulence = config.turbulence
        turbulence_name = "DNS / laminar" if turbulence is None else str(turbulence.model)
        sections.append(
            (
                "PHYSICS",
                [
                    ("density", f"{config.transport.density:.6g}", "kg/m^3"),
                    (
                        "kinematic viscosity",
                        f"{config.transport.kinematic_viscosity:.6e}",
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
        sink = getattr(solver, "logger", None)
        log_file_path = getattr(sink, "log_file_path", None)
        sections.append(
            (
                "MONITORING AND OUTPUT",
                [
                    ("solution directory", str(solver.solution_dir)),
                    ("log file", str(log_file_path or "disabled")),
                    ("log mode", str(getattr(sink, "mode", "simple"))),
                    (
                        "log interval",
                        getattr(getattr(sink, "schedule", None), "describe", lambda: "unknown")(),
                    ),
                    ("visualization", "VTK XML, cell-centred, appended binary"),
                    ("output compression", str(config.output.compression).upper()),
                    (
                        "output scheduling",
                        "asynchronous"
                        if config.output.asynchronous and not parallel.is_partitioned
                        else "synchronous",
                    ),
                    ("visualization cadence", solver._output_schedule.describe()),
                    (
                        "restart cadence",
                        (
                            solver._backup_config.schedule.describe()
                            if solver._backup_config.schedule is not None
                            else "disabled"
                        ),
                    ),
                    ("visualization ghost layers", str(config.output.ghost_layers)),
                    ("initialization time", f"{initialization_time:.3e}", "s"),
                ],
            )
        )
        return log_style.block_report("FVM SOLVER CONFIGURATION", sections)

    def log_solver_info(self, solver: Any, initialization_time: float) -> None:
        """Emit the formatted initialization report."""
        self.message(self.solver_info(solver, initialization_time), flush=True)

    def solver_state(self, solver: Any) -> None:
        """Emit the current high-level solver state."""
        self.section(
            "fvm solver  state",
            [
                ("case", str(solver._resolved_setup.case_name)),
                ("time", f"{solver.time:.5f}", "s"),
                ("step", f"{solver.step:,}"),
                ("cells, local", f"{solver.mesh_data['n_cells']:,}"),
                (
                    "algorithm",
                    str(solver._resolved_setup.pimple.algorithm),
                ),
            ],
        )

    def output_info(self, text: str) -> None:
        """Emit one visualization/backup output event."""
        self.info(text, flush=True)

    def timing(self, name: str, elapsed: float) -> None:
        """Record one completed phase with the profiler."""
        if self.profiler is not None:
            self.profiler.record(name, elapsed)

    def profile_report(self, record: dict[str, Any]) -> None:
        """Emit the per-step performance profile (debug mode, reported steps)."""
        if not self.debug or not self._step_reported:
            return
        rows: list[log_style.Row] = [("phases, max over ranks:", "")]
        for phase in record["phases"]:
            rows.extend(
                (
                    (f"  {phase['name']}, calls", f"{phase['calls']['max']:.0f}"),
                    (f"  {phase['name']}, mean", f"{phase['seconds']['mean']:.3f}", "s"),
                    (f"  {phase['name']}, max", f"{phase['seconds']['max']:.3f}", "s"),
                    (
                        f"  {phase['name']}, share",
                        f"{100.0 * phase['critical_path_fraction']:.1f}",
                        "%",
                    ),
                    (f"  {phase['name']}, rank imbalance", f"{phase['rank_imbalance']:.3f}"),
                )
            )
        if record["linear"]:
            rows.append(("linear solvers:", ""))
            for linear in record["linear"]:
                rows.extend(
                    (
                        (f"  {linear['equation']}, calls", f"{linear['calls']['max']:.0f}"),
                        (
                            f"  {linear['equation']}, iterations, max",
                            f"{linear['iterations']['max']:.0f}",
                        ),
                        (
                            f"  {linear['equation']}, setup, max",
                            f"{linear['setup_seconds']['max']:.3f}",
                            "s",
                        ),
                        (
                            f"  {linear['equation']}, solve, max",
                            f"{linear['solve_seconds']['max']:.3f}",
                            "s",
                        ),
                    )
                )
        memory = record["memory"]
        rows.extend(
            (
                ("memory:", ""),
                (
                    "  resident, all ranks",
                    f"{memory['aggregate_rss_end_bytes'] / _MIB:.1f}",
                    "MiB",
                ),
                (
                    "  peak during step",
                    f"{memory['aggregate_rss_max_observed_bytes'] / _MIB:.1f}",
                    "MiB",
                ),
                (
                    "  peak since start",
                    f"{memory['aggregate_peak_rss_end_bytes'] / _MIB:.1f}",
                    "MiB",
                ),
                ("  slowest rank", f"{record['step_seconds']['max']:.3f}", "s"),
            )
        )
        inventory = record.get("allocation_inventory")
        if inventory is not None:
            rows.append(("memory by subsystem:", ""))
            for name, values in inventory.items():
                label = _MEMORY_LABELS.get(name, name)
                rows.append((f"  {label}", f"{values['aggregate_bytes'] / _MIB:.1f}", "MiB"))
        self.section("PERFORMANCE PROFILE", rows, flush=True)

    def run_summary(self, *, status: str = "complete", failure=None) -> None:
        """Emit the closing wall-time summary with truthful terminal status."""
        mean = self._step_wall_time / self._steps if self._steps else 0.0
        self._emit("")
        title = "RUN COMPLETE" if status == "complete" else "RUN " + str(status).upper()
        rows: list[log_style.Row] = [
            ("status", str(status)),
            (self._count_label, f"{self._steps:,}"),
            ("wall time, total", f"{self._step_wall_time:.3e}", "s"),
            ("wall time, mean per step", f"{mean:.3f}", "s"),
        ]
        if failure is not None:
            rows.append(("error", f"{type(failure).__name__}: {failure}"))
        self.section(
            title,
            rows,
            flush=True,
        )

    def flush(self) -> None:
        """Flush the file sink."""
        if self._file is not None and not self._closed:
            self._file.flush()

    def close(self, *, status: str = "complete", failure=None) -> None:
        """Flush and close the file sink. This method is idempotent."""
        if self._closed:
            return
        self._step = None
        # Terminal status is a lifecycle record, not a debug-only detail.
        # Keep it visible in both simple and debug modes so failed runs never
        # look like successful early exits.
        self.run_summary(status=status, failure=failure)
        self.flush()
        if self._file is not None:
            self._file.close()
        self._closed = True


class Timer:
    """Per-owner wall-clock phases, in seconds, with a stack for nested phases.

    Create one timer per solver. Starting/stopping a phase never changes another
    owner's state. A missing phase returns zero elapsed seconds.
    """

    def __init__(self) -> None:
        """Create an empty phase store without starting measurements."""
        self._timers: dict[str, list[float]] = {}

    def start(self, name: str) -> None:
        """Start a named phase on this owner, retaining nested starts."""
        self._timers.setdefault(name, []).append(time.perf_counter())

    def stop(self, name: str) -> float:
        """Return elapsed seconds from the most recent matching start."""
        started = self._timers.get(name)
        if not started:
            return 0.0
        value = started.pop()
        if not started:
            del self._timers[name]
        return time.perf_counter() - value

    def log(self, name: str, *, sink: Any | None = None) -> float:
        """Stop a phase and report its elapsed seconds through the owning sink."""
        elapsed = self.stop(name)
        if sink is not None and elapsed > 0.0:
            sink.timing(name.strip(" -"), elapsed)
        return elapsed
