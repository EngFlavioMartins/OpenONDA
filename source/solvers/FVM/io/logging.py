"""Rank-aware console and file logging for the native FVM solver."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import getpass
import os
from pathlib import Path
import platform
import socket
import sys
import time
from typing import Any, TextIO

import numpy as np

from source.version import __version__

_WIDTH = 88
_HEADER_ROWS = 20
_MIB = 1024.0 * 1024.0

_RESIDUAL_LABELS = {
    "velocity": "Velocity residual",
    "U_increment": "Velocity increment",
    "U_x": "Velocity-x residual",
    "U_y": "Velocity-y residual",
    "U_z": "Velocity-z residual",
    "kinematic_pressure": "Pressure residual",
    "p_initial": "Pressure initial residual",
}
_RESIDUAL_ORDER = (
    "velocity",
    "U_increment",
    "U_x",
    "U_y",
    "U_z",
    "kinematic_pressure",
    "p_initial",
)

_MEMORY_LABELS = {
    "mesh_topology": "Mesh topology",
    "geometry": "Geometry",
    "solution_fields_history": "Fields and history",
    "turbulence_model": "Turbulence model",
    "linear_algorithm_workspaces": "Linear workspaces",
    "derived_diagnostics": "Derived diagnostics",
    "output_buffers": "Output buffers",
    "numpy_unique_total": "NumPy total",
    "native_python_petsc_rss_remainder": "Non-NumPy (Python, PETSc)",
}

_COLUMNS = (
    ("step", 6),
    ("time", 9),
    ("dt", 9),
    ("Co", 6),
    ("res(U)", 8),
    ("res(p)", 8),
    ("|div U|", 8),
    ("Cd", 8),
    ("s/step", 8),
)


def resolve_mode(default: str = "simple") -> str:
    """Return the log mode, letting ``FVM_LOG`` override the configured one."""
    mode = os.environ.get("FVM_LOG", default)
    if mode not in {"simple", "debug"}:
        raise ValueError(f"FVM_LOG must be 'simple' or 'debug'; got {mode!r}")
    return mode


def format_openonda_header(precision: str | None = "f64") -> str:
    """Return the OpenONDA FVM startup banner."""
    now = datetime.now()
    try:
        hostname = socket.gethostname()
    except Exception:
        hostname = "unknown"
    try:
        username = getpass.getuser()
    except Exception:
        username = "unknown"

    system_info = (
        f"{platform.system()}; python={platform.python_version()}; arch={platform.machine()}"
    )
    width = 91
    lines = [
        "",
        "/ / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / /",
        "* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * ",
        "   ░██████                                      ░██████   ░███    ░██ ░███████      ░███    ",
        "  ░██   ░██                                    ░██   ░██  ░████   ░██ ░██   ░██    ░██░██   ",
        " ░██     ░██ ░████████   ░███████  ░████████  ░██     ░██ ░██░██  ░██ ░██    ░██  ░██  ░██  ",
        " ░██     ░██ ░██    ░██ ░██    ░██ ░██    ░██ ░██     ░██ ░██ ░██ ░██ ░██    ░██ ░█████████ ",
        " ░██     ░██ ░██    ░██ ░█████████ ░██    ░██ ░██     ░██ ░██  ░██░██ ░██    ░██ ░██    ░██ ",
        "  ░██   ░██  ░███   ░██ ░██        ░██    ░██  ░██   ░██  ░██   ░████ ░██   ░██  ░██    ░██ ",
        "   ░██████   ░██░█████   ░███████  ░██    ░██   ░██████   ░██    ░███ ░███████   ░██    ░██ ",
        "             ░██                                                                            ",
        "             ░██                                                                            ",
        "* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * ",
        "| O pen       | " + "".ljust(width - 16) + "|",
        "| O perator   | "
        + "OpenONDA: Operator for Numerical Design & Aerodynamics.".ljust(width - 16)
        + "|",
        "| N umer.     | " + f"Version: {__version__}".ljust(width - 16) + "|",
        "| D esign     | " + "Website: https://github.com/EngFlavioMartins".ljust(width - 16) + "|",
        "| A erodyn.   | " + "FVM Solver: Finite Volume Method".ljust(width - 16) + "|",
        "* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * ",
        f"Build  : OpenONDA={__version__}",
        f"Arch   : {system_info}",
    ]
    if precision is not None:
        lines.append(f"Precision: {precision}")
    lines.extend(
        [
            "Exec   : FVM Solver",
            f"Date   : {now:%b %d %Y}",
            f"Time   : {now:%H:%M:%S}",
            f"Host   : {hostname}",
            f"User   : {username}",
            f"PID    : {os.getpid()}",
            "* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * ",
            "/ / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / /",
        ]
    )
    return "\n".join(lines)


def print_openonda_header(precision: str | None = "f64") -> None:
    """Print the FVM banner to the current console."""
    print(format_openonda_header(precision), flush=True)


@dataclass
class _StepRecord:
    """Raw per-step diagnostics, rendered by the simple or debug layout."""

    step: int
    time: float
    time_step_size: float
    residuals: dict[str, float] = field(default_factory=dict)
    continuity_max: float | None = None
    continuity_sum: float | None = None
    courant: float | None = None
    courant_target: float | None = None
    yplus: dict[str, dict[str, float]] = field(default_factory=dict)
    forces: dict[str, Any] = field(default_factory=dict)
    ibm_forces: dict[str, tuple[float, float]] = field(default_factory=dict)
    ibm_slip: float | None = None
    turbulence: tuple[float, float, float] | None = None
    turbulence_nu: float = 0.0
    warnings: tuple[str, ...] = ()
    elapsed: float = 0.0

    def drag(self) -> float | None:
        """Total drag coefficient over all patches and immersed bodies."""
        total = 0.0
        found = False
        for values in self.forces.values():
            total += float(values.get("coeffs", {}).get("Cd", 0.0))
            found = True
        for drag, _lift in self.ibm_forces.values():
            total += drag
            found = True
        return total if found else None


def _sections(record: _StepRecord) -> list[tuple[str, list[tuple[str, str]]]]:
    """Return the populated debug-mode sections of one step record."""
    sections: list[tuple[str, list[tuple[str, str]]]] = []

    if record.residuals:
        keys = [key for key in _RESIDUAL_ORDER if key in record.residuals]
        keys.extend(key for key in record.residuals if key not in keys)
        sections.append(
            (
                "Solver Convergence",
                [
                    (_RESIDUAL_LABELS.get(key, key), f"{float(record.residuals[key]):.3e}")
                    for key in keys
                ],
            )
        )

    if record.continuity_max is not None:
        sections.append(
            (
                "Conservation",
                [
                    ("Maximum |div U|", f"{record.continuity_max:.3e} 1/s"),
                    ("Boundary imbalance", f"{record.continuity_sum:.3e} m³/s"),
                ],
            )
        )

    if record.courant is not None:
        target = (
            "" if record.courant_target is None else f"  (target ≤ {record.courant_target:.3e})"
        )
        sections.append(
            ("Time Control", [("Maximum Courant number", f"{record.courant:.3e}{target}")])
        )

    if record.yplus:
        sections.append(
            (
                "Wall Resolution (y+)",
                [
                    (
                        name,
                        f"min={stats['min']:.3e}, max={stats['max']:.3e}, mean={stats['avg']:.3e}",
                    )
                    for name, stats in record.yplus.items()
                ],
            )
        )

    loads = [
        (
            patch,
            f"Cd={float(values.get('coeffs', {}).get('Cd', 0.0)):.4f}, "
            f"Cl={float(values.get('coeffs', {}).get('Cl', 0.0)):.4f}, "
            f"Cm={float(values.get('coeffs', {}).get('Cm', 0.0)):.4f}",
        )
        for patch, values in record.forces.items()
    ]
    loads.extend(
        (f"{name} (immersed)", f"Cd={drag:.4f}, Cl={lift:.4f}")
        for name, (drag, lift) in record.ibm_forces.items()
    )
    if record.ibm_slip is not None:
        loads.append(("Marker slip", f"{record.ibm_slip:.3e} m/s"))
    if loads:
        sections.append(("Aerodynamic Loads", loads))

    if record.turbulence is not None:
        minimum, maximum, mean = record.turbulence
        nu = record.turbulence_nu
        ratio_min = minimum / nu if nu > 0 else float("inf")
        ratio_max = maximum / nu if nu > 0 else float("inf")
        sections.append(
            (
                "Turbulence Diagnostics",
                [
                    ("nut minimum", f"{minimum:.3e} m²/s"),
                    ("nut maximum", f"{maximum:.3e} m²/s"),
                    ("nut mean", f"{mean:.3e} m²/s"),
                    ("nut/nu range", f"[{ratio_min:.3e}, {ratio_max:.3e}]"),
                ],
            )
        )

    if record.warnings:
        sections.append(
            ("Warnings", [(f"({index + 1})", text) for index, text in enumerate(record.warnings)])
        )

    return sections


class Logging:
    """Single FVM output sink and formatter.

    Rank zero tees messages to the live console and ``solution/<filename>``;
    worker ranks use a disabled instance, so call sites need no MPI guard.

    ``simple`` mode prints one table row per reported step; ``debug`` mode
    prints the full per-step block and the performance profile.
    """

    def __init__(
        self,
        case_dir: str | Path,
        *,
        config: Any | None = None,
        enabled: bool = True,
        filename: str | None = None,
        console: bool | None = None,
    ) -> None:
        if console is None:
            console = True if config is None else bool(config.console)
        if filename is None:
            filename = "fvm.log" if config is None else str(config.filename)

        self.enabled = bool(enabled)
        self.console = bool(console)
        self.mode = resolve_mode("simple" if config is None else str(config.mode))
        self.interval_steps = 1 if config is None else int(config.interval_steps)
        self.profiler: Any | None = None
        self.log_file_path: Path | None = None

        self._file: TextIO | None = None
        self._closed = False
        self._step: _StepRecord | None = None
        self._step_reported = True
        self._step_wall_time = 0.0
        self._steps = 0
        self._reported_steps = 0
        self._rows_since_header = 0
        self._needs_header = True
        self._drag_column = False
        self._columns_fixed = False

        if self.enabled:
            solution_dir = Path(case_dir).resolve() / "solution"
            solution_dir.mkdir(parents=True, exist_ok=True)
            self.log_file_path = solution_dir / filename
            self._file = self.log_file_path.open("w", buffering=1, encoding="utf-8")

    @property
    def debug(self) -> bool:
        """True when the full per-step diagnostics are being printed."""
        return self.mode == "debug"

    def _emit(self, text: str, *, flush: bool = False) -> None:
        if not self.enabled or self._closed:
            return
        if self.console:
            print(text, file=sys.stdout, flush=flush)
        if self._file is not None:
            print(text, file=self._file, flush=True)

    def message(self, text: str = "", *, flush: bool = False) -> None:
        """Emit one complete message to every configured sink."""
        self._emit(text, flush=flush)
        self._needs_header = True

    def info(self, text: str, *, flush: bool = False) -> None:
        """Emit an informational message."""
        self.message(f"[FVM][Info] {text}", flush=flush)

    def warning(self, text: str, *, flush: bool = True) -> None:
        """Emit a warning message."""
        self.message(f"[FVM][Warning] {text}", flush=flush)

    def debug_message(self, text: str, *, flush: bool = False) -> None:
        """Emit a message only in debug mode."""
        if self.debug:
            self.info(text, flush=flush)

    def header(self, precision: str | None = "f64") -> None:
        """Emit the OpenONDA FVM startup banner."""
        self.message(format_openonda_header(precision), flush=True)

    @staticmethod
    def _section(title: str, items: list[tuple[str, str]]) -> list[str]:
        lines = ["", "-" * _WIDTH, title, "-" * _WIDTH]
        label_width = max((len(label) for label, _value in items), default=0)
        lines.extend(f"  {label:<{label_width}}  : {value}" for label, value in items)
        return lines

    def section(self, title: str, items: list[tuple[str, str]]) -> None:
        """Emit one consistently formatted information section."""
        self.message("\n".join(self._section(title, items)))

    # -- Per-step diagnostics --------------------------------------------------

    def step_begin(self, step: int, time: float, time_step_size: float) -> None:
        """Open a new per-step record, flushing any left open by an abort."""
        self._flush_step()
        self._step = _StepRecord(
            step=int(step), time=float(time), time_step_size=float(time_step_size)
        )

    def _record(self, **values: Any) -> None:
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
        if residuals:
            self._record(residuals={key: float(value) for key, value in residuals.items()})

    def continuity_info(self, maximum: float, total: float) -> None:
        """Record global mass-conservation diagnostics."""
        self._record(continuity_max=float(maximum), continuity_sum=float(total))

    def courant_info(self, maximum: float, target: float | None = None) -> None:
        """Record the global Courant-number state."""
        self._record(
            courant=float(maximum),
            courant_target=None if target is None else float(target),
        )

    def yplus_info(self, yplus_stats: dict[str, dict[str, float]] | None) -> None:
        """Record wall-resolution diagnostics."""
        if yplus_stats:
            self._record(yplus=dict(yplus_stats))

    def force_info(self, forces: dict[str, Any]) -> None:
        """Record force coefficients for every configured patch."""
        if forces:
            self._record(forces=dict(forces))

    def ibm_force_info(self, forces: dict[str, tuple[float, float]], slip: float) -> None:
        """Record immersed-body force coefficients and marker slip."""
        if forces:
            self._record(ibm_forces=dict(forces), ibm_slip=float(slip))

    def turbulence_info(
        self,
        eddy_viscosity: np.ndarray | None,
        nu_molecular: float,
        *,
        statistics: tuple[float, float, float] | None = None,
    ) -> None:
        """Record turbulent-viscosity diagnostics."""
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
            turbulence_nu=float(nu_molecular),
        )

    def warnings_info(self, warnings: tuple[str, ...]) -> None:
        """Record acceptance-policy warnings raised by the step."""
        if warnings:
            self._record(warnings=tuple(warnings))

    def step_end(self, elapsed: float) -> None:
        """Close the open step and report it if the interval allows."""
        self._step_wall_time += float(elapsed)
        self._steps += 1
        if self._step is not None:
            self._step.elapsed = float(elapsed)
        self._flush_step()

    def _reportable(self, record: _StepRecord) -> bool:
        if record.warnings or self._reported_steps == 0 or self.interval_steps <= 1:
            return True
        return record.step % self.interval_steps == 0

    def _flush_step(self) -> None:
        record, self._step = self._step, None
        if record is None:
            return
        self._step_reported = self._reportable(record)
        if not self._step_reported:
            return
        self._reported_steps += 1
        if self.debug:
            self._emit(self._debug_block(record))
            self._emit(
                f"[FVM][Timing] step={record.step} step_s={record.elapsed:.3e} "
                f"cumulative_s={self._step_wall_time:.3e}",
                flush=True,
            )
        else:
            self._emit(self._simple_row(record), flush=True)

    def _debug_block(self, record: _StepRecord) -> str:
        title = f" TIME STEP  (step {record.step}, t = {record.time:.3e} s, Δt = {record.time_step_size:.3e} s)"
        bar = "=" * _WIDTH
        sep = "-" * _WIDTH
        sections = _sections(record)
        label_width = max(
            (len(label) for _title, items in sections for label, _value in items),
            default=0,
        )
        lines = ["", bar, title, bar]
        for index, (section_title, items) in enumerate(sections):
            if index > 0:
                lines.append(sep)
            lines.append(f"  {section_title}")
            lines.append(sep)
            lines.extend(f"  {label:<{label_width}}  : {value}" for label, value in items)
        lines.append(bar)
        return "\n".join(lines)

    def _simple_row(self, record: _StepRecord) -> str:
        drag = record.drag()
        if not self._columns_fixed:
            self._drag_column = drag is not None
            self._columns_fixed = True

        residuals = record.residuals
        cells = {
            "step": f"{record.step:d}",
            "time": f"{record.time:.3e}",
            "dt": f"{record.time_step_size:.3e}",
            "Co": "-" if record.courant is None else f"{record.courant:.3f}",
            "res(U)": "-" if "velocity" not in residuals else f"{residuals['velocity']:.2e}",
            "res(p)": "-"
            if "kinematic_pressure" not in residuals
            else f"{residuals['kinematic_pressure']:.2e}",
            "|div U|": ("-" if record.continuity_max is None else f"{record.continuity_max:.2e}"),
            "Cd": "" if drag is None else f"{drag:.4f}",
            "s/step": f"{record.elapsed:.2f}",
        }
        columns = [(name, width) for name, width in _COLUMNS if name != "Cd" or self._drag_column]

        lines = []
        if self._needs_header or self._rows_since_header >= _HEADER_ROWS:
            if self._reported_steps > 1:
                lines.append("")
            lines.append("  " + "  ".join(f"{name:>{width}}" for name, width in columns))
            lines.append("  " + "  ".join("-" * width for _name, width in columns))
            self._rows_since_header = 0
            self._needs_header = False
        lines.append("  " + "  ".join(f"{cells[name]:>{width}}" for name, width in columns))
        lines.extend(f"    ! {text}" for text in record.warnings)
        self._rows_since_header += 1
        return "\n".join(lines)

    # -- Startup and shutdown reports ------------------------------------------

    @staticmethod
    def solver_info(solver: Any, initialization_time: float) -> str:
        """Return a comprehensive FVM initialization report."""
        config = solver.setup
        mesh = solver.mesh_data
        parallel = solver.parallel
        partition = getattr(parallel, "partition", None)

        lines = ["", "=" * _WIDTH, "FVM SOLVER INFO", "=" * _WIDTH]
        lines.extend(
            Logging._section(
                "CONFIGURATION",
                [
                    ("Case Name", str(config.case_name)),
                    ("Algorithm", str(config.pimple.algorithm).upper()),
                    ("Execution Mode", str(config.execution.parallel_mode)),
                    ("MPI Ranks", str(parallel.size)),
                    ("Operator Backend", str(config.execution.operator_backend)),
                    ("Linear Backend", str(config.execution.linear_backend)),
                ],
            )
        )

        if partition is None:
            mesh_items = [
                ("Cells", f"{int(mesh['n_cells']):,}"),
                ("Faces", f"{int(mesh['n_faces']):,}"),
                ("Interior Faces", f"{int(mesh['n_interior_faces']):,}"),
                ("Boundary Patches", str(len(mesh["boundary"]))),
            ]
        else:
            mesh_items = [
                ("Global Cells", f"{int(partition.global_n_cells):,}"),
                ("Rank 0 Owned Cells", f"{int(parallel.n_owned):,}"),
                ("Rank 0 Local Faces", f"{int(mesh['n_faces']):,}"),
                ("Configured Patches", str(len(config.boundaries))),
            ]
        lines.extend(Logging._section("MESH", mesh_items))

        linear_method = config.linear.linear_solver
        pressure_method = config.linear.pressure_solver or linear_method
        momentum_method = config.linear.momentum_solver or linear_method
        lines.extend(
            Logging._section(
                "NUMERICS",
                [
                    ("Time Step Size", f"{config.time.time_step_size:.3e} s"),
                    ("End Time", f"{config.time.end_time:.3e} s"),
                    ("Time Scheme", str(config.schemes.time_scheme)),
                    ("Convection Scheme", str(config.schemes.convection_scheme)),
                    ("Gradient Scheme", str(config.schemes.gradient_scheme)),
                    ("Momentum Solver", str(momentum_method)),
                    ("Pressure Solver", str(pressure_method)),
                    ("Correctors", str(config.pimple.n_correctors)),
                    ("Outer Correctors", str(config.pimple.n_outer_correctors)),
                ],
            )
        )
        turbulence = config.turbulence
        turbulence_name = "DNS / laminar" if turbulence is None else str(turbulence.model)
        lines.extend(
            Logging._section(
                "PHYSICS",
                [
                    ("Density", f"{config.transport.density:.6g} kg/m³"),
                    ("Kinematic Viscosity", f"{config.transport.kinematic_viscosity:.6e} m²/s"),
                    ("Turbulence Model", turbulence_name),
                ],
            )
        )
        boundary_items = [
            (
                boundary.name,
                f"U={boundary.velocity_type}, p={boundary.pressure_type}, value={boundary.velocity_value}",
            )
            for boundary in config.boundaries
        ]
        lines.extend(Logging._section("BOUNDARY CONDITIONS", boundary_items))
        sink = getattr(solver, "logger", None)
        log_file_path = getattr(sink, "log_file_path", None)
        lines.extend(
            Logging._section(
                "MONITORING & I/O",
                [
                    ("Solution Directory", str(Path(solver.case_dir) / "solution")),
                    ("Log File", str(log_file_path or "disabled")),
                    ("Log Mode", str(getattr(sink, "mode", "simple"))),
                    ("Log Interval", f"{getattr(sink, 'interval_steps', 1)} steps"),
                    ("Visualization", "VTK XML, cell-centred, appended binary"),
                    ("Output Compression", str(config.output.compression).upper()),
                    (
                        "Output Scheduling",
                        "asynchronous"
                        if config.output.asynchronous and not parallel.is_partitioned
                        else "synchronous",
                    ),
                    ("Visualization Ghost Layers", str(config.output.ghost_layers)),
                    ("Initialization Time", f"{initialization_time:.3e} s"),
                ],
            )
        )
        lines.extend(["", "=" * _WIDTH, ""])
        return "\n".join(lines)

    def log_solver_info(self, solver: Any, initialization_time: float) -> None:
        """Emit the formatted initialization report."""
        self.message(self.solver_info(solver, initialization_time), flush=True)

    def solver_state(self, solver: Any) -> None:
        """Emit the current high-level solver state."""
        self.section(
            "FVM SOLVER STATE",
            [
                ("Case", str(solver.setup.case_name)),
                ("Time", f"{solver.time:.5f} s"),
                ("Step", str(solver.step)),
                ("Local Cells", f"{solver.mesh_data['n_cells']:,}"),
                ("Algorithm", str(solver.setup.pimple.algorithm)),
            ],
        )

    def output_info(self, text: str) -> None:
        """Emit one visualization/checkpoint output event."""
        self.info(text, flush=True)

    def timing(self, name: str, elapsed: float) -> None:
        """Record one completed phase with the profiler."""
        if self.profiler is not None:
            self.profiler.record(name, elapsed)

    def profile_report(self, record: dict[str, Any]) -> None:
        """Emit the per-step performance profile (debug mode, reported steps)."""
        if not self.debug or not self._step_reported:
            return
        rule = "-" * _WIDTH
        lines = [
            "",
            rule,
            "PERFORMANCE PROFILE   (this step; max over ranks)",
            rule,
            f"  {'Phase':<31} {'calls':>5} {'mean s':>10} {'max s':>10} {'share':>8} "
            f"{'max/mean':>8}",
        ]
        for phase in record["phases"]:
            lines.append(
                f"  {phase['name']:<31} "
                f"{phase['calls']['max']:5.0f} "
                f"{phase['seconds']['mean']:10.3f} "
                f"{phase['seconds']['max']:10.3f} "
                f"{100.0 * phase['critical_path_fraction']:7.1f}% "
                f"{phase['rank_imbalance']:8.3f}"
            )
        if record["linear"]:
            lines.extend([rule, "  Linear solvers"])
            for linear in record["linear"]:
                lines.append(
                    f"  {linear['equation']:<31} "
                    f"calls={linear['calls']['max']:.0f}, "
                    f"iterations(max)={linear['iterations']['max']:.0f}, "
                    f"setup(max)={linear['setup_seconds']['max']:.3f} s, "
                    f"solve(max)={linear['solve_seconds']['max']:.3f} s"
                )
        memory = record["memory"]
        lines.extend(
            [
                rule,
                f"  {'Resident, all ranks':<31}: "
                f"{memory['aggregate_rss_end_bytes'] / _MIB:.1f} MiB",
                f"  {'Peak during step':<31}: "
                f"{memory['aggregate_rss_max_observed_bytes'] / _MIB:.1f} MiB",
                f"  {'Peak since start':<31}: "
                f"{memory['aggregate_peak_rss_end_bytes'] / _MIB:.1f} MiB",
                f"  {'Slowest rank':<31}: {record['step_seconds']['max']:.3f} s",
            ]
        )
        inventory = record.get("allocation_inventory")
        if inventory is not None:
            lines.extend([rule, "  Memory by subsystem"])
            for name, values in inventory.items():
                label = _MEMORY_LABELS.get(name, name)
                lines.append(f"  {label:<31}: {values['aggregate_bytes'] / _MIB:.1f} MiB")
        lines.append(rule)
        self.message("\n".join(lines), flush=True)

    def run_summary(self) -> None:
        """Emit the closing wall-time summary."""
        if self._steps == 0 or self._step_wall_time <= 0.0:
            return
        mean = self._step_wall_time / self._steps
        self._emit("")
        self._emit(
            f"[FVM][RunTiming] steps={self._steps} total_s={self._step_wall_time:.3e} "
            f"mean_step_s={mean:.3f}",
            flush=True,
        )

    def flush(self) -> None:
        """Flush the file sink."""
        if self._file is not None and not self._closed:
            self._file.flush()

    def close(self) -> None:
        """Flush and close the file sink. This method is idempotent."""
        if self._closed:
            return
        self._flush_step()
        if not self.debug:
            self.run_summary()
        self.flush()
        if self._file is not None:
            self._file.close()
        self._closed = True


class Timer:
    """Named wall-clock timers for solver phases."""

    _timers: dict[str, float] = {}

    @staticmethod
    def start(name: str) -> None:
        """Start or restart a named timer."""
        Timer._timers[name] = time.perf_counter()

    @staticmethod
    def stop(name: str) -> float:
        """Stop a named timer and return its elapsed seconds."""
        started = Timer._timers.pop(name, None)
        return 0.0 if started is None else time.perf_counter() - started

    @staticmethod
    def log(name: str, *, sink: Any | None = None) -> float:
        """Stop a timer and record it with the configured sink."""
        elapsed = Timer.stop(name)
        if sink is not None and elapsed > 0.0:
            sink.timing(name.strip(" -"), elapsed)
        return elapsed
