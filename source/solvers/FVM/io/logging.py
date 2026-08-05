"""Rank-aware console and file logging for the native FVM solver."""

from __future__ import annotations

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

_BAR_WIDTH = 80
# Minimum width of the per-step diagnostics frame (matches the VPM solver's
# ``flow_diagnostics`` block, which uses a 62-column minimum).
_STEP_BAR_MIN = 64


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
    """Print the FVM banner to the current console.

    Kept as a compatibility helper; production solver output goes through
    :class:`Logging`.
    """
    print(format_openonda_header(precision), flush=True)


class Logging:
    """Single FVM output sink and VPM-style formatter.

    Rank zero tees messages to the live console and ``solution/fvm.log``.
    Worker ranks use a disabled instance, so call sites do not need their own
    MPI print guards. The console stream is resolved at emission time, which
    keeps pytest capture and user redirection working.
    """

    def __init__(
        self,
        case_dir: str | Path,
        *,
        enabled: bool = True,
        filename: str = "fvm.log",
        console: bool = True,
    ) -> None:
        self.enabled = bool(enabled)
        self.console = bool(console)
        self._file: TextIO | None = None
        self._closed = False
        self._step_wall_time = 0.0
        # Set by Solver after both the logger and MPI context exist. Worker
        # loggers are disabled for output but still feed local profiler data.
        self.profiler: Any | None = None
        # Open per-step diagnostics block: {step, flow_time, dt, sections}.
        # Sections are buffered here and rendered as one VPM-style framed block
        # by :meth:`step_timing`, so every colon aligns across the whole step.
        self._step: dict[str, Any] | None = None
        self.log_file_path: Path | None = None

        if self.enabled:
            solution_dir = Path(case_dir).resolve() / "solution"
            solution_dir.mkdir(parents=True, exist_ok=True)
            self.log_file_path = solution_dir / filename
            self._file = self.log_file_path.open(
                "w",
                buffering=1,
                encoding="utf-8",
            )

    def message(self, text: str = "", *, flush: bool = False) -> None:
        """Emit one complete message to every configured sink."""
        if not self.enabled or self._closed:
            return
        if self.console:
            print(text, file=sys.stdout, flush=flush)
        if self._file is not None:
            print(text, file=self._file, flush=True)

    def info(self, text: str, *, flush: bool = False) -> None:
        """Emit an informational message."""
        self.message(f"[INFO] {text}", flush=flush)

    def warning(self, text: str, *, flush: bool = True) -> None:
        """Emit a warning message."""
        self.message(f"(Warning) {text}", flush=flush)

    def header(self, precision: str | None = "f64") -> None:
        """Emit the OpenONDA FVM startup banner."""
        self.message(format_openonda_header(precision), flush=True)

    @staticmethod
    def _section(title: str, items: list[tuple[str, str]]) -> list[str]:
        lines = ["", "-" * 60, title, "-" * 60]
        label_width = max((len(label) for label, _value in items), default=0)
        lines.extend(f"  {label:<{label_width}}  : {value}" for label, value in items)
        return lines

    def section(self, title: str, items: list[tuple[str, str]]) -> None:
        """Emit one consistently formatted information section."""
        self.message("\n".join(self._section(title, items)))

    # -- Per-step diagnostics block (VPM ``flow_diagnostics`` style) -----------

    def _step_section(self, title: str, items: list[tuple[str, str]]) -> None:
        """Add a section to the open step block, or emit it immediately.

        When a step block is open (between :meth:`step_info` and
        :meth:`step_timing`) the section is buffered so the whole step renders
        as one aligned, framed block. Outside a step it falls back to the
        stand-alone :meth:`section` layout.
        """
        if not items:
            return
        if self._step is not None:
            self._step["sections"].append((title, list(items)))
        else:
            self.section(title, items)

    def _render_step_block(self) -> str:
        """Render the open step block as a single VPM-style framed report."""
        step = self._step
        assert step is not None
        title = (
            f" TIME STEP  (step {step['step']}, "
            f"t = {step['flow_time']:.3e} s, Δt = {step['dt']:.3e} s)"
        )
        width = max(len(title), _STEP_BAR_MIN)
        bar = "=" * width
        sep = "-" * width
        # One label width across every section so all colons line up.
        label_width = max(
            (len(label) for _title, items in step["sections"] for label, _value in items),
            default=0,
        )
        lines = ["", bar, title, bar]
        for index, (section_title, items) in enumerate(step["sections"]):
            if index > 0:
                lines.append(sep)
            lines.append(f"  {section_title}")
            lines.append(sep)
            lines.extend(f"  {label:<{label_width}}  : {value}" for label, value in items)
        lines.append(bar)
        return "\n".join(lines)

    def _flush_step_block(self) -> None:
        """Render and clear the open step block (no-op when none is open)."""
        if self._step is not None and self._step["sections"]:
            self.message(self._render_step_block())
        self._step = None

    @staticmethod
    def solver_info(solver: Any, initialization_time: float) -> str:
        """Return a comprehensive FVM initialization report."""
        config = solver.config
        mesh = solver.mesh_data
        parallel = solver.parallel
        partition = getattr(parallel, "partition", None)

        lines = ["", "=" * _BAR_WIDTH, "FVM SOLVER INFO", "=" * _BAR_WIDTH]
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
                ("Cells", f"{int(mesh['n_elements']):,}"),
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
                    ("Time Step Size", f"{config.time.delta_t:.3e} s"),
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
                    ("Kinematic Viscosity", f"{config.transport.nu:.6e} m²/s"),
                    ("Turbulence Model", turbulence_name),
                ],
            )
        )
        boundary_items = [
            (
                boundary.name,
                f"U={boundary.type_U}, p={boundary.type_p}, value={boundary.value_U}",
            )
            for boundary in config.boundaries
        ]
        lines.extend(Logging._section("BOUNDARY CONDITIONS", boundary_items))
        log_path = getattr(solver, "logger", None)
        log_file_path = getattr(log_path, "log_file_path", None)
        lines.extend(
            Logging._section(
                "MONITORING & I/O",
                [
                    ("Solution Directory", str(Path(solver.case_dir) / "solution")),
                    ("Log File", str(log_file_path or "disabled")),
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
        lines.extend(["", "=" * _BAR_WIDTH, ""])
        return "\n".join(lines)

    def log_solver_info(self, solver: Any, initialization_time: float) -> None:
        """Emit the formatted initialization report."""
        self.message(self.solver_info(solver, initialization_time), flush=True)

    def solver_state(self, solver: Any) -> None:
        """Emit the current high-level solver state."""
        self.section(
            "FVM SOLVER STATE",
            [
                ("Case", str(solver.config.case_name)),
                ("Time", f"{solver.flow_time:.5f} s"),
                ("Step", str(solver.time_step)),
                ("Local Cells", f"{solver.mesh_data['n_elements']:,}"),
                ("Algorithm", str(solver.config.pimple.algorithm)),
            ],
        )

    def step_info(self, flow_time: float, step: int, dt: float) -> None:
        """Open a new VPM-style per-step diagnostics block.

        The step header, convergence, conservation, wall-resolution and load
        sections are collected and rendered together by :meth:`step_timing` as
        one ``=``-framed block whose title carries the step and time — matching
        the VPM solver's ``flow_diagnostics`` output.
        """
        # Flush any block left open by an aborted step so nothing is dropped.
        self._flush_step_block()
        self._step = {
            "step": int(step),
            "flow_time": float(flow_time),
            "dt": float(dt),
            "sections": [],
        }

    def convergence_info(self, residuals: dict[str, float] | None) -> None:
        """Emit the nonlinear and linear convergence record."""
        if not residuals:
            return
        labels = {
            "p": "Pressure residual",
            "p_initial": "Pressure initial residual",
            "U": "Velocity residual",
            "U_increment": "Velocity increment",
            "U_x": "Velocity-x residual",
            "U_y": "Velocity-y residual",
            "U_z": "Velocity-z residual",
        }
        ordered = [
            key
            for key in ("U", "U_increment", "U_x", "U_y", "U_z", "p", "p_initial")
            if key in residuals
        ]
        ordered.extend(key for key in residuals if key not in ordered)
        items = [(labels.get(key, key), f"{float(residuals[key]):.3e}") for key in ordered]
        self._step_section("Solver Convergence", items)

    def continuity_info(self, maximum: float, total: float) -> None:
        """Emit global mass-conservation diagnostics."""
        self._step_section(
            "Conservation",
            [
                ("Maximum |div U|", f"{maximum:.3e} 1/s"),
                ("Boundary imbalance", f"{total:.3e} m³/s"),
            ],
        )

    def courant_info(self, maximum: float, dt: float, target: float) -> None:
        """Emit the global Courant-number state."""
        self._step_section(
            "Time Control",
            [("Maximum Courant number", f"{maximum:.3e}  (target ≤ {target:.3e})")],
        )

    def yplus_info(self, yplus_stats: dict[str, dict[str, float]] | None) -> None:
        """Emit wall-resolution diagnostics."""
        if not yplus_stats:
            return
        items = [
            (
                name,
                f"min={stats['min']:.3e}, max={stats['max']:.3e}, mean={stats['avg']:.3e}",
            )
            for name, stats in yplus_stats.items()
        ]
        self._step_section("Wall Resolution (y+)", items)

    def turbulence_info(
        self,
        nut: np.ndarray | None,
        nu_molecular: float,
        *,
        statistics: tuple[float, float, float] | None = None,
    ) -> None:
        """Emit turbulent-viscosity diagnostics."""
        if statistics is None:
            if nut is None:
                return
            values = np.asarray(nut)
            if values.size == 0:
                return
            minimum = float(np.min(values))
            maximum = float(np.max(values))
            mean = float(np.mean(values))
        else:
            minimum, maximum, mean = (float(value) for value in statistics)
        ratio_min = minimum / nu_molecular if nu_molecular > 0 else float("inf")
        ratio_max = maximum / nu_molecular if nu_molecular > 0 else float("inf")
        self._step_section(
            "Turbulence Diagnostics",
            [
                ("nut minimum", f"{minimum:.3e} m²/s"),
                ("nut maximum", f"{maximum:.3e} m²/s"),
                ("nut mean", f"{mean:.3e} m²/s"),
                ("nut/nu range", f"[{ratio_min:.3e}, {ratio_max:.3e}]"),
            ],
        )

    def force_info(self, forces: dict[str, Any]) -> None:
        """Emit force coefficients for every configured patch."""
        if not forces:
            return
        items = []
        for patch, values in forces.items():
            coefficients = values.get("coeffs", {})
            items.append(
                (
                    patch,
                    f"Cd={float(coefficients.get('Cd', 0.0)):.4f}, "
                    f"Cl={float(coefficients.get('Cl', 0.0)):.4f}, "
                    f"Cm={float(coefficients.get('Cm', 0.0)):.4f}",
                )
            )
        self._step_section("Aerodynamic Loads", items)

    def output_info(self, text: str) -> None:
        """Emit one visualization/checkpoint output event."""
        self.info(text, flush=True)

    def timing(self, name: str, elapsed: float, *, detailed: bool = False) -> None:
        """Emit a timing line, optionally gated by ``FVM_DETAILED_TIMING``."""
        if self.profiler is not None:
            self.profiler.record(name, elapsed)
        if detailed and os.environ.get("FVM_DETAILED_TIMING", "0") != "1":
            return
        self.message(f"  {name:<28}: {elapsed:.3e} s")

    def step_timing(self, elapsed: float) -> None:
        """Render the buffered step block, then the wall-time footer."""
        self._flush_step_block()
        self._step_wall_time += elapsed
        self.message(f"{'Time for this step:':<25}{elapsed:.3e} s")
        self.message(
            f"{'Total simulation time:':<25}{self._step_wall_time:.3e} s",
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
        # Render any step block left open by an aborted run before closing, so
        # the last step's diagnostics are never lost on shutdown.
        self._flush_step_block()
        self.flush()
        if self._file is not None:
            self._file.close()
        self._closed = True


class Timer:
    """Named wall-clock timers for profiling solver phases.

    Provides static ``start`` / ``stop`` / ``log`` methods that operate on
    a module-level dictionary of named timers.  Elapsed times are returned
    as floats (seconds) and can be optionally emitted through a
    :class:`Logging` sink.

    Timers are intentionally simple (no nesting, no cumulative totals) to
    keep the profiling overhead negligible.

    Examples
    --------
    >>> Timer.start("assembly")
    >>> ...  # do work
    >>> elapsed = Timer.stop("assembly")
    >>> print(f"Assembly took {elapsed:.3f} s")
    """

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
    def log(
        name: str,
        *,
        sink: Any | None = None,
        detailed: bool = False,
    ) -> float:
        """Stop a timer and optionally emit it through the configured sink."""
        elapsed = Timer.stop(name)
        if sink is not None and elapsed > 0.0:
            sink.timing(name.strip(" -"), elapsed, detailed=detailed)
        return elapsed
