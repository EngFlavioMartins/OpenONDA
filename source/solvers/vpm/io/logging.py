"""
Console/file logging for the solver: a line-buffered log stream, startup banner,
and the Logging helper.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: January 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

from datetime import datetime

# Expose a standard-library logger for external modules that import `logger` from
# this module (e.g., `from ..io.logging import logger`). Use a NullHandler by
# default so libraries don't configure global logging handlers when imported.
import logging as _stdio_logging
import os
import platform
from typing import Any

import numpy as np
import taichi as ti

from source import log_style
from source.version import __version__

from ..config import constants as constants_module
from ..config.constants import DEFAULT_CUTOFF_RADIUS_FACTOR

logger = _stdio_logging.getLogger("vpm")
logger.addHandler(_stdio_logging.NullHandler())
# =========================================================


class _LineBufferedLogStream:
    """Minimal stream adapter so print() writes are flushed to a log file in real time."""

    def __init__(self, file_obj: "_stdio_logging.IO[str]") -> None:
        self._file_obj = file_obj

    def write(self, data: str) -> int:
        if not data:
            return 0
        written = self._file_obj.write(data)
        self._file_obj.flush()
        return written

    def flush(self) -> None:
        self._file_obj.flush()

    def isatty(self) -> bool:
        return False

    @property
    def encoding(self) -> str:
        return getattr(self._file_obj, "encoding", "utf-8")

    def fileno(self) -> int:
        return self._file_obj.fileno()  # type: ignore[return-value]


class _TeeLogStream(_LineBufferedLogStream):
    """Line-buffered stream that mirrors writes to the original console."""

    def __init__(self, file_obj, console_stream) -> None:
        super().__init__(file_obj)
        self._console_stream = console_stream

    def write(self, data: str) -> int:
        if not data:
            return 0
        super().write(data)
        self._console_stream.write(data)
        self._console_stream.flush()
        return len(data)

    def flush(self) -> None:
        super().flush()
        self._console_stream.flush()

    def isatty(self) -> bool:
        return bool(getattr(self._console_stream, "isatty", lambda: False)())

    @property
    def encoding(self) -> str:
        return getattr(self._console_stream, "encoding", "utf-8")

    def fileno(self) -> int:
        return self._console_stream.fileno()


def print_openonda_header(precision="f32"):
    """Print a small, searchable run identity instead of a decorative banner."""
    backend = getattr(constants_module, "TAICHI_BACKEND", "UNKNOWN")
    print(
        f"OpenONDA {__version__} | VPM | {backend}/{precision} | "
        f"{datetime.now():%Y-%m-%d %H:%M:%S}\n"
        f"  {platform.system()} {platform.machine()} | "
        f"Python {platform.python_version()} | Taichi {ti.__version__}",
        flush=True,
    )


class _ActiveOutputRedirection:
    """Own one process-global VPM stdout/stderr redirection."""

    def __init__(
        self,
        stdout_original,
        stderr_original,
        stdout_redirected,
        stderr_redirected,
        file_handle,
    ) -> None:
        self.stdout_original = stdout_original
        self.stderr_original = stderr_original
        self.stdout_redirected = stdout_redirected
        self.stderr_redirected = stderr_redirected
        self.file_handle = file_handle
        self.closed = False

    def restore(self) -> None:
        """Restore streams if still owned by this redirection and close its file."""
        global _ACTIVE_OUTPUT_REDIRECTION

        if self.closed:
            if _ACTIVE_OUTPUT_REDIRECTION is self:
                _ACTIVE_OUTPUT_REDIRECTION = None
            return

        import sys

        if sys.stdout is self.stdout_redirected:
            sys.stdout = self.stdout_original
        if sys.stderr is self.stderr_redirected:
            sys.stderr = self.stderr_original

        try:
            self.file_handle.flush()
        finally:
            self.file_handle.close()
            self.closed = True
            if _ACTIVE_OUTPUT_REDIRECTION is self:
                _ACTIVE_OUTPUT_REDIRECTION = None


_ACTIVE_OUTPUT_REDIRECTION: _ActiveOutputRedirection | None = None
_OUTPUT_ATEXIT_REGISTERED = False


def _restore_active_output_redirection() -> None:
    """Close the one active VPM log redirection, if any."""
    if _ACTIVE_OUTPUT_REDIRECTION is not None:
        _ACTIVE_OUTPUT_REDIRECTION.restore()


class Logging:
    """
    Centralized logging class for VPM solver output.

    All methods are static and can be called directly without instantiation.
    Example:
        >>> Logging.flow_diagnostics(solver)
        >>> Logging.solver_summary(solver)
    """

    _routine_messages_enabled = True
    _active_step: int | None = None
    _last_block_section: str | None = None
    _last_progress_wall: float | None = None
    _progress_interval_seconds = 30.0
    _record_name_width = 11
    _detail_name_width = 13
    _event_detail_name_width = 26

    @staticmethod
    def set_routine_messages_enabled(enabled: bool) -> None:
        """Enable or suppress routine records while retaining warnings and errors."""
        Logging._routine_messages_enabled = bool(enabled)

    @staticmethod
    def message(text: str = "", *, flush: bool = False) -> None:
        """
        Emit a raw message to the solver console.

        Single choke-point for all free-form solver output so that the
        backing sink (stdout today, the stdlib ``logging`` module later)
        can be swapped in one place.

        Args:
            text: The message to print.
            flush: Force a flush of the underlying stream.
        """
        if not Logging._routine_messages_enabled:
            return
        print(text, flush=flush)

    @staticmethod
    def info(text: str, *, flush: bool = False) -> None:
        """Emit an informational event in the current solver block."""
        Logging.message(f"{'Event':<{Logging._record_name_width}} | {text}", flush=flush)

    @staticmethod
    def warning(text: str, *, flush: bool = False) -> None:
        """Emit a VPM warning even when routine output is suppressed."""
        context = "" if Logging._active_step is None else f" | step {Logging._active_step}"
        print(f"{'Warning':<{Logging._record_name_width}}{context} | {text}", flush=True)

    @staticmethod
    def _status_line(
        name: str,
        step: int,
        flow_time: float,
        *,
        total_steps: int | None,
        n_particles: int | None,
        wall_time: float | None = None,
    ) -> str:
        """Format progress and diagnostics headers with stable columns."""
        step_text = f"{step:,}" if total_steps is None else f"{step:,}/{total_steps:,}"
        elapsed = ""
        if wall_time is not None:
            elapsed = f" | elapsed={log_style.elapsed_time(wall_time)}"
        particle_text = "-" if n_particles is None else f"{n_particles:,}"
        return (
            f"{name:<{Logging._record_name_width}} | step={step_text:>11} | "
            f"t={flow_time:>10.4f} s{elapsed} | N={particle_text:>9}"
        )

    @staticmethod
    def _compact_record(name: str, topic: str, rows: tuple[log_style.Row, ...]) -> str:
        """Format an event or warning as a short aligned record."""
        lines = [f"{name:<{Logging._record_name_width}} | {topic}"]
        for row in rows:
            label, value = str(row[0]), str(row[1])
            unit = f" {row[2]}" if len(row) > 2 and row[2] else ""
            lines.append(f"  {label:<{Logging._event_detail_name_width}} | {value:>12}{unit}")
        return "\n".join(lines)

    @staticmethod
    def begin_step(step: int) -> None:
        """Set event context without printing an unaccepted physical state."""
        Logging.set_routine_messages_enabled(True)
        Logging._active_step = int(step)
        Logging._last_block_section = None

    @staticmethod
    def time_step(
        step: int,
        flow_time: float,
        wall_time: float,
        *,
        total_steps: int | None = None,
        n_particles: int | None = None,
    ) -> None:
        """Report accepted progress at most every 30 wall seconds, plus endpoints."""
        previous = Logging._last_progress_wall
        final = total_steps is not None and step == total_steps
        if (
            previous is not None
            and wall_time - previous < Logging._progress_interval_seconds
            and not final
        ):
            return
        Logging._last_progress_wall = float(wall_time)
        Logging.message(
            Logging._status_line(
                "Progress",
                step,
                flow_time,
                total_steps=total_steps,
                n_particles=n_particles,
                wall_time=wall_time,
            ),
            flush=True,
        )

    @staticmethod
    def _block_rows(
        title: str,
        rows: list[log_style.Row],
        *,
        flush: bool = False,
        force: bool = False,
    ) -> None:
        """Append rows to a block, showing an uppercase section title once."""
        if not force and not Logging._routine_messages_enabled:
            return
        normalized_title = title.upper()
        show_title = Logging._last_block_section != normalized_title
        print(
            log_style.block_section(normalized_title, rows, show_title=show_title),
            flush=flush,
        )
        Logging._last_block_section = normalized_title

    @staticmethod
    def section(title: str, *rows: log_style.Row, flush: bool = False) -> None:
        """Append one explicitly named section to the current solver block."""
        Logging._block_rows(title, list(rows), flush=flush)

    @staticmethod
    def _event_rows(topic: str, rows: tuple[log_style.Row, ...]) -> list[log_style.Row]:
        nested: list[log_style.Row] = [(topic, "")]
        for row in rows:
            label = f"  {row[0]}"
            if len(row) > 2:
                nested.append((label, row[1], row[2]))
            else:
                nested.append((label, row[1]))
        return nested

    @staticmethod
    def record(topic: str, *rows: log_style.Row, flush: bool = False) -> None:
        """Emit one time-dependent event without repeating static model details."""
        Logging.message(Logging._compact_record("Event", topic, rows), flush=flush)

    @staticmethod
    def warning_record(topic: str, *rows: log_style.Row, flush: bool = True) -> None:
        """Emit a VPM warning record that survives routine-message suppression."""
        del flush
        print(Logging._compact_record("Warning", topic, rows), flush=True)

    @staticmethod
    def flow_diagnostics(system):
        """Print already-computed diagnostics only when their sampler is due."""

        def quantity(value):
            return f"{value: .4e}" if np.isfinite(value) else " unavailable"

        def vector(values):
            return "[" + ", ".join(f"{value: .3e}" for value in values) + "]"

        step = int(getattr(system, "step", 0))
        flow_time = float(getattr(system, "time", 0.0))
        count = int(system.particles.n_particles_total)
        lines = [
            Logging._status_line(
                "Diagnostics",
                step,
                flow_time,
                total_steps=getattr(system, "_run_final_step", None),
                n_particles=count,
                wall_time=getattr(system, "wall_time", None),
            ),
            f"  {'Energy':<{Logging._detail_name_width}} | "
            f"E={quantity(system.total_kinetic_energy)} J   "
            f"dE/dt={quantity(system.kinetic_energy_rate)} J/s   "
            f"viscous={quantity(system.viscous_kinetic_energy_rate)} J/s",
            f"  {'Strength':<{Logging._detail_name_width}} | "
            f"sum|alpha|={quantity(system.vortex_strength_magnitude_sum)}   "
            f"net={vector(system.net_vortex_strength)} m^3/s",
            f"  {'Impulse':<{Logging._detail_name_width}} | "
            f"linear ={vector(system.total_linear_impulse)} m^4/s",
            f"  {'':<{Logging._detail_name_width}} | "
            f"angular={vector(system.total_angular_impulse)} m^5/s",
            f"  {'Field norms':<{Logging._detail_name_width}} | "
            f"enstrophy={quantity(system.total_enstrophy)} m^3/s^2   "
            f"helicity={quantity(system.total_helicity)} m^2/s^2",
        ]
        centroid = getattr(system, "vortex_centroid", None)
        if centroid is not None:
            lines.append(f"  {'Centroid':<{Logging._detail_name_width}} | {vector(centroid)} m")
        groups = getattr(system, "vortex_centroids_by_group", {})
        if len(groups) > 1:
            for group, centroid in groups.items():
                lines.append(
                    f"  {f'Group {group}':<{Logging._detail_name_width}} | {vector(centroid)} m"
                )
        Logging.message("\n".join(lines), flush=True)
        # Scheduled diagnostics are themselves a heartbeat. No extra progress
        # line is needed immediately afterward, including in fast simulations.
        if hasattr(system, "wall_time"):
            Logging._last_progress_wall = float(system.wall_time)
        if getattr(system, "vlm_solver", None) is not None:
            Logging.vlm_forces(system)

    @staticmethod
    def startup(system) -> None:
        """Show the run essentials; retain full configuration in the owned log."""
        Logging.message(
            f"{'Solver':<{Logging._record_name_width}} | {system.viscous_scheme} | "
            f"{type(system.induction).__name__.removesuffix('Induction')} | "
            f"{system.integrator.name} | {system.compute_device}/{system.precision}\n"
            f"  {'Time step':<{Logging._detail_name_width}} | "
            f"dt={system.time_step_size:.6g} s | steps={system.case.run.steps:,}\n"
            f"  {'Particles':<{Logging._detail_name_width}} | "
            f"capacity={system.setup.max_n_particles:,} | kernel={system.particle_kernel}\n"
            f"  {'Log':<{Logging._detail_name_width}} | {system._log_path / 'vpm.log'}",
            flush=True,
        )
        schedules = []
        for sampler in system.case.samplers.samples:
            schedule = getattr(sampler, "schedule", None)
            interval = getattr(schedule, "interval", None)
            if interval is not None:
                unit = "s" if type(schedule).__name__ == "EveryTime" else "steps"
                cadence = f"every {interval} {unit}"
            elif getattr(schedule, "is_final_only", False):
                cadence = "final only"
            else:
                cadence = "configured schedule"
            schedules.append(f"{type(sampler).__name__.removesuffix('Sampler')}: {cadence}")
        if schedules:
            Logging.message(
                f"  {'Output':<{Logging._detail_name_width}} | " + "; ".join(schedules),
                flush=True,
            )
        if _ACTIVE_OUTPUT_REDIRECTION is not None:
            _ACTIVE_OUTPUT_REDIRECTION.file_handle.write(Logging.solver_info(system) + "\n")
            _ACTIVE_OUTPUT_REDIRECTION.file_handle.flush()

    @staticmethod
    def run_finished(system, status: str, failure=None) -> None:
        """Keep terminal status visible, even after suppressed routine output."""
        particles = getattr(system, "particles", None)
        print(
            Logging._status_line(
                status.capitalize(),
                system.step,
                system.time,
                total_steps=getattr(system, "_run_final_step", None),
                n_particles=getattr(particles, "n_particles_total", None),
                wall_time=getattr(system, "wall_time", 0.0),
            ),
            flush=True,
        )
        if failure is not None:
            print(f"  {type(failure).__name__}: {failure}", flush=True)

    @staticmethod
    def _format_solver_config(system) -> list:
        """Return the solver-configuration rows."""
        rows: list[log_style.Row] = [
            ("flow model", getattr(system, "flow_model_description", system.flow_model)),
            ("integrator", getattr(getattr(system, "integrator", None), "name", "unknown")),
        ]
        induction = getattr(system, "induction", None)
        induction_name = induction.__class__.__name__ if induction is not None else "unknown"
        rows.append(("induction", induction_name.removesuffix("Induction")))
        axis = getattr(getattr(system, "setup", None), "axisymmetric_no_swirl_axis", None)
        if axis is not None:
            rows.append(("  symmetry", f"axisymmetric no-swirl about {axis}"))
        rows.extend(
            (
                ("compute device", system.compute_device),
                ("time step", f"{system.time_step_size:.3e}", "s"),
            )
        )
        return rows

    @staticmethod
    def _format_particle_system(system) -> list:
        """Return static particle-storage configuration rows."""
        setup = getattr(system, "setup", None)
        rows = [
            ("particles, maximum", f"{int(getattr(setup, 'max_n_particles', 0)):,}"),
            ("precision", str(getattr(system, "precision", "unknown"))),
        ]
        workspace_bytes = getattr(system, "fmm_workspace_bytes", None)
        if workspace_bytes is not None:
            rows.extend(
                (
                    ("FMM particle capacity", f"{int(getattr(setup, 'max_n_particles', 0)):,}"),
                    ("estimated FMM workspace", f"{workspace_bytes / 1024**2:.3f}", "MiB"),
                    ("selected device", str(getattr(system, "compute_device", "unknown"))),
                    ("selected precision", str(getattr(system, "precision", "unknown"))),
                    ("particle kernel", str(getattr(system, "particle_kernel", "unknown"))),
                )
            )
        return rows

    @staticmethod
    def _format_physics_model(system) -> list:
        """Return the physics-model rows."""
        if hasattr(system, "physics") and system.physics is not None:
            return system.physics.report_rows()
        return [("status", "not initialized")]

    @staticmethod
    def _format_viscous_time_step_size_limits(system) -> list:
        """Return lines showing Δt, stability/accuracy limit, and a warning if exceeded.

        Works whether or not particles are loaded — reads limits directly from
        the ViscousConfig methods (rwm_accuracy_dt, gbd_max_dt,
        dvh_required_dt) which require only ``particle_spacing`` and
        ``kinematic_viscosity`` to be set on the config.  Skips silently when those fields
        are absent.
        """
        lines: list[log_style.Row] = []
        try:
            visc_cfg = getattr(getattr(system, "setup", None), "viscous", None)
            if visc_cfg is None:
                return lines

            time_step_size = getattr(system, "time_step_size", None)
            if time_step_size is None:
                return lines

            scheme = getattr(system, "viscous_scheme", None) or getattr(visc_cfg, "scheme", None)
            particle_spacing = getattr(visc_cfg, "particle_spacing", None)
            kinematic_viscosity = getattr(visc_cfg, "kinematic_viscosity", None)

            # Derive limit and label from the ViscousConfig stability methods.
            limit: float | None = None
            limit_label: str = ""

            if (
                scheme == "RWM"
                and particle_spacing
                and particle_spacing > 0
                and kinematic_viscosity
                and kinematic_viscosity > 0
            ):
                limit = visc_cfg.rwm_accuracy_time_step_size()
                limit_label = "particle_spacing²/(4nu)"
            elif scheme == "GBD" and kinematic_viscosity and kinematic_viscosity > 0:
                try:
                    limit = visc_cfg.gbd_max_time_step_size()
                    limit_label = "particle_spacing²/(6nu)"
                except Exception:
                    pass
            elif scheme == "DVH" and kinematic_viscosity and kinematic_viscosity > 0:
                try:
                    limit = visc_cfg.dvh_required_time_step_size()
                    limit_label = "β·R_d²/(4nu)"
                except Exception:
                    pass

            if limit is not None and limit > 0:
                if scheme == "DVH":
                    lines.append(("required diffusion interval", f"{limit:.3e}", "s, pinned"))
                    lines.append(("  from", limit_label))
                    return lines
                exceeded = time_step_size > limit * (1.0 + 1e-6)
                lines.append(("stability limit", f"{limit:.3e}", "s"))
                lines.append(("  from", limit_label))
                lines.append(("  status", "EXCEEDS LIMIT" if exceeded else "ok"))
                if exceeded:
                    ratio = time_step_size / limit
                    lines.append(
                        (
                            "  warning",
                            f"time step is {ratio:.2f}x the {scheme} stability limit,"
                            " solution may be unstable",
                        )
                    )
        except Exception:
            pass
        return lines

    @staticmethod
    def _format_viscous_model(system) -> list:
        """Return the viscous-diffusion-model rows."""
        rows: list[log_style.Row] = [("scheme", system.viscous_scheme)]

        # Configured time step and stability/accuracy limit — always shown when
        # particle_spacing + kinematic_viscosity are set on the config.
        rows.extend(Logging._format_viscous_time_step_size_limits(system))

        if hasattr(system, "particles") and len(system.particles) > 0:
            kinematic_viscosity = system.particles.kinematic_viscosity_cpu()
            viscosities_t = system.particles.eddy_viscosity_cpu()
            viscosities_eff = system.particles.effective_viscosity_cpu()

            rows.append(("molecular viscosity", f"{kinematic_viscosity[0]:.3e}", "m^2/s"))

            if np.max(viscosities_t) > 1e-12:
                rows.extend(
                    (
                        ("turbulent viscosity, min", f"{np.min(viscosities_t):.3e}", "m^2/s"),
                        ("turbulent viscosity, mean", f"{np.mean(viscosities_t):.3e}", "m^2/s"),
                        ("turbulent viscosity, max", f"{np.max(viscosities_t):.3e}", "m^2/s"),
                        ("effective viscosity, min", f"{np.min(viscosities_eff):.3e}", "m^2/s"),
                        ("effective viscosity, mean", f"{np.mean(viscosities_eff):.3e}", "m^2/s"),
                        ("effective viscosity, max", f"{np.max(viscosities_eff):.3e}", "m^2/s"),
                    )
                )
        else:
            visc_cfg = getattr(getattr(system, "setup", None), "viscous", None)
            kinematic_viscosity = getattr(visc_cfg, "kinematic_viscosity", None)
            if kinematic_viscosity is not None and kinematic_viscosity > 0:
                rows.append(("molecular viscosity", f"{kinematic_viscosity:.3e}", "m^2/s"))
            else:
                rows.append(("molecular viscosity", "not configured"))
        return rows

    @staticmethod
    def _format_turbulence_model(system) -> list:
        """Return the turbulence-model rows."""
        if system.turbulence_model is not None:
            return system.turbulence_model.report_rows()
        if system.flow_model == "INVISCID":
            return [("status", "not applicable, inviscid, stretching only")]
        if system.flow_model == "POTENTIAL":
            return [("status", "not applicable, potential flow")]
        return [("status", "not applicable, direct numerical simulation")]

    @staticmethod
    def _format_monitoring_io(system) -> list:
        """Return the monitoring and output rows."""
        return [
            ("backup interval", f"{system.case.backup.interval_steps:,}", "steps"),
            ("backup directory", system._backup_path),
            ("log directory", system._log_path),
        ]

    @staticmethod
    def _format_vlm_mesh_lines(vlm) -> list:
        """Return the rows for a VLM solver whose mesh has been generated."""
        rows: list[log_style.Row] = [
            ("status", "active"),
            ("panels", f"{vlm.lattice.n_panels:,}"),
            ("panels, max", f"{vlm.max_n_panels:,}"),
            ("surfaces", f"{len(vlm.surfaces):,}"),
            ("precision", str(vlm.dtype)),
            ("linear solver", str(vlm.linear_solver)),
            ("density", f"{vlm.density:.3f}", "kg/m^3"),
            ("kinematic viscosity", f"{vlm.kinematic_viscosity:.3e}", "m^2/s"),
            ("force evaluation", str(vlm.force.method)),
        ]
        if len(vlm.surfaces) > 0:
            rows.append(("surfaces:", ""))
            for uid, (_aircraft, kinematics) in vlm.surfaces.items():
                rows.append((f"  {uid}", type(kinematics).__name__ if kinematics else "static"))
        return rows

    @staticmethod
    def _format_vlm_solver(system) -> list:
        """Return the VLM-solver rows."""
        if hasattr(system, "vlm_solver") and system.vlm_solver is not None:
            vlm = system.vlm_solver
            if vlm._mesh_generated:
                return Logging._format_vlm_mesh_lines(vlm)
            return [("status", "initialized, mesh not generated")]
        return [("status", "not initialized")]

    @staticmethod
    def _format_panel_data_lines(ps) -> list:
        """Return the rows for an active panel solver with geometry data."""
        lattice = getattr(ps, "lattice", None)
        n_panels = lattice.n_panels if lattice is not None else 0
        rows: list[log_style.Row] = [
            ("status", "active"),
            ("panels", f"{n_panels:,}"),
            ("panels, max", f"{ps.max_n_panels:,}"),
            ("precision", str(ps.float_dtype)),
        ]
        if hasattr(ps, "agglomerator") and ps.agglomerator is not None:
            rows.append(("agglomeration", f"enabled, target {ps.agglomeration_target}"))
        else:
            rows.append(("agglomeration", "disabled"))
        if hasattr(ps, "kutta") and ps.kutta is not None:
            rows.append(("kutta condition", f"enabled, {ps.kutta.n_te_panels} TE pairs"))
        else:
            rows.append(("kutta condition", "disabled"))
        return rows

    @staticmethod
    def _format_panel_solver(system) -> list:
        """Return the panel-solver rows."""
        if hasattr(system, "panel_solver") and system.panel_solver is not None:
            ps = system.panel_solver
            if getattr(ps, "lattice", None) is not None:
                return Logging._format_panel_data_lines(ps)
            return [("status", "initialized, no geometry")]
        return [("status", "not initialized")]

    @staticmethod
    def _format_stabilization_config(system) -> list:
        """Return the solution-check, stabilization, and particle-retention rows."""
        rows: list[log_style.Row] = []
        cfg = getattr(system.setup, "stabilization", None)
        stability_limit = system.health_limits.lagrangian_cfl.maximum
        if stability_limit is None:
            rows.append(("solution stability check", "disabled"))
        else:
            rows.append(("solution stability check", "enabled"))
            rows.append(("  maximum strain increment, infinity norm", f"{stability_limit:.3f}"))
        coefficient = getattr(cfg, "stretching_viscosity_coefficient", 0.0)
        if coefficient > 0.0:
            rows.append(("stretching viscosity", "enabled"))
            rows.append(("  c_stab", f"{coefficient:.3f}"))
        else:
            rows.append(("stretching viscosity", "disabled"))
        regularization_interval_steps = getattr(cfg, "regularization_interval_steps", 0)
        if regularization_interval_steps > 0:
            rows.append(("conservative filter", "enabled"))
            rows.append(("  interval", f"{regularization_interval_steps:,}", "steps"))
            rows.append(
                ("  grid spacing", f"{getattr(cfg, 'regularization_grid_spacing', 0.0):.3e}", "m")
            )
            capacity_spacing = getattr(cfg, "regularization_capacity_grid_spacing", None)
            if capacity_spacing is not None:
                rows.append(("  capacity grid spacing", f"{capacity_spacing:.3e}", "m"))
                rows.append(
                    (
                        "  capacity budget",
                        f"{100.0 * getattr(cfg, 'regularization_capacity_fraction', 1.0):.0f}",
                        "%",
                    )
                )
            core_radius = getattr(cfg, "regularization_core_radius", None)
            if core_radius is not None:
                rows.append(("  regenerated core", f"{core_radius:.3e}", "m"))
            capacity_core = getattr(cfg, "regularization_capacity_core_radius", None)
            if capacity_core is not None:
                rows.append(("  capacity core", f"{capacity_core:.3e}", "m"))
            radius_trigger = getattr(cfg, "regularization_core_radius_trigger", None)
            if radius_trigger is not None:
                rows.append(("  trigger, core radius", f"{radius_trigger:.3e}", "m"))
            divergence_trigger = getattr(cfg, "regularization_divergence_trigger", None)
            if divergence_trigger is not None:
                rows.append(("  trigger, divergence", f"{divergence_trigger:.3f}"))
            misalignment_trigger = getattr(cfg, "regularization_misalignment_trigger", None)
            if misalignment_trigger is not None:
                rows.append(("  trigger, misalignment", f"{misalignment_trigger:.1f}", "deg"))
        else:
            rows.append(("conservative filter", "disabled"))
        bounds = getattr(cfg, "remove_particles_by_bounds", None)
        if bounds is not None:
            xmin, xmax, ymin, ymax, zmin, zmax = bounds
            rows.append(("domain cutoff", "enabled"))
            rows.append(("  bounds, x", f"[{xmin:.3e}, {xmax:.3e}]", "m"))
            rows.append(("  bounds, y", f"[{ymin:.3e}, {ymax:.3e}]", "m"))
            rows.append(("  bounds, z", f"[{zmin:.3e}, {zmax:.3e}]", "m"))
        else:
            rows.append(("domain cutoff", "disabled"))
        return rows

    @staticmethod
    def solver_info(system) -> str:
        """Return the comprehensive configuration report printed once at time zero."""
        sections: list[tuple[str, list[log_style.Row]]] = [
            ("RUN", Logging._format_solver_config(system)),
            ("PARTICLE SYSTEM", Logging._format_particle_system(system)),
            ("PHYSICS MODEL", Logging._format_physics_model(system)),
            ("STABILIZATION", Logging._format_stabilization_config(system)),
            ("VISCOUS DIFFUSION", Logging._format_viscous_model(system)),
            ("TURBULENCE MODEL", Logging._format_turbulence_model(system)),
            ("OUTPUT AND MONITORING", Logging._format_monitoring_io(system)),
        ]
        if getattr(system, "vlm_solver", None) is not None:
            sections.append(("VORTEX-LATTICE METHOD", Logging._format_vlm_solver(system)))
        if getattr(system, "panel_solver", None) is not None:
            sections.append(("PANEL METHOD", Logging._format_panel_solver(system)))
        return log_style.block_report("VPM SOLVER CONFIGURATION", sections)

    @staticmethod
    def solver_summary(system) -> str:
        """Return the shorter VPM initialization summary."""
        rows: list[log_style.Row] = [
            ("flow model", getattr(system, "flow_model_description", system.flow_model)),
            ("integrator", getattr(getattr(system, "integrator", None), "name", "unknown")),
        ]
        axis = getattr(getattr(system, "setup", None), "axisymmetric_no_swirl_axis", None)
        if axis is not None:
            rows.append(("  symmetry", f"axisymmetric no-swirl about {axis}"))
        rows.extend(
            (
                ("compute device", system.compute_device),
                ("particle kernel", system.particle_kernel),
                ("viscous scheme", system.viscous_scheme),
                ("cutoff radius factor", str(DEFAULT_CUTOFF_RADIUS_FACTOR)),
                ("time step", f"{system.time_step_size:.2e}", "s"),
                ("turbulence:", ""),
            )
        )

        if (
            system.turbulence_model is not None
            and system.turbulence_model
            and hasattr(system.turbulence_model, "get_filter_info")
        ):
            filter_info = system.turbulence_model.get_filter_info()
            rows.extend(
                (
                    ("  grid filter particles", f"{filter_info['grid_filter_particles']:,}"),
                    ("  grid filter width", f"{filter_info['grid_filter_width']:.4f}"),
                    ("  test filter particles", f"{filter_info['test_filter_particles']:,}"),
                    ("  test filter width", f"{filter_info['test_filter_width']:.4f}"),
                    ("  max neighbours needed", f"{filter_info['max_neighbors_needed']:,}"),
                )
            )
        elif system.flow_model != "POTENTIAL":
            rows.append(("  model", system.flow_model))
        else:
            rows.append(("  model", "potential, no turbulence"))

        rows.extend(
            (
                ("state:", ""),
                ("  current step", f"{system.step:,}"),
                ("  simulation time", f"{system.time:.2e}", "s"),
                ("  wall time", f"{system.wall_time:.2e}", "s"),
                ("  vortex strength", f"{system.vortex_strength_magnitude_sum:.2e}", "m^3/s"),
                ("output:", ""),
                ("  backup interval", f"{system.case.backup.interval_steps:,}", "steps"),
                ("  backup prefix", "vpm"),
            )
        )
        return log_style.section("vpm solver  initialization summary", rows)

    @staticmethod
    def les_diagnostics(system):
        """
        Log detailed LES turbulence diagnostics.

        Args:
            system: Solver instance containing `LES` turbulence model object
        """
        # Only log if LES model is active
        if system.turbulence_model is None:
            return

        les = system.turbulence_model
        try:
            title = (
                "INITIAL TURBULENCE QUANTITIES"
                if int(getattr(system, "step", 0)) == 0
                else "TURBULENCE QUANTITIES"
            )
            Logging.section(
                title,
                ("Eddy viscosity", ""),
                ("  Minimum", f"{les.min_eddy_viscosity:.4e}", "m^2/s"),
                ("  Maximum", f"{les.max_eddy_viscosity:.4e}", "m^2/s"),
                ("Eddy-to-molecular viscosity ratio", ""),
                ("  Minimum", f"{les.min_eddy_viscosity_ratio:.4e}"),
                ("  Maximum", f"{les.max_eddy_viscosity_ratio:.4e}"),
                flush=True,
            )
        except Exception as error:
            Logging.warning(f"LES diagnostics failed: {error}", flush=True)

    @staticmethod
    def vlm_forces(system):
        """
        Log VLM surface forces and coefficients to console.

        Args:
            system: Solver instance with VLM solver
        """
        try:
            vlm = system.vlm_solver
            if vlm is None or not vlm._solved:
                return
            if not hasattr(vlm, "_last_forces"):
                return

            forces = vlm._last_forces
            Logging.section(
                "VORTEX-LATTICE LOADS",
                ("Lift coefficient", f"{forces['lift_coefficient']:.3e}"),
                ("Drag coefficient", f"{forces['drag_coefficient']:.3e}"),
                ("Active particles", f"{system.particles.n_particles_total:,}"),
                flush=True,
            )

        except Exception as error:
            Logging.warning(f"vlm force logging failed, {error}", flush=True)

    @staticmethod
    def particle_cleanup(percent, particles_before, particles_removed, particles_after):
        """Log one weak-particle-removal event.

        Args:
            percent: Percentage threshold used for removal
            particles_before: Number of particles before cleanup
            particles_removed: Number of particles removed
            particles_after: Number of particles after cleanup
        """
        removal_fraction = particles_removed / particles_before if particles_before else 0.0
        Logging.record(
            "weak particle pruning",
            ("threshold", f"{percent:.6g}", "%"),
            ("particles, before", f"{int(particles_before):,}"),
            ("particles, removed", f"{int(particles_removed):,}"),
            ("particles, after", f"{int(particles_after):,}"),
            ("fraction removed", f"{removal_fraction:.6f}"),
        )

    @staticmethod
    def step_timing(step_elapsed, total_elapsed, detailed_timing=None):
        """
        Log timing information for a completed simulation step.

        Args:
            step_elapsed: Time taken for the current step [s]
            total_elapsed: Cumulative simulation time [s]
            detailed_timing: Optional dictionary with per-operation durations
        """
        if not detailed_timing:
            return
        rows: list[log_style.Row] = [("Wall time, total", f"{total_elapsed:.6f}", "s")]
        for operation, duration in detailed_timing.items():
            fraction = duration / step_elapsed if step_elapsed > 0 else 0.0
            rows.append((operation, f"{duration:.6f}", "s"))
            rows.append(("  Share of step", f"{100.0 * fraction:.1f}", "%"))
        Logging.section("DETAILED WALL-CLOCK TIMING", *rows, flush=True)

    @staticmethod
    def stretching_time_step_size_warning(
        time_step_size: float,
        recommended_time_step_size: float,
        max_strain_rate: float,
    ) -> None:
        """Record an explicit-stretching stability-limit violation."""
        Logging.warning_record(
            "stretching exceeds its stability limit",
            ("time step", f"{time_step_size:.3e}", "s"),
            ("stability limit", f"{recommended_time_step_size:.3e}", "s"),
            ("strain rate, max", f"{max_strain_rate:.3e}", "1/s"),
        )

    def time_step_validation_summary(self: dict):
        """Print the time-step sizing validation report."""
        rows: list[log_style.Row] = [
            ("particle system:", ""),
            ("  particles", f"{int(self['n_particles_total']):,}"),
            ("  particle spacing, min", f"{self['min_particle_spacing']:.3e}", "m"),
            ("  particle spacing, mean", f"{self['mean_particle_spacing']:.3e}", "m"),
            ("  particle spacing, max", f"{self['max_particle_spacing']:.3e}", "m"),
            ("  spacing ratio, min over max", f"{self['particle_spacing_ratio']:.3f}"),
            ("flow:", ""),
            ("  velocity, max", f"{self['max_velocity_magnitude']:.3e}", "m/s"),
            ("  velocity, mean", f"{self['mean_velocity_magnitude']:.3e}", "m/s"),
            (
                "  velocity gradient, max",
                f"{self['max_velocity_gradient_magnitude']:.3e}",
                "1/s",
            ),
            ("  reynolds number", f"{self['reynolds_number']:.3e}"),
            ("viscosity:", ""),
            ("  molecular kinematic", f"{self['max_kinematic_viscosity']:.3e}", "m^2/s"),
            ("  eddy, max", f"{self['max_eddy_viscosity']:.3e}", "m^2/s"),
            ("  effective, max", f"{self['max_effective_viscosity']:.3e}", "m^2/s"),
            ("configuration:", ""),
            ("  scheme", self["viscous_scheme"]),
            ("  time step", f"{self['time_step_size']:.3e}", "s"),
            ("time step limits, safety factor 0.8:", ""),
        ]

        for scheme_name in ("CS", "RWM", "NONE"):
            scheme_data = self["schemes"][scheme_name]
            status = scheme_data["status"]
            rows.append(
                (f"  {scheme_name}, limit", f"{scheme_data['time_step_size_limit']:.3e}", "s")
            )
            rows.append(
                (f"  {scheme_name}, limiting factor", scheme_data.get("limiting_component", "n/a"))
            )
            rows.append((f"  {scheme_name}, status", status))

        current = self["schemes"][self["viscous_scheme"]]
        rows.append((f"component limits, scheme {self['viscous_scheme']}:", ""))
        rows.append(("  advection", f"{current['advection_time_step_size_limit']:.3e}", "s"))
        if "diffusion_time_step_size_limit" in current:
            rows.append(("  diffusion", f"{current['diffusion_time_step_size_limit']:.3e}", "s"))
        if "stretching_time_step_size_limit" in current:
            rows.append(("  stretching", f"{current['stretching_time_step_size_limit']:.3e}", "s"))

        if self["issues"]:
            rows.append(("issues:", ""))
            rows.extend((f"  ({index + 1})", issue) for index, issue in enumerate(self["issues"]))
        else:
            rows.append(("issues", "none detected"))

        print(log_style.record("vpm", "time step sizing validation", *rows, stamped=True) + "\n")

    @staticmethod
    def setup_output_redirection(solver: Any) -> None:
        """Configure process-global VPM output redirection.

        Only one solver can own ``sys.stdout``/``sys.stderr`` at a time. Before
        a new solver takes ownership, any previous VPM redirection is restored
        and its log file is closed. This prevents sequential solver construction
        from leaking one file descriptor per solver.

        The canonical log filename is always ``vpm.log``.
        """
        import atexit
        import sys

        global _ACTIVE_OUTPUT_REDIRECTION, _OUTPUT_ATEXIT_REGISTERED

        # A previous solver may have finished on a non-reporting step.  Startup
        # diagnostics for a newly constructed solver must never inherit that
        # step's suppression state.
        Logging.set_routine_messages_enabled(True)
        Logging._active_step = None
        Logging._last_block_section = None
        Logging._last_progress_wall = None

        if _ACTIVE_OUTPUT_REDIRECTION is not None:
            _ACTIVE_OUTPUT_REDIRECTION.restore()

        solver._stdout_original = sys.stdout  # type: ignore[attr-defined]
        solver._stderr_original = sys.stderr  # type: ignore[attr-defined]

        log_basename = "vpm.log"
        log_directory = solver._log_path
        os.makedirs(log_directory, exist_ok=True)

        solver.log_file_path = os.path.join(log_directory, log_basename)  # type: ignore[attr-defined]
        file_handle = open(  # noqa: SIM115
            solver.log_file_path,
            "w",
            buffering=1,
            encoding="utf-8",
        )
        solver._log_file_handle = file_handle  # type: ignore[attr-defined]

        stdout_redirected = _TeeLogStream(file_handle, solver._stdout_original)
        stderr_redirected = _TeeLogStream(file_handle, solver._stderr_original)

        sys.stdout = stdout_redirected  # type: ignore[assignment]
        sys.stderr = stderr_redirected  # type: ignore[assignment]

        redirection = _ActiveOutputRedirection(
            solver._stdout_original,
            solver._stderr_original,
            stdout_redirected,
            stderr_redirected,
            file_handle,
        )
        _ACTIVE_OUTPUT_REDIRECTION = redirection
        solver._restore_output_streams = redirection.restore  # type: ignore[attr-defined]

        if not _OUTPUT_ATEXIT_REGISTERED:
            atexit.register(_restore_active_output_redirection)
            _OUTPUT_ATEXIT_REGISTERED = True
