"""
Console/file logging for the solver: a line-buffered log stream, startup banner,
and the Logging helper.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: January 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

from datetime import datetime
import getpass

# Expose a standard-library logger for external modules that import `logger` from
# this module (e.g., `from ..io.logging import logger`). Use a NullHandler by
# default so libraries don't configure global logging handlers when imported.
import logging as _stdio_logging
import os
import platform
import socket
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
    """
    Print the OpenONDA solver header.

    Args:
        precision: Floating-point precision - 'f32' or 'f64'
    """
    now = datetime.now()
    date_str = now.strftime("%b %d %Y")
    time_str = now.strftime("%H:%M:%S")

    try:
        hostname = socket.gethostname()
    except Exception:
        hostname = "unknown"

    try:
        username = getpass.getuser()
    except Exception:
        username = "unknown"

    # Get Python and platform info
    python_version = platform.python_version()
    system_info = f"{platform.system()}; python={python_version}; arch={platform.machine()}"

    # Get hardware device name if available
    device_name = getattr(constants_module, "TAICHI_DEVICE_NAME", None)
    if device_name:
        system_info += f";device={device_name}"

    # Process ID
    # Process ID
    pid = os.getpid()

    width = 91

    s = "\n/ / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / /\n"
    s += "* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * \n"
    s += "   ░██████                                      ░██████   ░███    ░██ ░███████      ░███    \n"
    s += "  ░██   ░██                                    ░██   ░██  ░████   ░██ ░██   ░██    ░██░██   \n"
    s += " ░██     ░██ ░████████   ░███████  ░████████  ░██     ░██ ░██░██  ░██ ░██    ░██  ░██  ░██  \n"
    s += " ░██     ░██ ░██    ░██ ░██    ░██ ░██    ░██ ░██     ░██ ░██ ░██ ░██ ░██    ░██ ░█████████ \n"
    s += " ░██     ░██ ░██    ░██ ░█████████ ░██    ░██ ░██     ░██ ░██  ░██░██ ░██    ░██ ░██    ░██ \n"
    s += "  ░██   ░██  ░███   ░██ ░██        ░██    ░██  ░██   ░██  ░██   ░████ ░██   ░██  ░██    ░██ \n"
    s += "   ░██████   ░██░█████   ░███████  ░██    ░██   ░██████   ░██    ░███ ░███████   ░██    ░██ \n"
    s += "             ░██                                                                            \n"
    s += "             ░██                                                                            \n"
    s += "* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * \n"
    s += "| O pen       | " + "".ljust(width - 16) + "|\n"
    s += (
        "| O perator   | "
        + "OpenONDA: Operator for Numerical Design & Aerodynamics.".ljust(width - 16)
        + "|\n"
    )
    s += "| N umer.     | " + f"Version: {__version__}".ljust(width - 16) + "|\n"
    s += (
        "| D esign     | "
        + "Website: https://github.com/EngFlavioMartins".ljust(width - 16)
        + "|\n"
    )
    s += "| A erodyn.   | " + "VPM Solver: Vortex Particle Method".ljust(width - 16) + "|\n"
    s += "* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * \n"
    s += "/ / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / /\n"

    rows: list[log_style.Row] = [
        ("build", f"OpenONDA {__version__}, Taichi {ti.__version__}"),
        ("platform", system_info),
        ("backend", str(getattr(constants_module, "TAICHI_BACKEND", "UNKNOWN"))),
    ]
    if precision is not None:
        rows.append(("precision", precision))
    rows.extend(
        (
            ("executable", "VPM solver"),
            ("started", f"{date_str} {time_str}"),
            ("host", hostname),
            ("user", username),
            ("process", str(pid)),
        )
    )
    s += log_style.section("run", rows)

    print(s)


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
        """Emit an informational VPM message."""
        print(log_style.header("vpm", text, stamped=True), flush=flush)

    @staticmethod
    def warning(text: str, *, flush: bool = False) -> None:
        """Emit a VPM warning."""
        print(log_style.header("vpm", f"warning  {text}", stamped=True), flush=flush)

    @staticmethod
    def record(topic: str, *rows: log_style.Row, flush: bool = False) -> None:
        """Emit one routine VPM record: a topic header over indented detail rows."""
        Logging.message(log_style.record("vpm", topic, *rows, stamped=True), flush=flush)

    @staticmethod
    def warning_record(topic: str, *rows: log_style.Row, flush: bool = True) -> None:
        """Emit a VPM warning record that survives routine-message suppression."""
        print(
            log_style.record("vpm", f"warning  {topic}", *rows, stamped=True),
            flush=flush,
        )

    @staticmethod
    def flow_diagnostics(system):
        """
        Log flow diagnostics to console in a structured, section-based layout.

        Args:
            system: Solver instance with cached flow quantities
        """
        step = getattr(system, "step", None)
        current_time = getattr(system, "time", None)
        topic = "flow diagnostics"
        if step is not None and current_time is not None:
            topic = f"flow diagnostics, step {int(step):,}, t = {current_time:.3e} s"

        n_particles_total = getattr(getattr(system, "particles", None), "n_particles_total", None)
        rows: list[log_style.Row] = [
            ("integral quantities:", ""),
            (
                "  particles",
                f"{int(n_particles_total):,}" if n_particles_total is not None else "n/a",
            ),
            (
                "  vortex strength, sum of magnitudes",
                f"{system.vortex_strength_magnitude_sum:.3e}",
                "m^3/s",
            ),
        ]
        for axis, value in zip("xyz", system.net_vortex_strength, strict=False):
            rows.append((f"  vortex strength, net {axis}", f"{value:.3e}", "m^3/s"))
        # Linear and angular impulse are conserved invariants, so a small drift
        # is a genuine diagnostic that .3e would round away.
        for axis, value in zip("xyz", system.total_linear_impulse, strict=False):
            rows.append((f"  linear impulse, {axis}", f"{value:.6e}", "m^4/s"))
        for axis, value in zip("xyz", system.total_angular_impulse, strict=False):
            rows.append((f"  angular impulse, {axis}", f"{value:.6e}", "m^5/s"))
        rows.extend(
            (
                ("  enstrophy", f"{system.total_enstrophy:.3e}", "m^3/s^2"),
                ("  helicity", f"{system.total_helicity:.3e}", "m^2/s^2"),
                ("energy budget:", ""),
                ("  kinetic energy", f"{system.total_kinetic_energy:.3e}", "J"),
                (
                    "  modeled dissipation",
                    f"{system.viscous_kinetic_energy_rate:.3e}",
                    "J/s",
                ),
                ("  kinetic energy rate", f"{system.kinetic_energy_rate:.3e}", "J/s"),
            )
        )

        try:
            vortex_centroid = getattr(system, "vortex_centroid", None)
            geometry: list[log_style.Row] = []
            if vortex_centroid is not None:
                for axis, value in zip("xyz", vortex_centroid, strict=False):
                    geometry.append((f"  vortex centroid, {axis}", f"{value:.3e}", "m"))
            for group, centroid in system.vortex_centroids_by_group.items():
                for axis, value in zip("xyz", centroid, strict=False):
                    geometry.append((f"  group {group} centroid, {axis}", f"{value:.3e}", "m"))
            if geometry:
                rows.append(("vortex geometry:", ""))
                rows.extend(geometry)
        except Exception:
            pass

        try:
            history = getattr(system, "_diagnostics_history", None)
            if history is not None and len(history.get("observed_time_step_size", [])) > 0:
                observed = np.array(history["observed_time_step_size"])
                nonzero = observed[observed > 0]
                if nonzero.size > 0:
                    rows.extend(
                        (
                            ("time step:", ""),
                            ("  observed, mean", f"{nonzero.mean():.3e}", "s"),
                            ("  observed, median", f"{np.median(nonzero):.3e}", "s"),
                            ("  configured", f"{system.time_step_size:.3e}", "s"),
                        )
                    )
        except Exception:
            pass

        print(log_style.record("vpm", topic, *rows, stamped=True), flush=True)

        # Log VLM forces if VLM solver is present
        if hasattr(system, "vlm_solver") and system.vlm_solver is not None:
            Logging.vlm_forces(system)

    @staticmethod
    def _format_solver_config(system) -> list:
        """Return the solver-configuration rows."""
        rows: list[log_style.Row] = [
            ("flow model", getattr(system, "flow_model_description", system.flow_model)),
            ("time integration:", ""),
            ("  advection", system.advection_scheme),
            ("  stretching", system.stretching_scheme),
        ]
        if getattr(system, "stretching_conserve_moments", False):
            projection = "vortex strength + impulses"
            if getattr(system, "stretching_conserve_energy", False):
                projection += " + energy"
            rows.append(("  invariant projection", projection))
        if getattr(system, "stretching_reformulated", False):
            rows.append(("  formulation", "reformulated VPM"))
        vel_cfg = getattr(getattr(system, "setup", None), "velocity", None)
        if vel_cfg is not None and vel_cfg.method == "TREECODE":
            rows.append(("  velocity", f"treecode, Barnes-Hut, theta {vel_cfg.theta:g}"))
        else:
            rows.append(("  velocity", "direct, O(N^2)"))
        axis = getattr(getattr(system, "setup", None), "axisymmetric_no_swirl_axis", None)
        if axis is not None:
            rows.append(("  symmetry", f"axisymmetric no-swirl about {axis}"))
        rows.extend(
            (
                ("compute device", system.compute_device),
                ("time step", f"{system.time_step_size:.3e}", "s"),
                ("current step", f"{system.step:,}"),
                ("simulation time", f"{system.time:.3e}", "s"),
                ("wall time", f"{system.wall_time:.2f}", "s"),
            )
        )
        return rows

    @staticmethod
    def _format_particle_system(system) -> list:
        """Return the particle-system rows."""
        if hasattr(system, "particles") and system.particles is not None:
            return system.particles.report_rows()
        return [("status", "not initialized")]

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
        if system.viscous_scheme == "CS" and getattr(system, "stretching_conserve_moments", False):
            rows.append(("core moment projection", "vortex strength + both impulses"))

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
        name = (system.checkpoint_name or "").strip()
        prefix = f"vpm_{name}" if name else "vpm"
        return [
            ("snapshot interval", f"{system.checkpoint_interval_steps:,}", "steps"),
            ("snapshot prefix", prefix),
            ("logging interval", f"{system.setup.logging_interval_steps:,}", "steps"),
            ("timing report interval", f"{system.timing_interval_steps:,}", "steps"),
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
        """Return the residual-viscosity and particle-retention rows."""
        rows: list[log_style.Row] = []
        cfg = getattr(system.setup, "stabilization", None)
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
        """Return the comprehensive VPM solver report, one section per submodel."""
        sections = (
            ("vpm solver  configuration", Logging._format_solver_config(system)),
            ("vpm solver  particle system", Logging._format_particle_system(system)),
            ("vpm solver  physics model", Logging._format_physics_model(system)),
            ("vpm solver  stabilization", Logging._format_stabilization_config(system)),
            ("vpm solver  viscous diffusion", Logging._format_viscous_model(system)),
            ("vpm solver  turbulence", Logging._format_turbulence_model(system)),
            ("vpm solver  monitoring and output", Logging._format_monitoring_io(system)),
            ("vpm solver  vlm", Logging._format_vlm_solver(system)),
            ("vpm solver  panels", Logging._format_panel_solver(system)),
        )
        lines = [log_style.banner("vpm solver")]
        lines.extend(log_style.section(title, rows) for title, rows in sections)
        lines.append("")
        return "\n".join(lines)

    @staticmethod
    def solver_summary(system) -> str:
        """Return the shorter VPM initialization summary."""
        rows: list[log_style.Row] = [
            ("flow model", getattr(system, "flow_model_description", system.flow_model)),
            ("time integration:", ""),
            ("  advection", system.advection_scheme),
            ("  stretching", system.stretching_scheme),
        ]
        if getattr(system, "stretching_conserve_moments", False):
            projection = "vortex strength + impulses"
            if getattr(system, "stretching_conserve_energy", False):
                projection += " + energy"
            rows.append(("  invariant projection", projection))
        if getattr(system, "stretching_reformulated", False):
            rows.append(("  formulation", "reformulated VPM"))
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

        name = (system.checkpoint_name or "").strip()
        rows.extend(
            (
                ("state:", ""),
                ("  current step", f"{system.step:,}"),
                ("  simulation time", f"{system.time:.2e}", "s"),
                ("  wall time", f"{system.wall_time:.2e}", "s"),
                ("  vortex strength", f"{system.vortex_strength_magnitude_sum:.2e}", "m^3/s"),
                ("output:", ""),
                ("  snapshot interval", f"{system.checkpoint_interval_steps:,}", "steps"),
                ("  snapshot prefix", f"vpm_{name}" if name else "vpm"),
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
            Logging.record(
                "les diagnostics, classical smagorinsky",
                ("eddy viscosity, min", f"{les.min_eddy_viscosity:.4e}", "m^2/s"),
                ("eddy viscosity, max", f"{les.max_eddy_viscosity:.4e}", "m^2/s"),
                ("viscosity ratio, min", f"{les.min_eddy_viscosity_ratio:.4e}"),
                ("viscosity ratio, max", f"{les.max_eddy_viscosity_ratio:.4e}"),
                flush=True,
            )
        except Exception as error:
            Logging.warning(f"les diagnostics failed, {error}", flush=True)

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
            Logging.record(
                f"vlm forces, step {int(system.step):,}",
                ("lift coefficient", f"{forces['lift_coefficient']:.3e}"),
                ("drag coefficient", f"{forces['drag_coefficient']:.3e}"),
                ("particles, active", f"{system.particles.n_particles_total:,}"),
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
        rows: list[log_style.Row] = [
            ("step", f"{step_elapsed:.3e}", "s"),
            ("cumulative", f"{total_elapsed:.3e}", "s"),
        ]
        if detailed_timing and len(detailed_timing) > 0:
            rows.append(("phases:", ""))
            for operation, duration in detailed_timing.items():
                fraction = duration / step_elapsed if step_elapsed > 0 else 0.0
                rows.append((f"  {operation}", f"{duration:.6f}", "s"))
                rows.append((f"  {operation}, share", f"{100.0 * fraction:.1f}", "%"))
        Logging.record("timing", *rows, flush=True)

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

        Naming policy:
          - checkpoint_name empty/None  -> vpm.log
          - checkpoint_name provided    -> vpm_<checkpoint_name>.log
        """
        import atexit
        import sys

        global _ACTIVE_OUTPUT_REDIRECTION, _OUTPUT_ATEXIT_REGISTERED

        # A previous solver may have finished on a non-reporting step.  Startup
        # diagnostics for a newly constructed solver must never inherit that
        # step's suppression state.
        Logging.set_routine_messages_enabled(True)

        if _ACTIVE_OUTPUT_REDIRECTION is not None:
            _ACTIVE_OUTPUT_REDIRECTION.restore()

        log_mode = getattr(getattr(solver, "setup", None), "log_mode", "tee")
        solver._stdout_original = sys.stdout  # type: ignore[attr-defined]
        solver._stderr_original = sys.stderr  # type: ignore[attr-defined]

        if log_mode == "console":
            solver.log_file_path = None  # type: ignore[attr-defined]
            solver._log_file_handle = None  # type: ignore[attr-defined]
            solver._restore_output_streams = lambda: None  # type: ignore[attr-defined]
            return
        if log_mode not in {"file", "tee"}:
            raise ValueError(f"Unknown log_mode {log_mode!r}")

        checkpoint_name = (getattr(solver, "checkpoint_name", "") or "").strip()
        log_basename = f"vpm_{checkpoint_name}.log" if checkpoint_name else "vpm.log"
        log_directory = getattr(solver, "checkpoint_directory", None) or "solution"
        os.makedirs(log_directory, exist_ok=True)

        solver.log_file_path = os.path.join(log_directory, log_basename)  # type: ignore[attr-defined]
        file_handle = open(  # noqa: SIM115
            solver.log_file_path,
            "w",
            buffering=1,
            encoding="utf-8",
        )
        solver._log_file_handle = file_handle  # type: ignore[attr-defined]

        if log_mode == "tee":
            stdout_redirected = _TeeLogStream(file_handle, solver._stdout_original)
            stderr_redirected = _TeeLogStream(file_handle, solver._stderr_original)
        else:
            stdout_redirected = _LineBufferedLogStream(file_handle)
            stderr_redirected = stdout_redirected

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
