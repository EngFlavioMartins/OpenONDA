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
    s += f"Build  : OpenONDA={__version__} Taichi={ti.__version__}\n"
    s += f"Arch   : {system_info}\n"
    s += f"Backend: {getattr(constants_module, 'TAICHI_BACKEND', 'UNKNOWN')}\n"
    if precision is not None:
        s += f"Precision: {precision}\n"
    s += "Exec   : VPM Solver\n"
    s += f"Date   : {date_str}\n"
    s += f"Time   : {time_str}\n"
    s += f"Host   : {hostname}\n"
    s += f"User   : {username}\n"
    s += f"PID    : {pid}\n"
    s += "* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * \n"
    s += "/ / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / /\n"

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
        print(text, flush=flush)

    @staticmethod
    def info(text: str, *, flush: bool = False) -> None:
        """Emit an informational message (prefixed with ``[INFO]``)."""
        print(f"[INFO] {text}", flush=flush)

    @staticmethod
    def warning(text: str, *, flush: bool = False) -> None:
        """Emit a warning message (prefixed with ``(Warning)``)."""
        print(f"(Warning) {text}", flush=flush)

    @staticmethod
    def flow_diagnostics(system):
        """
        Log flow diagnostics to console in a structured, section-based layout.

        Args:
            system: Solver instance with cached flow quantities
        """
        lines = []
        step = getattr(system, "step", None)
        current_time = getattr(system, "time", None)

        if step is not None and current_time is not None:
            title = f" FLOW DIAGNOSTICS  (step {step}, t = {current_time:.3e} s)"
        else:
            title = " FLOW DIAGNOSTICS"

        bar = "=" * 80
        sep = "-" * 60

        lines.append("")
        lines.append(bar)
        lines.append(title)
        lines.append(bar)

        # -- Collect all sections ------------------------------------------
        sections = []
        label_w = 0

        # Integral Quantities
        n_particles = getattr(getattr(system, "particles", None), "n_particles", None)
        int_items = [
            ("Number of Particles", f"{int(n_particles):d}" if n_particles is not None else "n/a"),
            (
                "Total Circulation (\u03a3|\u0393|)",
                f"{system.total_vortex_strength_magnitude:.3e} m\u00b3/s",
            ),
            (
                "Total Circulation (\u03a3\u0393)",
                f"({system.total_strength[0]:.3e}, {system.total_strength[1]:.3e}, {system.total_strength[2]:.3e}) m\u00b3/s",
            ),
            (
                # High precision: linear impulse is a conserved invariant, so a
                # small drift is a genuine diagnostic that .3e would round away.
                "Linear Impulse",
                f"({system.total_linear_impulse[0]:.6e}, {system.total_linear_impulse[1]:.6e}, {system.total_linear_impulse[2]:.6e}) m\u2074/s",
            ),
            (
                "Angular Impulse",
                f"({system.total_angular_impulse[0]:.6e}, {system.total_angular_impulse[1]:.6e}, {system.total_angular_impulse[2]:.6e}) m\u2075/s",
            ),
            ("Total Enstrophy", f"{system.total_enstrophy:.3e} m\u00b3/s\u00b2"),
            ("Total Helicity", f"{system.total_helicity:.3e} m\u00b2/s\u00b2"),
        ]
        for label, _ in int_items:
            label_w = max(label_w, len(label))
        sections.append(("Integral Quantities", int_items))

        # Energy Budget
        E_current = system.total_kinetic_energy
        nu_enstrophy = system.vorticity_dissipation_rate
        dE_dt = system.kinetic_energy_dissipation_rate

        energy_items = [
            ("Total Energy, E", f"{E_current:.3e} J"),
            ("Modeled dissipation, -\u222b\u03bd_eff \u03c9\u00b2", f"{nu_enstrophy:.3e} J/s"),
            ("Energy decay rate, dE/dt", f"{dE_dt:.3e} J/s"),
        ]
        for label, _ in energy_items:
            label_w = max(label_w, len(label))
        sections.append(("Energy Budget", energy_items))

        # Vortex Geometry
        try:
            center = getattr(system, "centroid_of_circulation", None)
            centroids = system.centroids_of_circulation
            geo_items = []
            if center is not None:
                geo_items.append(
                    ("Center of Vorticity", f"({center[0]:.3e}, {center[1]:.3e}, {center[2]:.3e})")
                )
            for g, cg in centroids.items():
                geo_items.append(
                    (f"Group {g} centroid", f"({cg[0]:.3e}, {cg[1]:.3e}, {cg[2]:.3e})")
                )
            if geo_items:
                for label, _ in geo_items:
                    label_w = max(label_w, len(label))
                sections.append(("Vortex Geometry", geo_items))
        except Exception:
            pass

        # Time Step Diagnostics
        try:
            hist = getattr(system, "_diagnostics_history", None)
            time_step_size_items = []
            if hist is not None and len(hist.get("observed_time_step_size", [])) > 0:
                time_step_sizes = np.array(hist["observed_time_step_size"])
                dts_nz = time_step_sizes[time_step_sizes > 0]
                if dts_nz.size > 0:
                    time_step_size_items.append(
                        (
                            "VPM observed dt (mean, median)",
                            f"({dts_nz.mean():.3e}, {np.median(dts_nz):.3e}) s",
                        )
                    )
                    time_step_size_items.append(
                        ("VPM configured dt", f"{system.time_step_size:.3e} s")
                    )
            if time_step_size_items:
                for label, _ in time_step_size_items:
                    label_w = max(label_w, len(label))
                sections.append(("Time Step Diagnostics", time_step_size_items))
        except Exception:
            pass

        # -- Render sections ----------------------------------------------
        for sec_idx, (sec_name, items) in enumerate(sections):
            if sec_idx > 0:
                lines.append(sep)
            lines.append(f"  {sec_name}")
            lines.append(sep)
            for label, value in items:
                lines.append(f"  {label:<{label_w}}  : {value}")

        lines.append(bar)
        print("\n".join(lines), flush=True)

        # Log VLM forces if VLM solver is present
        if hasattr(system, "vlm_solver") and system.vlm_solver is not None:
            Logging.vlm_forces(system)

    @staticmethod
    def _format_solver_config(system) -> list:
        """Format solver configuration section."""
        lines = []
        lines.append("\n" + "=" * 80)
        lines.append("SOLVER CONFIGURATION")
        lines.append("-" * 60)
        lines.append(
            f"  Flow Model               : {getattr(system, 'flow_model_description', system.flow_model)}"
        )
        lines.append("  Time Integration:")
        lines.append(f"    Advection              : {system.advection_scheme}")
        lines.append(f"    Stretching             : {system.stretching_scheme}")
        if getattr(system, "stretching_conserve_moments", False):
            projection = "circulation + impulses"
            if getattr(system, "stretching_conserve_energy", False):
                projection += " + energy"
            lines.append(f"    Invariant projection   : {projection}")
        vel_cfg = getattr(getattr(system, "setup", None), "velocity", None)
        if vel_cfg is not None and vel_cfg.method == "TREECODE":
            lines.append(f"    Velocity               : Treecode (Barnes-Hut, θ={vel_cfg.theta})")
        else:
            lines.append("    Velocity               : Direct (O(N²))")
        axis = getattr(getattr(system, "setup", None), "axisymmetric_no_swirl_axis", None)
        if axis is not None:
            lines.append(f"    Symmetry               : Axisymmetric no-swirl about {axis}-axis")
        lines.append(f"  Processing Unit          : {system.compute_device}")
        lines.append(f"  Time Step Size           : {system.time_step_size:.3e} s")
        lines.append(f"  Current Time Step        : {system.step}")
        lines.append(f"  Simulation Time          : {system.time:.3e} s")
        lines.append(f"  Wall-clock Time          : {system.wall_time:.2f} s")
        return lines

    @staticmethod
    def _format_particle_system(system) -> list:
        """Format particle system section."""
        lines = []
        lines.append("\n" + "-" * 60)
        lines.append("PARTICLE SYSTEM")
        lines.append("-" * 60)
        if hasattr(system, "particles") and system.particles is not None:
            lines.append(str(system.particles))
        else:
            lines.append("  Status: Not initialized")
        return lines

    @staticmethod
    def _format_physics_model(system) -> list:
        """Format physics model section."""
        lines = []
        lines.append("\n" + "-" * 60)
        lines.append("PHYSICS MODEL")
        lines.append("-" * 60)
        if hasattr(system, "physics") and system.physics is not None:
            lines.append(str(system.physics))
        else:
            lines.append("  Status: Not initialized")
        return lines

    @staticmethod
    def _format_viscous_time_step_size_limits(system) -> list:
        """Return lines showing Δt, stability/accuracy limit, and a warning if exceeded.

        Works whether or not particles are loaded — reads limits directly from
        the ViscousConfig methods (rwm_accuracy_dt, gbd_max_dt,
        dvh_required_dt) which require only ``particle_spacing`` and
        ``kinematic_viscosity`` to be set on the config.  Skips silently when those fields
        are absent.
        """
        lines = []
        try:
            visc_cfg = getattr(getattr(system, "setup", None), "viscous", None)
            if visc_cfg is None:
                return lines

            time_step_size = getattr(system, "time_step_size", None)
            if time_step_size is None:
                return lines

            scheme = getattr(system, "viscous_scheme", None) or getattr(visc_cfg, "scheme", None)
            particle_spacing = getattr(visc_cfg, "particle_spacing", None)
            nu = getattr(visc_cfg, "kinematic_viscosity", None)

            # Derive limit and label from the ViscousConfig stability methods.
            limit: float | None = None
            limit_label: str = ""

            if scheme == "RWM" and particle_spacing and particle_spacing > 0 and nu and nu > 0:
                limit = visc_cfg.rwm_accuracy_time_step_size()
                limit_label = "particle_spacing²/(4nu)"
            elif scheme == "GBD" and nu and nu > 0:
                try:
                    limit = visc_cfg.gbd_max_time_step_size()
                    limit_label = "particle_spacing²/(6nu)"
                except Exception:
                    pass
            elif scheme == "DVH" and nu and nu > 0:
                try:
                    limit = visc_cfg.dvh_required_time_step_size()
                    limit_label = "β·R_d²/(4nu)"
                except Exception:
                    pass

            if limit is not None and limit > 0:
                if scheme == "DVH":
                    lines.append(
                        f"  Required diffusion interval ({limit_label}) : {limit:.3e} s  [PINNED]"
                    )
                    return lines
                exceeded = time_step_size > limit * (1.0 + 1e-6)
                status = "*** EXCEEDS LIMIT ***" if exceeded else "OK"
                lines.append(f"  Stability limit ({limit_label})    : {limit:.3e} s  [{status}]")
                if exceeded:
                    ratio = time_step_size / limit
                    lines.append(
                        f"  WARNING: Δt is {ratio:.2f}× the {scheme} stability limit "
                        f"— solution may be UNSTABLE."
                    )
        except Exception:
            pass
        return lines

    @staticmethod
    def _format_viscous_model(system) -> list:
        """Format viscous diffusion model section."""
        lines = []
        lines.append("\n" + "-" * 60)
        lines.append("VISCOUS DIFFUSION MODEL")
        lines.append("-" * 60)
        lines.append(f"  Scheme                   : {system.viscous_scheme}")
        if system.viscous_scheme == "CS" and getattr(system, "stretching_conserve_moments", False):
            lines.append("  Core moment projection   : circulation + both impulses")

        # Configured Δt and stability/accuracy limit — always shown when
        # particle_spacing + kinematic_viscosity are set on the config.
        lines.extend(Logging._format_viscous_time_step_size_limits(system))

        if hasattr(system, "particles") and len(system.particles) > 0:
            viscosities = system.particles.kinematic_viscosity_cpu()
            viscosities_t = system.particles.eddy_viscosity_cpu()
            viscosities_eff = system.particles.effective_viscosity_cpu()

            lines.append(f"  Molecular Viscosity (nu)    : {viscosities[0]:.3e} m²/s")

            if np.max(viscosities_t) > 1e-12:
                lines.append("  Turbulent Viscosity (nuₜ):")
                lines.append(f"    Min                    : {np.min(viscosities_t):.3e} m²/s")
                lines.append(f"    Max                    : {np.max(viscosities_t):.3e} m²/s")
                lines.append(f"    Mean                   : {np.mean(viscosities_t):.3e} m²/s")
                lines.append("  Effective Viscosity (nuₑ = nu + nuₜ):")
                lines.append(f"    Min                    : {np.min(viscosities_eff):.3e} m²/s")
                lines.append(f"    Max                    : {np.max(viscosities_eff):.3e} m²/s")
                lines.append(f"    Mean                   : {np.mean(viscosities_eff):.3e} m²/s")
        else:
            visc_cfg = getattr(getattr(system, "setup", None), "viscous", None)
            nu = getattr(visc_cfg, "kinematic_viscosity", None)
            if nu is not None and nu > 0:
                lines.append(f"  Molecular Viscosity (nu)    : {nu:.3e} m²/s")
            else:
                lines.append("  Viscosity (nu)            : (not configured)")
        return lines

    @staticmethod
    def _format_turbulence_model(system) -> list:
        """Format turbulence model section."""
        lines = []
        lines.append("\n" + "-" * 60)
        lines.append("TURBULENCE MODEL")
        lines.append("-" * 60)
        if system.turbulence_model is not None:
            lines.append(str(system.turbulence_model))
        elif system.flow_model == "INVISCID":
            lines.append("  Status: Not applicable (INVISCID — stretching only)")
        elif system.flow_model == "POTENTIAL":
            lines.append("  Status: Not applicable (Potential/Inviscid Flow)")
        else:
            lines.append("  Status: Not applicable (DNS - Direct Numerical Simulation)")
        return lines

    @staticmethod
    def _format_monitoring_io(system) -> list:
        """Format monitoring and I/O section."""
        lines = []
        name = (system.checkpoint_name or "").strip()
        prefix = f"vpm_{name}" if name else "vpm"
        lines.append("\n" + "-" * 60)
        lines.append("MONITORING & I/O")
        lines.append("-" * 60)
        lines.append(f"  Snapshot Frequency       : {system.checkpoint_interval_steps} steps")
        lines.append(f"  Snapshot Prefix          : {prefix}")
        logging_freq = system.setup.logging_interval_steps
        lines.append(f"  Logging Frequency        : {logging_freq} steps")
        timing_freq = system.timing_interval_steps
        lines.append(f"  Timing Report Frequency  : {timing_freq} steps")
        return lines

    @staticmethod
    def _format_vlm_mesh_lines(vlm) -> list:
        """Format lines for a VLM solver with mesh generated."""
        lines = [
            "  Status                   : Active",
            f"  Number of Panels         : {vlm.lattice.num_panels}",
            f"  Max Panels               : {vlm.max_panels}",
            f"  Number of Surfaces       : {len(vlm.surfaces)}",
            f"  Precision                : {vlm.dtype}",
            f"  Linear Solver            : {vlm.linear_solver}",
            f"  Density                  : {vlm.density:.3f} kg/m³",
            f"  Viscosity                : {vlm.viscosity:.3e} m²/s",
            f"  Force Evaluation Method  : {vlm.force.method}",
        ]
        if len(vlm.surfaces) > 0:
            lines.append("\n  SURFACES:")
            for uid, (_aircraft, kinematics) in vlm.surfaces.items():
                kin_type = type(kinematics).__name__ if kinematics else "Static"
                lines.append(f"    - {uid}: {kin_type}")
        return lines

    @staticmethod
    def _format_vlm_solver(system) -> list:
        """Format VLM solver section."""
        lines = []
        lines.append("\n" + "-" * 60)
        lines.append("VLM SOLVER")
        lines.append("-" * 60)
        if hasattr(system, "vlm_solver") and system.vlm_solver is not None:
            vlm = system.vlm_solver
            if vlm._mesh_generated:
                lines.extend(Logging._format_vlm_mesh_lines(vlm))
            else:
                lines.append("  Status                   : Initialized (Mesh not generated)")
        else:
            lines.append("  Status                   : Not initialized")
        return lines

    @staticmethod
    def _format_panel_data_lines(ps) -> list:
        """Format lines for an active panel solver with geometry data."""
        lattice = getattr(ps, "lattice", None)
        n_panels = lattice.num_panels if lattice is not None else 0
        lines = [
            "  Status                   : Active",
            f"  Number of Panels         : {n_panels}",
            f"  Max Panels               : {ps.max_panels}",
            f"  Precision                : {ps.float_dtype}",
        ]
        if hasattr(ps, "agglomerator") and ps.agglomerator is not None:
            lines.append(
                f"  Agglomeration            : Enabled (Target: {ps.agglomeration_target})"
            )
        else:
            lines.append("  Agglomeration            : Disabled")
        if hasattr(ps, "kutta") and ps.kutta is not None:
            lines.append(f"  Kutta Condition          : Enabled ({ps.kutta.n_te_panels} TE pairs)")
        else:
            lines.append("  Kutta Condition          : Disabled")
        return lines

    @staticmethod
    def _format_panel_solver(system) -> list:
        """Format panel solver section."""
        lines = []
        lines.append("\n" + "-" * 60)
        lines.append("PANEL SOLVER")
        lines.append("-" * 60)
        if hasattr(system, "panel_solver") and system.panel_solver is not None:
            ps = system.panel_solver
            lattice = getattr(ps, "lattice", None)
            if lattice is not None:
                lines.extend(Logging._format_panel_data_lines(ps))
            else:
                lines.append("  Status                   : Initialized (No Geometry)")
        else:
            lines.append("  Status                   : Not initialized")
        return lines

    @staticmethod
    def _format_stabilization_config(system) -> list:
        """Format residual-viscosity and particle-retention settings."""
        lines = []
        lines.append("\n" + "-" * 60)
        lines.append("STABILIZATION / PARTICLE RETENTION")
        lines.append("-" * 60)

        cfg = getattr(system.setup, "stabilization", None)
        coefficient = getattr(cfg, "stretching_viscosity_coefficient", 0.0)
        if coefficient > 0.0:
            lines.append("  Stretching viscosity     : Enabled")
            lines.append(f"    C_stab                 : {coefficient:.3f}")
        else:
            lines.append("  Stretching viscosity     : Disabled")
        regularization_interval_steps = getattr(cfg, "regularization_interval_steps", 0)
        if regularization_interval_steps > 0:
            lines.append("  Conservative filter     : Enabled")
            lines.append(f"    Check every           : {regularization_interval_steps:d} steps")
            lines.append(
                "    Grid spacing          : "
                f"{getattr(cfg, 'regularization_grid_spacing', 0.0):.3e} m"
            )
            capacity_spacing = getattr(cfg, "regularization_capacity_grid_spacing", None)
            if capacity_spacing is not None:
                lines.append(
                    "    Capacity grid         : "
                    f"{capacity_spacing:.3e} m at "
                    f"{100.0 * getattr(cfg, 'regularization_capacity_fraction', 1.0):.0f}% budget"
                )
            core_radius = getattr(cfg, "regularization_core_radius", None)
            if core_radius is not None:
                lines.append(f"    Regenerated core      : {core_radius:.3e} m")
            capacity_core = getattr(cfg, "regularization_capacity_core_radius", None)
            if capacity_core is not None:
                lines.append(f"    Capacity core         : {capacity_core:.3e} m")
            lines.append(
                "    Health triggers       : "
                f"div={getattr(cfg, 'regularization_divergence_trigger', 0.0):.3f}, "
                "angle="
                f"{getattr(cfg, 'regularization_misalignment_trigger', 0.0):.1f} deg"
            )
        else:
            lines.append("  Conservative filter     : Disabled")
        bounds = getattr(cfg, "remove_particles_by_bounds", None)
        if bounds is not None:
            xmin, xmax, ymin, ymax, zmin, zmax = bounds
            lines.append("  Domain cutoff            : Enabled")
            lines.append(f"    X: [{xmin:.3e}, {xmax:.3e}] m")
            lines.append(f"    Y: [{ymin:.3e}, {ymax:.3e}] m")
            lines.append(f"    Z: [{zmin:.3e}, {zmax:.3e}] m")
        else:
            lines.append("  Domain cutoff            : Disabled")
        return lines

    @staticmethod
    def solver_info(system) -> str:
        """
        Generate comprehensive solver information including all submodels.

        This method delegates to each component's __str__ method for detailed,
        well-formatted information about the entire solver state.

        Args:
            system: Solver instance

        Returns:
            str: Formatted comprehensive information string
        """
        lines = []
        lines.append("\n" + "=" * 80)
        lines.append("SOLVER INFO")
        lines.append("=" * 80)

        # Collect all sections
        lines.extend(Logging._format_solver_config(system))
        lines.extend(Logging._format_particle_system(system))
        lines.extend(Logging._format_physics_model(system))
        lines.extend(Logging._format_stabilization_config(system))
        lines.extend(Logging._format_viscous_model(system))
        lines.extend(Logging._format_turbulence_model(system))
        lines.extend(Logging._format_monitoring_io(system))
        lines.extend(Logging._format_vlm_solver(system))
        lines.extend(Logging._format_panel_solver(system))

        lines.append("\n" + "=" * 80 + "\n")
        return "\n".join(lines)

    @staticmethod
    def solver_summary(system) -> str:
        """
        Generate detailed solver initialization summary.

        Args:
            system: Solver instance

        Returns:
            str: Formatted summary string
        """
        total_strength = system.total_vortex_strength_magnitude

        lines = []
        lines.append("-" * 60)
        lines.append("VPM SOLVER INITIALIZATION SUMMARY")
        lines.append("-" * 60)

        # Basic configuration
        lines.append("CONFIGURATION:")
        lines.append(
            f"  Flow Model               : {getattr(system, 'flow_model_description', system.flow_model)}"
        )
        lines.append("  Time Integration:")
        lines.append(f"    Advection              : {system.advection_scheme}")
        lines.append(f"    Stretching             : {system.stretching_scheme}")
        if getattr(system, "stretching_conserve_moments", False):
            projection = "circulation + impulses"
            if getattr(system, "stretching_conserve_energy", False):
                projection += " + energy"
            lines.append(f"    Invariant projection   : {projection}")
        axis = getattr(getattr(system, "setup", None), "axisymmetric_no_swirl_axis", None)
        if axis is not None:
            lines.append(f"    Symmetry               : Axisymmetric no-swirl about {axis}-axis")
        lines.append(f"  Processing Unit          : {system.compute_device}")
        lines.append(f"  Particle Kernel          : {system.particle_kernel}")
        lines.append(f"  Viscous Scheme           : {system.viscous_scheme}")
        lines.append(f"  Cutoff Radius Factor     : {DEFAULT_CUTOFF_RADIUS_FACTOR}")
        lines.append(f"  Time Step Size           : {system.time_step_size:.2e} s")

        # Turbulence model information
        if (
            system.turbulence_model is not None
            and system.turbulence_model
            and hasattr(system.turbulence_model, "get_filter_info")
        ):
            filter_info = system.turbulence_model.get_filter_info()
            lines.append("")
            lines.append("TURBULENCE MODEL:")

            lines.append(f"  Grid Filter Particles    : {filter_info['grid_filter_particles']}")
            lines.append(f"  Grid Filter Width        : {filter_info['grid_filter_width']:.4f}")
            lines.append(f"  Test Filter Particles    : {filter_info['test_filter_particles']}")
            lines.append(f"  Test Filter Width        : {filter_info['test_filter_width']:.4f}")
            lines.append(f"  Max Neighbors Needed     : {filter_info['max_neighbors_needed']}")
        elif system.flow_model != "POTENTIAL":
            lines.append("")
            lines.append("TURBULENCE MODEL:")
            lines.append(f"  Model                    : {system.flow_model}")
        else:
            lines.append("")
            lines.append("TURBULENCE MODEL:")
            lines.append("  Model                    : POTENTIAL (No turbulence)")

        # Current simulation state
        lines.append("")
        lines.append("SIMULATION STATE:")
        lines.append(f"  Current Time Step        : {system.step}")
        lines.append(f"  Simulation Time          : {system.time:.2E} s")
        lines.append(f"  Wall-clock Time          : {system.wall_time:.2E} s")
        lines.append(f"  Total Strength           : {total_strength:.2E} m³/s")

        # I/O and monitoring
        name = (system.checkpoint_name or "").strip()
        prefix = f"vpm_{name}" if name else "vpm"
        lines.append("")
        lines.append("MONITORING & I/O:")
        lines.append(f"  Snapshot Frequency       : {system.checkpoint_interval_steps} steps")
        lines.append(f"  Snapshot Prefix          : {prefix}")

        lines.append("-" * 60)
        return "\n".join(lines)

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
        # Determine model name
        model_name = "Classical Smagorinsky"

        # Ensure diagnostics have been computed
        try:
            # Gather strain-rate diagnostics if available
            strain_min = getattr(les, "strain_rate_min", None)
            strain_max = getattr(les, "strain_rate_max", None)
            strain_mean = getattr(les, "strain_rate_mean", None)

            vol_min = getattr(les, "volume_filter_min", None)
            vol_max = getattr(les, "volume_filter_max", None)
            vol_mean = getattr(les, "volume_filter_mean", None)

            nut_min = getattr(les, "viscosities_t_min", 0.0)
            nut_max = getattr(les, "viscosities_t_max", 0.0)
            nut_ratio_min = getattr(les, "viscosities_t_ratio_min", 0.0)
            nut_ratio_max = getattr(les, "viscosities_t_ratio_max", 0.0)

            print("\n" + "-" * 60)
            print(f"LES diagnostics: {model_name}")
            print("-" * 60)

            if strain_min is not None:
                print("Strain rate magnitude |S| [1/s]:")
                print(f"  Min: {strain_min:.4e}")
                print(f"  Max: {strain_max:.4e}")
                print(f"  Mean: {strain_mean:.4e}")
                print()

            if vol_min is not None:
                print("Filter width Δ [m] (volume-based):")
                print(f"  Min: {vol_min:.4e}")
                print(f"  Max: {vol_max:.4e}")
                print(f"  Mean: {vol_mean:.4e}")
                print()

            print("Turbulent viscosity nut [m2/s]:")
            print(f"  Min: {nut_min:.4e}")
            print(f"  Max: {nut_max:.4e}")
            print(f"  Ratio nut/nu range: [{nut_ratio_min:.4e}, {nut_ratio_max:.4e}]")
            print("-" * 60, flush=True)

        except Exception as e:
            print(f"(warning) Failed to compute LES diagnostics: {e}", flush=True)

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
            n_p = system.particles.n_particles

            print("-" * 60)
            print("VLM FORCES")
            print("-" * 60)
            print(f"  Step                     : {system.step}")
            print(f"  Lift Coefficient (CL)    : {forces['CL']:.3e}")
            print(f"  Drag Coefficient (CD)    : {forces['CD']:.3e}")
            print(f"  Active Particles (n)     : {n_p}")
            print("-" * 60, flush=True)

        except Exception as e:
            try:
                forces = system.vlm_solver._last_forces
                print(f"  VLM: CL={forces['CL']:.3f} n={system.particles.n_particles}")
            except Exception:
                pass
            print(f"(warning) Failed to log VLM forces: {e}", flush=True)

    @staticmethod
    def particle_cleanup(percent, particles_before, particles_removed, particles_after):
        """
        Log particle cleanup information to console.

        Args:
            percent: Percentage threshold used for removal
            particles_before: Number of particles before cleanup
            particles_removed: Number of particles removed
            particles_after: Number of particles after cleanup
        """
        print()
        print("-" * 60)
        print("PARTICLE CLEANUP: Removing Weak Particles")
        print("-" * 60)
        print(f"Threshold:         {percent:.2f}% of maximum strength")
        print(f"Particles before:  {particles_before:,}")
        print(f"Particles removed: {particles_removed:,}")
        print(f"Particles after:   {particles_after:,}")
        print(f"Removal ratio:     {100 * particles_removed / particles_before:.2f}%")
        print("-" * 60)
        print()

    @staticmethod
    def step_timing(step_elapsed, total_elapsed, detailed_timing=None):
        """
        Log timing information for a completed simulation step.

        Args:
            step_elapsed: Time taken for the current step [s]
            total_elapsed: Cumulative simulation time [s]
            detailed_timing: Optional dictionary with per-operation durations
        """
        print(f"{'Time for this step:':<23}{step_elapsed:.3e} s")
        print(f"{'Total simulation time:':<23}{total_elapsed:.3e} s", flush=True)

        if detailed_timing and len(detailed_timing) > 0:
            print("  └- Detailed breakdown:")
            for operation, duration in detailed_timing.items():
                pct = 100.0 * duration / step_elapsed if step_elapsed > 0 else 0
                print(f"     {operation:<24}: {duration:6.3f}s ({pct:5.1f}%)")
            print()

    @staticmethod
    def stretching_time_step_size_warning(
        time_step_size: float, recommended_time_step_size: float, sigma_max: float
    ) -> None:
        """Warn that the vortex-stretching step exceeds its stability limit.

        Mirrors the viscous ``*** EXCEEDS LIMIT ***`` warning in
        :meth:`_format_viscous_dt_limits`, but for the explicit stretching
        integration, whose limit ``dt_rec = C/σ_max`` is set by the resolved
        strain rather than the grid. Printed when the integrator is the source
        of a blow-up, so the user knows the time-step must be reduced.
        """
        print(
            f"  WARNING: vortex-stretching Δt={time_step_size:.3e} s > recommended "
            f"{recommended_time_step_size:.3e} s (σ_max={sigma_max:.2f} s⁻¹) — reduce time-step.",
            flush=True,
        )

    def time_step_validation_summary(self: dict):
        """Print formatted time-step validation summary."""
        print("\n" + "-" * 60)
        print(" TIME-STEP SIZING VALIDATION")
        print("-" * 60)

        print("\nPARTICLE SYSTEM PROPERTIES:")
        print(f"  Number of particles:      {self['num_particles']}")
        print(f"  Particle spacing (min):   {self['h_min']:.3e} m")
        print(f"  Particle spacing (max):   {self['h_max']:.3e} m")
        print(f"  Particle spacing (mean):  {self['h_mean']:.3e} m")
        print(f"  Spacing ratio (min/max):  {self['h_ratio']:.3f}")

        print("\nFLOW PROPERTIES:")
        print(f"  Velocity (max):           {self['u_max']:.3e} m/s")
        print(f"  Velocity (mean):          {self['u_mean']:.3e} m/s")
        print(f"  Strain rate (max):        {self['grad_u_max']:.3e} 1/s")
        print(f"  Reynolds number:          {self['reynolds_number']:.3e}")

        print("\nVISCOSITY:")
        print(f"  Molecular viscosity:      {self['nu_molecular']:.3e} m²/s")
        print(f"  Turbulent viscosity (max):{self['nu_turbulent_max']:>14.3e} m²/s")
        print(f"  Effective viscosity (max):{self['nu_eff_max']:>14.3e} m²/s")

        print("\nCURRENT CONFIGURATION:")
        print(f"  Scheme:                   {self['current_scheme']}")
        print(f"  Time-step size:           {self['current_time_step_size']:.3e} s")

        print("\nRECOMMENDED TIME-STEP LIMITS (safety factor = 0.8):")
        print(
            f"  {'Scheme':<8} {'time_step_size_limit [s]':<18} {'Limiting Factor':<18} {'Status':<15}"
        )
        print(f"  {'-' * 8} {'-' * 17} {'-' * 17} {'-' * 14}")

        for scheme_name in ["CS", "RWM", "NONE"]:
            scheme_data = self["schemes"][scheme_name]
            status = scheme_data["status"]
            limiting = scheme_data.get("limiting_component", "N/A")
            time_step_size = scheme_data["dt_limit"]

            # Color-code status (basic: just text markers)
            status_marker = "✓ " if "stable" in status else "⚠ "

            print(
                f"  {scheme_name:<8} {time_step_size:>16.3e} {limiting:<18} {status_marker}{status:<13}"
            )

        print(f"\nCOMPONENT ANALYSIS (Current scheme: {self['current_scheme']}):")
        current_scheme_data = self["schemes"][self["current_scheme"]]
        print(
            f"  Advection limit:          {current_scheme_data['advection_time_step_size_component']:.3e} s"
        )
        if "dt_diff_component" in current_scheme_data:
            print(
                f"  Diffusion limit:          {current_scheme_data['diffusion_time_step_size_component']:.3e} s"
            )
        if "dt_stretch_component" in current_scheme_data:
            print(
                f"  Stretching limit:         {current_scheme_data['stretching_time_step_size_component']:.3e} s"
            )

        if self["issues"]:
            print("\n⚠️  WARNINGS/ISSUES:")
            for issue in self["issues"]:
                print(f"  → {issue}")
        else:
            print("\n✓ No issues detected.")

        print("-" * 60 + "\n")

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
