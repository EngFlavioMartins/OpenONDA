"""Log records of the VPM solver, written to the ``openonda.vpm`` logger.

The solver configures the logger with :func:`openonda.core.log_style.configure_logging`
to write ``solution/vpm.log`` and, optionally, the console.
"""

import logging
from typing import TYPE_CHECKING
import warnings

import taichi as ti

from openonda.core import log_style
from openonda.core.schedules import EverySteps, Schedule

if TYPE_CHECKING:
    from ..boundary_elements.vlm.solver.vlm_solver import VLMSolver
    from ..core.solver import VPMSolver

LOGGER = logging.getLogger("openonda.vpm")


def run_header(precision: str, backend: str) -> log_style.FormattedText:
    """Return the report that opens a VPM run."""
    return log_style.run_header(
        "VPM", precision, ("backend", backend), ("Taichi", str(ti.__version__))
    )


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
    _reported_step: int | None = None
    _schedule: Schedule = EverySteps(1)
    _time_step_size = 1.0
    _pending_sections: dict[str, tuple[log_style.Row, ...]] = {}

    @staticmethod
    def configure(schedule: Schedule, time_step_size: float) -> None:
        """Reset the step state and set the step-report cadence of a new solver."""
        Logging._routine_messages_enabled = True
        Logging._active_step = None
        Logging._reported_step = None
        Logging._pending_sections.clear()
        Logging._schedule = schedule
        Logging._time_step_size = float(time_step_size)

    @staticmethod
    def set_routine_messages_enabled(enabled: bool) -> None:
        """Suppress routine output without affecting warnings or numerical work."""
        Logging._routine_messages_enabled = bool(enabled)

    @staticmethod
    def message(text: str = "", *, flush: bool = False) -> None:
        """Write one complete shared report, or wrap plain text as an event."""
        if not Logging._routine_messages_enabled:
            return
        if not isinstance(text, log_style.FormattedText):
            text = log_style.block_section("events", [(text, "")])
        LOGGER.info(text)

    @staticmethod
    def info(text: str, *, flush: bool = False) -> None:
        """Record a module event."""
        if Logging._routine_messages_enabled:
            Logging.record(text, flush=flush)

    @staticmethod
    def debug(text: str) -> None:
        """Record module detail in the debug logging mode."""
        if LOGGER.isEnabledFor(logging.DEBUG):
            LOGGER.debug(log_style.block_section("events", [(text, "")]))

    @staticmethod
    def warning(text: str, *, flush: bool = True) -> None:
        """Immediately report a warning, including the attempted step if active."""
        if Logging._active_step is not None:
            text = f"VPM step {Logging._active_step}: {text}"
        LOGGER.warning(text)

    @staticmethod
    def runtime_warning(text: str, *, stacklevel: int = 2) -> None:
        """Report an operator warning through Python's warning filters."""
        warnings.warn(text, RuntimeWarning, stacklevel=stacklevel + 1)

    @staticmethod
    def error(text: str) -> None:
        """Immediately report a module error."""
        LOGGER.error(text)

    @staticmethod
    def begin_step(step: int) -> None:
        """Open a bounded buffer for a trial step; never announce acceptance here.

        Routine measurements from an unreported preceding step are discarded.
        Native scientific records keep their own sampling and persistence settings.
        One latest report per component is retained, independent of particle count.
        """
        Logging.set_routine_messages_enabled(True)
        Logging._active_step = int(step)
        Logging._pending_sections.clear()

    @staticmethod
    def report_due(step: int, time: float, total_steps: int | None = None) -> bool:
        """Return whether the accepted step is reported by the logging schedule."""
        return Logging._routine_messages_enabled and (
            Logging._schedule.is_due(step, time, Logging._time_step_size)
            or (total_steps is not None and step == total_steps)
        )

    @staticmethod
    def _report_step(
        step: int,
        flow_time: float,
        wall_time: float,
        total_steps: int | None,
        rows: list[log_style.Row],
    ) -> None:
        """Publish an accepted header and all queued sections in one sink write."""
        parts = []
        if Logging._reported_step != step:
            parts.append(log_style.step_header(step, flow_time, wall_time, total_steps=total_steps))
        if rows:
            parts.append(log_style.block_section("particles", rows))
        parts.extend(
            log_style.block_section(title, values)
            for title, values in sorted(
                Logging._pending_sections.items(),
                key=lambda item: log_style.step_section_order(item[0]),
            )
        )
        Logging._pending_sections.clear()
        Logging.message(log_style.FormattedText("\n".join(parts)), flush=True)
        Logging._reported_step = step

    @staticmethod
    def end_step(
        step: int,
        flow_time: float,
        wall_time: float,
        *,
        total_steps: int | None = None,
        n_particles: int | None = None,
    ) -> None:
        """Report an accepted step when the logging schedule selects it."""
        if not Logging.report_due(step, flow_time, total_steps):
            return
        rows = [] if n_particles is None else [("active particles", n_particles)]
        Logging._report_step(step, flow_time, wall_time, total_steps, rows)

    @staticmethod
    def section(title: str, *rows: log_style.Row, flush: bool = False) -> None:
        """Queue host measurements until reporting, or explicitly flush an event."""
        if not Logging._routine_messages_enabled:
            return
        if Logging._active_step is None or flush:
            Logging.message(log_style.block_section(title, rows), flush=flush)
        else:
            Logging._pending_sections[title] = rows

    @staticmethod
    def record(topic: str, *rows: log_style.Row, flush: bool = False) -> None:
        """Submit one component's latest measurements without formatting them."""
        Logging.section(topic if rows else "events", *(rows or ((topic, ""),)), flush=flush)

    @staticmethod
    def boundary_forces(component: str, totals: dict, surfaces: dict) -> None:
        """Report computed boundary loads through the shared record formatter.

        ``totals`` and each surface record supply lift/drag in N and their
        dimensionless coefficients. Optional moment coefficients and reference
        position (m) are reported only when the owning force model supplies them.
        This formatter never computes or substitutes physical measurements.
        """
        if len(surfaces) > 1:
            for name, forces in surfaces.items():
                Logging.record(
                    f"{component} surface {name} forces",
                    ("surface", name),
                    ("lift", forces["lift"], "N"),
                    ("drag", forces["drag"], "N"),
                    ("lift coefficient", forces["lift_coefficient"]),
                    ("drag coefficient", forces["drag_coefficient"]),
                    ("panels", forces["panel_count"]),
                )
        rows = [
            ("lift", totals["lift"], "N"),
            ("drag", totals["drag"], "N"),
            ("lift coefficient", totals["lift_coefficient"]),
            ("drag coefficient", totals["drag_coefficient"]),
        ]
        for name in (
            "side_force_coefficient",
            "rolling_moment_coefficient",
            "pitching_moment_coefficient",
            "pitching_moment_coefficient_quarter_chord",
            "yawing_moment_coefficient",
        ):
            if name in totals:
                rows.append((name.replace("_", " "), totals[name]))
        if "reference_point" in totals:
            rows.append(("reference point", totals["reference_point"], "m"))
        Logging.record(f"{component} forces", *rows, flush=True)

    @staticmethod
    def warning_record(topic: str, *rows: log_style.Row) -> None:
        """Immediately publish a structured warning without routine suppression."""
        details = [(topic, ""), *rows]
        if Logging._active_step is not None:
            details.insert(0, ("VPM step", Logging._active_step))
        LOGGER.warning(log_style.block_section(topic, details))

    @staticmethod
    def flow_diagnostics(system: "VPMSolver") -> None:
        """Report the sampler's existing host integrals; perform no field reductions."""
        if not Logging._routine_messages_enabled:
            return
        values = system._flow_integrals
        source = values["kinetic_energy_rate_source"]
        rate_label = (
            "viscous estimate / density"
            if source.endswith("viscous_rate")
            else "energy rate / density"
        )
        Logging._pending_sections["energy"] = (
            ("kinetic energy / density", values["total_kinetic_energy"], "m^5/s^2"),
            (rate_label, values["kinetic_energy_rate"], "m^5/s^3"),
            ("viscous contribution", values["viscous_kinetic_energy_rate"], "m^5/s^3"),
        )
        rows = [
            ("strength, magnitude sum", values["vortex_strength_magnitude_sum"], "m^3/s"),
            ("strength, net", tuple(values["net_vortex_strength"]), "m^3/s"),
            ("linear impulse / density", tuple(values["linear_impulse"]), "m^4/s"),
            ("angular impulse / density", tuple(values["angular_impulse"]), "m^5/s"),
            ("enstrophy", values["total_enstrophy"], "m^3/s^2"),
            ("helicity", values["total_helicity"], "m^4/s^2"),
        ]
        history = system._diagnostics_history
        if history["time"] and history["time"][-1] == system.time and history["vortex_centroid"]:
            rows.append(("centroid", tuple(history["vortex_centroid"][-1]), "m"))
        Logging._pending_sections["flow integrals"] = tuple(rows)
        wall_time = getattr(system, "elapsed_wall_time", getattr(system, "wall_time", 0.0))
        Logging._report_step(
            system.step, system.time, wall_time, getattr(system, "_run_final_step", None), []
        )
        if getattr(system, "vlm_solver", None) is not None:
            Logging.vlm_forces(system)

    @staticmethod
    def startup(system: "VPMSolver") -> None:
        """Publish the resolved configuration once through the common layout."""
        Logging.message(Logging.solver_info(system), flush=True)

    @staticmethod
    def run_finished(
        system: "VPMSolver", status: str, failure: BaseException | None = None
    ) -> None:
        """Report terminal state even when routine messages are suppressed."""
        labels = {
            "resolution_lost": "Stopped (resolution limit)",
            "unstable": "Stopped (invalid particle state)",
        }
        label = labels.get(status, status.capitalize())
        rows = [
            ("status", label),
            ("invalid step" if status == "unstable" else "accepted step", system.step),
            ("physical time", system.time, "s"),
            ("elapsed", log_style.elapsed_time(system.elapsed_wall_time)),
        ]
        if failure is not None:
            rows.append((type(failure).__name__, str(failure)))
        LOGGER.info(log_style.block_report("VPM run finished", [("run", rows)]))
        Logging._pending_sections.clear()
        Logging._active_step = None

    @staticmethod
    def _format_solver_config(system: "VPMSolver") -> list:
        """Return the solver-configuration rows."""
        rows: list[log_style.Row] = [
            ("flow model", getattr(system, "flow_model_description", system.flow_model)),
            ("integrator", getattr(getattr(system, "integrator", None), "name", "unknown")),
        ]
        induction = getattr(system, "induction", None)
        induction_name = induction.__class__.__name__ if induction is not None else "unknown"
        rows.append(("induction", induction_name.removesuffix("Induction")))
        axis = system.numerics.axisymmetric_no_swirl_axis
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
    def _format_particle_system(system: "VPMSolver") -> list:
        """Return static particle-storage configuration rows."""
        setup = system.numerics
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
    def _format_physics_model(system: "VPMSolver") -> list:
        """Return the physics-model rows."""
        if hasattr(system, "physics") and system.physics is not None:
            return system.physics.report_rows()
        return [("status", "not initialized")]

    @staticmethod
    def _format_viscous_time_step_size_limits(system: "VPMSolver") -> list:
        """Report configured diffusion intervals in seconds from resolved settings.

        RWM has an accuracy bound, GBD substeps its explicit molecular stage,
        and DVH uses a required diffusion interval. A GBD macro-step exceeding
        the molecular stage limit therefore does not itself indicate instability.
        """
        config = system.numerics.viscous
        if config.kinematic_viscosity is None or config.kinematic_viscosity <= 0.0:
            return []
        if config.scheme == "rwm" and config.particle_spacing is not None:
            return [
                ("accuracy limit", config.rwm_accuracy_time_step_size(), "s"),
                ("criterion", "particle spacing squared / (4 * kinematic viscosity)"),
            ]
        if config.scheme == "gbd" and config.gbd_grid_spacing is not None:
            return [
                ("molecular explicit stage limit", config.gbd_max_time_step_size(), "s"),
                ("criterion", "GBD grid spacing squared / (6 * kinematic viscosity)"),
                ("molecular diffusion", "substepped within each macro-step when needed"),
            ]
        if config.scheme == "dvh" and config.dvh_grid_spacing is not None:
            return [("required diffusion interval", config.dvh_required_time_step_size(), "s")]
        return []

    @staticmethod
    def _format_viscous_model(system: "VPMSolver") -> list:
        """Return the viscous-diffusion-model rows."""
        rows: list[log_style.Row] = [("scheme", system.viscous_scheme)]

        # Configured time step and stability/accuracy limit — always shown when
        # particle_spacing + kinematic_viscosity are set on the config.
        rows.extend(Logging._format_viscous_time_step_size_limits(system))

        return rows

    @staticmethod
    def _format_turbulence_model(system: "VPMSolver") -> list:
        """Return the turbulence-model rows."""
        if system.turbulence_model is not None:
            return system.turbulence_model.report_rows()
        if system.flow_model == "inviscid":
            return [("status", "not applicable, inviscid, stretching only")]
        if system.flow_model == "potential":
            return [("status", "not applicable, potential flow")]
        return [("status", "not applicable, direct numerical simulation")]

    @staticmethod
    def _format_monitoring_io(system: "VPMSolver") -> list:
        """Return the monitoring and output rows."""
        case = system.case

        def cadence(schedule: object) -> str:
            return "none" if schedule is None else schedule.describe()

        return [
            ("particle frames", cadence(case.output.schedule)),
            ("backups", cadence(case.backup.schedule)),
            ("solution directory", system.solution_dir),
            ("sample directory", system.samples_dir),
        ]

    @staticmethod
    def _format_vlm_mesh_lines(vlm: "VLMSolver") -> list:
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
    def _format_vlm_solver(system: "VPMSolver") -> list:
        """Return the VLM-solver rows."""
        if hasattr(system, "vlm_solver") and system.vlm_solver is not None:
            vlm = system.vlm_solver
            if vlm._mesh_generated:
                return Logging._format_vlm_mesh_lines(vlm)
            return [("status", "initialized, mesh not generated")]
        return [("status", "not initialized")]

    @staticmethod
    def _format_stabilization_config(system: "VPMSolver") -> list:
        """Return the solution-check, stabilization, and particle-retention rows."""
        rows: list[log_style.Row] = []
        cfg = system.numerics.stabilization
        stability_limit = system.state_limits.lagrangian_cfl.maximum
        if stability_limit is None:
            rows.append(("solution stability check", "disabled"))
        else:
            rows.append(("solution stability check", "enabled"))
            rows.append(("  maximum strain increment, infinity norm", f"{stability_limit:.3f}"))
        coefficient = getattr(cfg, "selective_eddy_viscosity_coefficient", 0.0)
        if coefficient > 0.0:
            rows.append(("selective eddy viscosity", "enabled"))
            rows.append(("  c_stab", f"{coefficient:.3f}"))
        else:
            rows.append(("selective eddy viscosity", "disabled"))
        regularization_interval_steps = getattr(cfg, "regularization_interval_steps", 0)
        if regularization_interval_steps > 0:
            rows.append(("conservative filter", "enabled"))
            rows.append(("  interval", f"{regularization_interval_steps:,}", "steps"))
            rows.append(
                ("  grid spacing", f"{getattr(cfg, 'regularization_grid_spacing', 0.0):.3e}", "m")
            )
            core_radius = getattr(cfg, "regularization_core_radius", None)
            if core_radius is not None:
                rows.append(("  regenerated core", f"{core_radius:.3e}", "m"))
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
            depth = getattr(cfg, "outflow_fringe_depth", None)
            if depth is not None:
                faces = ("x-", "x+", "y-", "y+", "z-", "z+")
                for face, width in zip(faces, depth, strict=True):
                    if width > 0.0:
                        rows.append((f"  outflow fringe, {face}", f"{width:.3e}", "m"))
                rows.append(
                    (
                        "  outflow fringe attenuation",
                        f"{cfg.outflow_fringe_attenuation:.3g}",
                        "e-folds",
                    )
                )
        else:
            rows.append(("domain cutoff", "disabled"))
        return rows

    @staticmethod
    def solver_info(system: "VPMSolver") -> str:
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
        return log_style.block_report("VPM SOLVER CONFIGURATION", sections)

    @staticmethod
    def solver_summary(system: "VPMSolver") -> str:
        """Return the shorter VPM initialization summary."""
        from ..config.constants import DEFAULT_CUTOFF_RADIUS_FACTOR

        rows: list[log_style.Row] = [
            ("flow model", getattr(system, "flow_model_description", system.flow_model)),
            ("integrator", getattr(getattr(system, "integrator", None), "name", "unknown")),
        ]
        axis = system.numerics.axisymmetric_no_swirl_axis
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
        elif system.flow_model != "potential":
            rows.append(("  model", system.flow_model))
        else:
            rows.append(("  model", "potential, no turbulence"))

        rows.extend(
            (
                ("state:", ""),
                ("  current step", f"{system.step:,}"),
                ("  simulation time", f"{system.time:.2e}", "s"),
                ("  evolution wall time", f"{system.wall_time:.2e}", "s"),
                ("  vortex strength", f"{system.vortex_strength_magnitude_sum:.2e}", "m^3/s"),
            )
        )
        return log_style.section("vpm solver  initialization summary", rows)

    @staticmethod
    def les_diagnostics(system: "VPMSolver") -> None:
        """
        Log detailed LES turbulence diagnostics.

        Args:
            system: Solver instance containing `LES` turbulence model object
        """
        if not Logging._routine_messages_enabled or system.turbulence_model is None:
            return

        les = system.turbulence_model
        Logging.section(
            "turbulence",
            ("eddy viscosity, minimum", les.min_eddy_viscosity, "m^2/s"),
            ("eddy viscosity, maximum", les.max_eddy_viscosity, "m^2/s"),
            ("eddy / molecular viscosity, minimum", les.min_eddy_viscosity_ratio),
            ("eddy / molecular viscosity, maximum", les.max_eddy_viscosity_ratio),
            flush=True,
        )

    @staticmethod
    def vlm_forces(system: "VPMSolver") -> None:
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
            force_components = ", ".join(f"{forces[f'force_{axis}']:.4e}" for axis in "xyz")
            Logging.section(
                "VORTEX-LATTICE LOADS",
                ("Force (x, y, z)", force_components, "N"),
                ("Lift coefficient", f"{forces['lift_coefficient']:.3e}"),
                ("Drag coefficient", f"{forces['drag_coefficient']:.3e}"),
                ("Coefficient reference pressure", f"{forces['dynamic_pressure']:.4e}", "Pa"),
                ("Coefficient reference area", f"{forces['reference_area']:.4e}", "m^2"),
                ("Active particles", f"{system.particles.n_particles_total:,}"),
                flush=True,
            )

        except Exception as error:
            Logging.warning(f"vlm force logging failed, {error}", flush=True)

    @staticmethod
    def particle_cleanup(
        percent: float, particles_before: int, particles_removed: int, particles_after: int
    ) -> None:
        """Log one weak-particle-removal event.

        Args:
            percent: Percentage threshold used for removal
            particles_before: Number of particles before cleanup
            particles_removed: Number of particles removed
            particles_after: Number of particles after cleanup
        """
        if not Logging._routine_messages_enabled:
            return
        removal_fraction = particles_removed / particles_before if particles_before else 0.0
        Logging.record(
            "weak particle pruning",
            ("threshold", percent, "%"),
            ("particles, before", particles_before),
            ("particles, removed", particles_removed),
            ("particles, after", particles_after),
            ("fraction removed", removal_fraction),
        )

    @staticmethod
    def step_timing(
        step_elapsed: float,
        detailed_timing: dict[str, float] | None = None,
        *,
        label: str = "Step wall time",
    ) -> None:
        """
        Log timing information for a completed simulation step.

        Args:
            step_elapsed: Time taken for the current step [s]
            detailed_timing: Optional dictionary with per-operation durations
        """
        rows: list[log_style.Row] = [(label, step_elapsed, "s")]
        detailed_timing = detailed_timing or {}
        for operation, duration in detailed_timing.items():
            fraction = duration / step_elapsed if step_elapsed > 0 else 0.0
            rows.append((operation, f"{duration:.6f}", "s"))
            rows.append(("  Share of step", f"{100.0 * fraction:.1f}", "%"))
        Logging.section("TIMING", *rows)
