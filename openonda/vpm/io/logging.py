"""Log reports of the VPM solver: the diagnostics of a step and the reports of a run.

The solver writes through a :class:`~openonda.core.log_style.StepLog` held in
``solver.logger``. The functions here build its sections and reports.
"""

from typing import TYPE_CHECKING

import taichi as ti

from openonda.core import log_style

if TYPE_CHECKING:
    from ..core.solver import VPMSolver
    from ..vlm.solver import VLMSolver

FLOW_MODEL_DESCRIPTIONS = {
    "dns": "DNS ::: (ω·∇)u + ν∇²ω",
    "les": "LES ::: (ω·∇)u + (ν+νt)∇²ω",
    "inviscid": "INV ::: (ω·∇)u",
}
"""Vorticity equation of each flow model, as the run section of the log shows it."""


def run_header(precision: str, backend: str) -> log_style.FormattedText:
    """Return the report that opens a VPM run."""
    return log_style.run_header(
        "VPM", precision, ("backend", backend), ("Taichi", str(ti.__version__))
    )


def flow_diagnostics(system: "VPMSolver") -> None:
    """Publish the step with the sampler's host integrals; perform no field reductions."""
    log = system.logger
    if log.muted:
        return
    values = system._flow_integrals
    source = values["kinetic_energy_rate_source"]
    rate_label = (
        "viscous estimate / density" if source.endswith("viscous_rate") else "energy rate / density"
    )
    energy = (
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
    log.publish(
        system.step,
        system.time,
        system.elapsed_wall_time,
        total_steps=None if system._external_clock else system._run_final_step,
        sections=(("energy", energy), ("flow integrals", tuple(rows))),
    )
    if system.vlm_solver is not None:
        vlm_forces(system)


def startup(system: "VPMSolver") -> None:
    """Write the resolved configuration once."""
    system.logger.report(solver_info(system))


def run_finished(system: "VPMSolver", status: str, failure: BaseException | None = None) -> None:
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
    system.logger.report(log_style.block_report("VPM run finished", [("run", rows)]))
    system.logger.abandon_step()


def _solver_rows(system: "VPMSolver") -> list[log_style.Row]:
    """Return the rows of the run section."""
    induction = system.induction
    rows: list[log_style.Row] = [
        ("flow model", FLOW_MODEL_DESCRIPTIONS[system.flow_model]),
        ("integrator", system.integrator.name),
        ("induction", type(induction).__name__.removesuffix("Induction")),
    ]
    # Only the slab induction runs its image fields on a mesh backend.
    mesh_backend = getattr(induction, "mesh_execution_backend", None)
    if mesh_backend is not None:
        rows.append(("  slab image fields", mesh_backend))
    rows.extend(
        (
            ("compute device", system.compute_device),
            ("time step", f"{system.time_step_size:.3e}", "s"),
        )
    )
    return rows


def _particle_system_rows(system: "VPMSolver") -> list[log_style.Row]:
    """Return the rows of the particle storage."""
    capacity = f"{system.numerics.max_n_particles:,}"
    rows: list[log_style.Row] = [
        ("particles, maximum", capacity),
        ("precision", str(system.precision)),
    ]
    if system.fmm_workspace_bytes is not None:
        rows.extend(
            (
                ("FMM particle capacity", capacity),
                ("estimated FMM workspace", f"{system.fmm_workspace_bytes / 1024**2:.3f}", "MiB"),
                ("selected device", str(system.compute_device)),
                ("selected precision", str(system.precision)),
                ("particle kernel", str(system.particle_kernel)),
            )
        )
    return rows


def _diffusion_interval_rows(system: "VPMSolver") -> list[log_style.Row]:
    """Return the configured diffusion intervals in s from the resolved settings.

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


def _turbulence_rows(system: "VPMSolver") -> list[log_style.Row]:
    """Return the rows of the turbulence model."""
    if system.turbulence_model is not None:
        return system.turbulence_model.report_rows()
    if system.spectral_closure is not None:
        return system.spectral_closure.report_rows()
    if system.flow_model == "inviscid":
        return [("status", "not applicable, inviscid, stretching only")]
    return [("status", "not applicable, direct numerical simulation")]


def _output_rows(system: "VPMSolver") -> list[log_style.Row]:
    """Return the rows of the monitoring and output section."""
    case = system.case

    def cadence(schedule: object) -> str:
        return "none" if schedule is None else schedule.describe()

    return [
        ("particle frames", cadence(case.output.schedule)),
        ("backups", cadence(case.backup.schedule)),
        ("solution directory", system.solution_dir),
        ("sample directory", system.samples_dir),
    ]


def _vlm_mesh_rows(vlm: "VLMSolver") -> list[log_style.Row]:
    """Return the rows for a VLM solver whose mesh has been generated."""
    rows: list[log_style.Row] = [
        ("status", "active"),
        ("panels", f"{vlm.lattice.n_panels:,}"),
        ("surfaces", f"{len(vlm.surfaces):,}"),
        ("precision", str(vlm.dtype)),
        ("density", f"{vlm.density:.3f}", "kg/m^3"),
        ("kinematic viscosity", f"{vlm.kinematic_viscosity:.3e}", "m^2/s"),
    ]
    if len(vlm.surfaces) > 0:
        rows.append(("surfaces:", ""))
        for uid, (_aircraft, kinematics) in vlm.surfaces.items():
            rows.append((f"  {uid}", type(kinematics).__name__ if kinematics else "static"))
    return rows


def _stabilization_rows(system: "VPMSolver") -> list[log_style.Row]:
    """Return the rows of the solution check, the stabilization and the particle retention."""
    rows: list[log_style.Row] = []
    cfg = system.numerics.stabilization
    stability_limit = system.state_limits.lagrangian_cfl.maximum
    if stability_limit is None:
        rows.append(("solution stability check", "disabled"))
    else:
        rows.append(("solution stability check", "enabled"))
        rows.append(("  maximum strain increment, infinity norm", f"{stability_limit:.3f}"))
    coefficient = cfg.selective_eddy_viscosity_coefficient or 0.0
    if coefficient > 0.0:
        rows.append(("selective eddy viscosity", "enabled"))
        rows.append(("  c_stab", f"{coefficient:.3f}"))
    else:
        rows.append(("selective eddy viscosity", "disabled"))
    if cfg.regularization_interval_steps > 0:
        rows.append(("conservative filter", "enabled"))
        rows.append(("  interval", f"{cfg.regularization_interval_steps:,}", "steps"))
        rows.append(("  grid spacing", f"{cfg.regularization_grid_spacing:.3e}", "m"))
        if cfg.regularization_core_radius is not None:
            rows.append(("  regenerated core", f"{cfg.regularization_core_radius:.3e}", "m"))
        if cfg.regularization_core_radius_trigger is not None:
            rows.append(
                ("  trigger, core radius", f"{cfg.regularization_core_radius_trigger:.3e}", "m")
            )
        if cfg.regularization_divergence_trigger is not None:
            rows.append(("  trigger, divergence", f"{cfg.regularization_divergence_trigger:.3f}"))
        if cfg.regularization_misalignment_trigger is not None:
            rows.append(
                (
                    "  trigger, misalignment",
                    f"{cfg.regularization_misalignment_trigger:.1f}",
                    "deg",
                )
            )
    else:
        rows.append(("conservative filter", "disabled"))
    bounds = cfg.remove_particles_by_bounds
    if bounds is not None:
        xmin, xmax, ymin, ymax, zmin, zmax = bounds
        rows.append(("domain cutoff", "enabled"))
        rows.append(("  bounds, x", f"[{xmin:.3e}, {xmax:.3e}]", "m"))
        rows.append(("  bounds, y", f"[{ymin:.3e}, {ymax:.3e}]", "m"))
        rows.append(("  bounds, z", f"[{zmin:.3e}, {zmax:.3e}]", "m"))
        depth = cfg.outflow_fringe_depth
        if depth is not None:
            faces = ("x-", "x+", "y-", "y+", "z-", "z+")
            for face, width in zip(faces, depth, strict=True):
                if width > 0.0:
                    rows.append((f"  outflow fringe, {face}", f"{width:.3e}", "m"))
            rows.append(
                ("  outflow fringe attenuation", f"{cfg.outflow_fringe_attenuation:.3g}", "e-folds")
            )
    else:
        rows.append(("domain cutoff", "disabled"))
    return rows


def solver_info(system: "VPMSolver") -> log_style.FormattedText:
    """Return the configuration report written once at time zero."""
    sections: list[tuple[str, list[log_style.Row]]] = [
        ("RUN", _solver_rows(system)),
        ("PARTICLE SYSTEM", _particle_system_rows(system)),
        ("PHYSICS MODEL", system.physics.report_rows()),
        ("STABILIZATION", _stabilization_rows(system)),
        (
            "VISCOUS DIFFUSION",
            [
                ("scheme", system.viscous_scheme),
                *_diffusion_interval_rows(system),
                *(
                    system.solenoidal_projection.report_rows()
                    if getattr(system, "solenoidal_projection", None) is not None
                    else []
                ),
            ],
        ),
        ("TURBULENCE MODEL", _turbulence_rows(system)),
        ("OUTPUT AND MONITORING", _output_rows(system)),
    ]
    vlm = system.vlm_solver
    if vlm is not None:
        sections.append(
            (
                "VORTEX-LATTICE METHOD",
                _vlm_mesh_rows(vlm)
                if vlm._mesh_generated
                else [("status", "initialized, mesh not generated")],
            )
        )
    return log_style.block_report("VPM SOLVER CONFIGURATION", sections)


def les_diagnostics(system: "VPMSolver") -> None:
    """Write the LES eddy-viscosity statistics of the solver's turbulence model."""
    les = system.turbulence_model
    if system.logger.muted or les is None:
        return
    system.logger.section(
        "turbulence",
        ("eddy viscosity, minimum", les.min_eddy_viscosity, "m^2/s"),
        ("eddy viscosity, maximum", les.max_eddy_viscosity, "m^2/s"),
        ("eddy / molecular viscosity, minimum", les.min_eddy_viscosity_ratio),
        ("eddy / molecular viscosity, maximum", les.max_eddy_viscosity_ratio),
        immediate=True,
    )


def vlm_forces(system: "VPMSolver") -> None:
    """Write the VLM surface forces and coefficients of the last solve."""
    vlm = system.vlm_solver
    if vlm is None or not vlm._solved or vlm._last_forces is None:
        return
    forces = vlm._last_forces
    force_components = ", ".join(f"{forces[f'force_{axis}']:.4e}" for axis in "xyz")
    system.logger.section(
        "VORTEX-LATTICE LOADS",
        ("Force (x, y, z)", force_components, "N"),
        ("Lift coefficient", f"{forces['lift_coefficient']:.3e}"),
        ("Drag coefficient", f"{forces['drag_coefficient']:.3e}"),
        ("Coefficient reference pressure", f"{forces['dynamic_pressure']:.4e}", "Pa"),
        ("Coefficient reference area", f"{forces['reference_area']:.4e}", "m^2"),
        ("Active particles", f"{system.particles.n_particles_total:,}"),
        immediate=True,
    )


def step_timing(
    system: "VPMSolver", step_elapsed: float, detailed_timing: dict[str, float] | None
) -> None:
    """Record the evolution wall time in s and, when profiled, the time of each operation."""
    rows: list[log_style.Row] = [("Evolution wall time", step_elapsed, "s")]
    for operation, duration in (detailed_timing or {}).items():
        fraction = duration / step_elapsed if step_elapsed > 0 else 0.0
        rows.append((operation, f"{duration:.6f}", "s"))
        rows.append(("  Share of step", f"{100.0 * fraction:.1f}", "%"))
    system.logger.section("TIMING", *rows)
