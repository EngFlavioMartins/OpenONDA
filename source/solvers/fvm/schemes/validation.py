"""Single source of truth for valid scheme names + fast config validation.

Catches typo'd / unsupported scheme selections at ``FVMSolver`` construction with a
clear, actionable message, instead of failing deep inside the first assembly
(``Unknown scheme: ...``) several seconds into a run.
"""

from __future__ import annotations

import numpy as np

from .boundaries import BOUNDARIES
from .limiters import LIMITERS

# Convection (div) schemes accepted by ``assemble.convection.assemble_convection_term``.
CONVECTION_SCHEMES = {"upwind", "central", "linear", "deferred", "lust", "linearupwind"} | set(
    LIMITERS
)

# Time (ddt) schemes resolved in ``solve.pimple_solver`` / ``assemble.momentum``.
TIME_SCHEMES = {"euler", "euler_implicit", "backward_euler", "backward", "bdf2"}

# Gradient schemes resolved by ``fields.gradients._resolve_gradient_fn``.
GRADIENT_SCHEMES = {"gauss", "lsq"}
LINEAR_SOLVERS = {"spsolve", "bicgstab", "gmres", "cg", "amg"}

# Turbulence models built by ``turbulence.create_model``.
TURBULENCE_MODELS = {
    "none",
    "iles",
    "dns",
    "smagorinsky",
    "equilibriumsmagorinsky",
    "equilibrium_smagorinsky",
    "wale",
    "sigma",
    "dynamicsmagorinsky",
    "dynamic_smagorinsky",
}

VELOCITY_BOUNDARY_TYPES = BOUNDARIES.names_for("velocity")
PRESSURE_BOUNDARY_TYPES = BOUNDARIES.names_for("kinematic_pressure")


def _check(value, valid, label, errors):
    if str(value).lower() not in valid:
        errors.append(f"  {label}={value!r} is not recognised; valid: {sorted(valid)}")


def validate_solver_params(solver, time=None) -> None:
    """Raise ``ValueError`` if any scheme name in the merged solver params is invalid."""
    errors: list[str] = []

    def finite_real(name: str, value, *, minimum=None, maximum=None, strict_min=False):
        if isinstance(value, bool) or not isinstance(value, int | float | np.integer | np.floating):
            errors.append(f"  {name}={value!r} must be a finite real number")
            return None
        value = float(value)
        if not np.isfinite(value):
            errors.append(f"  {name}={value!r} must be finite")
            return None
        if minimum is not None and (value <= minimum if strict_min else value < minimum):
            relation = ">" if strict_min else ">="
            errors.append(f"  {name}={value!r} must be {relation} {minimum}")
        if maximum is not None and value > maximum:
            errors.append(f"  {name}={value!r} must be <= {maximum}")
        return value

    algorithm = str(getattr(solver, "algorithm", "PIMPLE")).upper()
    if algorithm not in {"SIMPLE", "PIMPLE", "PISO"}:
        errors.append(
            f"  algorithm={algorithm!r} is not recognised; valid: ['PIMPLE', 'PISO', 'SIMPLE']"
        )
    failure_action = str(getattr(solver, "linear_failure_action", "raise")).lower()
    if failure_action not in {"raise", "direct_fallback"}:
        errors.append(
            f"  linear_failure_action must be 'raise' or 'direct_fallback'; got {failure_action!r}"
        )
    nullspace_method = str(getattr(solver, "pressure_nullspace_method", "auto")).lower()
    if nullspace_method not in {"auto", "reference", "petsc"}:
        errors.append(
            "  pressure_nullspace_method must be 'auto', 'reference', or 'petsc'; "
            f"got {nullspace_method!r}"
        )
    _check(
        getattr(solver, "convection_scheme", "deferred"),
        CONVECTION_SCHEMES,
        "convection_scheme",
        errors,
    )
    _check(getattr(solver, "time_scheme", "euler_implicit"), TIME_SCHEMES, "time_scheme", errors)
    _check(getattr(solver, "gradient_scheme", "gauss"), GRADIENT_SCHEMES, "gradient_scheme", errors)
    _check(getattr(solver, "linear_solver", "bicgstab"), LINEAR_SOLVERS, "linear_solver", errors)
    for name in ("momentum_solver", "pressure_solver"):
        value = getattr(solver, name, None)
        if value is not None:
            _check(value, LINEAR_SOLVERS, name, errors)
    effective_momentum_solver = getattr(solver, "momentum_solver", None) or getattr(
        solver, "linear_solver", "bicgstab"
    )
    if str(effective_momentum_solver).lower() == "amg":
        errors.append("  momentum_solver='amg' is unsupported; AMG is pressure-only")
    for name, minimum in (
        ("n_correctors", 1),
        ("n_outer_correctors", 1),
        ("min_outer_correctors", 1),
        ("n_nonorthogonal_correctors", 0),
    ):
        value = getattr(solver, name, minimum)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            errors.append(f"  {name}={value!r} must be an integer >= {minimum}")
    if algorithm == "PISO" and getattr(solver, "n_outer_correctors", 1) != 1:
        errors.append("  PISO requires n_outer_correctors == 1")
    if getattr(solver, "min_outer_correctors", 1) > getattr(solver, "n_outer_correctors", 1):
        errors.append("  min_outer_correctors cannot exceed n_outer_correctors")
    for name in ("velocity_relaxation", "pressure_relaxation"):
        finite_real(name, getattr(solver, name, 1.0), minimum=0.0, maximum=1.0, strict_min=True)
    for name in (
        "tolerance",
        "momentum_tolerance",
        "pressure_tolerance",
        "amg_reuse_tolerance",
        "ilu_drop_tolerance",
    ):
        finite_real(name, getattr(solver, name, 1e-6), minimum=0.0, strict_min=True)
    for name in (
        "momentum_relative_tolerance",
        "momentum_final_relative_tolerance",
        "pressure_relative_tolerance",
        "pressure_final_relative_tolerance",
    ):
        value = getattr(solver, name, 0.0)
        if value is not None:
            finite_real(name, value, minimum=0.0, maximum=1.0)
    for name in ("momentum_max_iterations", "pressure_max_iterations"):
        value = getattr(solver, name, 1)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            errors.append(f"  {name}={value!r} must be an integer >= 1")
    for name in ("amg_tolerance",):
        value = getattr(solver, name, None)
        if value is not None:
            finite_real(name, value, minimum=0.0, strict_min=True)
    for name in ("outer_residual_tolerance", "outer_continuity_tolerance"):
        value = getattr(solver, name, None)
        if value is not None:
            finite_real(name, value, minimum=0.0, strict_min=True)
    for name in ("amg_max_iterations",):
        value = getattr(solver, name, None)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 1
        ):
            errors.append(f"  {name}={value!r} must be an integer >= 1 when set")
    if time is not None:
        finite_real("time_step_size", time.time_step_size, minimum=0.0, strict_min=True)
        start = finite_real("start_time", time.start_time)
        end = finite_real("end_time", time.end_time)
        if start is not None and end is not None and not end > start:
            errors.append(
                f"  end_time={time.end_time!r} must be greater than start_time={time.start_time!r}"
            )
    if errors:
        raise ValueError("Invalid solver scheme selection:\n" + "\n".join(errors))


def validate_turbulence(config) -> None:
    """Validate the turbulence model name and its physical coefficients."""
    if config is None:
        return
    if str(config.model).lower() not in TURBULENCE_MODELS:
        raise ValueError(
            f"Unknown turbulence model {config.model!r}; valid: {sorted(TURBULENCE_MODELS)}"
        )
    name = str(config.model).lower()
    if name in {"equilibriumsmagorinsky", "equilibrium_smagorinsky"}:
        if (
            not np.isfinite(config.subgrid_kinetic_energy_coefficient)
            or float(config.subgrid_kinetic_energy_coefficient) < 0.0
        ):
            raise ValueError(
                "Equilibrium Smagorinsky subgrid_kinetic_energy_coefficient must be finite and non-negative"
            )
        if (
            not np.isfinite(config.subgrid_dissipation_coefficient)
            or float(config.subgrid_dissipation_coefficient) <= 0.0
        ):
            raise ValueError(
                "Equilibrium Smagorinsky subgrid_dissipation_coefficient must be finite and positive"
            )


def validate_acceptance_limits(limits) -> None:
    """Validate warning/abort threshold ordering and sustained window."""
    errors = []
    if not isinstance(limits.sustained_steps, int) or limits.sustained_steps < 1:
        errors.append("  sustained_steps must be an integer >= 1")
    for metric in (
        "max_continuity_error",
        "max_equation_residual",
        "max_courant_number",
        "max_velocity_magnitude",
    ):
        warning = getattr(limits, f"{metric}_warning")
        abort = getattr(limits, f"{metric}_abort")
        for label, value in (("warning", warning), ("abort", abort)):
            if value is not None and (not np.isfinite(value) or float(value) <= 0.0):
                errors.append(f"  {metric}_{label} must be finite and > 0 when set")
        if warning is not None and abort is not None and float(warning) > float(abort):
            errors.append(f"  {metric}_warning cannot exceed {metric}_abort")
    if errors:
        raise ValueError("Invalid FVM run acceptance limits:\n" + "\n".join(errors))


def validate_boundary_conditions(boundaries) -> None:
    """Reject BCs the complete pressure--velocity operator cannot honor."""
    errors = []
    for patch in boundaries:
        name = patch.get("name", "<unnamed>")
        velocity_type = patch.get("velocity_type")
        pressure_type = patch.get("pressure_type")
        if velocity_type not in VELOCITY_BOUNDARY_TYPES:
            errors.append(
                f"  patch {name!r}: velocity BC {velocity_type!r} unsupported; "
                f"valid: {sorted(VELOCITY_BOUNDARY_TYPES)}"
            )
        if pressure_type not in PRESSURE_BOUNDARY_TYPES:
            errors.append(
                f"  patch {name!r}: pressure BC {pressure_type!r} unsupported; "
                f"valid: {sorted(PRESSURE_BOUNDARY_TYPES)}"
            )
        if velocity_type in VELOCITY_BOUNDARY_TYPES:
            for operator in (
                "gradient",
                "convection",
                "diffusion",
                "pressure",
                "flux",
                "ghost",
                "diagnostics",
            ):
                BOUNDARIES.require(velocity_type, "velocity", operator)
        if pressure_type in PRESSURE_BOUNDARY_TYPES:
            for operator in (
                "gradient",
                "pressure",
                "flux",
                "ghost",
                "diagnostics",
            ):
                BOUNDARIES.require(pressure_type, "kinematic_pressure", operator)
    if errors:
        raise ValueError("Unsupported FVM boundary conditions:\n" + "\n".join(errors))
