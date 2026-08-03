"""Single source of truth for valid scheme names + fast config validation.

Catches typo'd / unsupported scheme selections at ``Solver`` construction with a
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
    "openfoamsmagorinsky",
    "openfoam_smagorinsky",
    "wale",
    "sigma",
    "dynamicsmagorinsky",
    "dynamic_smagorinsky",
}

VELOCITY_BOUNDARY_TYPES = BOUNDARIES.names_for("U")
PRESSURE_BOUNDARY_TYPES = BOUNDARIES.names_for("p")


def _check(value, valid, label, errors):
    if str(value).lower() not in valid:
        errors.append(f"  {label}={value!r} is not recognised; valid: {sorted(valid)}")


def validate_solver_params(solver, time=None) -> None:
    """Raise ``ValueError`` if any scheme name in the merged solver params is invalid."""
    errors: list[str] = []
    algorithm = str(getattr(solver, "algorithm", "PIMPLE")).upper()
    if algorithm not in {"SIMPLE", "PIMPLE", "PISO"}:
        errors.append(
            f"  algorithm={algorithm!r} is not recognised; valid: ['PIMPLE', 'PISO', 'SIMPLE']"
        )
    failure_policy = str(getattr(solver, "linear_failure_policy", "raise")).lower()
    if failure_policy not in {"raise", "direct_fallback"}:
        errors.append(
            f"  linear_failure_policy must be 'raise' or 'direct_fallback'; got {failure_policy!r}"
        )
    nullspace_policy = str(getattr(solver, "pressure_nullspace_policy", "auto")).lower()
    if nullspace_policy not in {"auto", "reference", "petsc"}:
        errors.append(
            "  pressure_nullspace_policy must be 'auto', 'reference', or 'petsc'; "
            f"got {nullspace_policy!r}"
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
    if getattr(solver, "momentum_solver", None) == "amg":
        errors.append("  momentum_solver='amg' is unsupported; AMG is pressure-only")
    for name, minimum in (
        ("n_correctors", 1),
        ("n_outer_correctors", 1),
        ("min_outer_correctors", 1),
        ("n_orthogonal_correctors", 0),
    ):
        value = getattr(solver, name, minimum)
        if not isinstance(value, int) or value < minimum:
            errors.append(f"  {name}={value!r} must be an integer >= {minimum}")
    if algorithm == "PISO" and getattr(solver, "n_outer_correctors", 1) != 1:
        errors.append("  PISO requires n_outer_correctors == 1")
    if getattr(solver, "min_outer_correctors", 1) > getattr(solver, "n_outer_correctors", 1):
        errors.append("  min_outer_correctors cannot exceed n_outer_correctors")
    for name in ("alpha_u", "alpha_p"):
        value = float(getattr(solver, name, 1.0))
        if not 0.0 < value <= 1.0:
            errors.append(f"  {name}={value!r} must satisfy 0 < {name} <= 1")
    for name in ("tolerance", "momentum_tol", "pressure_tol", "amg_reuse_tol", "ilu_drop_tol"):
        value = float(getattr(solver, name, 1e-6))
        if not value > 0.0:
            errors.append(f"  {name}={value!r} must be > 0")
    for name in ("momentum_maxiter", "pressure_maxiter"):
        value = getattr(solver, name, 1)
        if not isinstance(value, int) or value < 1:
            errors.append(f"  {name}={value!r} must be an integer >= 1")
    for name in ("amg_tol",):
        value = getattr(solver, name, None)
        if value is not None and float(value) <= 0.0:
            errors.append(f"  {name}={value!r} must be > 0 when set")
    for name in ("outer_residual_tolerance", "outer_continuity_tolerance"):
        value = getattr(solver, name, None)
        if value is not None and float(value) <= 0.0:
            errors.append(f"  {name}={value!r} must be > 0 when set")
    for name in ("amg_maxiter",):
        value = getattr(solver, name, None)
        if value is not None and (not isinstance(value, int) or value < 1):
            errors.append(f"  {name}={value!r} must be an integer >= 1 when set")
    time_scheme = str(getattr(solver, "time_scheme", "euler_implicit")).lower()
    if (
        time is not None
        and time_scheme in {"backward", "bdf2"}
        and bool(getattr(time, "adjust_timestep", False))
    ):
        errors.append(
            "  adaptive time stepping with BDF2 is unsupported until "
            "variable-step coefficients are implemented"
        )
    if time is not None:
        if not float(time.delta_t) > 0.0:
            errors.append(f"  delta_t={time.delta_t!r} must be > 0")
        if not float(time.end_time) > float(time.start_time):
            errors.append(
                f"  end_time={time.end_time!r} must be greater than start_time={time.start_time!r}"
            )
        if not isinstance(time.write_interval, int) or time.write_interval < 1:
            errors.append(f"  write_interval={time.write_interval!r} must be an integer >= 1")
        if bool(time.adjust_timestep):
            if not 0.0 < float(time.min_delta_t) <= float(time.max_delta_t):
                errors.append(
                    "  adaptive time-step bounds must satisfy 0 < min_delta_t <= max_delta_t"
                )
            if not float(time.max_cfl) > 0.0:
                errors.append(f"  max_cfl={time.max_cfl!r} must be > 0")
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
    if name in {"openfoamsmagorinsky", "openfoam_smagorinsky"}:
        if not np.isfinite(config.Ck) or float(config.Ck) < 0.0:
            raise ValueError("OpenFOAM Smagorinsky Ck must be finite and non-negative")
        if not np.isfinite(config.Ce) or float(config.Ce) <= 0.0:
            raise ValueError("OpenFOAM Smagorinsky Ce must be finite and positive")


def validate_acceptance_policy(policy) -> None:
    """Validate warning/abort threshold ordering and sustained window."""
    errors = []
    if not isinstance(policy.sustained_steps, int) or policy.sustained_steps < 1:
        errors.append("  sustained_steps must be an integer >= 1")
    for metric in ("continuity", "residual", "cfl", "velocity"):
        warning = getattr(policy, f"{metric}_warning")
        abort = getattr(policy, f"{metric}_abort")
        for label, value in (("warning", warning), ("abort", abort)):
            if value is not None and (not np.isfinite(value) or float(value) <= 0.0):
                errors.append(f"  {metric}_{label} must be finite and > 0 when set")
        if warning is not None and abort is not None and float(warning) > float(abort):
            errors.append(f"  {metric}_warning cannot exceed {metric}_abort")
    if errors:
        raise ValueError("Invalid FVM run acceptance policy:\n" + "\n".join(errors))


def validate_boundary_conditions(boundaries) -> None:
    """Reject BCs the complete pressure--velocity operator cannot honor."""
    errors = []
    for patch in boundaries:
        name = patch.get("name", "<unnamed>")
        type_u = patch.get("bc_type_U")
        type_p = patch.get("bc_type_p")
        if type_u not in VELOCITY_BOUNDARY_TYPES:
            errors.append(
                f"  patch {name!r}: velocity BC {type_u!r} unsupported; "
                f"valid: {sorted(VELOCITY_BOUNDARY_TYPES)}"
            )
        if type_p not in PRESSURE_BOUNDARY_TYPES:
            errors.append(
                f"  patch {name!r}: pressure BC {type_p!r} unsupported; "
                f"valid: {sorted(PRESSURE_BOUNDARY_TYPES)}"
            )
        if type_u in VELOCITY_BOUNDARY_TYPES:
            for operator in (
                "gradient",
                "convection",
                "diffusion",
                "pressure",
                "flux",
                "ghost",
                "diagnostics",
            ):
                BOUNDARIES.require(type_u, "U", operator)
        if type_p in PRESSURE_BOUNDARY_TYPES:
            for operator in (
                "gradient",
                "pressure",
                "flux",
                "ghost",
                "diagnostics",
            ):
                BOUNDARIES.require(type_p, "p", operator)
    if errors:
        raise ValueError("Unsupported FVM boundary conditions:\n" + "\n".join(errors))
