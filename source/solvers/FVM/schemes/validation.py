"""Single source of truth for valid scheme names + fast config validation.

Catches typo'd / unsupported scheme selections at ``Solver`` construction with a
clear, actionable message, instead of failing deep inside the first assembly
(``Unknown scheme: ...``) several seconds into a run.
"""

from __future__ import annotations

from .limiters import LIMITERS

# Convection (div) schemes accepted by ``assemble.convection.assemble_convection_term``.
CONVECTION_SCHEMES = {"upwind", "central", "linear", "deferred", "lust"} | set(LIMITERS)

# Time (ddt) schemes resolved in ``solve.pimple_solver`` / ``assemble.momentum``.
TIME_SCHEMES = {"euler", "euler_implicit", "backward_euler", "backward", "bdf2"}

# Gradient schemes resolved by ``fields.gradients._resolve_gradient_fn``.
GRADIENT_SCHEMES = {"gauss", "lsq"}

# Turbulence models built by ``turbulence.create_model``.
TURBULENCE_MODELS = {
    "none",
    "iles",
    "dns",
    "smagorinsky",
    "wale",
    "sigma",
    "dynamicsmagorinsky",
    "dynamic_smagorinsky",
}

VELOCITY_BOUNDARY_TYPES = {
    "fixedValue",
    "noSlip",
    "zeroGradient",
    "inletOutlet",
    "directionMixed",
    "empty",
    "slip",
    "symmetry",
}
PRESSURE_BOUNDARY_TYPES = {"fixedValue", "zeroGradient", "empty"}


def _check(value, valid, label, errors):
    if str(value).lower() not in valid:
        errors.append(f"  {label}={value!r} is not recognised; valid: {sorted(valid)}")


def validate_solver_params(solver, time=None) -> None:
    """Raise ``ValueError`` if any scheme name on a ``SolverParams`` is invalid."""
    errors: list[str] = []
    failure_policy = str(getattr(solver, "linear_failure_policy", "raise")).lower()
    if failure_policy not in {"raise", "direct_fallback"}:
        errors.append(
            f"  linear_failure_policy must be 'raise' or 'direct_fallback'; got {failure_policy!r}"
        )
    _check(
        getattr(solver, "convection_scheme", "deferred"),
        CONVECTION_SCHEMES,
        "convection_scheme",
        errors,
    )
    _check(getattr(solver, "time_scheme", "euler_implicit"), TIME_SCHEMES, "time_scheme", errors)
    _check(getattr(solver, "gradient_scheme", "gauss"), GRADIENT_SCHEMES, "gradient_scheme", errors)
    for name, minimum in (
        ("n_correctors", 1),
        ("n_outer_correctors", 1),
        ("n_orthogonal_correctors", 0),
    ):
        value = getattr(solver, name, minimum)
        if not isinstance(value, int) or value < minimum:
            errors.append(f"  {name}={value!r} must be an integer >= {minimum}")
    for name in ("alpha_u", "alpha_p"):
        value = float(getattr(solver, name, 1.0))
        if not 0.0 < value <= 1.0:
            errors.append(f"  {name}={value!r} must satisfy 0 < {name} <= 1")
    for name in ("tolerance", "momentum_tol", "pressure_tol", "amg_reuse_tol", "ilu_drop_tol"):
        value = float(getattr(solver, name, 1e-6))
        if not value > 0.0:
            errors.append(f"  {name}={value!r} must be > 0")
    for name in ("pressure_maxiter",):
        value = getattr(solver, name, 1)
        if not isinstance(value, int) or value < 1:
            errors.append(f"  {name}={value!r} must be an integer >= 1")
    for name in ("amg_tol",):
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
    if errors:
        raise ValueError("Invalid solver scheme selection:\n" + "\n".join(errors))


def validate_turbulence(config) -> None:
    """Raise ``ValueError`` if a turbulence model name is unrecognised."""
    if config is None:
        return
    if str(config.model).lower() not in TURBULENCE_MODELS:
        raise ValueError(
            f"Unknown turbulence model {config.model!r}; valid: {sorted(TURBULENCE_MODELS)}"
        )


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
    if errors:
        raise ValueError("Unsupported FVM boundary conditions:\n" + "\n".join(errors))
