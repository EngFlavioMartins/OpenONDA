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
    "none", "iles", "dns", "smagorinsky", "wale", "sigma",
    "dynamicsmagorinsky", "dynamic_smagorinsky",
}


def _check(value, valid, label, errors):
    if str(value).lower() not in valid:
        errors.append(f"  {label}={value!r} is not recognised; valid: {sorted(valid)}")


def validate_solver_params(solver) -> None:
    """Raise ``ValueError`` if any scheme name on a ``SolverParams`` is invalid."""
    errors: list[str] = []
    _check(getattr(solver, "convection_scheme", "deferred"), CONVECTION_SCHEMES, "convection_scheme", errors)
    _check(getattr(solver, "time_scheme", "euler_implicit"), TIME_SCHEMES, "time_scheme", errors)
    _check(getattr(solver, "gradient_scheme", "gauss"), GRADIENT_SCHEMES, "gradient_scheme", errors)
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
