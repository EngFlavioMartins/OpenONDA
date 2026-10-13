"""Boundary-condition validation against the registered boundary operators."""

from __future__ import annotations

from .boundaries import BOUNDARIES

VELOCITY_BOUNDARY_TYPES = BOUNDARIES.names_for("velocity")
PRESSURE_BOUNDARY_TYPES = BOUNDARIES.names_for("kinematic_pressure")


def validate_boundary_conditions(boundaries: list[dict]) -> None:
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
