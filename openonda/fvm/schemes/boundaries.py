"""Boundary-condition types of the FVM operators and their numerical treatment."""

from __future__ import annotations

from enum import Enum, auto


class BoundaryStrategy(Enum):
    """Enumeration of standard boundary-condition behaviours.

    Each member corresponds to one of the supported finite-volume boundary types
    and determines how the discrete operators (gradient, convection,
    diffusion, flux) treat the boundary faces.
    """

    FIXED_VALUE = auto()
    ZERO_GRADIENT = auto()
    EMPTY = auto()
    INLET_OUTLET = auto()
    SLIP = auto()
    CYCLIC = auto()
    FREESTREAM = auto()
    FIXED_FLUX_PRESSURE = auto()
    MIXED_VELOCITY = auto()


_SHARED = {
    "fixed_value": BoundaryStrategy.FIXED_VALUE,
    "zero_gradient": BoundaryStrategy.ZERO_GRADIENT,
    "empty": BoundaryStrategy.EMPTY,
    "cyclic": BoundaryStrategy.CYCLIC,
    "freestream": BoundaryStrategy.FREESTREAM,
}

# ``mixed_velocity`` is the velocity condition of the coupling: it prescribes
# the normal velocity and the normal gradient of the tangential velocity, and
# reconstructs the tangential face velocity from the evolving owner-cell state.
VELOCITY_BOUNDARY: dict[str, BoundaryStrategy] = {
    **_SHARED,
    "inlet_outlet": BoundaryStrategy.INLET_OUTLET,
    "slip": BoundaryStrategy.SLIP,
    "mixed_velocity": BoundaryStrategy.MIXED_VELOCITY,
}
# ``fixed_flux_pressure`` is the pressure partner of a prescribed-velocity
# boundary: the absolute pressure carries a momentum-compatible normal
# gradient, while the pressure correction is homogeneous Neumann so that the
# prescribed boundary flux is not changed.
PRESSURE_BOUNDARY: dict[str, BoundaryStrategy] = {
    **_SHARED,
    "fixed_flux_pressure": BoundaryStrategy.FIXED_FLUX_PRESSURE,
}
SCALAR_BOUNDARY: dict[str, BoundaryStrategy] = dict(_SHARED)

_FIELD_BOUNDARY = {
    "velocity": VELOCITY_BOUNDARY,
    "kinematic_pressure": PRESSURE_BOUNDARY,
    "scalar": SCALAR_BOUNDARY,
}


def boundary_strategy(name: str | None, field: str) -> BoundaryStrategy:
    """Return the numerical treatment of boundary type ``name`` for ``field``.

    Parameters
    ----------
    name : str or None
        Boundary-condition name, such as ``"fixed_value"``. ``None`` (no
        boundary condition set) is rejected like an unknown name.
    field : {'velocity', 'kinematic_pressure', 'scalar'}
        Field family.

    Raises
    ------
    ValueError
        If ``name`` is not a boundary type of ``field``.
    """
    table = _FIELD_BOUNDARY[field]
    strategy = table.get(name) if name is not None else None
    if strategy is None:
        raise ValueError(f"Boundary {name!r} is not a {field} boundary; supported: {sorted(table)}")
    return strategy


def validate_boundary_conditions(boundaries: list[dict]) -> None:
    """Reject patches whose velocity or pressure type is not supported."""
    errors = []
    for patch in boundaries:
        name = patch.get("name", "<unnamed>")
        for key, field, table in (
            ("velocity_type", "velocity", VELOCITY_BOUNDARY),
            ("pressure_type", "pressure", PRESSURE_BOUNDARY),
        ):
            if patch.get(key) not in table:
                errors.append(
                    f"  patch {name!r}: {field} BC {patch.get(key)!r} unsupported; "
                    f"valid: {sorted(table)}"
                )
    if errors:
        raise ValueError("Unsupported FVM boundary conditions:\n" + "\n".join(errors))
