"""Boundary conditions of the FVM mesh patches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from ._validation import finite_real, vector_or_field

VELOCITY_BOUNDARY_TYPES = frozenset(
    {
        "cyclic",
        "empty",
        "fixed_value",
        "freestream",
        "inlet_outlet",
        "no_slip",
        "slip",
        "symmetry",
        "zero_gradient",
    }
)
PRESSURE_BOUNDARY_TYPES = frozenset(
    {"cyclic", "empty", "fixed_flux_pressure", "fixed_value", "freestream", "zero_gradient"}
)


@dataclass(frozen=True, slots=True)
class BoundaryConfig:
    """Boundary condition of one named FVM patch.

    Parameters
    ----------
    name : str
        Mesh patch name; it must match a contiguous boundary range.
    velocity_type : str, default='fixed_value'
        Velocity condition: ``fixed_value``, ``zero_gradient``,
        ``inlet_outlet``, ``slip``, ``symmetry``, ``no_slip``,
        ``freestream``, ``empty`` or ``cyclic``.
    velocity_value : array-like, default=(0, 0, 0)
        Uniform ``(3,)`` or per-face ``(n_faces, 3)`` velocity in m/s.
    pressure_type : str, default='zero_gradient'
        Kinematic-pressure condition: ``fixed_value``, ``zero_gradient``,
        ``fixed_flux_pressure`` (the gradient that keeps the boundary flux of
        a prescribed velocity), ``freestream``, ``empty`` or ``cyclic``.
    kinematic_pressure_value : float, default=0.0
        Fixed kinematic pressure ``p/rho`` in m²/s².
    neighbour_patch : str or None
        Paired patch of a cyclic boundary.
    mesh_type : {'patch', 'wall', 'empty', 'cyclic'} or None
        Topological patch classification; ``None`` keeps the mesh value.

    Notes
    -----
    Face-count checks run after the mesh is constructed.
    """

    name: str
    velocity_type: str = "fixed_value"
    velocity_value: tuple[float, float, float] | np.ndarray = (0.0, 0.0, 0.0)
    pressure_type: str = "zero_gradient"
    kinematic_pressure_value: float = 0.0
    neighbour_patch: str | None = None
    mesh_type: Literal["patch", "wall", "empty", "cyclic"] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("BoundaryConfig.name must be a non-empty string")
        if self.velocity_type not in VELOCITY_BOUNDARY_TYPES:
            raise ValueError(
                f"BoundaryConfig.velocity_type must be one of {sorted(VELOCITY_BOUNDARY_TYPES)}; "
                f"got {self.velocity_type!r}"
            )
        if self.pressure_type not in PRESSURE_BOUNDARY_TYPES:
            raise ValueError(
                f"BoundaryConfig.pressure_type must be one of {sorted(PRESSURE_BOUNDARY_TYPES)}; "
                f"got {self.pressure_type!r}"
            )
        object.__setattr__(
            self,
            "velocity_value",
            vector_or_field("BoundaryConfig.velocity_value", self.velocity_value),
        )
        object.__setattr__(
            self,
            "kinematic_pressure_value",
            finite_real("BoundaryConfig.kinematic_pressure_value", self.kinematic_pressure_value),
        )
        if self.neighbour_patch is not None and (
            not isinstance(self.neighbour_patch, str) or not self.neighbour_patch.strip()
        ):
            raise ValueError("BoundaryConfig.neighbour_patch must be a non-empty string or None")
        if self.mesh_type not in {None, "patch", "wall", "empty", "cyclic"}:
            raise ValueError("BoundaryConfig.mesh_type must be patch, wall, empty, cyclic, or None")

    @staticmethod
    def inlet(name: str, velocity: tuple[float, float, float]) -> BoundaryConfig:
        """Return a fixed-velocity inlet with zero-gradient pressure."""
        return BoundaryConfig(name=name, velocity_type="fixed_value", velocity_value=velocity)

    @staticmethod
    def outlet(name: str, kinematic_pressure: float = 0.0) -> BoundaryConfig:
        """Return an inlet-outlet velocity boundary at fixed kinematic pressure in m²/s²."""
        return BoundaryConfig(
            name=name,
            velocity_type="inlet_outlet",
            pressure_type="fixed_value",
            kinematic_pressure_value=kinematic_pressure,
        )

    @staticmethod
    def freestream(
        name: str, velocity: tuple[float, float, float], kinematic_pressure: float = 0.0
    ) -> BoundaryConfig:
        """Return an incompressible external-flow boundary."""
        return BoundaryConfig(
            name=name,
            velocity_type="freestream",
            velocity_value=velocity,
            pressure_type="freestream",
            kinematic_pressure_value=kinematic_pressure,
        )

    @staticmethod
    def cyclic(name: str, neighbour_patch: str) -> BoundaryConfig:
        """Return one side of a periodic patch pair."""
        return BoundaryConfig(
            name=name,
            velocity_type="cyclic",
            pressure_type="cyclic",
            neighbour_patch=neighbour_patch,
            mesh_type="cyclic",
        )

    @staticmethod
    def wall(name: str) -> BoundaryConfig:
        """Return a no-slip wall."""
        return BoundaryConfig(name=name, velocity_type="fixed_value", mesh_type="wall")

    @staticmethod
    def slip(name: str) -> BoundaryConfig:
        """Return an impermeable zero-shear boundary."""
        return BoundaryConfig(name=name, velocity_type="slip", mesh_type="patch")

    @staticmethod
    def empty(name: str) -> BoundaryConfig:
        """Return an empty patch of an extruded two-dimensional mesh."""
        return BoundaryConfig(
            name=name, velocity_type="empty", pressure_type="empty", mesh_type="empty"
        )
