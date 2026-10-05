"""Serializable velocity histories evaluated on native accepted clocks."""

from dataclasses import dataclass
import math

import numpy as np

from source.simulation.parallel import collective_phase


def apply_initial_velocity(solver, field) -> None:
    """Admit every local physical field before entering native halo exchange."""
    with collective_phase(solver.parallel.comm, "initial velocity field admission"):
        count = solver.mesh_data["n_cells"]
        values = np.asarray(field(solver.geo_data["cell_centre"][:count]), dtype=float)
        if values.shape != (count, 3) or not np.isfinite(values).all():
            raise ValueError(f"Initial velocity field must be finite with shape ({count}, 3)")
    with collective_phase(solver.parallel.comm, "initial velocity field installation"):
        solver.set_initial_velocity(values)


@dataclass(frozen=True)
class VelocityRamp:
    """Hold an initial velocity, then make a C2 transition to a final velocity.

    Velocities use m/s and transition endpoints use seconds. The quintic
    transition has zero first and second derivatives at both endpoints.
    """

    initial: tuple[float, float, float]
    final: tuple[float, float, float]
    start_time: float
    end_time: float

    def __post_init__(self):
        for name in ("initial", "final"):
            values = np.asarray(getattr(self, name), dtype=float)
            if values.shape != (3,) or not np.isfinite(values).all():
                raise ValueError(f"VelocityRamp.{name} must contain three finite velocities")
            object.__setattr__(self, name, tuple(float(value) for value in values))
        if not all(math.isfinite(value) for value in (self.start_time, self.end_time)):
            raise ValueError("VelocityRamp transition times must be finite")
        if not 0 <= self.start_time < self.end_time:
            raise ValueError("VelocityRamp requires 0 <= start_time < end_time")

    def at(self, time: float) -> np.ndarray:
        fraction = np.clip((time - self.start_time) / (self.end_time - self.start_time), 0.0, 1.0)
        blend = fraction**3 * (10.0 + fraction * (-15.0 + 6.0 * fraction))
        first = np.asarray(self.initial)
        return first + blend * (np.asarray(self.final) - first)


@dataclass(frozen=True)
class VelocityBoundary:
    """Apply a velocity history to named FVM patches.

    ``normal_only`` prescribes the normal velocity with zero tangential
    normal gradient. A configured slip patch recovers native slip when its
    prescribed normal velocity reaches zero.
    """

    patches: tuple[str, ...]
    velocity: VelocityRamp
    normal_only: bool = False

    def __post_init__(self):
        patches = tuple(self.patches)
        if not patches or any(not isinstance(name, str) or not name for name in patches):
            raise ValueError("VelocityBoundary requires named patches")
        if len(set(patches)) != len(patches):
            raise ValueError("VelocityBoundary patches must be unique")
        if not isinstance(self.velocity, VelocityRamp):
            raise TypeError("VelocityBoundary.velocity must be a VelocityRamp")
        if not isinstance(self.normal_only, bool):
            raise TypeError("VelocityBoundary.normal_only must be a bool")
        object.__setattr__(self, "patches", patches)


def apply_velocity_boundaries(solver, time: float) -> None:
    """Install current traces collectively before the native implicit step."""
    conditions = solver._resolved_setup.velocity_boundaries
    previous = solver.__dict__.setdefault("_velocity_boundary_values", {})
    configured = {item.name: item for item in solver._resolved_setup.boundaries}
    for index, condition in enumerate(conditions):
        velocity = condition.velocity.at(time)
        if index in previous and np.array_equal(previous[index], velocity):
            continue
        for name in condition.patches:
            normals = solver.get_boundary_face_normal(name)
            if condition.normal_only:
                normal_velocity = normals @ velocity
                solver.set_normal_velocity_tangential_gradient_boundary_condition(
                    normal_velocity, np.zeros_like(normals), name
                )
                restore_slip = configured[name].velocity_type == "slip" and bool(
                    np.all(np.abs(normal_velocity) <= 1e-12)
                )
                if solver.parallel.comm is not None:
                    restore_slip = solver.parallel.comm.bcast(
                        restore_slip if solver.parallel.is_root else None, root=0
                    )
                if restore_slip:
                    for boundary in solver.boundaries:
                        if boundary["name"] == name:
                            boundary["velocity_type"] = "slip"
                            for key in (
                                "normal_velocity_field",
                                "tangential_gradient_field",
                                "max_removed_tangential_gradient_normal_component",
                            ):
                                boundary.pop(key, None)
            else:
                solver.set_dirichlet_velocity_boundary_condition_vec(
                    np.tile(velocity, (len(normals), 1)), name
                )
        previous[index] = velocity
