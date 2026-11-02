"""Elastically mounted body: a rigid mesh driven by the fluid force on its walls.

The body translates on linear springs and dampers along chosen directions,

``m x'' + b x' + k x = F . d``

for each direction ``d``, with ``F`` the pressure and viscous force on the
mounted wall patches. The equation is integrated exactly over every interval
with the force held at its value at the last accepted FVM state (a staggered
scheme, first order in the force), so the increments over the FVM substeps of
a coupling interval compose to the increment over the whole interval.

:class:`ElasticMount` is the immutable configuration, passed as the
``kinematics`` of :class:`~openonda.fvm.motion.MeshMotion`. The solver holds
the structural state in a :class:`MountedBody`, one per solver, which the
restart files save and restore.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math

import numpy as np
from scipy.linalg import expm

from openonda.core.kinematics import RigidMotion, finite_vector

__all__ = ["ElasticMount", "MountedBody"]


def _per_direction(name: str, value: float | Sequence[float], n: int, *, minimum: float) -> tuple:
    """Return one finite coefficient per direction, all at least ``minimum``."""
    values = np.atleast_1d(np.asarray(value, dtype=np.float64))
    if values.shape == (1,):
        values = np.repeat(values, n)
    if values.shape != (n,) or not np.all(np.isfinite(values)) or np.any(values < minimum):
        raise ValueError(
            f"ElasticMount.{name} must be one finite value, or one per direction, >= {minimum}"
        )
    return tuple(float(item) for item in values)


@dataclass(frozen=True, slots=True)
class ElasticMount(RigidMotion):
    """Linear spring–damper mount of a rigid body, driven by the fluid force on its walls.

    Parameters
    ----------
    mass : float
        Positive body mass in kg (per modelled span for a planar case).
    stiffness : float or sequence of float
        Positive spring stiffness in N/m, one value or one per direction.
    damping : float or sequence of float, default=0
        Non-negative damping coefficient in N·s/m, one value or one per direction.
    directions : sequence of array-like, default=((1, 0, 0), (0, 1, 0))
        Orthogonal unit directions of the translational degrees of freedom; the
        body cannot move in any other direction.
    patch_names : sequence of str or None, default=None
        Wall patches whose force drives the body; ``None`` selects all walls.
    rotation_centre : array-like, shape (3,), default=(0, 0, 0)
        Reference point of the body in m; it translates with the body.
    initial_displacement, initial_velocity : sequence of float or None
        Spring extension in m and body velocity in m/s along the directions at
        the start. The mesh as constructed is the start configuration, so the
        spring equilibrium lies at ``-initial_displacement`` from it.

    Notes
    -----
    ``ElasticMount`` itself is a prescribed motion only in the trivial sense:
    its own velocity and increments give the free response from the initial
    state without any fluid force. The FVM solver replaces it by a
    :class:`MountedBody`, which carries the forced response.
    """

    mass: float
    stiffness: float | tuple[float, ...]
    damping: float | tuple[float, ...] = 0.0
    directions: tuple[tuple[float, float, float], ...] = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    patch_names: tuple[str, ...] | None = None
    rotation_centre: tuple[float, float, float] = (0.0, 0.0, 0.0)
    initial_displacement: tuple[float, ...] | None = None
    initial_velocity: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        mass = float(self.mass)
        if isinstance(self.mass, bool) or not math.isfinite(mass) or mass <= 0.0:
            raise ValueError("ElasticMount.mass must be finite and positive")
        object.__setattr__(self, "mass", mass)
        directions = tuple(
            finite_vector("ElasticMount.directions", item, unit=True)
            for item in np.atleast_2d(np.asarray(self.directions, dtype=np.float64))
        )
        if not 1 <= len(directions) <= 3:
            raise ValueError("ElasticMount.directions must hold one to three directions")
        matrix = np.array(directions)
        if not np.allclose(matrix @ matrix.T, np.eye(len(directions)), atol=1.0e-12):
            raise ValueError("ElasticMount.directions must be mutually orthogonal")
        object.__setattr__(self, "directions", directions)
        n = len(directions)
        object.__setattr__(
            self, "stiffness", _per_direction("stiffness", self.stiffness, n, minimum=0.0)
        )
        if any(value <= 0.0 for value in self.stiffness):
            raise ValueError("ElasticMount.stiffness must be positive")
        object.__setattr__(self, "damping", _per_direction("damping", self.damping, n, minimum=0.0))
        if self.patch_names is not None:
            names = tuple(self.patch_names)
            if not names or not all(isinstance(name, str) and name for name in names):
                raise ValueError("ElasticMount.patch_names must be None or non-empty patch names")
            object.__setattr__(self, "patch_names", names)
        object.__setattr__(
            self,
            "rotation_centre",
            finite_vector("ElasticMount.rotation_centre", self.rotation_centre),
        )
        for name in ("initial_displacement", "initial_velocity"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _per_direction(name, value, n, minimum=-math.inf))

    @property
    def n_degrees_of_freedom(self) -> int:
        """Number of translational degrees of freedom."""
        return len(self.directions)

    def direction_matrix(self) -> np.ndarray:
        """Unit directions as rows, shape ``(n, 3)``."""
        return np.array(self.directions, dtype=np.float64)

    def initial_state(self) -> tuple[np.ndarray, np.ndarray]:
        """Initial generalized displacement in m and velocity in m/s, shape ``(n,)`` each."""
        n = self.n_degrees_of_freedom
        displacement = (
            np.zeros(n)
            if self.initial_displacement is None
            else np.array(self.initial_displacement)
        )
        velocity = np.zeros(n) if self.initial_velocity is None else np.array(self.initial_velocity)
        return displacement, velocity

    def response(
        self, elapsed: float, displacement: np.ndarray, velocity: np.ndarray, force: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Exact state after ``elapsed`` seconds under a constant generalized force.

        Each degree of freedom is the linear system ``s' = A s + g`` with
        ``s = (x, v)``; its solution is ``exp(A t) s0 + A^-1 (exp(A t) - I) g``.
        """
        n = self.n_degrees_of_freedom
        if elapsed == 0.0:
            # The accepted state itself, bit for bit, as a restart needs it
            return np.array(displacement, dtype=np.float64), np.array(velocity, dtype=np.float64)
        new_displacement = np.empty(n)
        new_velocity = np.empty(n)
        for i in range(n):
            matrix = np.array(
                [[0.0, 1.0], [-self.stiffness[i] / self.mass, -self.damping[i] / self.mass]]
            )
            transition = expm(matrix * float(elapsed))
            forcing = np.array([0.0, force[i] / self.mass])
            state = transition @ np.array([displacement[i], velocity[i]]) + np.linalg.solve(
                matrix, (transition - np.eye(2)) @ forcing
            )
            new_displacement[i], new_velocity[i] = state
        return new_displacement, new_velocity

    def _free(self, time: float) -> tuple[np.ndarray, np.ndarray]:
        displacement, velocity = self.initial_state()
        return self.response(time, displacement, velocity, np.zeros(self.n_degrees_of_freedom))

    def get_velocity(self, time: float) -> np.ndarray:
        """Velocity in m/s of the free response from the initial state, without fluid force."""
        return self.direction_matrix().T @ self._free(time)[1]

    def get_angular_velocity(self, time: float) -> np.ndarray:
        """The mount does not rotate."""
        return np.zeros(3)

    def increment(self, time: float, time_step_size: float) -> tuple[np.ndarray, np.ndarray]:
        """Displacement of the free response over ``[t, t + dt]``, without fluid force."""
        displacement = self._free(time + time_step_size)[0] - self._free(time)[0]
        return np.eye(3), self.direction_matrix().T @ displacement


class MountedBody(RigidMotion):
    """The structural state of an :class:`ElasticMount` inside one solver.

    The accepted state (time, displacement, velocity and the generalized
    fluid force at that time) defines the trajectory of every later time
    exactly, so the velocity and increments at any time are pure functions
    of the accepted state and compose over any splitting of an interval.

    Parameters
    ----------
    mount : ElasticMount
        The mount configuration.
    time : float
        Start time in s of the initial state.
    """

    def __init__(self, mount: ElasticMount, time: float) -> None:
        self.mount = mount
        self.rotation_centre = mount.rotation_centre
        self._directions = mount.direction_matrix()
        self.reset(time)

    def reset(self, time: float) -> None:
        """Return to the initial state of the mount at ``time``, without fluid force."""
        self.time = float(time)
        self.displacement, self.velocity = self.mount.initial_state()
        self.force = np.zeros(self.mount.n_degrees_of_freedom)

    def response(self, time: float) -> tuple[np.ndarray, np.ndarray]:
        """Generalized displacement and velocity at ``time`` from the accepted state."""
        return self.mount.response(time - self.time, self.displacement, self.velocity, self.force)

    def get_velocity(self, time: float) -> np.ndarray:
        """Body velocity in m/s at ``time``."""
        return self._directions.T @ self.response(time)[1]

    def get_angular_velocity(self, time: float) -> np.ndarray:
        """The body does not rotate."""
        return np.zeros(3)

    def increment(self, time: float, time_step_size: float) -> tuple[np.ndarray, np.ndarray]:
        """Body displacement over ``[t, t + dt]`` on the trajectory of the accepted state."""
        displacement = self.response(time + time_step_size)[0] - self.response(time)[0]
        return np.eye(3), self._directions.T @ displacement

    def accept(
        self, time: float, fluid_force: np.ndarray, placed_time: float | None = None
    ) -> None:
        """Commit the state at ``time`` with the fluid force in N of that state.

        ``placed_time`` is the time the mesh was placed at for the step, which
        a driver landing on an event time may round differently from the
        accepted ``time``; the state is the response at the placed time, so
        the mesh velocity of the step is kept exactly.
        """
        self.displacement, self.velocity = self.response(
            time if placed_time is None else placed_time
        )
        self.time = float(time)
        self.force = self._directions @ np.asarray(fluid_force, dtype=np.float64).reshape(3)

    def state_arrays(self) -> dict[str, np.ndarray]:
        """Copies of the accepted state arrays, keyed for a restart file."""
        return {
            "mesh_mount_displacement": np.array(self.displacement),
            "mesh_mount_velocity": np.array(self.velocity),
            "mesh_mount_force": np.array(self.force),
        }

    def restore(self, time: float, state: dict[str, np.ndarray]) -> None:
        """Place the accepted state at ``time`` from :meth:`state_arrays` data."""
        n = self.mount.n_degrees_of_freedom
        arrays = []
        for name in ("mesh_mount_displacement", "mesh_mount_velocity", "mesh_mount_force"):
            array = np.array(state[name], dtype=np.float64)
            if array.shape != (n,) or not np.all(np.isfinite(array)):
                raise ValueError(f"Restart state {name} must hold {n} finite values")
            arrays.append(array)
        self.time = float(time)
        self.displacement, self.velocity, self.force = arrays
