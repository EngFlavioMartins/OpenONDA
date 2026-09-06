"""VPM physical workspaces that are shared by induction and diffusion."""

from __future__ import annotations

import taichi as ti

from ..config.constants import MAX_N_PARTICLES
from .base import PhysicsBase
from .diffusion.core_spreading import apply_core_spreading
from .diffusion.grid import _GridDiffusionMixin
from .diffusion.random_walk import apply_random_walk
from .events import NullPhysicsEventObserver, PhysicsEventObserver


@ti.data_oriented
class PhysicsEngine(PhysicsBase, _GridDiffusionMixin):
    """Own reusable VPM fields and physical operators.

    Particle induction is supplied by an :class:`InductionMethod`; this class
    no longer selects or advances an advection/stretching scheme.  It remains
    the shared device workspace for field evaluation, diffusion and coupling
    helpers used by the solver and induction backends.
    """

    def __init__(
        self,
        particle_kernel: str = "GAUSSIAN",
        max_n_particles: int = MAX_N_PARTICLES,
        accumulator_dtype: ti.types = ti.f32,
        max_evaluation_points: int = 200000,
        event_observer: PhysicsEventObserver | None = None,
    ):
        super().__init__(particle_kernel, max_n_particles, accumulator_dtype, max_evaluation_points)
        self._event_observer = event_observer or NullPhysicsEventObserver()
        self._init_grid_diffusion()

        # External stage providers may use either a device callback or a host
        # callback.  They are consumed by StageRHS, never by the RK engine.
        self.velocity_override = None
        self.velocity_override_gradient = None
        self.body_velocity = None
        self.body_velocity_gradient = None
        self.body_velocity_field = None
        self.body_velocity_gradient_field = None

        self._diffusion = _DiffusionHandler(self)

    def report_rows(self) -> list:
        """Return the physics-model configuration as log detail rows."""
        return [
            ("kernel", str(self.particle_kernel)),
            ("particles, max", f"{self.max_n_particles:,}"),
        ]

    def core_spreading_diffusion(self, particles, time_step_size: float):
        """Apply Gaussian core spreading."""
        self._diffusion.core_spreading_diffusion(particles, time_step_size)

    def random_walk_method_diffusion(
        self,
        particles,
        time_step_size: float,
        *,
        random_seed: int,
        accepted_step: int,
    ):
        """Apply random-walk diffusion."""
        self._diffusion.random_walk_method_diffusion(
            particles,
            time_step_size,
            random_seed=random_seed,
            accepted_step=accepted_step,
        )


class _DiffusionHandler:
    """Delegate split diffusion operators to the owning physics workspace."""

    def __init__(self, parent: PhysicsEngine):
        self._parent = parent

    def core_spreading_diffusion(self, particles, time_step_size: float):
        apply_core_spreading(self._parent, particles, time_step_size)

    def random_walk_method_diffusion(
        self,
        particles,
        time_step_size: float,
        *,
        random_seed: int,
        accepted_step: int,
    ):
        apply_random_walk(
            self._parent,
            particles,
            time_step_size,
            random_seed=random_seed,
            accepted_step=accepted_step,
        )


__all__ = ["PhysicsEngine"]
