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
        """Allocate the shared VPM physics workspace.

        Parameters
        ----------
        particle_kernel : str, default="GAUSSIAN"
            Kernel family used by inherited particle-field evaluation.
        max_n_particles : int, default=MAX_N_PARTICLES
            Fixed particle capacity for device fields.
        accumulator_dtype : taichi scalar type, default=ti.f32
            Precision of reusable accumulation fields.
        max_evaluation_points : int, default=200000
            Capacity for arbitrary target-point velocity/gradient evaluation.
        event_observer : PhysicsEventObserver, optional
            Diagnostic sink for warnings and numerical events.  A silent
            :class:`NullPhysicsEventObserver` is used when omitted.

        Notes
        -----
        Construction allocates device workspaces and diffusion support but
        does not move particles, advance time, or select a Runge--Kutta stage.
        External velocity/body hooks are initialized to ``None`` and are
        consumed by :mod:`stage_rhs`.
        """
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
        """Apply deterministic Gaussian core-spreading diffusion.

        Parameters
        ----------
        particles : Particles
            Mutable particle container whose active core radii and strengths
            are updated according to the configured viscosity.
        time_step_size : float
            Physical elapsed time in seconds.

        Notes
        -----
        This is a split operator called by the accepted-step integrator; it
        mutates particle device fields and does not return a new container.
        """
        self._diffusion.core_spreading_diffusion(particles, time_step_size)

    def random_walk_method_diffusion(
        self,
        particles,
        time_step_size: float,
        *,
        random_seed: int,
        accepted_step: int,
    ):
        """Apply stochastic random-walk diffusion to active particles.

        Parameters
        ----------
        particles : Particles
            Mutable particle container to update.
        time_step_size : float
            Physical elapsed time in seconds.
        random_seed : int
            Base seed for deterministic replay of the stochastic operator.
        accepted_step : int
            Accepted-step index mixed into the seed so rejected/candidate RK
            stages do not consume a hidden random sequence.
        """
        self._diffusion.random_walk_method_diffusion(
            particles,
            time_step_size,
            random_seed=random_seed,
            accepted_step=accepted_step,
        )


class _DiffusionHandler:
    """Delegate split diffusion operators to the owning physics workspace."""

    def __init__(self, parent: PhysicsEngine):
        """Bind the handler to its owning :class:`PhysicsEngine`."""
        self._parent = parent

    def core_spreading_diffusion(self, particles, time_step_size: float):
        """Run core spreading through the parent workspace."""
        apply_core_spreading(self._parent, particles, time_step_size)

    def random_walk_method_diffusion(
        self,
        particles,
        time_step_size: float,
        *,
        random_seed: int,
        accepted_step: int,
    ):
        """Run random-walk diffusion through the parent workspace."""
        apply_random_walk(
            self._parent,
            particles,
            time_step_size,
            random_seed=random_seed,
            accepted_step=accepted_step,
        )


__all__ = ["PhysicsEngine"]
