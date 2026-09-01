"""Direct O(N²) induction adapter."""

from __future__ import annotations

import taichi as ti

from ...kernels.base import RadialVortexKernel, make_vortex_kernel


@ti.data_oriented
class DirectInduction:
    """Adapt the existing exact direct kernels to :class:`InductionMethod`.

    The adapter owns no particle state.  It receives the complete temporary
    stage fields on every call and writes only the caller-provided outputs.
    The canonical production strength equation is the pairwise transposed
    equation (mode ``1`` in the legacy kernel factory); the mode integer is
    intentionally private to this migration adapter.
    """

    def __init__(
        self,
        physics=None,
        max_n_particles: int | None = None,
        kernel: RadialVortexKernel | None = None,
    ) -> None:
        self.method = "DIRECT"
        self.kernel = make_vortex_kernel("GAUSSIAN") if kernel is None else kernel
        self.physics = None
        self.max_n_particles = int(max_n_particles or 1)
        self._strain_rate = None
        if physics is not None:
            self.bind(physics)

    def bind(self, physics, *, kernel: RadialVortexKernel | None = None):
        """Bind this immutable construction object to one physics workspace."""
        self.physics = physics
        if kernel is not None:
            self.kernel = kernel
        capacity = physics.max_n_particles if self.max_n_particles == 1 else self.max_n_particles
        if capacity < 1:
            raise ValueError("max_n_particles must be positive")
        self.max_n_particles = capacity
        self._strain_rate = ti.Matrix.field(
            3, 3, dtype=physics.accumulator_dtype, shape=(capacity,)
        )
        return self

    def evaluate_stage(
        self,
        *,
        position,
        vortex_strength,
        core_radius,
        count: int,
        velocity_out,
        vortex_strength_rate_out,
        velocity_gradient_out=None,
        stage_time: float = 0.0,
    ) -> None:
        """Evaluate one supplied stage without reading accepted particle state."""
        del stage_time
        if self.physics is None:
            raise RuntimeError("DirectInduction must be bound to a PhysicsEngine before evaluation")
        count = int(count)
        if count < 0 or count > self.max_n_particles:
            raise ValueError(f"stage count {count} exceeds induction capacity {self.max_n_particles}")
        if count == 0:
            return

        self.physics.compute_velocities_kernel(
            position,
            vortex_strength,
            core_radius,
            velocity_out,
            self.physics._zero_velocity,
            count,
        )
        for start in range(0, count, 4096):
            target_count = min(4096, count - start)
            self.physics.compute_stretching_rate_batch_kernel(
                position,
                vortex_strength,
                core_radius,
                vortex_strength_rate_out,
                1,
                start,
                target_count,
                count,
            )

        if velocity_gradient_out is not None:
            self.physics.compute_velocity_gradients_kernel(
                position,
                vortex_strength,
                core_radius,
                velocity_gradient_out,
                self._strain_rate,
                count,
            )


__all__ = ["DirectInduction"]
