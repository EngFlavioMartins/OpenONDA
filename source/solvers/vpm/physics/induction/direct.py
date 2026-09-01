"""Direct O(N²) induction adapter."""

from __future__ import annotations

import taichi as ti


@ti.data_oriented
class DirectInduction:
    """Adapt the existing exact direct kernels to :class:`InductionMethod`.

    The adapter owns no particle state.  It receives the complete temporary
    stage fields on every call and writes only the caller-provided outputs.
    The canonical production strength equation is the pairwise transposed
    equation (mode ``1`` in the legacy kernel factory); the mode integer is
    intentionally private to this migration adapter.
    """

    def __init__(self, physics, max_n_particles: int | None = None) -> None:
        self.physics = physics
        capacity = physics.max_n_particles if max_n_particles is None else int(max_n_particles)
        if capacity < 1:
            raise ValueError("max_n_particles must be positive")
        self.max_n_particles = capacity
        self._strain_rate = ti.Matrix.field(
            3, 3, dtype=physics.accumulator_dtype, shape=(capacity,)
        )

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
