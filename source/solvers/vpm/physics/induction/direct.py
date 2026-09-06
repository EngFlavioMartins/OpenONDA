"""Direct O(N²) induction adapter."""

from __future__ import annotations

from typing import Self

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

    supported_kernels = frozenset(
        {"GAUSSIAN", "HIGH_ORDER_GAUSSIAN", "SUPER_GAUSSIAN", "WINCKELMANS"}
    )
    supported_devices = frozenset({"AUTO", "CPU", "VULKAN", "CUDA", "METAL"})
    supports_gradient = True
    supports_variable_core_radius = True
    supports_f64 = True
    device_resident = True
    strength_rate_mode = "PAIRWISE_TRANSPOSED"
    supports_target_fields = True

    def __init__(self) -> None:
        self.method = "DIRECT"
        self.kernel = make_vortex_kernel("GAUSSIAN")
        self.physics = None
        self.max_n_particles = 1
        self._strain_rate = None

    def build(self) -> Self:
        """Return a fresh unbound runtime evaluator for an immutable case."""
        return type(self)()

    def bind(self, physics: object, *, kernel: RadialVortexKernel | None = None) -> Self:
        """Bind this immutable construction object to one physics workspace."""
        self.physics = physics
        physics.configure_velocity("DIRECT")
        if kernel is not None:
            self.kernel = kernel
        capacity = physics.max_n_particles
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
        position: object,
        vortex_strength: object,
        core_radius: object,
        count: int,
        velocity_out: object,
        vortex_strength_rate_out: object,
        velocity_gradient_out: object | None = None,
        strength_rate_enabled: bool = True,
        stage_time: float = 0.0,
    ) -> None:
        """Evaluate one supplied stage without reading accepted particle state."""
        del stage_time
        if self.physics is None:
            raise RuntimeError("DirectInduction must be bound to a PhysicsEngine before evaluation")
        count = int(count)
        if count < 0 or count > self.max_n_particles:
            raise ValueError(
                f"stage count {count} exceeds induction capacity {self.max_n_particles}"
            )
        if count == 0:
            return

        if strength_rate_enabled and velocity_gradient_out is None:
            self.physics.compute_velocity_and_stretching_rate_kernel(
                position,
                vortex_strength,
                core_radius,
                velocity_out,
                vortex_strength_rate_out,
                self.physics._zero_velocity,
                count,
            )
        else:
            self.physics.compute_velocities_kernel(
                position,
                vortex_strength,
                core_radius,
                velocity_out,
                self.physics._zero_velocity,
                count,
            )
        if not strength_rate_enabled:
            self.physics._zero_vec3_field(vortex_strength_rate_out, count)
        elif velocity_gradient_out is not None:
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

    def evaluate_targets(
        self,
        *,
        target_position,
        source_position,
        source_vortex_strength,
        source_core_radius,
        target_velocity,
        target_velocity_gradient,
        target_count: int,
        source_count: int,
        include_freestream: bool,
        background_velocity,
    ) -> None:
        """Evaluate arbitrary targets through the direct backend contract."""
        if self.physics is None:
            raise RuntimeError("DirectInduction must be bound before target evaluation")
        if target_velocity is not None:
            self.physics.compute_target_velocity_kernel(
                target_position,
                source_position,
                source_vortex_strength,
                source_core_radius,
                target_velocity,
                background_velocity if include_freestream else self.physics._zero_velocity,
                int(target_count),
                int(source_count),
            )
        if target_velocity_gradient is not None:
            self.physics.compute_target_velocity_gradient_kernel(
                target_position,
                source_position,
                source_vortex_strength,
                source_core_radius,
                target_velocity_gradient,
                int(target_count),
                int(source_count),
            )


__all__ = ["DirectInduction"]
