"""Treecode induction adapter used during the VPM migration."""

from __future__ import annotations

import taichi as ti


@ti.data_oriented
class TreecodeInduction:
    """Adapt the existing LBVH treecode to the common stage contract.

    This migration adapter deliberately keeps the treecode implementation
    private.  Its hierarchy is rebuilt from the supplied stage position and
    strength fields, so a strength-changing RK stage cannot reuse stale
    moments.
    """

    def __init__(
        self,
        physics=None,
        theta: float = 0.3,
        multipole_order: int = 1,
        sort_particle_targets: bool = False,
        traversal_block_dim: int = 128,
        max_n_particles: int | None = None,
    ) -> None:
        if not 0.0 < float(theta) < 2.0:
            raise ValueError("treecode theta must be in (0, 2)")
        self.physics = None
        self.theta = float(theta)
        self.multipole_order = int(multipole_order)
        self.sort_particle_targets = bool(sort_particle_targets)
        self.traversal_block_dim = int(traversal_block_dim)
        if self.multipole_order not in (1, 2, 3):
            raise ValueError("treecode multipole_order must be 1, 2, or 3")
        if self.traversal_block_dim < 0:
            raise ValueError("treecode traversal_block_dim must be non-negative")
        self.max_n_particles = int(max_n_particles or 1)
        if physics is not None:
            self.bind(physics)

    def bind(self, physics):
        """Bind this construction object to one physics workspace."""
        self.physics = physics
        if self.max_n_particles == 1:
            self.max_n_particles = physics.max_n_particles
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
        """Evaluate one complete stage from the supplied source fields."""
        del stage_time
        if self.physics is None:
            raise RuntimeError("TreecodeInduction must be bound to a PhysicsEngine before evaluation")
        count = int(count)
        if count < 0 or count > self.max_n_particles:
            raise ValueError(f"stage count {count} exceeds treecode capacity")
        if count == 0:
            return

        tree = self.physics._get_or_create_treecode(count, self.theta)
        tree.build(position, vortex_strength, core_radius, count)
        self.physics._target_tree_key = None
        tree.compute_velocities_gpu(background_field=self.physics._zero_velocity)
        self.physics._copy_vec3(tree.velocity, velocity_out, count)
        tree.compute_velocity_gradients_gpu()
        self.physics.gradient_contraction_rate_kernel(
            tree.velocity_gradient,
            vortex_strength,
            vortex_strength_rate_out,
            1,
            count,
        )
        if velocity_gradient_out is not None:
            self.physics._copy_mat3(tree.velocity_gradient, velocity_gradient_out, count)


__all__ = ["TreecodeInduction"]
