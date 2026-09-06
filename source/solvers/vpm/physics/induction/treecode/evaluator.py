"""Barnes--Hut induction evaluator behind the common VPM stage contract."""

from typing import Self

import numpy as np
import taichi as ti

from ....kernels.base import RadialVortexKernel, make_vortex_kernel

_TREECODE_THETA = 0.1
_TREECODE_MULTIPOLE_ORDER = 1
_TREECODE_SORT_PARTICLE_TARGETS = False
_TREECODE_TRAVERSAL_BLOCK_DIM = 128


@ti.kernel
def _rate_from_gradient(
    gradient: ti.template(), strength: ti.template(), output: ti.template(), count: ti.i32
):
    """Contract the hierarchical velocity gradient with Γᵀ for each target."""
    for i in range(count):
        output[i] = gradient[i].transpose() @ strength[i]


@ti.data_oriented
class TreecodeInduction:
    """Evaluate stage velocity and a consistent gradient-derived rate hierarchically.

    The LBVH workspace is rebuilt from every supplied stage state.  This keeps
    geometry and strength moments synchronized when an RK stage changes either
    position or vortex strength.  The transposed rate is contracted from the
    same hierarchical velocity gradient used for optional diagnostics; no
    direct pairwise rate fallback is hidden behind the treecode interface.
    """

    supported_kernels = frozenset({"GAUSSIAN", "WINCKELMANS"})
    supported_devices = frozenset({"AUTO", "CPU", "VULKAN", "CUDA", "METAL"})
    supports_gradient = True
    supports_variable_core_radius = True
    supports_target_fields = True
    # The LBVH fields are intentionally f32 to keep the device workspace
    # bounded; reject a nominal f64 case at the immutable configuration edge.
    supports_f64 = False
    device_resident = True
    strength_rate_mode = "HIERARCHICAL_GRADIENT"

    def __init__(self) -> None:
        self.method = "TREECODE"
        self.kernel = make_vortex_kernel("GAUSSIAN")
        self.physics = None
        self.theta = _TREECODE_THETA
        self.multipole_order = _TREECODE_MULTIPOLE_ORDER
        self.sort_particle_targets = _TREECODE_SORT_PARTICLE_TARGETS
        self.traversal_block_dim = _TREECODE_TRAVERSAL_BLOCK_DIM
        self.max_n_particles = 1
        self.diagnostics = {
            "strength_rate_mode": self.strength_rate_mode,
            "stage_evaluations": 0,
            "gradient_evaluations": 0,
            "hierarchical_strength_rates": 0,
            "direct_strength_rate_fallbacks": 0,
        }

    @classmethod
    def _for_testing(
        cls,
        *,
        theta: float = _TREECODE_THETA,
        multipole_order: int = _TREECODE_MULTIPOLE_ORDER,
        sort_particle_targets: bool = _TREECODE_SORT_PARTICLE_TARGETS,
        traversal_block_dim: int = _TREECODE_TRAVERSAL_BLOCK_DIM,
    ) -> Self:
        """Construct a non-public evaluator for controlled qualification studies."""
        if not 0.0 < float(theta) < 2.0:
            raise ValueError("treecode theta must be in (0, 2)")
        if int(multipole_order) not in (1, 2, 3):
            raise ValueError("treecode multipole_order must be 1, 2, or 3")
        if int(traversal_block_dim) < 0:
            raise ValueError("treecode traversal_block_dim must be non-negative")
        instance = cls()
        instance.theta = float(theta)
        instance.multipole_order = int(multipole_order)
        instance.sort_particle_targets = bool(sort_particle_targets)
        instance.traversal_block_dim = int(traversal_block_dim)
        return instance

    def build(self) -> Self:
        """Return a fresh unbound runtime evaluator for an immutable case."""
        return type(self)._for_testing(
            theta=self.theta,
            multipole_order=self.multipole_order,
            sort_particle_targets=self.sort_particle_targets,
            traversal_block_dim=self.traversal_block_dim,
        )

    def bind(self, physics: object, *, kernel: RadialVortexKernel | None = None) -> Self:
        """Bind this construction object to one physics workspace."""
        self.physics = physics
        physics.configure_velocity(
            "TREECODE",
            self.theta,
            multipole_order=self.multipole_order,
            sort_particle_targets=self.sort_particle_targets,
            traversal_block_dim=self.traversal_block_dim,
        )
        if kernel is not None:
            self.kernel = kernel
        self.max_n_particles = physics.max_n_particles
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
        """Evaluate one complete stage from the supplied source fields."""
        del stage_time
        if self.physics is None:
            raise RuntimeError("TreecodeInduction must be bound before evaluation")
        count = int(count)
        if count < 0 or count > self.max_n_particles:
            raise ValueError(f"stage count {count} exceeds treecode capacity")
        if count == 0:
            return

        self.diagnostics["stage_evaluations"] += 1

        tree = self.physics._get_or_create_treecode(count, self.theta)
        tree.build(position, vortex_strength, core_radius, count)
        self.physics._target_tree_key = None
        self.physics.configure_velocity(
            "TREECODE",
            self.theta,
            multipole_order=self.multipole_order,
            sort_particle_targets=self.sort_particle_targets,
            traversal_block_dim=self.traversal_block_dim,
        )
        if strength_rate_enabled or velocity_gradient_out is not None:
            tree.compute_velocity_and_gradient_gpu()
            self.diagnostics["gradient_evaluations"] += 1
        else:
            tree.compute_velocities_gpu(background_field=self.physics._zero_velocity)
        self.physics._copy_vec3(tree.velocity, velocity_out, count)
        if velocity_gradient_out is not None:
            self.physics._copy_mat3(tree.velocity_gradient, velocity_gradient_out, count)

        if strength_rate_enabled:
            _rate_from_gradient(
                tree.velocity_gradient, tree.vortex_strength, vortex_strength_rate_out, count
            )
            self.diagnostics["hierarchical_strength_rates"] += 1
        else:
            self.physics._zero_vec3_field(vortex_strength_rate_out, count)

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
        """Evaluate target fields with the same LBVH operator as RK stages."""
        if self.physics is None:
            raise RuntimeError("TreecodeInduction must be bound before target evaluation")
        target_count = int(target_count)
        source_count = int(source_count)
        tree = self.physics._get_or_create_treecode(max(target_count, source_count), self.theta)
        tree.build(source_position, source_vortex_strength, source_core_radius, source_count)
        self.physics._target_tree_key = None
        target_np = self.physics._download_vector_field(target_position, target_count)
        background_np = None
        if include_freestream:
            background_np = np.asarray(
                [
                    background_velocity[None][0],
                    background_velocity[None][1],
                    background_velocity[None][2],
                ],
                dtype=np.float32,
            )
        if target_velocity is not None and target_velocity_gradient is not None:
            velocity, gradient = tree.compute_target_velocity_and_gradients(
                target_np, background_np
            )
            self.physics._upload_vector_array(velocity, target_velocity, target_count)
            self.physics._upload_matrix_array(gradient, target_velocity_gradient, target_count)
        elif target_velocity is not None:
            velocity = tree.compute_target_velocity(target_np, background_np)
            self.physics._upload_vector_array(velocity, target_velocity, target_count)
        elif target_velocity_gradient is not None:
            gradient = tree.compute_target_velocity_gradient(target_np)
            self.physics._upload_matrix_array(gradient, target_velocity_gradient, target_count)


__all__ = ["TreecodeInduction"]
