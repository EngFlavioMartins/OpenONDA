"""Barnes--Hut induction evaluator behind the common VPM stage contract."""

from typing import Self

import taichi as ti

from ....kernels.base import RadialVortexKernel, make_vortex_kernel


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
    supports_gradient = True
    supports_variable_core_radius = True
    # The LBVH fields are intentionally f32 to keep the device workspace
    # bounded; reject a nominal f64 case at the immutable configuration edge.
    supports_f64 = False
    device_resident = True

    def __init__(
        self,
        physics: object | None = None,
        theta: float = 0.3,
        multipole_order: int = 1,
        sort_particle_targets: bool = False,
        traversal_block_dim: int = 128,
        max_n_particles: int | None = None,
        kernel: RadialVortexKernel | None = None,
    ) -> None:
        self.method = "TREECODE"
        self.kernel = make_vortex_kernel("GAUSSIAN") if kernel is None else kernel
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
        self.diagnostics = {
            "stage_evaluations": 0,
            "gradient_evaluations": 0,
            "hierarchical_strength_rates": 0,
            "direct_strength_rate_fallbacks": 0,
        }
        if physics is not None:
            self.bind(physics)

    def build(self) -> Self:
        """Return a fresh unbound runtime evaluator for an immutable case."""
        return type(self)(
            theta=self.theta,
            multipole_order=self.multipole_order,
            sort_particle_targets=self.sort_particle_targets,
            traversal_block_dim=self.traversal_block_dim,
            max_n_particles=self.max_n_particles,
            kernel=self.kernel,
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
        if self.max_n_particles == 1:
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


__all__ = ["TreecodeInduction"]
