"""Regularized vortex FMM stage evaluator."""

from __future__ import annotations

import numpy as np
import taichi as ti

from ....kernels.base import RadialVortexKernel, make_vortex_kernel
from .diagnostics import FMMDiagnostics
from .interaction_lists import well_separated
from .multipoles import multipole_velocity, p2m
from .tree import FMMTree


@ti.data_oriented
class FMMInduction:
    """Evaluate the canonical stage rates with a deterministic FMM hierarchy.

    The hierarchy is rebuilt from the supplied stage sources.  Its near/far
    metadata and multipole ownership are independent of the legacy treecode;
    the conservative particle rate uses the exact pair operator.  Monopole and
    first-moment far-field translations use the common singular Biot--Savart
    limit, while near interactions use the supplied radial kernel exactly.
    """

    def __init__(
        self,
        physics=None,
        *,
        tolerance: float = 1.0e-4,
        kernel: RadialVortexKernel | None = None,
        max_n_particles: int | None = None,
    ) -> None:
        if not 0.0 < float(tolerance) < 1.0:
            raise ValueError("FMM tolerance must lie in (0, 1)")
        self.method = "FMM"
        self.tolerance = float(tolerance)
        self.kernel = make_vortex_kernel("GAUSSIAN") if kernel is None else kernel
        self.physics = None
        self.max_n_particles = int(max_n_particles or 1)
        self.tree = FMMTree(leaf_capacity=1)
        self.diagnostics = FMMDiagnostics()
        if physics is not None:
            self.bind(physics)

    def bind(self, physics, *, kernel: RadialVortexKernel | None = None):
        self.physics = physics
        physics.configure_velocity("DIRECT")
        if kernel is not None:
            self.kernel = kernel
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
        del stage_time
        if self.physics is None:
            raise RuntimeError("FMMInduction must be bound before evaluation")
        count = int(count)
        if count < 0 or count > self.max_n_particles:
            raise ValueError(f"stage count {count} exceeds FMM capacity {self.max_n_particles}")
        if count == 0:
            return

        self.tree.build(position, vortex_strength, core_radius, count)
        self.diagnostics.hierarchy_builds += 1
        self._evaluate_hierarchy_velocity(
            position, vortex_strength, core_radius, count, velocity_out
        )
        if velocity_gradient_out is not None:
            self._evaluate_exact_gradient(
                position, vortex_strength, core_radius, count, velocity_gradient_out
            )

        # Always use the documented conservative pairwise transpose for Γ̇.
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

    def _evaluate_hierarchy_velocity(
        self, position, vortex_strength, core_radius, count: int, velocity_out
    ) -> None:
        del position, vortex_strength, core_radius
        position_np = self.tree.position
        strength_np = self.tree.vortex_strength
        core_np = self.tree.core_radius
        velocity_np = np.zeros((count, 3), dtype=np.float64)
        target_cells = np.empty(count, dtype=object)
        for cell in self.tree.cells:
            target_cells[cell.indices] = cell

        for target_index in range(count):
            target_cell = target_cells[target_index]
            target_core = core_np[target_index]
            for source_cell in self.tree.cells:
                if well_separated(source_cell, target_cell, self.tolerance):
                    multipole = p2m(
                        position_np[source_cell.indices],
                        strength_np[source_cell.indices],
                        source_cell.centre,
                    )
                    velocity_np[target_index] += multipole_velocity(
                        multipole, position_np[target_index] - source_cell.centre
                    )
                    self.diagnostics.m2l_interactions += 1
                    continue

                source_indices = source_cell.indices
                displacement = position_np[target_index] - position_np[source_indices]
                pair_velocity = self.kernel.velocity_pair(
                    displacement,
                    strength_np[source_indices],
                    target_core,
                    core_np[source_indices],
                )
                if target_index in source_indices:
                    pair_velocity[source_indices == target_index] = 0.0
                velocity_np[target_index] += np.sum(pair_velocity, axis=0)
                self.diagnostics.p2p_interactions += len(source_indices)

        dtype = np.asarray(velocity_out.to_numpy()).dtype
        padded = np.zeros((self.max_n_particles, 3), dtype=dtype)
        padded[:count] = velocity_np
        velocity_out.from_numpy(padded)

    def _evaluate_exact_gradient(
        self, position, vortex_strength, core_radius, count: int, gradient_out
    ) -> None:
        del position, vortex_strength, core_radius
        position_np = self.tree.position
        strength_np = self.tree.vortex_strength
        core_np = self.tree.core_radius
        gradient_np = np.zeros((count, 3, 3), dtype=np.float64)
        for target_index in range(count):
            displacement = position_np[target_index] - position_np
            pair_gradient = self.kernel.gradient_pair(
                displacement,
                strength_np,
                core_np[target_index],
                core_np,
            )
            pair_gradient[target_index] = 0.0
            gradient_np[target_index] = np.sum(pair_gradient, axis=0)
        dtype = np.asarray(gradient_out.to_numpy()).dtype
        padded = np.zeros((self.max_n_particles, 3, 3), dtype=dtype)
        padded[:count] = gradient_np
        gradient_out.from_numpy(padded)


__all__ = ["FMMInduction"]
