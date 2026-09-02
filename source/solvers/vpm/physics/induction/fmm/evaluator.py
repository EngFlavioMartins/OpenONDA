"""Host reference FMM stage evaluator with explicit upward/downward passes."""

from __future__ import annotations

from typing import Self

import numpy as np
import taichi as ti

from ....kernels.base import RadialVortexKernel, make_vortex_kernel
from ..base import StrengthRateMode
from .diagnostics import FMMDiagnostics
from .interaction_lists import well_separated
from .local_expansions import l2l, m2l
from .multipoles import m2m, p2m
from .near_field import p2p_velocity_gradient
from .tree import FMMNode, FMMTree


@ti.data_oriented
class FMMInduction:
    """Evaluate a coupled stage with a deterministic host FMM reference.

    This implementation executes a complete low-order cell hierarchy:

    ``P2M → M2M → M2L → L2L → L2P`` plus exact regularized near-field ``P2P``.

    The reference backend intentionally keeps its source arrays on NumPy so
    its translations can be independently inspected and tested. Production
    GPU deployment remains a separate backend concern; this class no longer
    hides a direct stretching or direct-gradient fallback behind the FMM name.
    """

    supported_kernels = frozenset(
        {"GAUSSIAN", "HIGH_ORDER_GAUSSIAN", "SUPER_GAUSSIAN", "WINCKELMANS"}
    )
    supports_gradient = True
    supports_variable_core_radius = True
    supports_f64 = True
    # This reference implementation stages the active prefix through
    # reusable host buffers. It is not a device-resident production backend.
    device_resident = False
    strength_rate_mode = "HIERARCHICAL_GRADIENT"

    def __init__(
        self,
        physics: object | None = None,
        *,
        tolerance: float = 1.0e-4,
        kernel: RadialVortexKernel | None = None,
        max_n_particles: int | None = None,
        leaf_capacity: int = 8,
        strength_rate_mode: StrengthRateMode = "HIERARCHICAL_GRADIENT",
    ) -> None:
        if not 0.0 < float(tolerance) < 1.0:
            raise ValueError("FMM tolerance must lie in (0, 1)")
        if int(leaf_capacity) < 1:
            raise ValueError("FMM leaf_capacity must be positive")
        self.method = "FMM"
        self.tolerance = float(tolerance)
        self.kernel = make_vortex_kernel("GAUSSIAN") if kernel is None else kernel
        self.physics = None
        self.max_n_particles = int(max_n_particles or 1)
        self.leaf_capacity = int(leaf_capacity)
        normalized_rate_mode = strength_rate_mode.upper()
        if normalized_rate_mode != self.strength_rate_mode:
            raise ValueError(
                "FMMInduction supports only strength_rate_mode="
                f"{self.strength_rate_mode}; exact pairwise rates require DirectInduction"
            )
        self.strength_rate_mode = normalized_rate_mode
        self.tree = FMMTree(leaf_capacity=self.leaf_capacity)
        self.diagnostics = FMMDiagnostics(strength_rate_mode=self.strength_rate_mode)
        if physics is not None:
            self.bind(physics)

    def build(self) -> Self:
        """Return a fresh unbound runtime evaluator for an immutable case."""
        return type(self)(
            tolerance=self.tolerance,
            kernel=self.kernel,
            max_n_particles=self.max_n_particles,
            leaf_capacity=self.leaf_capacity,
            strength_rate_mode=self.strength_rate_mode,
        )

    def bind(self, physics: object, *, kernel: RadialVortexKernel | None = None) -> Self:
        """Bind this runtime evaluator to the shared particle precision context."""
        self.physics = physics
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
        """Evaluate velocity, gradient, and strength rate from one stage state."""
        del stage_time
        if self.physics is None:
            raise RuntimeError("FMMInduction must be bound before evaluation")
        count = int(count)
        if count < 0 or count > self.max_n_particles:
            raise ValueError(f"stage count {count} exceeds FMM capacity {self.max_n_particles}")
        if count == 0:
            return

        # Use the shared bounded transfer helpers when the stage is backed by
        # Taichi fields.  ``Field.to_numpy()`` would download the complete
        # capacity (which can be much larger than the active cloud) on every
        # RK stage and is the main avoidable memory spike of this host backend.
        position_np = self.physics._download_vector_field(position, count)
        strength_np = self.physics._download_vector_field(vortex_strength, count)
        radius_np = self.physics._download_scalar_field(core_radius, count)
        self.tree.build(position_np, strength_np, radius_np, count)
        self.diagnostics.hierarchy_builds += 1
        self.diagnostics.host_particle_transfers += 1
        multipoles = self._upward_pass()
        velocity_np, gradient_np = self._downward_pass(multipoles)

        self._write_vector(velocity_out, velocity_np)
        if velocity_gradient_out is not None:
            self._write_gradient(velocity_gradient_out, gradient_np)

        if strength_rate_enabled:
            strength_np = self.tree.vortex_strength
            rate_np = np.einsum("nji,nj->ni", gradient_np, strength_np)
            self.diagnostics.hierarchical_strength_rates += 1
            self.diagnostics.last_uncorrected_rate_defect = float(
                np.linalg.norm(rate_np.sum(axis=0))
            )
            # Normalize the net-rate defect by the sum of individual rate
            # magnitudes, which remains meaningful when cancellation is strong.
            self.diagnostics.last_strength_rate_norm = float(np.linalg.norm(rate_np, axis=1).sum())
            self.diagnostics.last_relative_rate_defect = (
                self.diagnostics.last_uncorrected_rate_defect
                / max(self.diagnostics.last_strength_rate_norm, np.finfo(float).eps)
            )
            self._write_vector(vortex_strength_rate_out, rate_np)
        else:
            self.physics._zero_vec3_field(vortex_strength_rate_out, count)

    def _upward_pass(self) -> dict[int, dict[str, np.ndarray]]:
        """Build P2M moments at leaves and translate them through M2M."""
        assert self.tree.root is not None
        multipoles: dict[int, dict[str, np.ndarray]] = {}
        for index in range(len(self.tree.nodes) - 1, -1, -1):
            node = self.tree.nodes[index]
            if node.children:
                children = [multipoles[child] for child in node.children]
                centres = [self.tree.nodes[child].centre for child in node.children]
                multipoles[index] = m2m(children, centres, node.centre)
                self.diagnostics.m2m_operations += 1
            else:
                multipoles[index] = p2m(
                    self.tree.position[node.indices],
                    self.tree.vortex_strength[node.indices],
                    node.centre,
                )
                self.diagnostics.p2m_operations += 1
        return multipoles

    def _downward_pass(
        self, multipoles: dict[int, dict[str, np.ndarray]]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Accumulate dual-tree locals, propagate them, and evaluate leaves."""
        count = len(self.tree.position)
        velocity = np.zeros((count, 3), dtype=np.float64)
        gradient = np.zeros((count, 3, 3), dtype=np.float64)
        assert self.tree.root is not None
        locals_by_node = {index: _zero_local() for index in range(len(self.tree.nodes))}
        near_pairs: list[tuple[int, int]] = []
        self._resolve_node_pair(
            multipoles,
            self.tree.root,
            self.tree.root,
            locals_by_node,
            near_pairs,
        )
        # FMM local coefficients are evaluated only after a distinct top-down
        # pass. Nodes are stored preorder, so parents precede their children.
        for node_index, node in enumerate(self.tree.nodes):
            for child_index in node.children:
                child_local = l2l(
                    locals_by_node[node_index],
                    self.tree.nodes[child_index].centre - node.centre,
                )
                locals_by_node[child_index] = _add_local(locals_by_node[child_index], child_local)
                self.diagnostics.l2l_operations += 1
                if (
                    np.linalg.norm(child_local["value"]) > 0.0
                    or np.linalg.norm(child_local["gradient"]) > 0.0
                ):
                    self.diagnostics.nonzero_l2l_operations += 1

        near_by_target: dict[int, list[int]] = {}
        for target_index, source_index in near_pairs:
            near_by_target.setdefault(target_index, []).append(source_index)
        for target_index, target in enumerate(self.tree.nodes):
            if target.children:
                continue
            self._evaluate_leaf(
                target_index,
                target,
                locals_by_node[target_index],
                near_by_target.get(target_index, []),
                velocity,
                gradient,
            )
        self.diagnostics.gradient_evaluations += 1
        return velocity, gradient

    def _resolve_node_pair(
        self,
        multipoles: dict[int, dict[str, np.ndarray]],
        target_index: int,
        source_index: int,
        locals_by_node: dict[int, dict[str, np.ndarray]],
        near_pairs: list[tuple[int, int]],
    ) -> None:
        """Resolve one target/source pair with dual-tree descent."""
        target = self.tree.nodes[target_index]
        source = self.tree.nodes[source_index]
        if target_index != source_index and well_separated(
            source, target, self.tolerance, self.kernel
        ):
            locals_by_node[target_index] = _add_local(
                locals_by_node[target_index],
                m2l(multipoles[source_index], target.centre - source.centre),
            )
            self.diagnostics.m2l_interactions += 1
            return

        if not target.children and not source.children:
            near_pairs.append((target_index, source_index))
            return

        if target.children and (not source.children or target.half_width >= source.half_width):
            for child_index in target.children:
                self._resolve_node_pair(
                    multipoles,
                    child_index,
                    source_index,
                    locals_by_node,
                    near_pairs,
                )
            return

        for child_index in source.children:
            self._resolve_node_pair(
                multipoles,
                target_index,
                child_index,
                locals_by_node,
                near_pairs,
            )

    def _evaluate_leaf(
        self,
        target_index: int,
        target: FMMNode,
        local: dict[str, np.ndarray],
        source_node_indices: list[int],
        velocity: np.ndarray,
        gradient: np.ndarray,
    ) -> None:
        """Evaluate one target leaf's propagated local and exact near pairs."""
        target_indices = target.indices
        target_position = self.tree.position[target_indices]
        near_velocity = np.zeros((len(target_indices), 3), dtype=np.float64)
        near_gradient = np.zeros((len(target_indices), 3, 3), dtype=np.float64)
        for source_index in source_node_indices:
            source = self.tree.nodes[source_index]
            particle_indices = source.indices
            source_velocity, source_gradient = p2p_velocity_gradient(
                self.kernel,
                target_position,
                self.tree.position[particle_indices],
                self.tree.vortex_strength[particle_indices],
                self.tree.core_radius[target_indices],
                self.tree.core_radius[particle_indices],
                exclude_self=source_index == target_index,
            )
            near_velocity += source_velocity
            near_gradient += source_gradient
            self.diagnostics.p2p_interactions += len(target_indices) * len(particle_indices)

        far_velocity, far_gradient = _l2p_batch(local, target_position - target.centre)
        velocity[target_indices] += far_velocity + near_velocity
        gradient[target_indices] += far_gradient + near_gradient
        self.diagnostics.l2p_evaluations += len(target_indices)

    def _write_vector(self, output, values: np.ndarray) -> None:
        if hasattr(self.physics, "_upload_vector_array"):
            self.physics._upload_vector_array(values, output, len(values))
            return
        dtype = getattr(self.physics, "np_dtype", np.float32)
        padded = np.zeros((self.max_n_particles, 3), dtype=dtype)
        padded[: len(values)] = values
        output.from_numpy(padded)

    def _write_gradient(self, output, values: np.ndarray) -> None:
        if hasattr(self.physics, "_upload_matrix_array"):
            self.physics._upload_matrix_array(values, output, len(values))
            return
        dtype = getattr(self.physics, "np_dtype", np.float32)
        padded = np.zeros((self.max_n_particles, 3, 3), dtype=dtype)
        padded[: len(values)] = values
        output.from_numpy(padded)


def _zero_local() -> dict[str, np.ndarray]:
    return {
        "value": np.zeros(3, dtype=np.float64),
        "gradient": np.zeros((3, 3), dtype=np.float64),
        "displacement": np.zeros(3, dtype=np.float64),
    }


def _add_local(left: dict[str, np.ndarray], right: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Add two first-order local expansions coefficient-wise."""
    return {
        "value": np.asarray(left["value"], dtype=np.float64)
        + np.asarray(right["value"], dtype=np.float64),
        "gradient": np.asarray(left["gradient"], dtype=np.float64)
        + np.asarray(right["gradient"], dtype=np.float64),
        "displacement": np.zeros(3, dtype=np.float64),
    }


def _l2p_batch(
    local: dict[str, np.ndarray], displacement: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate a first-order local expansion for all particles in a leaf."""
    displacement = np.asarray(displacement, dtype=np.float64)
    local_gradient = np.asarray(local["gradient"], dtype=np.float64)
    velocity = np.asarray(local["value"], dtype=np.float64) + displacement @ local_gradient.T
    gradient = np.broadcast_to(local_gradient, (len(displacement), 3, 3)).copy()
    return velocity, gradient


__all__ = ["FMMInduction"]
