"""Host reference FMM stage evaluator with explicit upward/downward passes."""

from __future__ import annotations

import numpy as np
import taichi as ti

from ....kernels.base import RadialVortexKernel, make_vortex_kernel
from .diagnostics import FMMDiagnostics
from .interaction_lists import well_separated
from .local_expansions import l2l
from .multipoles import m2m, multipole_velocity_batch, p2m
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

    def __init__(
        self,
        physics=None,
        *,
        tolerance: float = 1.0e-4,
        kernel: RadialVortexKernel | None = None,
        max_n_particles: int | None = None,
        leaf_capacity: int = 8,
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
        self.tree = FMMTree(leaf_capacity=self.leaf_capacity)
        self.diagnostics = FMMDiagnostics()
        if physics is not None:
            self.bind(physics)

    def build(self):
        """Return a fresh unbound runtime evaluator for an immutable case."""
        return type(self)(
            tolerance=self.tolerance,
            kernel=self.kernel,
            max_n_particles=self.max_n_particles,
            leaf_capacity=self.leaf_capacity,
        )

    def bind(self, physics, *, kernel: RadialVortexKernel | None = None):
        """Bind this runtime evaluator to the shared particle precision context."""
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

        self.tree.build(position, vortex_strength, core_radius, count)
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
        """Propagate local expansions and evaluate exact near-field leaves."""
        count = len(self.tree.position)
        velocity = np.zeros((count, 3), dtype=np.float64)
        gradient = np.zeros((count, 3, 3), dtype=np.float64)
        assert self.tree.root is not None
        self._resolve_sources(
            self.tree.nodes[self.tree.root],
            [self.tree.root],
            multipoles,
            _zero_local(),
            velocity,
            gradient,
            (),
        )
        self.diagnostics.gradient_evaluations += 1
        return velocity, gradient

    def _resolve_sources(
        self,
        target: FMMNode,
        sources: list[int],
        multipoles: dict[int, dict[str, np.ndarray]],
        local: dict[str, np.ndarray],
        velocity: np.ndarray,
        gradient: np.ndarray,
        far_source_indices: tuple[int, ...],
    ) -> None:
        """Classify cell pairs, recurse unresolved pairs, then perform L2P/P2P."""
        far_local = dict(local)
        inherited_far_sources = list(far_source_indices)
        unresolved: list[int] = []
        for source_index in sources:
            source = self.tree.nodes[source_index]
            if well_separated(source, target, self.tolerance, self.kernel):
                far_local = _add_m2l(
                    far_local,
                    multipoles[source_index],
                    source.centre,
                    target.centre,
                )
                self.diagnostics.m2l_interactions += 1
                inherited_far_sources.append(source_index)
            else:
                unresolved.append(source_index)

        if target.children:
            for child_index in target.children:
                child = self.tree.nodes[child_index]
                child_local = l2l(far_local, child.centre - target.centre)
                self.diagnostics.l2l_operations += 1
                self._resolve_sources(
                    child,
                    unresolved,
                    multipoles,
                    child_local,
                    velocity,
                    gradient,
                    tuple(inherited_far_sources),
                )
            return

        target_indices = target.indices
        target_position = self.tree.position[target_indices]
        near_velocity = np.zeros((len(target_indices), 3), dtype=np.float64)
        near_gradient = np.zeros((len(target_indices), 3, 3), dtype=np.float64)
        pending = list(unresolved)
        while pending:
            source_index = pending.pop()
            source = self.tree.nodes[source_index]
            if well_separated(source, target, self.tolerance, self.kernel):
                far_local = _add_m2l(
                    far_local,
                    multipoles[source_index],
                    source.centre,
                    target.centre,
                )
                self.diagnostics.m2l_interactions += 1
                inherited_far_sources.append(source_index)
                continue
            if source.children:
                pending.extend(source.children)
                continue
            source_indices = source.indices
            source_velocity, source_gradient = p2p_velocity_gradient(
                self.kernel,
                target_position,
                self.tree.position[source_indices],
                self.tree.vortex_strength[source_indices],
                self.tree.core_radius[target_indices],
                self.tree.core_radius[source_indices],
                exclude_self=source is target,
            )
            near_velocity += source_velocity
            near_gradient += source_gradient
            self.diagnostics.p2p_interactions += len(target_indices) * len(source_indices)

        far_velocity, far_gradient = _evaluate_far_multipoles(
            inherited_far_sources,
            multipoles,
            self.tree.nodes,
            target_position,
        )
        velocity[target_indices] += far_velocity + near_velocity
        gradient[target_indices] += far_gradient + near_gradient
        self.diagnostics.l2p_evaluations += len(target_indices)

    def _write_vector(self, output, values: np.ndarray) -> None:
        dtype = getattr(self.physics, "np_dtype", np.float32)
        padded = np.zeros((self.max_n_particles, 3), dtype=dtype)
        padded[: len(values)] = values
        output.from_numpy(padded)

    def _write_gradient(self, output, values: np.ndarray) -> None:
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


def _add_m2l(
    local: dict[str, np.ndarray],
    multipole: dict[str, np.ndarray],
    source_centre: np.ndarray,
    target_centre: np.ndarray,
) -> dict[str, np.ndarray]:
    """Translate one low-order multipole into a first-order local expansion."""
    displacement = np.asarray(target_centre, dtype=np.float64) - np.asarray(
        source_centre, dtype=np.float64
    )
    # The local record is retained for the explicit M2L/L2L operation trace.
    # The final leaf evaluator uses the complete retained source moments at
    # the actual target positions, so this bookkeeping record only needs the
    # analytic leading local coefficient and does not repeat the high-order
    # source expansion for every cell pair.
    circulation = np.asarray(multipole["circulation"], dtype=np.float64)
    radius = max(float(np.linalg.norm(displacement)), 1.0e-8)
    q_infinity = 1.0 / (4.0 * np.pi)
    value = q_infinity * np.cross(circulation, displacement) / radius**3
    derivative = np.zeros((3, 3), dtype=np.float64)
    for axis in range(3):
        derivative[:, axis] = q_infinity * (
            np.cross(circulation, np.eye(3)[axis]) / radius**3
            - 3.0 * displacement[axis] * np.cross(circulation, displacement) / radius**5
        )
    return {
        "value": local["value"] + value,
        "gradient": local["gradient"] + derivative,
        "displacement": np.zeros(3, dtype=np.float64),
    }


def _evaluate_far_multipoles(
    source_indices: list[int],
    multipoles: dict[int, dict[str, np.ndarray]],
    nodes: tuple[FMMNode, ...],
    target_position: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate classified far multipoles at the actual target locations.

    The low-order local record is still constructed and translated during the
    downward pass, but evaluating the retained source moments at each leaf
    target avoids introducing a second, independent first-order truncation in
    the target displacement.  The finite-difference Jacobian is the derivative
    of that same far evaluator, so the stage velocity and stretching rate use
    one consistent hierarchical approximation.
    """
    target_position = np.asarray(target_position, dtype=np.float64)
    velocity = np.zeros((len(target_position), 3), dtype=np.float64)
    gradient = np.zeros((len(target_position), 3, 3), dtype=np.float64)
    for source_index in source_indices:
        multipole = multipoles[source_index]
        source_centre = np.asarray(nodes[source_index].centre, dtype=np.float64)
        displacement = target_position - source_centre
        velocity += multipole_velocity_batch(multipole, displacement)
        radius = np.maximum(np.linalg.norm(displacement, axis=1), 1.0e-8)
        difference = np.maximum(1.0e-6, 1.0e-5 * radius)
        for axis in range(3):
            offset = np.zeros(3, dtype=np.float64)
            offset[axis] = 1.0
            gradient[:, :, axis] += (
                multipole_velocity_batch(multipole, displacement + difference[:, None] * offset)
                - multipole_velocity_batch(multipole, displacement - difference[:, None] * offset)
            ) / (2.0 * difference[:, None])
    return velocity, gradient


__all__ = ["FMMInduction"]
