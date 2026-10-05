"""Device-resident Cartesian FMM for coupled VPM stages.

The workspace reuses the verified device LBVH for hierarchy construction only.
Evaluation is a separate dual-tree path with P2M, M2M, M2L, L2L, L2P, and an
exact kernel-specific near-field P2P pass. Particle state never crosses to
NumPy during an RK stage.

The source and local expansions use a fixed p=3 Cartesian basis. The internal
admissibility test keeps the omitted Taylor and regularisation tails in the
near list, while analytic local derivatives supply the complete velocity
gradient. Expansion and traversal settings are not part of the public API.
"""

from contextlib import contextmanager
import math
import time
from typing import NamedTuple, Self

import numpy as np
import taichi as ti

from ....io.logging import Logging
from ....kernels.base import RadialVortexKernel, make_vortex_kernel
from ..base import _STRETCHING_MODES, normalize_stretching_scheme
from ..stretching import stretching_rate
from ..treecode.lbvh import _TRAVERSAL_BATCH_SIZE, TaichiTreecode, _DeviceFields
from .diagnostics import FMMDiagnostics

_EXPANSION_ORDER = 3
_MULTI_INDICES = (
    (0, 0, 0),
    (0, 0, 1),
    (0, 1, 0),
    (1, 0, 0),
    (0, 0, 2),
    (0, 1, 1),
    (0, 2, 0),
    (1, 0, 1),
    (1, 1, 0),
    (2, 0, 0),
    (0, 0, 3),
    (0, 1, 2),
    (0, 2, 1),
    (0, 3, 0),
    (1, 0, 2),
    (1, 1, 1),
    (1, 2, 0),
    (2, 0, 1),
    (2, 1, 0),
    (3, 0, 0),
)
_MOMENT_COUNT = len(_MULTI_INDICES)
_LOCAL_COUNT = len(_MULTI_INDICES)
_DERIVATIVE_ORDER = 2 * _EXPANSION_ORDER
_DERIVATIVE_INDICES = tuple(
    (a, b, total - a - b)
    for total in range(_DERIVATIVE_ORDER + 1)
    for a in range(total + 1)
    for b in range(total - a + 1)
)
_DERIVATIVE_COUNT = len(_DERIVATIVE_INDICES)
_MAX_DERIVATIVE_TERMS = 8
_M2L_BATCH_SIZE = 32768
# Image locals are optional execution scratch, unlike the physical-source
# operator. Dense source/target overlap must use the established target walk,
# not grow eight pair arrays until a small accelerator runs out of memory.
_MAX_IMAGE_PAIR_CAPACITY = 1 << 22
_FMM_LEAF_CAPACITY = 32
_MAX_TREE_LEVELS = 96
# Start each list at 32 pairs per particle, then grow from observed demand.
# Overflowed final lists are never consumed by the field passes.
_PAIR_CAPACITY_FACTOR = 32
_MAX_PAIR_CAPACITY = (1 << 30) - 1  # A queued pair can produce two i32 entries.
_MAX_LIST_GROWTH_RETRIES = 4
_EPSILON_SQUARED = 1.0e-24
_ONE_OVER_FOUR_PI = 0.07957747154594767
_GEOMETRIC_SEPARATION_FACTOR = 3.0
_VELOCITY_TAIL_RELATIVE_TOLERANCE = 1.0e-5
_GRADIENT_TAIL_RELATIVE_TOLERANCE = 1.0e-5


class _ListCapacities(NamedTuple):
    m2l: int
    near: int
    queue: int

    @property
    def storage_bytes(self) -> int:
        # M2L sources become ordered near sources after the far-field pass.
        return 4 * (self.m2l + max(self.m2l, self.near) + 2 * self.near + 4 * self.queue)


def _normalize_list_capacities(value) -> _ListCapacities:
    values = value if isinstance(value, tuple) else (value,) * 3
    if len(values) != 3:
        raise ValueError("FMM scratch requires three interaction-list capacities")
    capacities = _ListCapacities(*(int(item) for item in values))
    if any(not 1 <= item <= _MAX_PAIR_CAPACITY for item in capacities):
        raise ValueError("FMM interaction-list capacity is outside the safe i32 range")
    return capacities


class _InteractionListCapacityError(RuntimeError):
    """A scratch-storage shortage, raised before any list is consumed."""

    def __init__(self, capacity, *, m2l, near, queue_a, queue_b, queue_peak=0):
        self.capacities = _normalize_list_capacities(capacity)
        self.counts = _ListCapacities(m2l, near, max(queue_a, queue_b, queue_peak))
        self.required_pairs = max(self.counts)
        super().__init__(
            "FMM interaction-list capacity was exceeded "
            f"(capacities={self.capacities}, observed={self.counts})"
        )


def _double_factorial(value: int) -> int:
    """Return ``value!!`` for the derivative-table coefficient builder.

    Parameters
    ----------
    value : int
        Non-negative integer whose same-parity factors are multiplied.

    Returns
    -------
    int
        Product ``value * (value - 2) * ...`` down to one or two. The helper
        is evaluated during host-side FMM metadata construction only.
    """
    result = 1
    for factor in range(value, 0, -2):
        result *= factor
    return result


def _translation_tables():
    """Build fixed analytic ``1/(4*pi*r)`` derivative metadata through order six.

    Returns
    -------
    tuple[numpy.ndarray, ...]
        Lookup indices, term counts, coefficients, Cartesian exponents, and
        radial powers consumed by the Taichi M2L kernels.
    """
    lookup = np.full(
        (_DERIVATIVE_ORDER + 1,) * 3,
        -1,
        dtype=np.int32,
    )
    term_count = np.zeros(_DERIVATIVE_COUNT, dtype=np.int32)
    coefficient = np.zeros((_DERIVATIVE_COUNT, _MAX_DERIVATIVE_TERMS), dtype=np.float32)
    exponent = np.zeros((_DERIVATIVE_COUNT, _MAX_DERIVATIVE_TERMS, 3), dtype=np.int32)
    radial_step = np.zeros((_DERIVATIVE_COUNT, _MAX_DERIVATIVE_TERMS), dtype=np.int32)
    for derivative_index, alpha in enumerate(_DERIVATIVE_INDICES):
        lookup[alpha] = derivative_index
        order = sum(alpha)
        slot = 0
        for px in range(alpha[0] // 2 + 1):
            for py in range(alpha[1] // 2 + 1):
                for pz in range(alpha[2] // 2 + 1):
                    p = (px, py, pz)
                    contraction_count = sum(p)
                    remainder = tuple(alpha[axis] - 2 * p[axis] for axis in range(3))
                    numerator = math.prod(math.factorial(value) for value in alpha)
                    denominator = math.prod(
                        math.factorial(remainder[axis]) * math.factorial(p[axis])
                        for axis in range(3)
                    )
                    sign = -1.0 if (order - contraction_count) % 2 else 1.0
                    coefficient[derivative_index, slot] = (
                        sign
                        * _double_factorial(2 * (order - contraction_count) - 1)
                        * numerator
                        / (2**contraction_count * denominator)
                        * _ONE_OVER_FOUR_PI
                    )
                    exponent[derivative_index, slot] = remainder
                    radial_step[derivative_index, slot] = order - contraction_count
                    slot += 1
        term_count[derivative_index] = slot
    return lookup, term_count, coefficient, exponent, radial_step


@ti.data_oriented
class FMMDeviceWorkspace:
    """Preallocated f32 fields implementing the fixed-order device FMM.

    The workspace stores source moments, local
    expansions, interaction lists, near-field pairs, output rates, and scalar
    diagnostics for one fixed particle capacity. The algorithm uses a
    Cartesian expansion of order three, exact kernel-specific P2P interactions
    for non-admissible pairs, and analytic local derivatives for gradients.

    Parameters
    ----------
    max_n_particles : int
        Fixed particle capacity. Allocation scales with capacity, not active
        count.
    radial_factors : callable
        Kernel-specific finite P2P velocity/Jacobian factors in m⁻³ and m⁻⁵.
    kernel_name : {"GAUSSIAN", "WINCKELMANS"}
        Radial kernel used by the shared strict LBVH target traversal. Other
        FMM kernels use the exact direct arbitrary-target operator.
    velocity_tail_cutoff, gradient_tail_cutoff : float
        Dimensionless regularization-tail multipliers used by admissibility.
    max_evaluation_points : int
        Maximum arbitrary-target batch size. Target traversal storage scales
        with the smaller of this value and particle capacity.

    Notes
    -----
    All numerical fields use f32. The workspace is not thread-safe and must
    be used by one induction evaluator at a time.
    """

    def __init__(
        self,
        max_n_particles: int,
        radial_factors,
        kernel_name: str,
        velocity_tail_cutoff: float,
        gradient_tail_cutoff: float,
        max_evaluation_points: int,
        *,
        max_pairs: int | None = None,
        _list_capacities: _ListCapacities | None = None,
    ) -> None:
        """Allocate fixed-capacity f32 storage for one device FMM evaluator.

        Parameters
        ----------
        max_n_particles : int
            Maximum active source/target count. Device fields are allocated
            for this capacity, so increasing it changes memory use.
        radial_factors : callable
            Taichi function of ``(r/sigma, sigma, with_gradient)`` returning
            the finite induced velocity and Jacobian factors in m⁻³ and m⁻⁵.
        kernel_name : str
            Radial-kernel name used by the shared LBVH target traversal.
        velocity_tail_cutoff, gradient_tail_cutoff : float
            Dimensionless tail thresholds used when deciding whether a cell
            interaction is admissible for velocity and gradient evaluation.
        max_evaluation_points : int
            Maximum arbitrary-target batch size. Larger queries are processed
            in consecutive batches without changing their results.
        max_pairs : int or None, default=None
            Equal capacity of each interaction list and traversal queue.
            None uses the initial per-particle sizing heuristic.

        Notes
        -----
        Allocation has a device-side memory cost and is not thread-safe. All
        numerical fields use ``ti.f32``; the caller must reuse this workspace
        only with a compatible fixed-capacity evaluator.
        """
        self.max_n_particles = int(max_n_particles)
        self.max_nodes = 2 * self.max_n_particles
        if max_pairs is not None and _list_capacities is not None:
            raise ValueError("specify either equal or independent FMM scratch capacities")
        initial = _PAIR_CAPACITY_FACTOR * self.max_n_particles if max_pairs is None else max_pairs
        self.list_capacities = _normalize_list_capacities(
            initial if _list_capacities is None else _list_capacities
        )
        self.max_m2l_pairs, self.max_near_pairs, self.max_queue_pairs = self.list_capacities
        self.m2l_batch_size = min(_M2L_BATCH_SIZE, self.max_m2l_pairs)
        self.radial_factors = radial_factors
        self.velocity_tail_cutoff = float(velocity_tail_cutoff)
        self.gradient_tail_cutoff = float(gradient_tail_cutoff)
        self.target_batch_capacity = min(max(1, int(max_evaluation_points)), _TRAVERSAL_BATCH_SIZE)
        self.kernel_name = kernel_name
        self.tree = TaichiTreecode(
            max_n_particles=self.max_n_particles,
            max_nodes=self.max_nodes,
            theta=0.1,
            max_leaf_size=1,
            kernel_type=kernel_name if kernel_name in {"GAUSSIAN", "WINCKELMANS"} else "GAUSSIAN",
            multipole_order=1,
            sort_particle_targets=False,
            traversal_block_dim=0,
            device_sort_only=False,
            hierarchy_only=True,
            max_evaluation_points=self.target_batch_capacity,
        )
        fields = _DeviceFields()
        self.multipole = fields.vector(3, dtype=ti.f32, shape=self.max_nodes * _MOMENT_COUNT)
        self.local = fields.vector(3, dtype=ti.f32, shape=self.max_nodes * _LOCAL_COUNT)
        # Expansions are needed only on the interaction tree, which stops at
        # maximal <=32-particle cells, not at every singleton LBVH leaf.
        self._active_internal_nodes = fields.scalar(dtype=ti.i32, shape=self.max_nodes)
        self._active_leaf_nodes = fields.scalar(dtype=ti.i32, shape=self.max_n_particles)
        self._particle_cell = fields.scalar(dtype=ti.i32, shape=self.max_n_particles)
        self._level_node_count = fields.scalar(dtype=ti.i32, shape=_MAX_TREE_LEVELS)
        self._level_node_start = fields.scalar(dtype=ti.i32, shape=_MAX_TREE_LEVELS)
        self._level_node_cursor = fields.scalar(dtype=ti.i32, shape=_MAX_TREE_LEVELS)
        self._active_internal_count = fields.scalar(dtype=ti.i32, shape=())
        self._active_leaf_count = fields.scalar(dtype=ti.i32, shape=())
        self._coefficient_a = fields.scalar(dtype=ti.i32, shape=_MOMENT_COUNT)
        self._coefficient_b = fields.scalar(dtype=ti.i32, shape=_MOMENT_COUNT)
        self._coefficient_c = fields.scalar(dtype=ti.i32, shape=_MOMENT_COUNT)
        self._derivative_lookup = fields.scalar(
            dtype=ti.i32,
            shape=(
                _DERIVATIVE_ORDER + 1,
                _DERIVATIVE_ORDER + 1,
                _DERIVATIVE_ORDER + 1,
            ),
        )
        self._derivative_term_count = fields.scalar(dtype=ti.i32, shape=_DERIVATIVE_COUNT)
        self._derivative_coefficient = fields.scalar(
            dtype=ti.f32,
            shape=(_DERIVATIVE_COUNT, _MAX_DERIVATIVE_TERMS),
        )
        self._derivative_exponent = fields.vector(
            3,
            dtype=ti.i32,
            shape=(_DERIVATIVE_COUNT, _MAX_DERIVATIVE_TERMS),
        )
        self._derivative_radial_step = fields.scalar(
            dtype=ti.i32,
            shape=(_DERIVATIVE_COUNT, _MAX_DERIVATIVE_TERMS),
        )
        self._m2l_derivative = fields.scalar(
            dtype=ti.f32,
            shape=(self.m2l_batch_size, _DERIVATIVE_COUNT),
        )
        self.m2l_target = fields.scalar(dtype=ti.i32, shape=self.max_m2l_pairs)
        self.m2l_source = fields.scalar(
            dtype=ti.i32, shape=max(self.max_m2l_pairs, self.max_near_pairs)
        )
        self.near_target = fields.scalar(dtype=ti.i32, shape=self.max_near_pairs)
        self.near_source = fields.scalar(dtype=ti.i32, shape=self.max_near_pairs)
        self._m2l_count = fields.scalar(dtype=ti.i32, shape=())
        self._near_count = fields.scalar(dtype=ti.i32, shape=())
        # Metal lacks the i64 atomic required by a monolithic exact counter.
        # Two u32 limbs retain every near P2P interaction count without
        # relying on lossy f32 accumulation. A single list entry contains at
        # most 32*32 interactions, so carry detection is unambiguous.
        self._p2p_particle_count_low = fields.scalar(dtype=ti.u32, shape=())
        self._p2p_particle_count_high = fields.scalar(dtype=ti.u32, shape=())
        self._list_error = fields.scalar(dtype=ti.i32, shape=())
        self._nonzero_l2l_count = fields.scalar(dtype=ti.i32, shape=())
        self._queue_target_a = fields.scalar(dtype=ti.i32, shape=self.max_queue_pairs)
        self._queue_source_a = fields.scalar(dtype=ti.i32, shape=self.max_queue_pairs)
        self._queue_target_b = fields.scalar(dtype=ti.i32, shape=self.max_queue_pairs)
        self._queue_source_b = fields.scalar(dtype=ti.i32, shape=self.max_queue_pairs)
        self._queue_count_a = fields.scalar(dtype=ti.i32, shape=())
        self._queue_count_b = fields.scalar(dtype=ti.i32, shape=())
        self._queue_peak_count = fields.scalar(dtype=ti.i32, shape=())
        self._near_target_count = fields.scalar(dtype=ti.i32, shape=self.max_nodes)
        self._near_scan_a = fields.scalar(dtype=ti.i32, shape=self.max_nodes)
        self._near_scan_b = fields.scalar(dtype=ti.i32, shape=self.max_nodes)
        self.velocity = fields.vector(3, dtype=ti.f32, shape=self.max_n_particles)
        self.gradient = fields.matrix(3, 3, dtype=ti.f32, shape=self.max_n_particles)
        self.rate = fields.vector(3, dtype=ti.f32, shape=self.max_n_particles)
        self._rate_sum = fields.vector(3, dtype=ti.f32, shape=())
        self._rate_norm_sum = fields.scalar(dtype=ti.f32, shape=())
        self._rate_defect = fields.scalar(dtype=ti.f32, shape=())
        fields.finalize()
        self._device_fields = fields
        coefficient_array = np.asarray(_MULTI_INDICES, dtype=np.int32)
        self._coefficient_a.from_numpy(coefficient_array[:, 0])
        self._coefficient_b.from_numpy(coefficient_array[:, 1])
        self._coefficient_c.from_numpy(coefficient_array[:, 2])
        lookup, term_count, coefficient, exponent, radial_step = _translation_tables()
        self._derivative_lookup.from_numpy(lookup)
        self._derivative_term_count.from_numpy(term_count)
        self._derivative_coefficient.from_numpy(coefficient)
        self._derivative_exponent.from_numpy(exponent)
        self._derivative_radial_step.from_numpy(radial_step)
        self.profile_passes = False
        self.source_multipole_generation = 0
        self.last_phase_seconds = {
            "tree_build": 0.0,
            "upward": 0.0,
            "interaction_lists": 0.0,
            "m2l": 0.0,
            "downward": 0.0,
            "near_field": 0.0,
            "strength_rate": 0.0,
        }

    def destroy(self) -> None:
        """Release the disposable FMM and hierarchy scratch fields."""
        self.tree.destroy()
        self._device_fields.destroy()

    @property
    def max_pairs(self) -> int:
        """Largest list capacity, retained for aggregate scratch reporting."""
        return max(self.list_capacities)

    @ti.func
    def _factorial(self, value: ti.i32) -> ti.f32:
        result = ti.cast(1.0, ti.f32)
        for i in ti.static(range(1, _DERIVATIVE_ORDER + 1)):
            if i <= value:
                result *= ti.cast(i, ti.f32)
        return result

    @ti.func
    def _small_power(self, value: ti.f32, exponent: ti.i32) -> ti.f32:
        result = ti.cast(1.0, ti.f32)
        for power in ti.static(range(_DERIVATIVE_ORDER)):
            if power < exponent:
                result *= value
        return result

    # Keep this description outside the Taichi function body. Taichi 1.7.4
    # misidentifies this bound helper as nested when its first statement is a
    # docstring, which prevents the enclosing kernels from compiling.
    @ti.func
    def _inverse_r_derivative(
        self, displacement: ti.template(), derivative_index: ti.i32
    ) -> ti.f32:
        radius_sq = displacement.dot(displacement)
        inverse_radius = ti.rsqrt(ti.max(radius_sq, ti.cast(_EPSILON_SQUARED, ti.f32)))
        inverse_radius_sq = inverse_radius * inverse_radius
        value = ti.cast(0.0, ti.f32)
        for term in ti.static(range(_MAX_DERIVATIVE_TERMS)):
            if term < self._derivative_term_count[derivative_index]:
                exponent = self._derivative_exponent[derivative_index, term]
                monomial = (
                    self._small_power(displacement[0], exponent[0])
                    * self._small_power(displacement[1], exponent[1])
                    * self._small_power(displacement[2], exponent[2])
                )
                radial = inverse_radius
                for step in ti.static(range(_DERIVATIVE_ORDER)):
                    if step < self._derivative_radial_step[derivative_index, term]:
                        radial *= inverse_radius_sq
                value += self._derivative_coefficient[derivative_index, term] * monomial * radial
        return value

    @ti.func
    def _well_separated(self, target: ti.i32, source: ti.i32) -> ti.i32:
        result = 0
        if target != source:
            displacement = self.tree.node_centre[target] - self.tree.node_centre[source]
            distance = ti.sqrt(displacement.dot(displacement))
            cell_radius = self.tree.node_half_size[target] + self.tree.node_half_size[source]
            pair_core_radius_bound = 0.5 * (
                self.tree.node_max_radius[target] + self.tree.node_max_radius[source]
            )
            velocity_regularization_radius = self.velocity_tail_cutoff * pair_core_radius_bound
            gradient_regularization_radius = self.gradient_tail_cutoff * pair_core_radius_bound
            if (
                distance > _GEOMETRIC_SEPARATION_FACTOR * cell_radius
                and distance > cell_radius + velocity_regularization_radius
                and distance > cell_radius + gradient_regularization_radius
            ):
                result = 1
        return result

    @ti.func
    def _is_fmm_leaf(self, node: ti.i32) -> ti.i32:
        return 1 if self.tree.node_particle_count[node] <= _FMM_LEAF_CAPACITY else 0

    @ti.func
    def _is_active_cell(self, node: ti.i32) -> ti.i32:
        parent = self.tree.node_parent[node]
        active = self._is_fmm_leaf(node) == 0
        if parent < 0:  # noqa: SIM114 - do not read a negative parent in Taichi.
            active = True
        elif self._is_fmm_leaf(parent) == 0:
            active = True
        return 1 if active else 0

    @ti.kernel
    def _count_active_cells(self, node_count: ti.i32):
        for level in range(_MAX_TREE_LEVELS):
            self._level_node_count[level] = 0
        self._active_leaf_count[None] = 0
        for node in range(node_count):
            if self._is_fmm_leaf(node) == 0:
                ti.atomic_add(self._level_node_count[self.tree.node_depth[node]], 1)

    @ti.kernel
    def _initialize_active_cell_offsets(self):
        total = 0
        ti.loop_config(serialize=True)
        for level in range(_MAX_TREE_LEVELS):
            self._level_node_start[level] = total
            self._level_node_cursor[level] = total
            total += self._level_node_count[level]
        self._active_internal_count[None] = total

    @ti.kernel
    def _write_active_cell_schedule(self, node_count: ti.i32):
        for node in range(node_count):
            if self._is_fmm_leaf(node) == 0:
                destination = ti.atomic_add(self._level_node_cursor[self.tree.node_depth[node]], 1)
                self._active_internal_nodes[destination] = node
            elif self._is_active_cell(node) != 0:
                destination = ti.atomic_add(self._active_leaf_count[None], 1)
                self._active_leaf_nodes[destination] = node
                first = self.tree.node_particle_start[node]
                for slot in range(first, first + self.tree.node_particle_count[node]):
                    self._particle_cell[slot] = node

    @ti.kernel
    def _zero_pass(self, node_count: ti.i32, particle_count: ti.i32):
        self._nonzero_l2l_count[None] = 0
        for node in range(node_count):
            if self._is_active_cell(node) != 0:
                for coefficient in ti.static(range(_MOMENT_COUNT)):
                    self.multipole[node * _MOMENT_COUNT + coefficient] = ti.Vector([0.0, 0.0, 0.0])
                for coefficient in ti.static(range(_LOCAL_COUNT)):
                    self.local[node * _LOCAL_COUNT + coefficient] = ti.Vector([0.0, 0.0, 0.0])
        for particle in range(particle_count):
            self.velocity[particle] = ti.Vector([0.0, 0.0, 0.0])
            self.gradient[particle] = ti.Matrix.zero(ti.f32, 3, 3)
            self.rate[particle] = ti.Vector([0.0, 0.0, 0.0])

    @ti.kernel
    def _p2m_pass(self, count: ti.i32):
        for leaf_slot in range(self._active_leaf_count[None]):
            node = self._active_leaf_nodes[leaf_slot]
            first = self.tree.node_particle_start[node]
            for coefficient in range(_MOMENT_COUNT):
                a = self._coefficient_a[coefficient]
                b = self._coefficient_b[coefficient]
                c = self._coefficient_c[coefficient]
                moment = ti.Vector([0.0, 0.0, 0.0])
                for slot in range(first, first + self.tree.node_particle_count[node]):
                    particle = self.tree.sorted_indices[slot]
                    offset = self.tree.position[particle] - self.tree.node_centre[node]
                    scale = (
                        self._small_power(offset[0], a)
                        * self._small_power(offset[1], b)
                        * self._small_power(offset[2], c)
                        / self._factorial(a)
                        / self._factorial(b)
                        / self._factorial(c)
                    )
                    moment += self.tree.vortex_strength[particle] * scale
                self.multipole[node * _MOMENT_COUNT + coefficient] = moment

    @ti.kernel
    def _m2m_level(self, count: ti.i32, level: ti.i32):
        for internal_slot in range(self._level_node_count[level]):
            node = self._active_internal_nodes[self._level_node_start[level] + internal_slot]
            if self.tree.node_depth[node] == level:
                node_base = node * _MOMENT_COUNT
                for alpha_index in range(_MOMENT_COUNT):
                    alpha_a = self._coefficient_a[alpha_index]
                    alpha_b = self._coefficient_b[alpha_index]
                    alpha_c = self._coefficient_c[alpha_index]
                    translated = ti.Vector([0.0, 0.0, 0.0])
                    for child_slot in ti.static(range(2)):
                        child = (
                            self.tree.node_left[node]
                            if child_slot == 0
                            else self.tree.node_right[node]
                        )
                        child_base = child * _MOMENT_COUNT
                        offset = self.tree.node_centre[child] - self.tree.node_centre[node]
                        for beta_index in range(_MOMENT_COUNT):
                            beta_a = self._coefficient_a[beta_index]
                            beta_b = self._coefficient_b[beta_index]
                            beta_c = self._coefficient_c[beta_index]
                            if beta_a <= alpha_a and beta_b <= alpha_b and beta_c <= alpha_c:
                                da = alpha_a - beta_a
                                db = alpha_b - beta_b
                                dc = alpha_c - beta_c
                                scale = (
                                    self._small_power(offset[0], da)
                                    * self._small_power(offset[1], db)
                                    * self._small_power(offset[2], dc)
                                    / self._factorial(da)
                                    / self._factorial(db)
                                    / self._factorial(dc)
                                )
                                translated += self.multipole[child_base + beta_index] * scale
                    self.multipole[node_base + alpha_index] = translated

    @ti.kernel
    def _initialize_interaction_lists(self):
        self._m2l_count[None] = 0
        self._near_count[None] = 0
        self._p2p_particle_count_low[None] = ti.u32(0)
        self._p2p_particle_count_high[None] = ti.u32(0)
        self._list_error[None] = 0
        root = self.tree._root[None]
        self._queue_target_a[0] = root
        self._queue_source_a[0] = root
        self._queue_count_a[None] = 1
        self._queue_count_b[None] = 0
        self._queue_peak_count[None] = 1

    @ti.func
    def _record_p2p_particle_count(self, particle_pairs: ti.i32):
        """Atomically add a bounded P2P count to the portable u64 diagnostic."""
        increment = ti.cast(particle_pairs, ti.u32)
        previous = ti.atomic_add(self._p2p_particle_count_low[None], increment)
        if previous > ti.u32(0xFFFFFFFF) - increment:
            ti.atomic_add(self._p2p_particle_count_high[None], ti.u32(1))

    @ti.func
    def _append_m2l_pair(self, target: ti.i32, source: ti.i32):
        slot = ti.atomic_add(self._m2l_count[None], 1)
        if slot < self.max_m2l_pairs:
            self.m2l_target[slot] = target
            self.m2l_source[slot] = source
        else:
            ti.atomic_max(self._list_error[None], 1)

    @ti.func
    def _append_near_pair(self, target: ti.i32, source: ti.i32):
        slot = ti.atomic_add(self._near_count[None], 1)
        if slot < self.max_near_pairs:
            self.near_target[slot] = target
            self.near_source[slot] = source
        else:
            ti.atomic_max(self._list_error[None], 1)
        particle_pairs = (
            self.tree.node_particle_count[target] * self.tree.node_particle_count[source]
        )
        if target == source:
            particle_pairs -= self.tree.node_particle_count[target]
        self._record_p2p_particle_count(particle_pairs)

    @ti.func
    def _append_queue_pair(
        self,
        target: ti.i32,
        source: ti.i32,
        target_queue: ti.template(),
        source_queue: ti.template(),
        queue_count: ti.template(),
    ):
        slot = ti.atomic_add(queue_count[None], 1)
        if slot < self.max_queue_pairs:
            target_queue[slot] = target
            source_queue[slot] = source
        else:
            ti.atomic_max(self._list_error[None], 1)

    @ti.func
    def _process_dual_tree_pair(
        self,
        target: ti.i32,
        source: ti.i32,
        target_queue: ti.template(),
        source_queue: ti.template(),
        queue_count: ti.template(),
    ):
        target_is_leaf = self._is_fmm_leaf(target)
        source_is_leaf = self._is_fmm_leaf(source)
        if self._well_separated(target, source) == 1:
            self._append_m2l_pair(target, source)
        elif target_is_leaf == 1 and source_is_leaf == 1:
            # Singleton self pairs still contribute a finite velocity Jacobian.
            self._append_near_pair(target, source)
        else:
            split_target = source_is_leaf == 1
            if target_is_leaf == 0 and source_is_leaf == 0:
                split_target = self.tree.node_half_size[target] >= self.tree.node_half_size[source]
            if split_target:
                self._append_queue_pair(
                    self.tree.node_left[target],
                    source,
                    target_queue,
                    source_queue,
                    queue_count,
                )
                self._append_queue_pair(
                    self.tree.node_right[target],
                    source,
                    target_queue,
                    source_queue,
                    queue_count,
                )
            else:
                self._append_queue_pair(
                    target,
                    self.tree.node_left[source],
                    target_queue,
                    source_queue,
                    queue_count,
                )
                self._append_queue_pair(
                    target,
                    self.tree.node_right[source],
                    target_queue,
                    source_queue,
                    queue_count,
                )

    @ti.kernel
    def _dual_tree_a_to_b(self):
        self._queue_peak_count[None] = ti.max(
            self._queue_peak_count[None], self._queue_count_a[None]
        )
        source_count = 0
        if self._queue_peak_count[None] <= self.max_queue_pairs:
            self._queue_count_b[None] = 0
            source_count = self._queue_count_a[None]
        for pair in range(source_count):
            self._process_dual_tree_pair(
                self._queue_target_a[pair],
                self._queue_source_a[pair],
                self._queue_target_b,
                self._queue_source_b,
                self._queue_count_b,
            )

    @ti.kernel
    def _dual_tree_b_to_a(self):
        self._queue_peak_count[None] = ti.max(
            self._queue_peak_count[None], self._queue_count_b[None]
        )
        source_count = 0
        if self._queue_peak_count[None] <= self.max_queue_pairs:
            self._queue_count_a[None] = 0
            source_count = self._queue_count_b[None]
        for pair in range(source_count):
            self._process_dual_tree_pair(
                self._queue_target_b[pair],
                self._queue_source_b[pair],
                self._queue_target_a,
                self._queue_source_a,
                self._queue_count_a,
            )

    @ti.kernel
    def _finalize_interaction_lists(self):
        self._queue_peak_count[None] = ti.max(
            self._queue_peak_count[None], self._queue_count_a[None], self._queue_count_b[None]
        )
        if (self._queue_count_a[None] != 0 or self._queue_count_b[None] != 0) and self._list_error[
            None
        ] == 0:
            self._list_error[None] = 2

    def _build_interaction_lists(self, pass_count: int) -> None:
        """Build a partitioning dual-tree list with fixed device passes."""
        self._initialize_interaction_lists()
        for pass_index in range(pass_count):
            if pass_index % 2 == 0:
                self._dual_tree_a_to_b()
            else:
                self._dual_tree_b_to_a()
        self._finalize_interaction_lists()

    @ti.kernel
    def _m2l_derivative_pass(
        self,
        pair_start: ti.i32,
        batch_count: ti.i32,
    ):
        """Cache all analytic derivatives needed by one bounded M2L batch."""
        for local_pair in range(batch_count):
            pair = pair_start + local_pair
            target = self.m2l_target[pair]
            source = self.m2l_source[pair]
            displacement = self.tree.node_centre[target] - self.tree.node_centre[source]
            for derivative_index in range(_DERIVATIVE_COUNT):
                self._m2l_derivative[local_pair, derivative_index] = self._inverse_r_derivative(
                    displacement, derivative_index
                )

    @ti.kernel
    def _m2l_accumulate_pass(self, pair_start: ti.i32, batch_count: ti.i32):
        """Contract p=3 source moments into p=3 target locals."""
        for local_pair in range(batch_count):
            pair = pair_start + local_pair
            target = self.m2l_target[pair]
            source = self.m2l_source[pair]
            source_base = source * _MOMENT_COUNT
            target_base = target * _LOCAL_COUNT
            for beta_index in range(_LOCAL_COUNT):
                beta_a = self._coefficient_a[beta_index]
                beta_b = self._coefficient_b[beta_index]
                beta_c = self._coefficient_c[beta_index]
                translated = ti.Vector([0.0, 0.0, 0.0])
                for alpha_index in range(_MOMENT_COUNT):
                    alpha_a = self._coefficient_a[alpha_index]
                    alpha_b = self._coefficient_b[alpha_index]
                    alpha_c = self._coefficient_c[alpha_index]
                    derivative_index = self._derivative_lookup[
                        alpha_a + beta_a,
                        alpha_b + beta_b,
                        alpha_c + beta_c,
                    ]
                    sign = 1.0
                    if (alpha_a + alpha_b + alpha_c) % 2 == 1:
                        sign = -1.0
                    translated += (
                        sign
                        * self.multipole[source_base + alpha_index]
                        * self._m2l_derivative[local_pair, derivative_index]
                    )
                translated /= (
                    self._factorial(beta_a) * self._factorial(beta_b) * self._factorial(beta_c)
                )
                for component in ti.static(range(3)):
                    ti.atomic_add(
                        self.local[target_base + beta_index][component],
                        translated[component],
                    )

    @ti.kernel
    def _l2l_level(self, count: ti.i32, level: ti.i32):
        for internal_slot in range(self._level_node_count[level]):
            node = self._active_internal_nodes[self._level_node_start[level] + internal_slot]
            if self.tree.node_depth[node] == level:
                for child_slot in ti.static(range(2)):
                    child = (
                        self.tree.node_left[node] if child_slot == 0 else self.tree.node_right[node]
                    )
                    offset = self.tree.node_centre[child] - self.tree.node_centre[node]
                    parent_base = node * _LOCAL_COUNT
                    child_base = child * _LOCAL_COUNT
                    translation_norm_sq = 0.0
                    for beta_index in range(_LOCAL_COUNT):
                        beta_a = self._coefficient_a[beta_index]
                        beta_b = self._coefficient_b[beta_index]
                        beta_c = self._coefficient_c[beta_index]
                        translated = ti.Vector([0.0, 0.0, 0.0])
                        for gamma_index in range(_LOCAL_COUNT):
                            gamma_a = self._coefficient_a[gamma_index]
                            gamma_b = self._coefficient_b[gamma_index]
                            gamma_c = self._coefficient_c[gamma_index]
                            if gamma_a >= beta_a and gamma_b >= beta_b and gamma_c >= beta_c:
                                da = gamma_a - beta_a
                                db = gamma_b - beta_b
                                dc = gamma_c - beta_c
                                scale = (
                                    self._factorial(gamma_a)
                                    / self._factorial(beta_a)
                                    / self._factorial(da)
                                    * self._factorial(gamma_b)
                                    / self._factorial(beta_b)
                                    / self._factorial(db)
                                    * self._factorial(gamma_c)
                                    / self._factorial(beta_c)
                                    / self._factorial(dc)
                                    * self._small_power(offset[0], da)
                                    * self._small_power(offset[1], db)
                                    * self._small_power(offset[2], dc)
                                )
                                translated += self.local[parent_base + gamma_index] * scale
                        self.local[child_base + beta_index] += translated
                        translation_norm_sq += translated.dot(translated)
                    if translation_norm_sq > 0.0:
                        ti.atomic_add(self._nonzero_l2l_count[None], 1)

    @ti.kernel
    def _l2p_pass(self, count: ti.i32):
        for slot in range(count):
            particle = self.tree.sorted_indices[slot]
            node = self._particle_cell[slot]
            base = node * _LOCAL_COUNT
            offset = self.tree.position[particle] - self.tree.node_centre[node]
            x, y, z = offset[0], offset[1], offset[2]
            # The local is now evaluated at a multi-particle cell instead of
            # being translated down to a singleton centre. All cubic terms
            # must therefore contribute to the velocity and its Jacobian.
            a_x = (
                self.local[base + 3]
                + 2.0 * self.local[base + 9] * offset[0]
                + self.local[base + 8] * offset[1]
                + self.local[base + 7] * offset[2]
                + 3.0 * self.local[base + 19] * x * x
                + 2.0 * self.local[base + 18] * x * y
                + 2.0 * self.local[base + 17] * x * z
                + self.local[base + 16] * y * y
                + self.local[base + 15] * y * z
                + self.local[base + 14] * z * z
            )
            a_y = (
                self.local[base + 2]
                + self.local[base + 8] * offset[0]
                + 2.0 * self.local[base + 6] * offset[1]
                + self.local[base + 5] * offset[2]
                + self.local[base + 18] * x * x
                + 2.0 * self.local[base + 16] * x * y
                + self.local[base + 15] * x * z
                + 3.0 * self.local[base + 13] * y * y
                + 2.0 * self.local[base + 12] * y * z
                + self.local[base + 11] * z * z
            )
            a_z = (
                self.local[base + 1]
                + self.local[base + 7] * offset[0]
                + self.local[base + 5] * offset[1]
                + 2.0 * self.local[base + 4] * offset[2]
                + self.local[base + 17] * x * x
                + self.local[base + 15] * x * y
                + 2.0 * self.local[base + 14] * x * z
                + self.local[base + 12] * y * y
                + 2.0 * self.local[base + 11] * y * z
                + 3.0 * self.local[base + 10] * z * z
            )
            a_xx = (
                2.0 * self.local[base + 9]
                + 6.0 * self.local[base + 19] * x
                + 2.0 * self.local[base + 18] * y
                + 2.0 * self.local[base + 17] * z
            )
            a_xy = (
                self.local[base + 8]
                + 2.0 * self.local[base + 18] * x
                + 2.0 * self.local[base + 16] * y
                + self.local[base + 15] * z
            )
            a_xz = (
                self.local[base + 7]
                + 2.0 * self.local[base + 17] * x
                + self.local[base + 15] * y
                + 2.0 * self.local[base + 14] * z
            )
            a_yy = (
                2.0 * self.local[base + 6]
                + 2.0 * self.local[base + 16] * x
                + 6.0 * self.local[base + 13] * y
                + 2.0 * self.local[base + 12] * z
            )
            a_yz = (
                self.local[base + 5]
                + self.local[base + 15] * x
                + 2.0 * self.local[base + 12] * y
                + 2.0 * self.local[base + 11] * z
            )
            a_zz = (
                2.0 * self.local[base + 4]
                + 2.0 * self.local[base + 14] * x
                + 2.0 * self.local[base + 11] * y
                + 6.0 * self.local[base + 10] * z
            )
            self.velocity[particle] = ti.Vector([a_y[2] - a_z[1], a_z[0] - a_x[2], a_x[1] - a_y[0]])
            self.gradient[particle] = ti.Matrix(
                [
                    [a_xy[2] - a_xz[1], a_yy[2] - a_yz[1], a_yz[2] - a_zz[1]],
                    [a_xz[0] - a_xx[2], a_yz[0] - a_xy[2], a_zz[0] - a_xz[2]],
                    [a_xx[1] - a_xy[0], a_xy[1] - a_yy[0], a_xz[1] - a_yz[0]],
                ]
            )

    @ti.kernel
    def _count_near_targets(self, node_count: ti.i32, pair_count: ti.i32):
        for node in range(node_count):
            self._near_target_count[node] = 0
        for pair in range(pair_count):
            ti.atomic_add(self._near_target_count[self.near_target[pair]], 1)

    @ti.kernel
    def _copy_near_counts(self, node_count: ti.i32):
        for node in range(node_count):
            self._near_scan_a[node] = self._near_target_count[node]

    @ti.kernel
    def _scan_near_counts(
        self,
        source: ti.template(),
        destination: ti.template(),
        node_count: ti.i32,
        stride: ti.i32,
    ):
        """Perform one parallel inclusive-scan step over target-node counts."""
        for node in range(node_count):
            value = source[node]
            if node >= stride:
                value += source[node - stride]
            destination[node] = value

    @ti.kernel
    def _initialize_near_cursor(
        self,
        inclusive_count: ti.template(),
        cursor: ti.template(),
        node_count: ti.i32,
    ):
        for node in range(node_count):
            cursor[node] = inclusive_count[node] - self._near_target_count[node]

    @ti.kernel
    def _order_near_sources(self, cursor: ti.template(), pair_count: ti.i32):
        """Group source cells by target cell in the reusable M2L source array."""
        for pair in range(pair_count):
            target = self.near_target[pair]
            destination = ti.atomic_add(cursor[target], 1)
            self.m2l_source[destination] = self.near_source[pair]

    @ti.kernel
    def _near_field_target_pass(self, inclusive_count: ti.template(), particle_count: ti.i32):
        """Accumulate exact near interactions once per target particle."""
        for target_slot in range(particle_count):
            target = self.tree.sorted_indices[target_slot]
            target_node = self._particle_cell[target_slot]
            pair_count = self._near_target_count[target_node]
            pair_start = inclusive_count[target_node] - pair_count
            velocity = self.velocity[target]
            gradient = self.gradient[target]
            for pair in range(pair_start, pair_start + pair_count):
                source_node = self.m2l_source[pair]
                source_start = self.tree.node_particle_start[source_node]
                source_count = self.tree.node_particle_count[source_node]
                for source_slot in range(source_start, source_start + source_count):
                    source = self.tree.sorted_indices[source_slot]
                    displacement = self.tree.position[target] - self.tree.position[source]
                    radius = ti.sqrt(displacement.dot(displacement))
                    sigma = 0.5 * (self.tree.core_radius[target] + self.tree.core_radius[source])
                    source_strength = self.tree.vortex_strength[source]
                    factors = self.radial_factors(radius / sigma, sigma, True)
                    velocity += source_strength.cross(displacement) * factors[0]
                    cross_value = displacement.cross(source_strength)
                    for row in ti.static(range(3)):
                        for column in ti.static(range(3)):
                            skew_value = 0.0
                            if row == 0 and column == 1:
                                skew_value = -source_strength[2]
                            elif row == 0 and column == 2:
                                skew_value = source_strength[1]
                            elif row == 1 and column == 0:
                                skew_value = source_strength[2]
                            elif row == 1 and column == 2:
                                skew_value = -source_strength[0]
                            elif row == 2 and column == 0:
                                skew_value = -source_strength[1]
                            elif row == 2 and column == 1:
                                skew_value = source_strength[0]
                            gradient[row, column] += (
                                factors[0] * skew_value
                                + factors[1] * cross_value[row] * displacement[column]
                            )
            self.velocity[target] = velocity
            self.gradient[target] = gradient

    def _near_field_pass(self, node_count: int, particle_count: int, pair_count: int) -> None:
        """Build target adjacency and evaluate exact near fields without output atomics."""
        self._count_near_targets(node_count, pair_count)
        self._copy_near_counts(node_count)
        source = self._near_scan_a
        destination = self._near_scan_b
        stride = 1
        while stride < node_count:
            self._scan_near_counts(source, destination, node_count, stride)
            source, destination = destination, source
            stride *= 2
        self._initialize_near_cursor(source, destination, node_count)
        self._order_near_sources(destination, pair_count)
        self._near_field_target_pass(source, particle_count)

    @ti.kernel
    def _empty_target_pass(
        self,
        target_velocity: ti.template(),
        target_gradient: ti.template(),
        background_velocity: ti.template(),
        target_count: ti.i32,
        write_velocity: ti.template(),
        write_gradient: ti.template(),
    ):
        """Write freestream velocity and zero gradient for an empty source cloud."""
        for target in range(target_count):
            if ti.static(write_velocity):
                target_velocity[target] = background_velocity[None]
            if ti.static(write_gradient):
                target_gradient[target] = ti.Matrix.zero(ti.f32, 3, 3)

    @ti.kernel
    def _reset_rate_diagnostics(self):
        self._rate_sum[None] = ti.Vector([0.0, 0.0, 0.0])
        self._rate_norm_sum[None] = 0.0
        self._rate_defect[None] = 0.0

    @ti.kernel
    def _rate_pass(self, count: ti.i32, stretching_mode: ti.i32):
        for particle in range(count):
            rate = stretching_rate(
                self.gradient[particle], self.tree.vortex_strength[particle], stretching_mode
            )
            self.rate[particle] = rate
            for component in ti.static(range(3)):
                ti.atomic_add(self._rate_sum[None][component], rate[component])
            ti.atomic_add(self._rate_norm_sum[None], ti.sqrt(rate.dot(rate)))

    @ti.kernel
    def _finalize_rate_diagnostics(self):
        self._rate_defect[None] = ti.sqrt(self._rate_sum[None].dot(self._rate_sum[None]))

    def evaluate(
        self, position, vortex_strength, core_radius, count: int, stretching_mode: int
    ) -> None:
        """Run all FMM passes for one particle stage.

        Parameters
        ----------
        position : ti.Vector.field
            Source/target positions with logical shape ``(count, 3)`` in m.
        vortex_strength : ti.Vector.field
            Particle-strength vectors ``Gamma`` with shape ``(count, 3)`` in m³/s.
        core_radius : ti.field
            Core radii with shape ``(count,)`` in m.
        count : int
            Active particle count; only the prefix is read.
        stretching_mode : int
            Internal code for the direct, transposed, or mixed strength-rate
            contraction.

        Raises
        ------
        RuntimeError
            If the fixed interaction-list capacity is exceeded.

        Notes
        -----
        The method mutates workspace outputs and diagnostics. It rebuilds the
        hierarchy from the supplied stage so Runge--Kutta position, strength,
        and core-radius changes are represented in the source moments.
        """
        count = int(count)
        phase_start = time.perf_counter()
        self.tree.build(position, vortex_strength, core_radius, count)
        if self.profile_passes:
            ti.sync()
            self.last_phase_seconds["tree_build"] = time.perf_counter() - phase_start
        node_count = 2 * count - 1
        phase_start = time.perf_counter()
        level_count = self.prepare_source_multipoles(count)
        if self.profile_passes:
            ti.sync()
            self.last_phase_seconds["upward"] = time.perf_counter() - phase_start
        phase_start = time.perf_counter()
        self._build_interaction_lists(2 * level_count)
        list_error = int(self._list_error[None])
        if list_error == 1:
            raise _InteractionListCapacityError(
                self.list_capacities,
                m2l=int(self._m2l_count[None]),
                near=int(self._near_count[None]),
                queue_a=int(self._queue_count_a[None]),
                queue_b=int(self._queue_count_b[None]),
                queue_peak=int(self._queue_peak_count[None]),
            )
        if list_error:
            raise RuntimeError("FMM dual-tree traversal did not finish within its pass limit")
        if self.profile_passes:
            ti.sync()
            self.last_phase_seconds["interaction_lists"] = time.perf_counter() - phase_start
        m2l_count = int(self._m2l_count[None])
        phase_start = time.perf_counter()
        for pair_start in range(0, m2l_count, self.m2l_batch_size):
            batch_count = min(self.m2l_batch_size, m2l_count - pair_start)
            self._m2l_derivative_pass(pair_start, batch_count)
            self._m2l_accumulate_pass(pair_start, batch_count)
        if self.profile_passes:
            ti.sync()
            self.last_phase_seconds["m2l"] = time.perf_counter() - phase_start
        phase_start = time.perf_counter()
        for level in range(level_count):
            self._l2l_level(count, level)
        self._l2p_pass(count)
        if self.profile_passes:
            ti.sync()
            self.last_phase_seconds["downward"] = time.perf_counter() - phase_start
        phase_start = time.perf_counter()
        self._near_field_pass(node_count, count, int(self._near_count[None]))
        if self.profile_passes:
            ti.sync()
            self.last_phase_seconds["near_field"] = time.perf_counter() - phase_start
        phase_start = time.perf_counter()
        self._reset_rate_diagnostics()
        self._rate_pass(count, stretching_mode)
        self._finalize_rate_diagnostics()
        ti.sync()
        if self.profile_passes:
            self.last_phase_seconds["strength_rate"] = time.perf_counter() - phase_start

    def prepare_source_multipoles(self, count: int) -> int:
        """Prepare p=3 moments on the active cells of an already-built source tree.

        The source tree must represent the current position, strength and core
        fields. Below maximal interaction leaves no moment is valid or needed:
        consumers must direct-sum those cells rather than descend further.
        This also clears local/output scratch for a subsequent particle solve.
        Field identity is not a validity token; callers must rerun this method
        after changing or rebuilding sources.

        Returns the hierarchy level count. The generation counter advances
        after the ordered device work is submitted, for explicit immutable
        source scopes; it does not detect external in-place mutations.
        """
        count = int(count)
        if count < 1 or count > self.max_n_particles:
            raise ValueError("source count must fit the positive FMM active prefix")
        node_count = 2 * count - 1
        level_count = int(self.tree._max_depth[None]) + 1
        if level_count > _MAX_TREE_LEVELS:
            raise RuntimeError("FMM source tree exceeds the active-cell level capacity")
        self._count_active_cells(node_count)
        self._initialize_active_cell_offsets()
        self._write_active_cell_schedule(node_count)
        self._zero_pass(node_count, count)
        self._p2m_pass(count)
        for level in range(level_count - 1, -1, -1):
            self._m2m_level(count, level)
        self.source_multipole_generation += 1
        return level_count

    def evaluate_targets(
        self,
        target_position,
        source_position,
        source_vortex_strength,
        source_core_radius,
        target_velocity,
        target_gradient,
        target_count: int,
        source_count: int,
        background_velocity,
        reuse_source_tree: bool = False,
    ) -> None:
        """Evaluate arbitrary targets without transferring particle fields to the host.

        The source LBVH is rebuilt from the supplied active source prefix.
        Targets then use the same strict device tree traversal as the qualified
        treecode, without host staging or a second hierarchy build. Particle
        stages remain fixed-order p=3 FMM evaluations.

        Target batches reuse the LBVH traversal stack, so target count is not
        limited by particle capacity. Only active source and target prefixes
        are read or written.
        """
        target_count = int(target_count)
        source_count = int(source_count)
        if target_count < 0:
            raise ValueError("target_count must be non-negative")
        if source_count < 0 or source_count > self.max_n_particles:
            raise ValueError(
                f"source count {source_count} exceeds FMM capacity {self.max_n_particles}"
            )
        if target_velocity is None and target_gradient is None:
            raise ValueError("at least one target output is required")
        if target_count == 0:
            return
        write_velocity = target_velocity is not None
        write_gradient = target_gradient is not None
        velocity_output = self.velocity if target_velocity is None else target_velocity
        gradient_output = self.gradient if target_gradient is None else target_gradient
        if source_count == 0:
            self._empty_target_pass(
                velocity_output,
                gradient_output,
                background_velocity,
                target_count,
                write_velocity,
                write_gradient,
            )
            return

        if not reuse_source_tree:
            self.tree.build(
                source_position,
                source_vortex_strength,
                source_core_radius,
                source_count,
            )
        self.tree.compute_external_target_fields(
            target_position,
            velocity_output,
            gradient_output,
            background_velocity,
            target_count,
            self.target_batch_capacity,
            write_velocity=write_velocity,
            write_gradient=write_gradient,
        )

    def p2p_particle_count(self) -> int:
        """Return the exact accumulated near-field interaction count.

        The two device ``u32`` limbs form one unsigned 64-bit diagnostic
        counter. This host-side reconstruction happens after stage completion
        and does not participate in FMM arithmetic.
        """
        return (int(self._p2p_particle_count_high[None]) << 32) | int(
            self._p2p_particle_count_low[None]
        )


@ti.data_oriented
class FMMInduction:
    """Device-resident fixed-order FMM induction backend.

    The backend performs P2M, M2M, M2L, L2L, L2P, and kernel-specific near-field
    P2P passes on device-resident particle fields. It computes a velocity
    gradient from the same expansion and conditions that gradient into the
    requested particle-strength rate. The stretching formulation is
    independent of the FMM approximation.

    Supported production combinations are CPU, Vulkan, Metal, CUDA, and AUTO
    resolution; f32 precision; and Gaussian, high-order Gaussian, super-Gaussian, or
    Winckelmans radial kernels. Arbitrary target queries reuse the source LBVH,
    p=3 multipoles, and exact regularized near interactions on the device.
    """

    # AUTO is accepted as a request to resolve a backend at solver construction;
    # the resolved backend is checked again before any FMM workspace is built.
    # Only the backends exercised by the production qualification are advertised.
    supported_devices = frozenset({"AUTO", "CPU", "VULKAN", "METAL", "CUDA"})
    supported_kernels = frozenset(
        {"GAUSSIAN", "HIGH_ORDER_GAUSSIAN", "SUPER_GAUSSIAN", "WINCKELMANS"}
    )
    supports_gradient = True
    supports_variable_core_radius = True
    supports_f64 = False
    supports_target_fields = True
    device_resident = True
    # Independent of the caller's query allocation. The local-polynomial and
    # dual-traversal scratch must remain bounded even for million-point queries.
    max_image_block_targets = 8192
    # Optional execution scratch, never an accuracy or image-tail parameter.
    max_image_geometry_bytes = 64 * 1024 * 1024

    def __init__(self, *, stretching_scheme: str = "TRANSPOSED") -> None:
        """Create an unbound FMM evaluator.

        Parameters
        ----------
        stretching_scheme : {"DIRECT", "TRANSPOSED", "MIXED"}, default="TRANSPOSED"
            Formulation used to conditions the computed velocity gradient into
            ``dGamma/dt``. It does not change hierarchy construction or
            induced velocity.

        Notes
        -----
        Construction allocates no FMM workspace. :meth:`bind` performs the
        precision check and allocates one-source scratch, which grows with the
        active source count.

        Raises
        ------
        ValueError
            If ``stretching_scheme`` is unsupported.
        """
        self.stretching_scheme = normalize_stretching_scheme(stretching_scheme)
        self._stretching_mode = _STRETCHING_MODES[self.stretching_scheme]
        self.method = "FMM"
        self.physics = None
        self.kernel: RadialVortexKernel = make_vortex_kernel("GAUSSIAN")
        self.max_n_particles = 1
        self.workspace: FMMDeviceWorkspace | None = None
        self._target_workspace = None
        self._image_geometry_cache = None
        self._source_moments_ready = False
        self._reclaim_stage_cache = None
        self.diagnostics = FMMDiagnostics(stretching_scheme=self.stretching_scheme)

    def build(self) -> Self:
        """Return a fresh unbound FMM evaluator preserving the scheme."""
        return type(self)(stretching_scheme=self.stretching_scheme)

    @property
    def supports_image_blocks(self) -> bool:
        """Enable the image-local path only on its qualified runtime/kernel set.

        Other device backends retain the established strict target traversal.
        In particular, the block work counters currently require i64 atomics,
        which are not supported by every advertised self-FMM backend.
        """
        return (
            ti.lang.impl.get_runtime().prog is not None
            and ti.lang.impl.current_cfg().arch in (ti.cpu, ti.cuda)
            and self.kernel.name in {"GAUSSIAN", "WINCKELMANS"}
        )

    def bind(self, physics: object, *, kernel: RadialVortexKernel | None = None) -> Self:
        """Bind and allocate the evaluator for one f32 physics workspace.

        Parameters
        ----------
        physics : PhysicsEngine
            Runtime VPM workspace. Its capacity, particle kernel, device
            functions, and f32 accumulator dtype determine allocation.
        kernel : RadialVortexKernel or None, default=None
            Optional already-constructed kernel. If omitted, the kernel named
            by ``physics.particle_kernel`` is created.

        Returns
        -------
        FMMInduction
            This bound evaluator.

        Raises
        ------
        ValueError
            If the workspace uses f64 accumulators.

        Notes
        -----
        Binding allocates one-source scratch. Later evaluations grow scratch
        to fit active sources without changing the declared particle ceiling.
        """
        if physics.accumulator_dtype != ti.f32:
            raise ValueError("FMMInduction currently supports precision='f32' only")
        self._release_image_geometry()
        self._release_target_workspace()
        if self.workspace is not None:
            previous = self.workspace
            self.workspace = None
            ti.sync()
            previous.destroy()
        self._last_tree_key = None
        self._fixed_source_key = None
        self._source_moments_ready = False
        self.physics = physics
        self.kernel = make_vortex_kernel(physics.particle_kernel) if kernel is None else kernel
        self.max_n_particles = int(physics.max_n_particles)
        self._velocity_tail_cutoff, self._gradient_tail_cutoff = (
            self.kernel.dimensionless_tail_cutoffs(
                _VELOCITY_TAIL_RELATIVE_TOLERANCE,
                _GRADIENT_TAIL_RELATIVE_TOLERANCE,
            )
        )
        self._radial_factors = physics._kernel_functions["radial_factors_"]
        self._max_evaluation_points = physics.max_evaluation_points
        self.workspace = FMMDeviceWorkspace(
            1,
            self._radial_factors,
            self.kernel.name,
            self._velocity_tail_cutoff,
            self._gradient_tail_cutoff,
            self._max_evaluation_points,
        )
        return self

    @contextmanager
    def stage_workspace(self, count: int, *, reclaim):
        """Reserve native scratch before a composed operator allocates its cache.

        Reclaim that disposable cache only on source or interaction-list growth.
        The callback must preserve stage inputs and any completed host results.
        """
        previous = self._reclaim_stage_cache
        self._reclaim_stage_cache = reclaim
        try:
            self._ensure_workspace(count)
            yield
        finally:
            self._reclaim_stage_cache = previous

    def _ensure_workspace(self, source_count: int) -> None:
        """Grow disposable scratch to fit the active source prefix."""
        if source_count > self.max_n_particles:
            raise ValueError(
                f"source count {source_count} exceeds FMM capacity {self.max_n_particles}"
            )
        current = self.workspace
        if current is None:
            raise RuntimeError("FMMInduction must be bound before evaluation")
        if source_count <= current.max_n_particles:
            return
        if getattr(self, "_fixed_source_key", None) is not None:
            raise RuntimeError("cannot grow FMM workspace inside fixed-source target scope")
        capacity = min(self.max_n_particles, max(source_count, 2 * current.max_n_particles))
        pairs = _ListCapacities(
            *(max(_PAIR_CAPACITY_FACTOR * capacity, item) for item in current.list_capacities)
        )
        self._replace_workspace(capacity, pairs)

    def _replace_workspace(self, capacity: int, max_pairs: int | _ListCapacities) -> None:
        """Replace only disposable scratch; stage inputs and published outputs stay intact."""
        if getattr(self, "_fixed_source_key", None) is not None:
            raise RuntimeError("cannot grow FMM workspace inside fixed-source target scope")
        current = self.workspace
        if current is None:
            raise RuntimeError("FMMInduction must be bound before evaluation")
        pairs = _normalize_list_capacities(max_pairs)
        profile_passes = current.profile_passes
        if self._reclaim_stage_cache is not None:
            self._reclaim_stage_cache()
        ti.sync()
        # Target kernels close over the source workspace's fields. Release
        # them before destroying those fields, never retarget a compiled solver.
        self._release_target_workspace()
        self.workspace = None
        self._last_tree_key = None
        self._source_moments_ready = False
        current.destroy()
        del current
        self.workspace = FMMDeviceWorkspace(
            capacity,
            self._radial_factors,
            self.kernel.name,
            self._velocity_tail_cutoff,
            self._gradient_tail_cutoff,
            self._max_evaluation_points,
            _list_capacities=pairs,
        )
        self.workspace.profile_passes = profile_passes

    def _release_target_workspace(self) -> None:
        if self._image_geometry_cache is not None:
            self._image_geometry_cache.invalidate()
        target = self._target_workspace
        self._target_workspace = None
        if target is not None:
            target.destroy()

    def _release_image_geometry(self) -> None:
        cache = self._image_geometry_cache
        if cache is not None:
            cache.close()
            self._image_geometry_cache = None

    def close(self) -> None:
        """Release this evaluator's scratch, never caller physics/particle fields.

        Call before resetting its Taichi runtime. A wrapper which borrows this
        evaluator must not close it implicitly; the evaluator's solver chooses
        teardown. Rebinding a closed evaluator is supported.
        """
        if getattr(self, "_fixed_source_key", None) is not None:
            raise RuntimeError("cannot close FMM scratch inside an immutable-source scope")
        self._release_image_geometry()
        self._release_target_workspace()
        current = self.workspace
        if current is not None:
            ti.sync()
            self.workspace = None
            current.destroy()
        self._last_tree_key = None
        self._source_moments_ready = False

    @contextmanager
    def _fixed_image_targets(self, position, count, tile_capacity, *, read_fields, write_fields):
        """Internal immutable-target scope, separate from source freshness.

        Only standard autonomous backend bindings qualify. Aliased/unknown
        fields and custom methods keep the original fresh-preparation path.
        No geometry validity escapes this context, although its bounded solver
        survives to avoid global Taichi JIT invalidation between image stages.
        """
        from ..reuse_backends import FMMReuseConditions, _standard_methods
        from .target_geometry import TargetGeometryCache, disjoint_fields, scratch_fields
        from .targets import FMMTargetEvaluator

        target = self._target_workspace
        standard_target = target is None or (
            _standard_methods(target, FMMTargetEvaluator)
            and _standard_methods(target.tree, TaichiTreecode)
        )
        writable = tuple(write_fields) + scratch_fields(
            self,
            self.workspace,
            getattr(self.workspace, "tree", None),
            target,
            getattr(target, "tree", None),
        )
        cache = self._image_geometry_cache
        retained = scratch_fields(None if cache is None else cache.storage)
        if (
            not standard_target
            or FMMReuseConditions(self)() is None
            or not disjoint_fields(read_fields, writable + retained)
            or not disjoint_fields(retained, writable)
        ):
            self.diagnostics.image_geometry_fallback_scopes += 1
            yield
            return
        if cache is not None and cache.max_bytes != self.max_image_geometry_bytes:
            self._release_image_geometry()
            cache = None
        if cache is None:
            cache = self._image_geometry_cache = TargetGeometryCache(
                self.max_image_geometry_bytes, self.diagnostics
            )
        with cache.scope(position, int(count), int(tile_capacity)):
            yield

    def _replace_target_workspace(self, targets: int, images: int, pairs: int | None = None):
        from .targets import FMMTargetEvaluator

        if pairs is None:
            pairs = min(32 * targets, _MAX_IMAGE_PAIR_CAPACITY)
        if not 1 <= pairs <= _MAX_IMAGE_PAIR_CAPACITY:
            raise ValueError("Image interaction scratch exceeds its bounded capacity")
        self._release_target_workspace()
        self._target_workspace = FMMTargetEvaluator(
            self.workspace, targets, max_images=images, max_pairs=pairs
        )
        return self._target_workspace

    def _grow_interaction_lists(self, error: _InteractionListCapacityError) -> None:
        """Grow from observed demand, without changing FMM accuracy or particle capacity."""
        current = self.workspace
        assert current is not None
        proposed = _ListCapacities(
            *(
                max(
                    _PAIR_CAPACITY_FACTOR * current.max_n_particles,
                    math.ceil(1.5 * capacity),
                    math.ceil(1.25 * required),
                )
                if required > capacity
                else capacity
                for capacity, required in zip(current.list_capacities, error.counts, strict=True)
            )
        )
        if max(proposed) > _MAX_PAIR_CAPACITY:
            raise RuntimeError("FMM interaction lists exceed the safe i32 capacity") from error
        Logging.runtime_warning(
            f"Growing FMM interaction-list storage from {current.list_capacities} to {proposed}; "
            "retrying the unchanged particle stage",
            stacklevel=2,
        )
        self._replace_workspace(current.max_n_particles, proposed)
        self.diagnostics.interaction_list_resizes += 1

    def estimated_workspace_bytes(
        self,
        max_n_particles: int,
        max_evaluation_points: int | None = None,
        *,
        max_pairs: int | None = None,
        _list_capacities: _ListCapacities | None = None,
    ) -> int:
        """Estimate fixed FMM and hierarchy field arrays for a capacity.

        Parameters
        ----------
        max_n_particles : int
            Positive source-particle capacity used to size hierarchy,
            interaction-list, and particle-output arrays.
        max_evaluation_points : int or None, default=None
            Maximum simultaneous arbitrary-target batch size. ``None`` uses
            the bound workspace's batch capacity when available, otherwise
            ``max_n_particles``.
        max_pairs : int or None, default=None
            Equal per-list scratch capacity; None uses the initial sizing heuristic.

        Returns
        -------
        int
            Approximate bytes for the fixed FMM and LBVH field arrays,
            including target traversal stacks. Taichi allocator metadata,
            compiled kernels, and driver allocations are excluded.

        Raises
        ------
        ValueError
            If ``max_n_particles`` is less than one.
        """
        capacity = int(max_n_particles)
        if capacity < 1:
            raise ValueError("max_n_particles must be positive")
        if max_evaluation_points is None:
            evaluation_capacity = (
                self.workspace.target_batch_capacity if self.workspace is not None else capacity
            )
        else:
            evaluation_capacity = int(max_evaluation_points)
        if evaluation_capacity < 1:
            raise ValueError("max_evaluation_points must be positive")
        evaluation_capacity = min(evaluation_capacity, _TRAVERSAL_BATCH_SIZE)
        node_count = 2 * capacity
        if max_pairs is not None and _list_capacities is not None:
            raise ValueError("specify either equal or independent FMM scratch capacities")
        initial = _PAIR_CAPACITY_FACTOR * capacity if max_pairs is None else max_pairs
        pairs = _normalize_list_capacities(
            initial if _list_capacities is None else _list_capacities
        )
        coefficient_bytes = node_count * 3 * 4 * (_MOMENT_COUNT + _LOCAL_COUNT)
        interaction_bytes = pairs.storage_bytes + 4  # Queue high-water counter.
        near_adjacency_bytes = node_count * 3 * 4
        active_schedule_bytes = (node_count + 2 * capacity + 3 * _MAX_TREE_LEVELS + 2) * 4
        output_bytes = capacity * (3 + 9 + 3) * 4
        # ``hierarchy_only`` retains only source-tree state plus the target
        # traversal stack.  The unused particle stack and target outputs are
        # one-element stubs, so they are intentionally not capacity-scaled.
        source_particle_bytes = capacity * (3 + 3 + 1) * 4
        node_metadata_bytes = node_count * (3 + 1 + 3 + 3 + 3 + 6 + 2 + 6) * 4
        sort_capacity = 1 << (capacity - 1).bit_length()
        sort_and_leaf_bytes = capacity * (1 + 7) * 4 + (sort_capacity - capacity) * 2 * 4
        target_stack_depth = (
            self.workspace.tree.max_stack_depth if self.workspace is not None else 48
        )
        target_stack_bytes = evaluation_capacity * target_stack_depth * 4
        derivative_cache_bytes = min(_M2L_BATCH_SIZE, pairs.m2l) * _DERIVATIVE_COUNT * 4
        return int(
            coefficient_bytes
            + interaction_bytes
            + near_adjacency_bytes
            + active_schedule_bytes
            + output_bytes
            + source_particle_bytes
            + node_metadata_bytes
            + sort_and_leaf_bytes
            + target_stack_bytes
            + derivative_cache_bytes
        )

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
        """Evaluate velocity, gradient, and strength rate for one RK stage.

        Parameters
        ----------
        position : object
            Stage positions, logical shape ``(count, 3)``, in m.
        vortex_strength : object
            Stage particle-strength vectors ``Gamma``, shape ``(count, 3)``, in m³/s.
        core_radius : object
            Stage core radii, shape ``(count,)``, in m.
        count : int
            Active particle prefix.
        velocity_out : object
            Output velocity field, shape ``(count, 3)``, in m/s.
        vortex_strength_rate_out : object
            Output strength-rate field, shape ``(count, 3)``, in m³/s².
        velocity_gradient_out : object or None, default=None
            Optional output gradient, shape ``(count, 3, 3)``, in 1/s.
        strength_rate_enabled : bool, default=True
            Copy the FMM stretching rate when true; otherwise explicitly zero
            the output rate.
        stage_time : float, default=0.0
            Stage time in seconds, accepted for the common conditions and unused
            by this autonomous backend.

        Raises
        ------
        RuntimeError
            If unbound, traversal is incomplete, or bounded scratch growth
            cannot accommodate an interaction list. No partial output is copied.
        ValueError
            If ``count`` exceeds the bound capacity.

        Notes
        -----
        Source stage fields are read-only. Output fields, workspace storage,
        and :attr:`diagnostics` are mutated. The stage gradient is computed
        even when only a strength rate is requested because it defines that
        discrete rate.
        """
        del stage_time
        if self.physics is None or self.workspace is None:
            raise RuntimeError("FMMInduction must be bound before evaluation")
        if getattr(self, "_fixed_source_key", None) is not None:
            raise RuntimeError("cannot advance an FMM stage inside an immutable-source scope")
        count = int(count)
        if count < 0 or count > self.max_n_particles:
            raise ValueError(f"stage count {count} exceeds FMM capacity {self.max_n_particles}")
        if count == 0:
            return
        self._ensure_workspace(count)
        self._source_moments_ready = False
        for attempt in range(_MAX_LIST_GROWTH_RETRIES + 1):
            try:
                self.workspace.evaluate(
                    position, vortex_strength, core_radius, count, self._stretching_mode
                )
                break
            except _InteractionListCapacityError as error:
                if attempt == _MAX_LIST_GROWTH_RETRIES:
                    raise RuntimeError(
                        "FMM interaction-list storage still insufficient after bounded growth; "
                        "no stage output was published"
                    ) from error
                self._grow_interaction_lists(error)
        self._last_tree_key = (position, vortex_strength, core_radius, count)
        self._source_moments_ready = True
        self.physics._copy_vec3(self.workspace.velocity, velocity_out, count)
        if velocity_gradient_out is not None:
            self.physics._copy_mat3(self.workspace.gradient, velocity_gradient_out, count)
        if strength_rate_enabled:
            self.physics._copy_vec3(self.workspace.rate, vortex_strength_rate_out, count)
        else:
            self.physics._zero_vec3_field(vortex_strength_rate_out, count)
        m2l_count = int(self.workspace._m2l_count[None])
        near_count = int(self.workspace._near_count[None])
        self.diagnostics.stage_evaluations += 1
        self.diagnostics.hierarchy_builds += 1
        self.diagnostics.p2m_operations += count
        active_internal_count = int(self.workspace._active_internal_count[None])
        active_leaf_count = int(self.workspace._active_leaf_count[None])
        self.diagnostics.m2m_operations += active_internal_count
        self.diagnostics.m2l_interactions += m2l_count
        self.diagnostics.p2p_interactions += self.workspace.p2p_particle_count()
        self.diagnostics.l2l_operations += 2 * active_internal_count
        self.diagnostics.nonzero_l2l_operations += int(self.workspace._nonzero_l2l_count[None])
        self.diagnostics.l2p_evaluations += count
        self.diagnostics.gradient_evaluations += 1
        self.diagnostics.hierarchical_strength_rates += int(strength_rate_enabled)
        self.diagnostics.host_particle_transfers = 0
        self.diagnostics.last_uncorrected_rate_defect = float(self.workspace._rate_defect[None])
        self.diagnostics.last_strength_rate_norm = float(self.workspace._rate_norm_sum[None])
        self.diagnostics.last_relative_rate_defect = (
            self.diagnostics.last_uncorrected_rate_defect
            / max(self.diagnostics.last_strength_rate_norm, np.finfo(np.float32).eps)
        )
        self.diagnostics.peak_node_count = max(self.diagnostics.peak_node_count, 2 * count - 1)
        self.diagnostics.last_active_cell_count = active_internal_count + active_leaf_count
        self.diagnostics.last_active_leaf_count = active_leaf_count
        self.diagnostics.peak_interaction_list_count = max(
            self.diagnostics.peak_interaction_list_count, m2l_count + near_count
        )
        self.diagnostics.device_memory_estimate_bytes = self._estimate_memory_bytes()
        if self.workspace.profile_passes:
            phase = self.workspace.last_phase_seconds
            self.diagnostics.last_tree_build_seconds = phase["tree_build"]
            self.diagnostics.last_upward_pass_seconds = phase["upward"]
            self.diagnostics.last_interaction_list_seconds = phase["interaction_lists"]
            self.diagnostics.last_m2l_seconds = phase["m2l"]
            self.diagnostics.last_downward_pass_seconds = phase["downward"]
            self.diagnostics.last_near_field_seconds = phase["near_field"]
            self.diagnostics.last_strength_rate_seconds = phase["strength_rate"]

    def evaluate_gaussian_source_vorticity(
        self, *, position, vortex_strength, core_radius, count, vorticity_out
    ) -> None:
        """Reconstruct f32 Gaussian vorticity at the supplied source centres.

        Sum only the supplied physical sources, including self terms, with
        pair-mean core radii. No images or background field are added. Rebuild
        the hierarchy because unchanged field identities can hold new values.
        """
        with self.fixed_source_targets(position, vortex_strength, core_radius, count):
            self.workspace.tree.compute_gaussian_particle_vorticity(vorticity_out, count)

    @contextmanager
    def fixed_source_targets(self, position, strength, radius, count, *, reuse_current_tree=False):
        """Reuse one LBVH for target batches while source fields are immutable.

        The caller owns a short evaluation scope and must not mutate any of the
        three source fields inside it. The scope ends before the next RK stage.
        """
        if self.workspace is None:
            raise RuntimeError("FMMInduction must be bound before fixed-source targets")
        if getattr(self, "_fixed_source_key", None) is not None:
            raise RuntimeError("nested fixed-source target scopes are unsupported")
        if int(count) < 0:
            raise ValueError("source count must be non-negative")
        key = (position, strength, radius, int(count))
        self._ensure_workspace(int(count))
        previous = getattr(self, "_last_tree_key", None)
        same_tree = (
            previous is not None
            and all(previous[i] is key[i] for i in range(3))
            and previous[3] == key[3]
        )
        if int(count) > 0 and (not reuse_current_tree or not same_tree):
            self._source_moments_ready = False
            self.workspace.tree.build(*key)
        self._last_tree_key = key
        self._fixed_source_key = key
        try:
            yield
        finally:
            self._fixed_source_key = None

    def evaluate_image_block(
        self,
        *,
        source_position,
        source_vortex_strength,
        source_core_radius,
        source_count: int,
        target_position,
        target_start: int,
        target_count: int,
        images,
        target_velocity,
        target_velocity_gradient,
    ) -> None:
        """Evaluate every reflected/translated source in one convergence block.

        The output is in physical coordinates, including reflection parity.
        No tail decision is made here. One target-local expansion is shared by
        a cell and by all images in the block; regularized near pairs remain
        exact. Source moments may only be reused inside an explicit immutable-
        source scope. Outside one, a fresh scope is opened. Target geometry is
        copied afresh unless a separate built-in immutable-target image scope
        is active: an immutable source alone never freezes targets.
        Disposable overflow retries publish neither partial nor stale output.
        Dense blocks exceeding the fixed scratch bound are declined for the
        wrapper's unchanged strict target traversal, not approximated or cut.
        """
        from .targets import TargetBlockNotWorthwhile, TargetInteractionCapacityError

        if self.physics is None or self.workspace is None:
            raise RuntimeError("FMMInduction must be bound before image evaluation")
        if not self.supports_image_blocks:
            raise NotImplementedError(
                "This kernel retains its existing exact image target operator"
            )
        count, start, sources = int(target_count), int(target_start), int(source_count)
        images = tuple(images)
        if count < 0 or start < 0 or not 0 <= sources <= self.max_n_particles:
            raise ValueError("image evaluation requires valid active prefixes")
        if count > self.max_image_block_targets:
            raise ValueError("image target tile exceeds the bounded workspace capacity")
        if not images or any(not math.isfinite(shift) for shift, _ in images):
            raise ValueError("image evaluation requires a nonempty finite image block")
        if target_velocity is None and target_velocity_gradient is None:
            raise ValueError("at least one target output is required")
        if count == 0:
            return
        fixed = getattr(self, "_fixed_source_key", None)
        key = (source_position, source_vortex_strength, source_core_radius, sources)
        if fixed is None:
            with self.fixed_source_targets(*key):
                self.evaluate_image_block(
                    source_position=source_position,
                    source_vortex_strength=source_vortex_strength,
                    source_core_radius=source_core_radius,
                    source_count=sources,
                    target_position=target_position,
                    target_start=start,
                    target_count=count,
                    images=images,
                    target_velocity=target_velocity,
                    target_velocity_gradient=target_velocity_gradient,
                )
            return
        if not all(fixed[i] is key[i] for i in range(3)) or fixed[3] != sources:
            raise RuntimeError("fixed-source image scope received different source fields")
        if sources == 0:
            self.workspace._empty_target_pass(
                target_velocity if target_velocity is not None else self.workspace.velocity,
                target_velocity_gradient
                if target_velocity_gradient is not None
                else self.workspace.gradient,
                self.physics._zero_velocity,
                count,
                target_velocity is not None,
                target_velocity_gradient is not None,
            )
            return
        if not self._source_moments_ready:
            self.workspace.prepare_source_multipoles(sources)
            self._source_moments_ready = True
        target = self._target_workspace
        if target is None or target.max_targets < count or target.max_images < len(images):
            target = self._replace_target_workspace(
                max(count, 1 if target is None else target.max_targets),
                max(256, len(images), 1 if target is None else target.max_images),
            )
        self.diagnostics.device_memory_estimate_bytes = self._estimate_memory_bytes()
        for attempt in range(_MAX_LIST_GROWTH_RETRIES + 1):
            cache = self._image_geometry_cache
            if cache is None:
                target.prepare_targets(target_position, count, target_start=start)
                self.diagnostics.image_target_geometry_builds += 1
            else:
                cache.prepare(target, target_position, count, target_start=start)
            try:
                target.evaluate_image_block(
                    images, target_velocity, target_velocity_gradient, self.physics._zero_velocity
                )
                return
            except TargetInteractionCapacityError as error:
                if error.required_pairs > _MAX_IMAGE_PAIR_CAPACITY:
                    raise TargetBlockNotWorthwhile(
                        required_pairs=error.required_pairs,
                        capacity=_MAX_IMAGE_PAIR_CAPACITY,
                        diagnostics=dict(getattr(target, "last_diagnostics", {})),
                    ) from error
                if attempt == _MAX_LIST_GROWTH_RETRIES:
                    raise RuntimeError(
                        "FMM image lists still insufficient after bounded growth; "
                        "no image-block output was published"
                    ) from error
                pairs = max(
                    math.ceil(1.5 * target.max_pairs), math.ceil(1.25 * error.required_pairs)
                )
                pairs = min(pairs, _MAX_IMAGE_PAIR_CAPACITY)
                if pairs > _MAX_PAIR_CAPACITY:
                    raise RuntimeError("FMM image lists exceed the safe i32 capacity") from error
                Logging.runtime_warning(
                    f"Growing FMM image-list storage from {target.max_pairs} to {pairs} "
                    "pairs per list; retrying the unchanged image block",
                    stacklevel=2,
                )
                target = self._replace_target_workspace(
                    target.max_targets, target.max_images, pairs
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
        """Evaluate arbitrary target fields with a device-resident hierarchy.

        Parameters
        ----------
        target_position : object
            Target points, shape ``(target_count, 3)``, in m.
        source_position, source_vortex_strength, source_core_radius : object
            Source fields with shapes ``(source_count, 3)``, ``(source_count,
            3)``, and ``(source_count,)`` in m, m³/s, and m.
        target_velocity : object or None
            Optional velocity output, shape ``(target_count, 3)``, in m/s.
        target_velocity_gradient : object or None
            Optional gradient output, shape ``(target_count, 3, 3)``, in 1/s.
        target_count, source_count : int
            Active target and source counts.
        include_freestream : bool
            Add ``background_velocity`` to velocity output when true.
        background_velocity : object
            Freestream vector in m/s.

        Raises
        ------
        RuntimeError
            If the evaluator is not bound.

        Notes
        -----
        Particle stages use the p=3 FMM. Arbitrary targets reuse its source
        LBVH with the qualified strict tree traversal at theta=0.1, so no
        particle field is downloaded and no second hierarchy is built for
        Gaussian and Winckelmans kernels. High-order Gaussian and
        super-Gaussian target queries retain the exact regularized direct
        operator because the shared LBVH does not implement those leaf kernels.
        """
        if self.physics is None or self.workspace is None:
            raise RuntimeError("FMMInduction must be bound before target evaluation")
        if self.kernel.name not in {"GAUSSIAN", "WINCKELMANS"}:
            self._last_tree_key = None
            self._source_moments_ready = False
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
            return
        fixed = getattr(self, "_fixed_source_key", None)
        if fixed is not None and not (
            fixed[0] is source_position
            and fixed[1] is source_vortex_strength
            and fixed[2] is source_core_radius
            and fixed[3] == int(source_count)
        ):
            raise RuntimeError("fixed-source target scope received different source fields")
        self._ensure_workspace(int(source_count))
        if fixed is None:
            self._source_moments_ready = False
        self.workspace.evaluate_targets(
            target_position,
            source_position,
            source_vortex_strength,
            source_core_radius,
            target_velocity,
            target_velocity_gradient,
            int(target_count),
            int(source_count),
            background_velocity if include_freestream else self.physics._zero_velocity,
            reuse_source_tree=fixed is not None,
        )
        if fixed is None:
            self._last_tree_key = (
                source_position,
                source_vortex_strength,
                source_core_radius,
                int(source_count),
            )

    def _estimate_memory_bytes(self) -> int:
        """Return the current capacity-based workspace estimate in bytes."""
        if self.workspace is None:
            return self.estimated_workspace_bytes(self.max_n_particles)
        total = self.estimated_workspace_bytes(
            self.workspace.max_n_particles,
            self.workspace.target_batch_capacity,
            _list_capacities=self.workspace.list_capacities,
        )
        target = self._target_workspace
        if target is not None:
            estimate = getattr(target, "estimated_memory_bytes", None)
            if estimate is not None:
                total += int(estimate())
        if self._image_geometry_cache is not None:
            total += self._image_geometry_cache.allocated_bytes
        return total


__all__ = ["FMMDeviceWorkspace", "FMMInduction"]
