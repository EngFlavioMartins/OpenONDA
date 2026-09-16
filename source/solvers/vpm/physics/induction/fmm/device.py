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

import math
import time
from typing import Self

import numpy as np
import taichi as ti

from ....kernels.base import RadialVortexKernel, make_vortex_kernel
from ..base import _STRETCHING_MODES, normalize_stretching_scheme
from ..stretching import stretching_rate
from ..treecode.lbvh import TaichiTreecode
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
_FMM_LEAF_CAPACITY = 32
_MAX_TREE_LEVELS = 96
# The dual-tree queues and final near/far lists each have independent storage.
# Qualification on volumetric, sheet-like, and filamentary clouds reaches at
# most 16.1 pairs per particle in either final list. A factor of 32 retains a
# twofold margin. Capacity exhaustion remains a hard error.
_PAIR_CAPACITY_FACTOR = 32
_EPSILON_SQUARED = 1.0e-24
_ONE_OVER_FOUR_PI = 0.07957747154594767
_GEOMETRIC_SEPARATION_FACTOR = 3.0
_VELOCITY_TAIL_RELATIVE_TOLERANCE = 1.0e-5
_GRADIENT_TAIL_RELATIVE_TOLERANCE = 1.0e-5


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

    This important internal runtime object stores source moments, local
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

        Notes
        -----
        Allocation has a device-side memory cost and is not thread-safe. All
        numerical fields use ``ti.f32``; the caller must reuse this workspace
        only with a compatible fixed-capacity evaluator.
        """
        self.max_n_particles = int(max_n_particles)
        self.max_nodes = 2 * self.max_n_particles
        self.max_pairs = _PAIR_CAPACITY_FACTOR * self.max_n_particles
        self.m2l_batch_size = min(_M2L_BATCH_SIZE, self.max_pairs)
        self.radial_factors = radial_factors
        self.velocity_tail_cutoff = float(velocity_tail_cutoff)
        self.gradient_tail_cutoff = float(gradient_tail_cutoff)
        self.target_batch_capacity = min(
            self.max_n_particles,
            max(1, int(max_evaluation_points)),
        )
        self.tree = TaichiTreecode(
            max_n_particles=self.max_n_particles,
            max_nodes=self.max_nodes,
            theta=0.1,
            max_leaf_size=1,
            kernel_type=kernel_name if kernel_name in {"GAUSSIAN", "WINCKELMANS"} else "GAUSSIAN",
            multipole_order=1,
            sort_particle_targets=False,
            traversal_block_dim=0,
            device_sort_only=True,
            hierarchy_only=True,
            max_evaluation_points=self.target_batch_capacity,
        )
        self.multipole = ti.Vector.field(3, dtype=ti.f32, shape=self.max_nodes * _MOMENT_COUNT)
        self.local = ti.Vector.field(3, dtype=ti.f32, shape=self.max_nodes * _LOCAL_COUNT)
        self._coefficient_a = ti.field(dtype=ti.i32, shape=_MOMENT_COUNT)
        self._coefficient_b = ti.field(dtype=ti.i32, shape=_MOMENT_COUNT)
        self._coefficient_c = ti.field(dtype=ti.i32, shape=_MOMENT_COUNT)
        self._derivative_lookup = ti.field(
            dtype=ti.i32,
            shape=(
                _DERIVATIVE_ORDER + 1,
                _DERIVATIVE_ORDER + 1,
                _DERIVATIVE_ORDER + 1,
            ),
        )
        self._derivative_term_count = ti.field(dtype=ti.i32, shape=_DERIVATIVE_COUNT)
        self._derivative_coefficient = ti.field(
            dtype=ti.f32,
            shape=(_DERIVATIVE_COUNT, _MAX_DERIVATIVE_TERMS),
        )
        self._derivative_exponent = ti.Vector.field(
            3,
            dtype=ti.i32,
            shape=(_DERIVATIVE_COUNT, _MAX_DERIVATIVE_TERMS),
        )
        self._derivative_radial_step = ti.field(
            dtype=ti.i32,
            shape=(_DERIVATIVE_COUNT, _MAX_DERIVATIVE_TERMS),
        )
        self._m2l_derivative = ti.field(
            dtype=ti.f32,
            shape=(self.m2l_batch_size, _DERIVATIVE_COUNT),
        )
        self.m2l_target = ti.field(dtype=ti.i32, shape=self.max_pairs)
        self.m2l_source = ti.field(dtype=ti.i32, shape=self.max_pairs)
        self.near_target = ti.field(dtype=ti.i32, shape=self.max_pairs)
        self.near_source = ti.field(dtype=ti.i32, shape=self.max_pairs)
        self._m2l_count = ti.field(dtype=ti.i32, shape=())
        self._near_count = ti.field(dtype=ti.i32, shape=())
        # Metal lacks the i64 atomic required by a monolithic exact counter.
        # Two u32 limbs retain every near P2P interaction count without
        # relying on lossy f32 accumulation. A single list entry contains at
        # most 32*32 interactions, so carry detection is unambiguous.
        self._p2p_particle_count_low = ti.field(dtype=ti.u32, shape=())
        self._p2p_particle_count_high = ti.field(dtype=ti.u32, shape=())
        self._list_error = ti.field(dtype=ti.i32, shape=())
        self._nonzero_l2l_count = ti.field(dtype=ti.i32, shape=())
        self._queue_target_a = ti.field(dtype=ti.i32, shape=self.max_pairs)
        self._queue_source_a = ti.field(dtype=ti.i32, shape=self.max_pairs)
        self._queue_target_b = ti.field(dtype=ti.i32, shape=self.max_pairs)
        self._queue_source_b = ti.field(dtype=ti.i32, shape=self.max_pairs)
        self._queue_count_a = ti.field(dtype=ti.i32, shape=())
        self._queue_count_b = ti.field(dtype=ti.i32, shape=())
        self._near_target_count = ti.field(dtype=ti.i32, shape=self.max_nodes)
        self._near_scan_a = ti.field(dtype=ti.i32, shape=self.max_nodes)
        self._near_scan_b = ti.field(dtype=ti.i32, shape=self.max_nodes)
        self.velocity = ti.Vector.field(3, dtype=ti.f32, shape=self.max_n_particles)
        self.gradient = ti.Matrix.field(3, 3, dtype=ti.f32, shape=self.max_n_particles)
        self.rate = ti.Vector.field(3, dtype=ti.f32, shape=self.max_n_particles)
        self._rate_sum = ti.Vector.field(3, dtype=ti.f32, shape=())
        self._rate_norm_sum = ti.field(dtype=ti.f32, shape=())
        self._rate_defect = ti.field(dtype=ti.f32, shape=())
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
        self.last_phase_seconds = {
            "tree_build": 0.0,
            "upward": 0.0,
            "interaction_lists": 0.0,
            "m2l": 0.0,
            "downward": 0.0,
            "near_field": 0.0,
            "strength_rate": 0.0,
        }

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

    @ti.kernel
    def _zero_pass(self, node_count: ti.i32, particle_count: ti.i32):
        self._nonzero_l2l_count[None] = 0
        for node in range(node_count):
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
        for slot in range(count):
            particle = self.tree.sorted_indices[slot]
            self.multipole[slot * _MOMENT_COUNT] = self.tree.vortex_strength[particle]

    @ti.kernel
    def _m2m_level(self, count: ti.i32, level: ti.i32):
        for internal_slot in range(count - 1):
            node = count + internal_slot
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
        if slot < self.max_pairs:
            self.m2l_target[slot] = target
            self.m2l_source[slot] = source
        else:
            ti.atomic_max(self._list_error[None], 1)

    @ti.func
    def _append_near_pair(self, target: ti.i32, source: ti.i32):
        slot = ti.atomic_add(self._near_count[None], 1)
        if slot < self.max_pairs:
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
        if slot < self.max_pairs:
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
        self._queue_count_b[None] = 0
        source_count = ti.min(self._queue_count_a[None], self.max_pairs)
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
        self._queue_count_a[None] = 0
        source_count = ti.min(self._queue_count_b[None], self.max_pairs)
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
        if self._queue_count_a[None] != 0 or self._queue_count_b[None] != 0:
            self._list_error[None] = 1

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
        for internal_slot in range(count - 1):
            node = count + internal_slot
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
            node = slot
            base = node * _LOCAL_COUNT
            offset = self.tree.position[particle] - self.tree.node_centre[node]
            a_x = (
                self.local[base + 3]
                + 2.0 * self.local[base + 9] * offset[0]
                + self.local[base + 8] * offset[1]
                + self.local[base + 7] * offset[2]
            )
            a_y = (
                self.local[base + 2]
                + self.local[base + 8] * offset[0]
                + 2.0 * self.local[base + 6] * offset[1]
                + self.local[base + 5] * offset[2]
            )
            a_z = (
                self.local[base + 1]
                + self.local[base + 7] * offset[0]
                + self.local[base + 5] * offset[1]
                + 2.0 * self.local[base + 4] * offset[2]
            )
            a_xx = 2.0 * self.local[base + 9]
            a_xy = self.local[base + 8]
            a_xz = self.local[base + 7]
            a_yy = 2.0 * self.local[base + 6]
            a_yz = self.local[base + 5]
            a_zz = 2.0 * self.local[base + 4]
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
            target_node = target_slot
            parent = self.tree.node_parent[target_node]
            while parent >= 0 and self._is_fmm_leaf(parent) == 1:
                target_node = parent
                parent = self.tree.node_parent[target_node]
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
        self._zero_pass(node_count, count)
        self._p2m_pass(count)
        level_count = min(_MAX_TREE_LEVELS, int(self.tree._max_depth[None]) + 1)
        for level in range(level_count - 1, -1, -1):
            self._m2m_level(count, level)
        if self.profile_passes:
            ti.sync()
            self.last_phase_seconds["upward"] = time.perf_counter() - phase_start
        phase_start = time.perf_counter()
        self._build_interaction_lists(2 * level_count)
        if int(self._list_error[None]) != 0:
            raise RuntimeError(
                "FMM interaction-list capacity was exceeded "
                f"(capacity={self.max_pairs}, m2l={int(self._m2l_count[None])}, "
                f"near={int(self._near_count[None])}, "
                f"queue_a={int(self._queue_count_a[None])}, "
                f"queue_b={int(self._queue_count_b[None])})"
            )
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
    gradient from the same expansion and contracts that gradient into the
    requested particle-strength rate. The stretching formulation is
    independent of the FMM approximation.

    Supported production combinations are CPU, Vulkan, Metal, and AUTO
    resolution; f32 precision; and Gaussian, high-order Gaussian, super-Gaussian, or
    Winckelmans radial kernels. Arbitrary target queries reuse the source LBVH,
    p=3 multipoles, and exact regularized near interactions on the device.
    """

    # AUTO is accepted as a request to resolve a backend at solver construction;
    # the resolved backend is checked again before any FMM workspace is built.
    # Only the backends exercised by the production qualification are advertised.
    supported_devices = frozenset({"AUTO", "CPU", "VULKAN", "METAL"})
    supported_kernels = frozenset(
        {"GAUSSIAN", "HIGH_ORDER_GAUSSIAN", "SUPER_GAUSSIAN", "WINCKELMANS"}
    )
    supports_gradient = True
    supports_variable_core_radius = True
    supports_f64 = False
    supports_target_fields = True
    device_resident = True

    def __init__(self, *, stretching_scheme: str = "TRANSPOSED") -> None:
        """Create an unbound FMM evaluator.

        Parameters
        ----------
        stretching_scheme : {"DIRECT", "TRANSPOSED", "MIXED"}, default="TRANSPOSED"
            Formulation used to contract the computed velocity gradient into
            ``dGamma/dt``. It does not change hierarchy construction or
            induced velocity.

        Notes
        -----
        Construction allocates no FMM workspace. :meth:`bind` performs the
        precision check and allocates fields sized to the physics capacity.

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
        self.diagnostics = FMMDiagnostics(stretching_scheme=self.stretching_scheme)

    def build(self) -> Self:
        """Return a fresh unbound FMM evaluator preserving the scheme."""
        return type(self)(stretching_scheme=self.stretching_scheme)

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
        Binding allocates a fixed-order workspace proportional to capacity;
        it may consume substantially more memory than the active count alone
        suggests.
        """
        if physics.accumulator_dtype != ti.f32:
            raise ValueError("FMMInduction currently supports precision='f32' only")
        self.physics = physics
        self.kernel = make_vortex_kernel(physics.particle_kernel) if kernel is None else kernel
        self.max_n_particles = int(physics.max_n_particles)
        velocity_tail_cutoff, gradient_tail_cutoff = self.kernel.dimensionless_tail_cutoffs(
            _VELOCITY_TAIL_RELATIVE_TOLERANCE,
            _GRADIENT_TAIL_RELATIVE_TOLERANCE,
        )
        self.workspace = FMMDeviceWorkspace(
            self.max_n_particles,
            physics._kernel_functions["radial_factors_"],
            self.kernel.name,
            velocity_tail_cutoff,
            gradient_tail_cutoff,
            physics.max_evaluation_points,
        )
        return self

    def estimated_workspace_bytes(
        self, max_n_particles: int, max_evaluation_points: int | None = None
    ) -> int:
        """Estimate fixed FMM and hierarchy field payloads for a capacity.

        Parameters
        ----------
        max_n_particles : int
            Positive source-particle capacity used to size hierarchy,
            interaction-list, and particle-output arrays.
        max_evaluation_points : int or None, default=None
            Maximum simultaneous arbitrary-target batch size. ``None`` uses
            the bound workspace's batch capacity when available, otherwise
            ``max_n_particles``.

        Returns
        -------
        int
            Approximate bytes for the fixed FMM and LBVH field payloads,
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
        evaluation_capacity = min(capacity, evaluation_capacity)
        node_count = 2 * capacity
        max_pairs = _PAIR_CAPACITY_FACTOR * capacity
        coefficient_bytes = node_count * 3 * 4 * (_MOMENT_COUNT + _LOCAL_COUNT)
        interaction_bytes = max_pairs * 8 * 4
        near_adjacency_bytes = node_count * 3 * 4
        output_bytes = capacity * (3 + 9 + 3) * 4
        # ``hierarchy_only`` retains only source-tree state plus the target
        # traversal stack.  The legacy particle stack and target outputs are
        # one-element stubs, so they are intentionally not capacity-scaled.
        source_particle_bytes = capacity * (3 + 3 + 1) * 4
        node_metadata_bytes = node_count * (3 + 1 + 3 + 3 + 3 + 6 + 2 + 6) * 4
        sort_and_leaf_bytes = capacity * (1 + 7) * 4
        target_stack_depth = (
            self.workspace.tree.max_stack_depth if self.workspace is not None else 48
        )
        target_stack_bytes = evaluation_capacity * target_stack_depth * 4
        derivative_cache_bytes = min(_M2L_BATCH_SIZE, max_pairs) * _DERIVATIVE_COUNT * 4
        return int(
            coefficient_bytes
            + interaction_bytes
            + near_adjacency_bytes
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
            Stage time in seconds, accepted for the common contract and unused
            by this autonomous backend.

        Raises
        ------
        RuntimeError
            If :meth:`bind` has not been called or an interaction list
            overflows its fixed capacity.
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
        count = int(count)
        if count < 0 or count > self.max_n_particles:
            raise ValueError(f"stage count {count} exceeds FMM capacity {self.max_n_particles}")
        if count == 0:
            return
        self.workspace.evaluate(
            position, vortex_strength, core_radius, count, self._stretching_mode
        )
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
        self.diagnostics.m2m_operations += max(count - 1, 0)
        self.diagnostics.m2l_interactions += m2l_count
        self.diagnostics.p2p_interactions += self.workspace.p2p_particle_count()
        self.diagnostics.l2l_operations += 2 * max(count - 1, 0)
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
        )

    def _estimate_memory_bytes(self) -> int:
        """Return the current capacity-based workspace estimate in bytes."""
        if self.workspace is None:
            return self.estimated_workspace_bytes(self.max_n_particles)
        return self.estimated_workspace_bytes(
            self.max_n_particles,
            self.workspace.target_batch_capacity,
        )


__all__ = ["FMMDeviceWorkspace", "FMMInduction"]
