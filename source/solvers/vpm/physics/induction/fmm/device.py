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
from ..base import StrengthRateMode
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
_PAIR_CAPACITY_FACTOR = 64
_EPSILON_SQUARED = 1.0e-24
_ONE_OVER_FOUR_PI = 0.07957747154594767
_GEOMETRIC_SEPARATION_FACTOR = 3.0
_VELOCITY_TAIL_RELATIVE_TOLERANCE = 1.0e-5
_GRADIENT_TAIL_RELATIVE_TOLERANCE = 1.0e-5


def _double_factorial(value: int) -> int:
    result = 1
    for factor in range(value, 0, -2):
        result *= factor
    return result


def _translation_tables():
    """Build fixed analytic ``1/r`` derivative metadata through order six."""
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
    """Preallocated Taichi fields and fixed-order FMM passes."""

    def __init__(
        self,
        max_n_particles: int,
        q_kernel,
        zeta_kernel,
        velocity_tail_cutoff: float,
        gradient_tail_cutoff: float,
    ) -> None:
        self.max_n_particles = int(max_n_particles)
        self.max_nodes = 2 * self.max_n_particles
        self.max_pairs = max(64, _PAIR_CAPACITY_FACTOR * self.max_n_particles)
        self.m2l_batch_size = min(_M2L_BATCH_SIZE, self.max_pairs)
        self.q_kernel = q_kernel
        self.zeta_kernel = zeta_kernel
        self.velocity_tail_cutoff = float(velocity_tail_cutoff)
        self.gradient_tail_cutoff = float(gradient_tail_cutoff)
        self.tree = TaichiTreecode(
            max_n_particles=self.max_n_particles,
            max_nodes=self.max_nodes,
            theta=0.3,
            max_leaf_size=1,
            kernel_type="GAUSSIAN",
            multipole_order=1,
            sort_particle_targets=False,
            traversal_block_dim=0,
            device_sort_only=True,
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
        self._p2p_particle_count = ti.field(dtype=ti.i64, shape=())
        self._list_error = ti.field(dtype=ti.i32, shape=())
        self._nonzero_l2l_count = ti.field(dtype=ti.i32, shape=())
        self._queue_target_a = ti.field(dtype=ti.i32, shape=self.max_pairs)
        self._queue_source_a = ti.field(dtype=ti.i32, shape=self.max_pairs)
        self._queue_target_b = ti.field(dtype=ti.i32, shape=self.max_pairs)
        self._queue_source_b = ti.field(dtype=ti.i32, shape=self.max_pairs)
        self._queue_count_a = ti.field(dtype=ti.i32, shape=())
        self._queue_count_b = ti.field(dtype=ti.i32, shape=())
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

    @ti.func
    def _inverse_r_derivative(
        self, displacement: ti.template(), derivative_index: ti.i32
    ) -> ti.f32:
        """Evaluate a table-driven analytic derivative of ``1/(4*pi*r)``."""
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
        for node in range(self.max_nodes):
            if node >= count and node < 2 * count - 1 and self.tree.node_depth[node] == level:
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
        self._p2p_particle_count[None] = 0
        self._list_error[None] = 0
        root = self.tree._root[None]
        self._queue_target_a[0] = root
        self._queue_source_a[0] = root
        self._queue_count_a[None] = 1
        self._queue_count_b[None] = 0

    @ti.func
    def _append_m2l_pair(self, target: ti.i32, source: ti.i32):
        slot = ti.atomic_add(self._m2l_count[None], 1)
        if slot < self.max_pairs:
            self.m2l_target[slot] = target
            self.m2l_source[slot] = source
        else:
            self._list_error[None] = 1

    @ti.func
    def _append_near_pair(self, target: ti.i32, source: ti.i32):
        slot = ti.atomic_add(self._near_count[None], 1)
        if slot < self.max_pairs:
            self.near_target[slot] = target
            self.near_source[slot] = source
        else:
            self._list_error[None] = 1
        particle_pairs = (
            self.tree.node_particle_count[target] * self.tree.node_particle_count[source]
        )
        if target == source:
            particle_pairs -= self.tree.node_particle_count[target]
        ti.atomic_add(self._p2p_particle_count[None], ti.cast(particle_pairs, ti.i64))

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
            self._list_error[None] = 1

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
            if target != source or self.tree.node_particle_count[target] > 1:
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
        for node in range(self.max_nodes):
            if node >= count and node < 2 * count - 1 and self.tree.node_depth[node] == level:
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
    def _near_field_pass(self, pair_count: ti.i32):
        for pair in range(pair_count):
            target_node = self.near_target[pair]
            source_node = self.near_source[pair]
            target_start = self.tree.node_particle_start[target_node]
            source_start = self.tree.node_particle_start[source_node]
            target_count = self.tree.node_particle_count[target_node]
            source_count = self.tree.node_particle_count[source_node]
            for target_slot in range(target_start, target_start + target_count):
                target = self.tree.sorted_indices[target_slot]
                for source_slot in range(source_start, source_start + source_count):
                    source = self.tree.sorted_indices[source_slot]
                    if target != source:
                        displacement = self.tree.position[target] - self.tree.position[source]
                        radius_sq = displacement.dot(displacement)
                        if radius_sq > ti.cast(_EPSILON_SQUARED, ti.f32):
                            radius = ti.sqrt(radius_sq)
                            sigma = 0.5 * (
                                self.tree.core_radius[target] + self.tree.core_radius[source]
                            )
                            rho = radius / sigma
                            q_value = self.q_kernel(rho)
                            zeta_value = self.zeta_kernel(rho)
                            inv_r2 = 1.0 / radius_sq
                            inv_r3 = inv_r2 / radius
                            source_strength = self.tree.vortex_strength[source]
                            velocity = source_strength.cross(displacement) * q_value * inv_r3
                            for component in ti.static(range(3)):
                                ti.atomic_add(self.velocity[target][component], velocity[component])
                            term1 = q_value * inv_r3
                            term2 = (
                                3.0 * q_value * inv_r3 * inv_r2
                                - zeta_value / (sigma * sigma * sigma) * inv_r2
                            )
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
                                    ti.atomic_add(
                                        self.gradient[target][row, column],
                                        term1 * skew_value
                                        + term2 * cross_value[row] * displacement[column],
                                    )

    @ti.kernel
    def _reset_rate_diagnostics(self):
        self._rate_sum[None] = ti.Vector([0.0, 0.0, 0.0])
        self._rate_norm_sum[None] = 0.0
        self._rate_defect[None] = 0.0

    @ti.kernel
    def _rate_pass(self, count: ti.i32):
        for particle in range(count):
            rate = self.gradient[particle].transpose() @ self.tree.vortex_strength[particle]
            self.rate[particle] = rate
            for component in ti.static(range(3)):
                ti.atomic_add(self._rate_sum[None][component], rate[component])
            ti.atomic_add(self._rate_norm_sum[None], ti.sqrt(rate.dot(rate)))

    @ti.kernel
    def _finalize_rate_diagnostics(self):
        self._rate_defect[None] = ti.sqrt(self._rate_sum[None].dot(self._rate_sum[None]))

    def evaluate(self, position, vortex_strength, core_radius, count: int) -> None:
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
        self._near_field_pass(int(self._near_count[None]))
        if self.profile_passes:
            ti.sync()
            self.last_phase_seconds["near_field"] = time.perf_counter() - phase_start
        phase_start = time.perf_counter()
        self._reset_rate_diagnostics()
        self._rate_pass(count)
        self._finalize_rate_diagnostics()
        ti.sync()
        if self.profile_passes:
            self.last_phase_seconds["strength_rate"] = time.perf_counter() - phase_start


@ti.data_oriented
class FMMInduction:
    """Device-resident hierarchical-gradient FMM induction backend."""

    # AUTO is accepted as a request to resolve a backend at solver construction;
    # the resolved backend is checked again before any FMM workspace is built.
    # Only the backends exercised by the production qualification are advertised.
    supported_devices = frozenset({"AUTO", "CPU", "VULKAN"})
    supported_kernels = frozenset(
        {"GAUSSIAN", "HIGH_ORDER_GAUSSIAN", "SUPER_GAUSSIAN", "WINCKELMANS"}
    )
    supports_gradient = True
    supports_variable_core_radius = True
    supports_f64 = False
    supports_target_fields = True
    device_resident = True
    strength_rate_mode: StrengthRateMode = "HIERARCHICAL_GRADIENT"

    def __init__(self) -> None:
        self.method = "FMM"
        self.physics = None
        self.kernel: RadialVortexKernel = make_vortex_kernel("GAUSSIAN")
        self.max_n_particles = 1
        self.workspace: FMMDeviceWorkspace | None = None
        self.diagnostics = FMMDiagnostics(strength_rate_mode=self.strength_rate_mode)

    def build(self) -> Self:
        """Return a fresh unbound FMM evaluator for an immutable case setup."""
        return type(self)()

    def bind(self, physics: object, *, kernel: RadialVortexKernel | None = None) -> Self:
        """Bind the evaluator to one single-precision VPM physics workspace."""
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
            physics._kernel_functions["q_"],
            physics._kernel_functions["zeta_"],
            velocity_tail_cutoff,
            gradient_tail_cutoff,
        )
        return self

    def estimated_workspace_bytes(self, max_n_particles: int) -> int:
        """Return the fixed FMM workspace allocation for a particle capacity."""
        capacity = int(max_n_particles)
        if capacity < 1:
            raise ValueError("max_n_particles must be positive")
        node_count = 2 * capacity
        max_pairs = max(64, _PAIR_CAPACITY_FACTOR * capacity)
        coefficient_bytes = node_count * 3 * 4 * (_MOMENT_COUNT + _LOCAL_COUNT)
        interaction_bytes = max_pairs * 8 * 4
        output_bytes = capacity * (3 + 9 + 3) * 4
        return int(coefficient_bytes + interaction_bytes + output_bytes)

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
        """Evaluate velocity, gradient, and strength rate for one RK stage."""
        del stage_time
        if self.physics is None or self.workspace is None:
            raise RuntimeError("FMMInduction must be bound before evaluation")
        count = int(count)
        if count < 0 or count > self.max_n_particles:
            raise ValueError(f"stage count {count} exceeds FMM capacity {self.max_n_particles}")
        if count == 0:
            return
        self.workspace.evaluate(position, vortex_strength, core_radius, count)
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
        self.diagnostics.p2p_interactions += int(self.workspace._p2p_particle_count[None])
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
        """Evaluate arbitrary target fields through the FMM backend boundary.

        The production FMM workspace currently has a particle-target pass but
        no dual-tree arbitrary-target pass.  Keep that limitation explicit at
        the backend boundary and use the shared regularized target kernels as
        a bounded correctness fallback; PhysicsBase no longer silently
        bypasses the selected induction method.
        """
        if self.physics is None:
            raise RuntimeError("FMMInduction must be bound before target evaluation")
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

    def _estimate_memory_bytes(self) -> int:
        return self.estimated_workspace_bytes(self.max_n_particles)


__all__ = ["FMMDeviceWorkspace", "FMMInduction"]
