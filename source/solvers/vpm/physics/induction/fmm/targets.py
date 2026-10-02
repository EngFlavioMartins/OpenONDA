"""Bounded source-to-local evaluation for arbitrary device-resident targets.

The physical-source hierarchy is owned by :class:`FMMDeviceWorkspace`. This
evaluator preserves the existing strict target operator's source monopoles
and shares a high-order local expansion over each admissible target cell.
Regularised source-only-core near pairs remain exact. The particle self-FMM's
cubic source multipoles are a separate operator and are not changed here.
No source or target particle field is staged through NumPy during evaluation.

This internal path does not own source freshness: callers must rebuild the
source tree after every mutation or hold an explicit immutable-source scope.
"""

from contextlib import contextmanager
import math
from time import perf_counter

import numpy as np
import taichi as ti

from ..treecode.lbvh import TaichiTreecode, _OwnedFields
from .device import (
    _FMM_LEAF_CAPACITY,
    _M2L_BATCH_SIZE,
    _MAX_PAIR_CAPACITY,
    _double_factorial,
)

# Unlike a particle-only FMM, the local polynomial is evaluated away from its
# centre.  Keep a stricter geometric test for its Jacobian truncation error.
_TARGET_RADIUS_RATIO = 0.04
_LOCAL_ORDER = 7
_DERIVATIVE_ORDER = _LOCAL_ORDER
_MULTI_INDICES = tuple(
    (a, b, degree - a - b)
    for degree in range(_LOCAL_ORDER + 1)
    for a in range(degree + 1)
    for b in range(degree - a + 1)
)
_LOCAL_COUNT = len(_MULTI_INDICES)
_DERIVATIVE_INDICES = tuple(
    (a, b, degree - a - b)
    for degree in range(_DERIVATIVE_ORDER + 1)
    for a in range(degree + 1)
    for b in range(degree - a + 1)
)
_DERIVATIVE_COUNT = len(_DERIVATIVE_INDICES)
_DERIVATIVE_TERMS = max(
    math.prod(value // 2 + 1 for value in alpha) for alpha in _DERIVATIVE_INDICES
)
_MAX_DUAL_TREE_LEVELS = 2 * 96 + 1
_INITIAL_PAIRS_PER_TARGET = 32
_TARGET_START_CAPACITY = 1024
_NEAR_LANES = 64


def _target_derivative_tables():
    lookup = np.full((_DERIVATIVE_ORDER + 1,) * 3, -1, dtype=np.int32)
    counts = np.zeros(_DERIVATIVE_COUNT, dtype=np.int32)
    coefficients = np.zeros((_DERIVATIVE_COUNT, _DERIVATIVE_TERMS), dtype=np.float32)
    exponents = np.zeros((_DERIVATIVE_COUNT, _DERIVATIVE_TERMS, 3), dtype=np.int32)
    radial_steps = np.zeros((_DERIVATIVE_COUNT, _DERIVATIVE_TERMS), dtype=np.int32)
    for index, alpha in enumerate(_DERIVATIVE_INDICES):
        lookup[alpha] = index
        order = sum(alpha)
        slot = 0
        for px in range(alpha[0] // 2 + 1):
            for py in range(alpha[1] // 2 + 1):
                for pz in range(alpha[2] // 2 + 1):
                    p = (px, py, pz)
                    contraction = sum(p)
                    remainder = tuple(alpha[a] - 2 * p[a] for a in range(3))
                    numerator = math.prod(math.factorial(value) for value in alpha)
                    denominator = math.prod(
                        math.factorial(remainder[a]) * math.factorial(p[a]) for a in range(3)
                    )
                    coefficients[index, slot] = (
                        (-1.0 if (order - contraction) % 2 else 1.0)
                        * _double_factorial(2 * (order - contraction) - 1)
                        * numerator
                        / (2**contraction * denominator * 4 * math.pi)
                    )
                    exponents[index, slot] = remainder
                    radial_steps[index, slot] = order - contraction
                    slot += 1
        counts[index] = slot
    return lookup, counts, coefficients, exponents, radial_steps


class TargetInteractionCapacityError(RuntimeError):
    """Disposable target scratch is too small; no caller output was published."""

    def __init__(self, capacity, required):
        self.required_pairs = int(required)
        super().__init__(f"target FMM needs {required} interaction entries, capacity is {capacity}")


class TargetBlockNotWorthwhile(RuntimeError):  # noqa: N818 -- an optional fast-path decline.
    """A bounded local-expansion trial declined without publishing any output.

    The caller must use its unchanged legacy operator for this complete tile
    and block. This is a cost decision, never an image truncation or health gate.
    """

    def __init__(self, required_pairs, capacity, diagnostics=None):
        self.required_pairs = int(required_pairs)
        self.capacity = int(capacity)
        self.diagnostics = {} if diagnostics is None else dict(diagnostics)
        super().__init__(
            f"image block exceeds bounded local work: {required_pairs} pairs, budget {capacity}"
        )


@ti.data_oriented
class FMMTargetEvaluator:
    """Evaluate velocity/Jacobians from one already-prepared source workspace.

    ``source_workspace`` must outlive this object.  Replacing source scratch
    therefore requires destroying this target evaluator first.  A target
    workspace is reusable only by one execution stream at a time.
    """

    def __init__(self, source_workspace, max_targets: int, *, max_pairs=None, max_images=513):
        self.tree = None
        self._fields = None
        self._pair_fields = None
        try:
            self._initialize(source_workspace, max_targets, max_pairs=max_pairs, max_images=max_images)
        except BaseException as error:
            # Construction may fail after the hierarchy or coefficient fields
            # were allocated, but before the caller receives this object.
            try:
                self.destroy()
            except BaseException as cleanup_error:
                error.add_note(f"Target FMM cleanup also failed: {cleanup_error!r}")
            raise

    def _initialize(self, source_workspace, max_targets, *, max_pairs, max_images):
        self.source = source_workspace
        self.legacy_radial_kernel = source_workspace.kernel_name in {"GAUSSIAN", "WINCKELMANS"}
        self.source_separation = 2.0 / math.sqrt(float(source_workspace.tree.theta_sq))
        # Arbitrary targets previously used the stricter LBVH blob-tail gate,
        # not the particle FMM's 1e-5 regularization threshold.
        self.target_core_cutoff = max(
            float(source_workspace.tree.regularization_tail_cutoff[None]),
            source_workspace.velocity_tail_cutoff,
            source_workspace.gradient_tail_cutoff,
        )
        self.max_targets = int(max_targets)
        if self.max_targets < 1:
            raise ValueError("target capacity must be positive")
        self.max_images = int(max_images)
        if self.max_images < 1:
            raise ValueError("image capacity must be positive")
        if max_pairs is not None and not 1 <= int(max_pairs) <= _MAX_PAIR_CAPACITY:
            raise ValueError("target interaction capacity is outside the safe i32 range")
        self.tree = TaichiTreecode(
            max_n_particles=self.max_targets,
            max_nodes=2 * self.max_targets,
            max_leaf_size=1,
            multipole_order=1,
            hierarchy_only=True,
            max_evaluation_points=1,
        )
        fields = _OwnedFields()
        self._fields = fields
        self.zero_strength = fields.vector(3, dtype=ti.f32, shape=self.max_targets)
        self.zero_radius = fields.scalar(dtype=ti.f32, shape=self.max_targets)
        self.query_position = fields.vector(3, dtype=ti.f32, shape=self.max_targets)
        self.image_shift = fields.scalar(dtype=ti.f32, shape=self.max_images)
        self.image_odd = fields.scalar(dtype=ti.i32, shape=self.max_images)
        self.image_count = fields.scalar(dtype=ti.i32, shape=())
        self.block_mode = fields.scalar(dtype=ti.i32, shape=())
        self.leaf_nodes = fields.scalar(dtype=ti.i32, shape=self.max_targets)
        self.node_leaf = fields.scalar(dtype=ti.i32, shape=2 * self.max_targets)
        self.node_bound_radius = fields.scalar(dtype=ti.f32, shape=2 * self.max_targets)
        self.leaf_count = fields.scalar(dtype=ti.i32, shape=())
        self.frontier_count = fields.scalar(dtype=ti.i32, shape=())
        self.local = fields.vector(3, dtype=ti.f32, shape=2 * self.max_targets * _LOCAL_COUNT)
        self.local_present = fields.scalar(dtype=ti.i32, shape=2 * self.max_targets)
        self.near_count = fields.scalar(dtype=ti.i32, shape=2 * self.max_targets)
        self.scan_a = fields.scalar(dtype=ti.i32, shape=2 * self.max_targets)
        self.scan_b = fields.scalar(dtype=ti.i32, shape=2 * self.max_targets)
        self.cursor = fields.scalar(dtype=ti.i32, shape=2 * self.max_targets)
        self.local_alpha = fields.vector(3, dtype=ti.i32, shape=_LOCAL_COUNT)
        self.local_inverse_factorial = fields.scalar(dtype=ti.f32, shape=_LOCAL_COUNT)
        self.derivative_lookup = fields.scalar(dtype=ti.i32, shape=(_DERIVATIVE_ORDER + 1,) * 3)
        self.derivative_term_count = fields.scalar(dtype=ti.i32, shape=_DERIVATIVE_COUNT)
        self.derivative_coeff = fields.scalar(
            dtype=ti.f32, shape=(_DERIVATIVE_COUNT, _DERIVATIVE_TERMS)
        )
        self.derivative_exponent = fields.vector(
            3, dtype=ti.i32, shape=(_DERIVATIVE_COUNT, _DERIVATIVE_TERMS)
        )
        self.derivative_radial_steps = fields.scalar(
            dtype=ti.i32, shape=(_DERIVATIVE_COUNT, _DERIVATIVE_TERMS)
        )
        self.m2l_count = fields.scalar(dtype=ti.i32, shape=())
        self.near_pair_count = fields.scalar(dtype=ti.i32, shape=())
        self.direct_work = fields.scalar(dtype=ti.i64, shape=())
        self.monopole_work = fields.scalar(dtype=ti.i64, shape=())
        self.monopole_pair_count = fields.scalar(dtype=ti.i32, shape=())
        self.legacy_subtree_work = fields.scalar(dtype=ti.i64, shape=())
        self.error = fields.scalar(dtype=ti.i32, shape=())
        self.velocity = fields.vector(3, dtype=ti.f32, shape=self.max_targets)
        self.gradient = fields.matrix(3, 3, dtype=ti.f32, shape=self.max_targets)
        self.near_partial_velocity = fields.vector(3, dtype=ti.f32, shape=(self.max_targets, _NEAR_LANES))
        self.near_partial_gradient = fields.matrix(3, 3, dtype=ti.f32, shape=(self.max_targets, _NEAR_LANES))
        self.target_path_capacity = int(self.tree.max_tree_depth_guard) + 1
        # Level-major storage coalesces neighbouring Morton targets' reads.
        # Root-first traversal keeps shared ancestors at the same SIMD loop
        # iteration even when target leaves have unequal depths. Source lists,
        # source opening decisions and per-node lane partitions are unchanged.
        self.target_path = fields.scalar(
            dtype=ti.i32, shape=(self.target_path_capacity, self.max_targets)
        )
        self.target_path_length = fields.scalar(dtype=ti.i32, shape=self.max_targets)
        self.target_path_error = fields.scalar(dtype=ti.i32, shape=())
        fields.finalize()
        self.local_alpha.from_numpy(np.asarray(_MULTI_INDICES, dtype=np.int32))
        self.local_inverse_factorial.from_numpy(
            np.asarray(
                [1.0 / math.prod(math.factorial(a) for a in alpha) for alpha in _MULTI_INDICES],
                dtype=np.float32,
            )
        )
        for field, array in zip(
            (
                self.derivative_lookup,
                self.derivative_term_count,
                self.derivative_coeff,
                self.derivative_exponent,
                self.derivative_radial_steps,
            ),
            _target_derivative_tables(),
            strict=True,
        ):
            field.from_numpy(array)
        self._allocate_pairs(
            _INITIAL_PAIRS_PER_TARGET * self.max_targets if max_pairs is None else int(max_pairs)
        )
        self.last_diagnostics = {}
        self._prepared_count = 0
        self.target_geometry_builds = 0

    def _allocate_pairs(self, capacity):
        capacity = int(capacity)
        if not 1 <= capacity <= _MAX_PAIR_CAPACITY:
            raise RuntimeError("target FMM interaction capacity exceeds safe i32 range")
        ti.sync()
        if self._pair_fields is not None:
            previous = self._pair_fields
            self._pair_fields = None
            previous.destroy()
        fields = _OwnedFields()
        self._pair_fields = fields
        self.m2l_target = fields.scalar(dtype=ti.i32, shape=capacity)
        self.m2l_source = fields.scalar(dtype=ti.i32, shape=capacity)
        self.m2l_image = fields.scalar(dtype=ti.i32, shape=capacity)
        self.near_target = fields.scalar(dtype=ti.i32, shape=capacity)
        self.near_source = fields.scalar(dtype=ti.i32, shape=capacity)
        self.near_image = fields.scalar(dtype=ti.i32, shape=capacity)
        self.near_legacy = fields.scalar(dtype=ti.i32, shape=capacity)
        self.ordered_near_source = fields.scalar(dtype=ti.i32, shape=capacity)
        self.ordered_near_image = fields.scalar(dtype=ti.i32, shape=capacity)
        self.ordered_near_legacy = fields.scalar(dtype=ti.i32, shape=capacity)
        self.frontier_source_a = fields.scalar(dtype=ti.i32, shape=capacity)
        self.frontier_source_b = fields.scalar(dtype=ti.i32, shape=capacity)
        self.frontier_target_a = fields.scalar(dtype=ti.i32, shape=capacity)
        self.frontier_target_b = fields.scalar(dtype=ti.i32, shape=capacity)
        self.frontier_image_a = fields.scalar(dtype=ti.i32, shape=capacity)
        self.frontier_image_b = fields.scalar(dtype=ti.i32, shape=capacity)
        self.batch_capacity = min(capacity, _M2L_BATCH_SIZE)
        self.derivatives = fields.scalar(
            dtype=ti.f32, shape=(self.batch_capacity, _DERIVATIVE_COUNT)
        )
        fields.finalize()
        self.max_pairs = capacity

    def destroy(self):
        owners = (self._pair_fields, self._fields, self.tree)
        self._pair_fields = self._fields = self.tree = None
        if not any(owner is not None for owner in owners):
            return
        failures = []
        try:
            ti.sync()
        except BaseException as error:
            failures.append(error)
        # Release independent owners even if synchronization or one release
        # fails, and never retain a handle to already-destroyed scratch.
        for owner in owners:
            if owner is not None:
                try:
                    owner.destroy()
                except BaseException as error:
                    failures.append(error)
        if failures:
            for failure in failures[1:]:
                failures[0].add_note(f"Additional target FMM cleanup failure: {failure!r}")
            raise failures[0]

    def estimated_memory_bytes(self):
        """Owned dense field payload, excluding allocator/compiler overhead."""
        from taichi.lang.field import Field

        total = 0
        seen = set()
        for owner in (self, self.tree):
            for field in vars(owner).values():
                if isinstance(field, Field) and id(field) not in seen:
                    seen.add(id(field))
                    components = getattr(field, "n", 1) * getattr(field, "m", 1)
                    itemsize = np.dtype(ti.lang.util.to_numpy_type(field.dtype)).itemsize
                    total += math.prod(field.shape) * components * itemsize
        return total

    @contextmanager
    def _profile_phase(self, name):
        enabled = bool(getattr(self.source, "profile_passes", False))
        if enabled:
            ti.sync()
            started = perf_counter()
        try:
            yield
        finally:
            if enabled:
                ti.sync()
                passes = self.last_diagnostics.setdefault("passes_seconds", {})
                passes[name] = passes.get(name, 0.0) + perf_counter() - started

    @ti.kernel
    def _copy_target_slice(self, position: ti.template(), start: ti.i32, count: ti.i32):
        for point in range(count):
            self.query_position[point] = position[start + point]

    @ti.kernel
    def _set_image_transform(self, shift: ti.f32, odd: ti.i32):
        self.image_shift[0] = shift
        self.image_odd[0] = odd
        self.image_count[None] = 1
        self.block_mode[None] = 0

    @ti.func
    def _transform(self, point: ti.template(), image: ti.i32):
        result = point
        if self.image_odd[image]:
            result[2] = self.image_shift[image] - point[2]
        else:
            result[2] = point[2] - self.image_shift[image]
        return result

    @ti.kernel
    def _prepare_leaves(self, count: ti.i32):
        self.leaf_count[None] = 0
        for node in range(2 * count - 1):
            self.node_leaf[node] = -1
            radius = 0.0
            if count > 1:
                # The rounded AABB midpoint need not bisect its endpoints at
                # large translated coordinates. Its half-diagonal alone can
                # therefore under-enclose real targets about the stored centre.
                centre = self.tree.node_centre[node]
                extent = ti.max(
                    ti.abs(self.tree._node_aabb_min[node] - centre),
                    ti.abs(self.tree._node_aabb_max[node] - centre),
                )
                radius = extent.norm()
            self.node_bound_radius[node] = radius
        for node in range(2 * count - 1):
            parent = self.tree.node_parent[node]
            maximal_leaf = 0
            if self.tree.node_particle_count[node] <= _TARGET_START_CAPACITY:
                if parent < 0:  # noqa: SIM114 -- avoid negative Taichi field access.
                    maximal_leaf = 1
                elif self.tree.node_particle_count[parent] > _TARGET_START_CAPACITY:
                    maximal_leaf = 1
            if maximal_leaf:
                leaf = ti.atomic_add(self.leaf_count[None], 1)
                self.leaf_nodes[leaf] = node
                self.node_leaf[node] = leaf

    @ti.kernel
    def _initialise(self, node_count: ti.i32):
        self.error[None] = 0
        self.m2l_count[None] = 0
        self.near_pair_count[None] = 0
        self.direct_work[None] = 0
        self.monopole_work[None] = 0
        self.monopole_pair_count[None] = 0
        self.legacy_subtree_work[None] = 0
        self.frontier_count[None] = 0
        for leaf in range(node_count):
            self.near_count[leaf] = 0
            self.local_present[leaf] = 0
            for coefficient in range(_LOCAL_COUNT):
                self.local[leaf * _LOCAL_COUNT + coefficient] = ti.Vector.zero(ti.f32, 3)

    @ti.func
    def _legacy_accept(self, source: ti.i32, displacement: ti.template()):
        distance_sq = displacement.dot(displacement)
        distance = ti.sqrt(distance_sq)
        diameter = 2.0 * self.source.tree.node_half_size[source]
        return (
            distance > ti.max(1e-8, self.source.tree.node_avg_radius[source])
            and diameter * diameter / distance_sq < self.source.tree.theta_sq
            and self.source.tree._node_core_is_admissible(
                source, distance, self.source.tree.node_max_radius[source]
            ) != 0
        )

    @ti.func
    def _admissibility(self, target: ti.i32, source: ti.i32, image: ti.i32):
        displacement = (
            self._transform(self.tree.node_centre[target], image)
            - self.source.tree.node_com[source]
        )
        distance = displacement.norm()
        target_radius = self.node_bound_radius[target]
        source_radius = self.source.tree.node_half_size[source]
        source_extent = source_radius + (
            self.source.tree.node_com[source] - self.source.tree.node_centre[source]
        ).norm()
        core = self.source.tree.node_max_radius[source]
        net_strength = self.source.tree.node_net_vortex_strength[source]
        # Packet bounds classify the *legacy* pointwise decision. A mixed
        # packet must not open a source for points that would accept it: even
        # more accurate individual image terms can spoil cancellation of the
        # original operator's errors in the complete image sum.
        padding = 8 * 1.1920928955078125e-7 * (
            distance + target_radius + source_extent + core
            + ti.abs(self.tree.node_centre[target]).sum()
            + ti.abs(self.source.tree.node_com[source]).sum()
            + ti.abs(self.image_shift[image])
        )
        minimum_distance = ti.max(0.0, distance - target_radius - padding)
        maximum_distance = distance + target_radius + padding
        outside_tail = minimum_distance > source_extent + self.target_core_cutoff * core
        common_cores = (
            core - self.source.tree.node_min_radius[source]
            <= 1e-5 * ti.max(self.source.tree.node_avg_radius[source], 1e-12)
        )
        source_ok = (
            minimum_distance > self.source_separation * source_radius
            and (common_cores or outside_tail)
            and minimum_distance > ti.max(1e-8, self.source.tree.node_avg_radius[source])
            # Preserve the strict legacy target traversal's cancellation gate.
            and net_strength.dot(net_strength) > 1e-24
        )
        # Local polynomials represent the singular field. Common-core nodes
        # inside the tail are admissible only for the regularised M2P path.
        target_ok = outside_tail and target_radius + padding <= _TARGET_RADIUS_RATIO * distance
        none_accept = (
            maximum_distance <= self.source_separation * source_radius
            or maximum_distance <= ti.max(1e-8, self.source.tree.node_avg_radius[source])
            or net_strength.dot(net_strength) <= 1e-24
            or (
                not common_cores
                and maximum_distance <= source_extent
                + self.source.tree.regularization_tail_cutoff[None] * core
            )
        )
        if target_radius == 0.0:
            source_ok = self._legacy_accept(source, displacement)
            none_accept = not source_ok
        return ti.Vector([
            ti.cast(source_ok, ti.i32), ti.cast(target_ok, ti.i32), ti.cast(none_accept, ti.i32)
        ])

    @ti.kernel
    def _initialise_frontier(
        self, leaf_start: ti.i32, leaf_count: ti.i32,
        image_start: ti.i32, image_count: ti.i32, root_override: ti.i32,
    ):
        for work in range(leaf_count * image_count):
            if work < self.max_pairs:
                leaf, image = work // image_count, image_start + work % image_count
                target = root_override
                if target < 0:
                    target = self.leaf_nodes[leaf_start + leaf]
                self.frontier_source_a[work] = self.source.tree._root[None]
                self.frontier_target_a[work] = target
                self.frontier_image_a[work] = image
            else:
                ti.atomic_max(self.error[None], 1)

    @ti.kernel
    def _advance_frontier(
        self, source_in: ti.template(), target_in: ti.template(), image_in: ti.template(),
        source_out: ti.template(), target_out: ti.template(), image_out: ti.template(), count: ti.i32,
    ):
        self.frontier_count[None] = 0
        for work in range(count):
            if self.error[None] == 0:
                source, target, image = source_in[work], target_in[work], image_in[work]
                admissible = self._admissibility(target, source, image)
                source_leaf = self.source.tree.node_is_leaf[source] != 0
                target_leaf = self.tree.node_particle_count[target] <= _FMM_LEAF_CAPACITY
                if admissible[0] and admissible[1]:
                    pair = ti.atomic_add(self.m2l_count[None], 1)
                    if pair < self.max_pairs:
                        self.m2l_target[pair] = target
                        self.m2l_source[pair] = source
                        self.m2l_image[pair] = image
                        self.local_present[target] = 1
                    else:
                        ti.atomic_max(self.error[None], 1)
                elif admissible[0] or not admissible[2] or (source_leaf and target_leaf):
                    pair = ti.atomic_add(self.near_pair_count[None], 1)
                    encoded_source = source
                    legacy = 0
                    if not admissible[0] and not admissible[2]:
                        legacy = 1
                        ti.atomic_add(
                            self.legacy_subtree_work[None],
                            ti.cast(self.tree.node_particle_count[target], ti.i64),
                        )
                    elif admissible[0]:
                        # Negative entries are accepted legacy monopoles,
                        # evaluated once per target, not exact source lists.
                        encoded_source = -source - 1
                        ti.atomic_add(self.monopole_pair_count[None], 1)
                        ti.atomic_add(
                            self.monopole_work[None],
                            ti.cast(self.tree.node_particle_count[target], ti.i64),
                        )
                    else:
                        particle_work = ti.cast(self.tree.node_particle_count[target], ti.i64) * ti.cast(
                            self.source.tree.node_particle_count[source], ti.i64
                        )
                        ti.atomic_add(self.direct_work[None], particle_work)
                    if pair < self.max_pairs:
                        self.near_target[pair] = target
                        self.near_source[pair] = encoded_source
                        self.near_image[pair] = image
                        self.near_legacy[pair] = legacy
                        ti.atomic_add(self.near_count[target], 1)
                    else:
                        ti.atomic_max(self.error[None], 1)
                else:
                    destination = ti.atomic_add(self.frontier_count[None], 2)
                    if destination + 1 < self.max_pairs:
                        split_target = source_leaf
                        if not source_leaf and self.tree.node_particle_count[target] > 1:
                            split_target = (
                                self.source_separation * self.source.tree.node_half_size[source]
                                < self.tree.node_half_size[target]
                            )
                        if split_target:
                            source_out[destination] = source
                            source_out[destination + 1] = source
                            target_out[destination] = self.tree.node_left[target]
                            target_out[destination + 1] = self.tree.node_right[target]
                        else:
                            source_out[destination] = self.source.tree.node_left[source]
                            source_out[destination + 1] = self.source.tree.node_right[source]
                            target_out[destination] = target
                            target_out[destination + 1] = target
                        image_out[destination] = image
                        image_out[destination + 1] = image
                    else:
                        ti.atomic_max(self.error[None], 1)

    def _walk_sources(self, leaf_start, leaf_count, image_start, image_count, root_override):
        count = leaf_count * image_count
        if count > self.max_pairs:
            self.error[None] = 1
            return
        self._initialise_frontier(leaf_start, leaf_count, image_start, image_count, root_override)
        current = (self.frontier_source_a, self.frontier_target_a, self.frontier_image_a)
        following = (self.frontier_source_b, self.frontier_target_b, self.frontier_image_b)
        for _level in range(_MAX_DUAL_TREE_LEVELS):
            if int(self.error[None]) or not count:
                return
            self._advance_frontier(*current, *following, count)
            count = int(self.frontier_count[None])
            current, following = following, current
        if count and not int(self.error[None]):
            self.error[None] = 4

    @ti.func
    def _angular_derivative(self, direction: ti.template(), derivative: ti.i32):
        value = 0.0
        for term in ti.static(range(_DERIVATIVE_TERMS)):
            if term < self.derivative_term_count[derivative]:
                exponent = self.derivative_exponent[derivative, term]
                monomial = 1.0
                for power in ti.static(range(_DERIVATIVE_ORDER)):
                    for axis in ti.static(range(3)):
                        if power < exponent[axis]:
                            monomial *= direction[axis]
                value += self.derivative_coeff[derivative, term] * monomial
        return value

    @ti.func
    def _inverse_r_derivative(self, displacement: ti.template(), derivative: ti.i32):
        inverse = ti.rsqrt(ti.max(displacement.dot(displacement), 1e-24))
        direction = displacement * inverse
        first_exponent = self.derivative_exponent[derivative, 0]
        order = first_exponent[0] + first_exponent[1] + first_exponent[2]
        result = self._angular_derivative(direction, derivative) * inverse
        for power in ti.static(range(_DERIVATIVE_ORDER)):
            if power < order:
                result *= inverse
        return result

    @ti.kernel
    def _compute_derivatives(self, start: ti.i32, count: ti.i32):
        for pair in range(count):
            leaf = self.m2l_target[start + pair]
            source = self.m2l_source[start + pair]
            displacement = (
                self._transform(self.tree.node_centre[leaf], self.m2l_image[start + pair])
                - self.source.tree.node_com[source]
            )
            direction = displacement * ti.rsqrt(ti.max(displacement.dot(displacement), 1e-24))
            limit = _DERIVATIVE_COUNT
            if self.tree.node_half_size[leaf] == 0.0:
                # A singleton is evaluated exactly at its centre; potential
                # degrees above two cannot affect velocity or its Jacobian.
                limit = 10
            self.derivatives[pair, 0] = 1.0 / (4.0 * math.pi)
            for derivative in range(1, limit):
                beta = self.derivative_exponent[derivative, 0]
                axis = 0
                if beta[0] == 0:
                    axis = 1
                    if beta[1] == 0:
                        axis = 2
                total = 0.0
                # Differentiate (1 + 2 d.h + h.h)^(-1/2). Lower total
                # degrees precede this coefficient, so this Cartesian
                # recurrence replaces hundreds of monomial contractions.
                for j in ti.static(range(3)):
                    if beta[j] > 0:
                        lower = beta
                        lower[j] -= 1
                        previous = self.derivative_lookup[lower[0], lower[1], lower[2]]
                        factor = ti.cast(2 * beta[j], ti.f32)
                        if j == axis:
                            factor -= 1.0
                        total -= factor * direction[j] * self.derivatives[pair, previous]
                    if beta[j] > 1:
                        lower = beta
                        lower[j] -= 2
                        previous = self.derivative_lookup[lower[0], lower[1], lower[2]]
                        factor = (beta[j] - 1) * beta[j]
                        if j == axis:
                            factor = (beta[j] - 1) ** 2
                        total -= factor * self.derivatives[pair, previous]
                self.derivatives[pair, derivative] = total

    @ti.kernel
    def _translate(self, start: ti.i32, count: ti.i32):
        for pair in range(count):
            leaf = self.m2l_target[start + pair]
            source = self.m2l_source[start + pair]
            displacement = (
                self._transform(self.tree.node_centre[leaf], self.m2l_image[start + pair])
                - self.source.tree.node_com[source]
            )
            inverse = ti.rsqrt(ti.max(displacement.dot(displacement), 1e-24))
            local_scale = self.tree.node_half_size[leaf]
            if local_scale == 0.0:
                local_scale = 1.0
            local_count = _LOCAL_COUNT
            if self.tree.node_half_size[leaf] == 0.0:
                local_count = 10
            for beta in range(local_count):
                ba, bb, bc = self.local_alpha[beta]
                total = self.source.tree.node_net_vortex_strength[source] * (
                    inverse * self.derivatives[pair, beta] * self.local_inverse_factorial[beta]
                )
                # Locals use dimensionless offset (x-centre)/cell_radius.
                # Never form high inverse powers which can overflow merely
                # because the same physical problem uses smaller length units.
                for power in ti.static(range(_LOCAL_ORDER)):
                    if power < ba + bb + bc:
                        total *= inverse * local_scale
                if self.block_mode[None] and self.image_odd[self.m2l_image[start + pair]]:
                    # A is an axial vector.  Composing the polynomial with a
                    # z reflection contributes (-1)^beta_z; det(R) R applies
                    # to its vector coefficient.  Curl then produces R u and
                    # R J R, exactly the slab wrapper's full-vector parity.
                    total[0] = -total[0]
                    total[1] = -total[1]
                    if bc % 2:
                        total = -total
                for component in ti.static(range(3)):
                    ti.atomic_add(
                        self.local[leaf * _LOCAL_COUNT + beta][component], total[component]
                    )

    @ti.kernel
    def _copy_counts(self, count: ti.i32):
        for leaf in range(count):
            self.scan_a[leaf] = self.near_count[leaf]

    @ti.kernel
    def _scan_counts(
        self, previous: ti.template(), following: ti.template(), count: ti.i32, stride: ti.i32
    ):
        for leaf in range(count):
            value = previous[leaf]
            if leaf >= stride:
                value += previous[leaf - stride]
            following[leaf] = value

    @ti.kernel
    def _initialise_cursor(self, inclusive: ti.template(), count: ti.i32):
        for leaf in range(count):
            self.cursor[leaf] = inclusive[leaf] - self.near_count[leaf]

    @ti.kernel
    def _order_near(self, count: ti.i32):
        for pair in range(count):
            leaf = self.near_target[pair]
            destination = ti.atomic_add(self.cursor[leaf], 1)
            self.ordered_near_source[destination] = self.near_source[pair]
            self.ordered_near_image[destination] = self.near_image[pair]
            self.ordered_near_legacy[destination] = self.near_legacy[pair]

    @ti.func
    def _differentiate_local(self, base: ti.i32, offset: ti.template(), derivative: ti.template()):
        result = ti.Vector.zero(ti.f32, 3)
        for index in ti.static(range(_LOCAL_COUNT)):
            if ti.static(
                _MULTI_INDICES[index][0] >= derivative[0]
                and _MULTI_INDICES[index][1] >= derivative[1]
                and _MULTI_INDICES[index][2] >= derivative[2]
            ):
                factor = ti.cast(1.0, ti.f32)
                for axis in ti.static(range(3)):
                    for _exponent in ti.static(
                        range(_MULTI_INDICES[index][axis] - derivative[axis])
                    ):
                        factor *= offset[axis]
                    factor *= ti.static(
                        math.factorial(_MULTI_INDICES[index][axis])
                        / math.factorial(_MULTI_INDICES[index][axis] - derivative[axis])
                    )
                result += factor * self.local[base + index]
        return result

    @ti.func
    def _local_derivatives(self, base: ti.i32, offset: ti.template()):
        powers = ti.Matrix.zero(ti.f32, 3, _LOCAL_ORDER + 1)
        for axis in ti.static(range(3)):
            powers[axis, 0] = 1.0
            for degree in ti.static(range(1, _LOCAL_ORDER + 1)):
                powers[axis, degree] = powers[axis, degree - 1] * offset[axis]
        ax, ay, az = ti.Vector.zero(ti.f32, 3), ti.Vector.zero(ti.f32, 3), ti.Vector.zero(ti.f32, 3)
        axx, axy, axz = ti.Vector.zero(ti.f32, 3), ti.Vector.zero(ti.f32, 3), ti.Vector.zero(ti.f32, 3)
        ayy, ayz, azz = ti.Vector.zero(ti.f32, 3), ti.Vector.zero(ti.f32, 3), ti.Vector.zero(ti.f32, 3)
        for index in range(_LOCAL_COUNT):
            a, b, c = self.local_alpha[index]
            value = self.local[base + index]
            x, y, z = powers[0, a], powers[1, b], powers[2, c]
            if a:
                ax += (a * powers[0, a - 1] * y * z) * value
                if b:
                    axy += (a * b * powers[0, a - 1] * powers[1, b - 1] * z) * value
                if c:
                    axz += (a * c * powers[0, a - 1] * y * powers[2, c - 1]) * value
            if b:
                ay += (b * x * powers[1, b - 1] * z) * value
                if c:
                    ayz += (b * c * x * powers[1, b - 1] * powers[2, c - 1]) * value
            if c:
                az += (c * x * y * powers[2, c - 1]) * value
            if a > 1:
                axx += (a * (a - 1) * powers[0, a - 2] * y * z) * value
            if b > 1:
                ayy += (b * (b - 1) * x * powers[1, b - 2] * z) * value
            if c > 1:
                azz += (c * (c - 1) * x * y * powers[2, c - 2]) * value
        return ax, ay, az, axx, axy, axz, ayy, ayz, azz

    @ti.func
    def _node_local_fields(self, node: ti.i32, position: ti.template()):
        velocity = ti.Vector.zero(ti.f32, 3)
        gradient = ti.Matrix.zero(ti.f32, 3, 3)
        if self.local_present[node]:
            base = node * _LOCAL_COUNT
            centre = self.tree.node_centre[node]
            if self.block_mode[None] == 0:
                centre = self._transform(centre, 0)
            scale = self.tree.node_half_size[node]
            if scale == 0.0:
                scale = 1.0
            offset = (position - centre) / scale
            ax, ay, az = self.local[base + 3], self.local[base + 2], self.local[base + 1]
            axx, axy, axz = 2 * self.local[base + 9], self.local[base + 8], self.local[base + 7]
            ayy, ayz, azz = 2 * self.local[base + 6], self.local[base + 5], 2 * self.local[base + 4]
            if self.tree.node_half_size[node] != 0.0:
                ax, ay, az, axx, axy, axz, ayy, ayz, azz = self._local_derivatives(base, offset)
            velocity = ti.Vector([ay[2] - az[1], az[0] - ax[2], ax[1] - ay[0]]) / scale
            gradient = ti.Matrix(
                [
                    [axy[2] - axz[1], ayy[2] - ayz[1], ayz[2] - azz[1]],
                    [axz[0] - axx[2], ayz[0] - axy[2], azz[0] - axz[2]],
                    [axx[1] - axy[0], axy[1] - ayy[0], axz[1] - ayz[0]],
                ]
            ) / (scale * scale)
        return velocity, gradient

    @ti.func
    def _monopole_query_fields(self, source_node: ti.i32, query: ti.template()):
        source_position = self.source.tree.node_com[source_node]
        strength = self.source.tree.node_net_vortex_strength[source_node]
        displacement = query - source_position
        radius = displacement.norm()
        core = self.source.tree.node_avg_radius[source_node]
        # Match the legacy regularised monopole, including its safer far-field
        # arithmetic: evaluating blob sigma^-5 first can overflow at small
        # physical units even when the final gradient is representable.
        r2 = radius * radius
        r3 = r2 * radius
        r5 = r3 * r2
        cross = displacement.cross(strength)
        velocity, gradient = ti.Vector.zero(ti.f32, 3), ti.Matrix.zero(ti.f32, 3, 3)
        if ti.static(self.legacy_radial_kernel):
            q = self.source.tree.q_kernel(radius / core)
            zeta = self.source.tree.zeta_kernel(radius / core) / (core * core * core)
            velocity = -q * cross / r3
            gradient = (q / r3) * self.source.tree.skew(strength) + (
                3.0 * q / r5 - zeta / r2
            ) * cross.outer_product(displacement)
        else:
            # Internal users can still exercise other radial families, whose
            # source hierarchy uses Gaussian metadata but not Gaussian physics.
            # Production image-block dispatch currently admits only G/W.
            factors = self.source.radial_factors(radius / core, core, True)
            velocity = -factors[0] * cross
            gradient = factors[0] * self.source.tree.skew(strength) + factors[
                1
            ] * cross.outer_product(displacement)
        return velocity, gradient

    @ti.func
    def _physical_image_fields(self, velocity: ti.template(), gradient: ti.template(), image: ti.i32):
        v, j = velocity, gradient
        if self.block_mode[None] and self.image_odd[image]:
            v[2] = -v[2]
            for a, b in ti.static(ti.ndrange(3, 3)):
                if ti.static((a == 2) != (b == 2)):
                    j[a, b] = -j[a, b]
        return v, j

    @ti.func
    def _monopole_fields(self, source_node: ti.i32, position: ti.template(), image: ti.i32):
        query = position
        if self.block_mode[None]:
            query = self._transform(position, image)
        velocity, gradient = self._monopole_query_fields(source_node, query)
        return self._physical_image_fields(velocity, gradient, image)

    @ti.func
    def _exact_query_node_fields(self, source_node: ti.i32, query: ti.template()):
        velocity = ti.Vector.zero(ti.f32, 3)
        gradient = ti.Matrix.zero(ti.f32, 3, 3)
        first = self.source.tree.node_particle_start[source_node]
        size = self.source.tree.node_particle_count[source_node]
        for source_slot in range(first, first + size):
            source = self.source.tree.sorted_indices[source_slot]
            source_position = self.source.tree.position[source]
            strength = self.source.tree.vortex_strength[source]
            displacement = query - source_position
            radius = displacement.norm()
            core = self.source.tree.core_radius[source]
            factors = self.source.radial_factors(radius / core, core, True)
            cross = displacement.cross(strength)
            velocity -= factors[0] * cross
            gradient += factors[0] * self.source.tree.skew(strength) + factors[
                1
            ] * cross.outer_product(displacement)
        return velocity, gradient

    @ti.func
    def _exact_node_fields(self, source_node: ti.i32, position: ti.template(), image: ti.i32):
        query = position
        if self.block_mode[None]:
            query = self._transform(position, image)
        velocity, gradient = self._exact_query_node_fields(source_node, query)
        return self._physical_image_fields(velocity, gradient, image)

    @ti.func
    def _legacy_subtree_fields(self, root: ti.i32, position: ti.template(), image: ti.i32):
        """Preserve pointwise source acceptance inside a mixed packet.

        Parent/right-sibling advancement is stackless and confined to this
        subtree. Each terminal source node contributes once, in left-first
        order; the caller retains all contributions in private block scratch.
        """
        query = position
        if self.block_mode[None]:
            query = self._transform(position, image)
        velocity, gradient = ti.Vector.zero(ti.f32, 3), ti.Matrix.zero(ti.f32, 3, 3)
        node = root
        while node >= 0:
            accepted = self._legacy_accept(node, query - self.source.tree.node_com[node])
            if accepted or self.source.tree.node_is_leaf[node]:
                v, j = ti.Vector.zero(ti.f32, 3), ti.Matrix.zero(ti.f32, 3, 3)
                if accepted:
                    v, j = self._monopole_query_fields(node, query)
                else:
                    v, j = self._exact_query_node_fields(node, query)
                velocity += v
                gradient += j
                # Find the next unvisited sibling, stopping at this job's root.
                previous = node
                node = -1
                while previous != root:
                    parent = self.source.tree.node_parent[previous]
                    if self.source.tree.node_left[parent] == previous:
                        node = self.source.tree.node_right[parent]
                        break
                    previous = parent
            else:
                node = self.source.tree.node_left[node]
        return self._physical_image_fields(velocity, gradient, image)

    @ti.func
    def _near_pairs_fields(self, begin: ti.i32, end: ti.i32, position: ti.template()):
        velocity, gradient = ti.Vector.zero(ti.f32, 3), ti.Matrix.zero(ti.f32, 3, 3)
        for pair in range(begin, end):
            source_node = self.ordered_near_source[pair]
            image = self.ordered_near_image[pair]
            v, j = ti.Vector.zero(ti.f32, 3), ti.Matrix.zero(ti.f32, 3, 3)
            if self.ordered_near_legacy[pair]:
                v, j = self._legacy_subtree_fields(source_node, position, image)
            elif source_node < 0:
                v, j = self._monopole_fields(-source_node - 1, position, image)
            else:
                v, j = self._exact_node_fields(source_node, position, image)
            velocity += v
            gradient += j
        return velocity, gradient

    @ti.kernel
    def _evaluate_local_particles(self, count: ti.i32):
        for slot in range(count):
            target = self.tree.sorted_indices[slot]
            position = self.tree.position[target]
            if self.block_mode[None] == 0:
                position = self._transform(position, 0)
            velocity = ti.Vector.zero(ti.f32, 3)
            gradient = ti.Matrix.zero(ti.f32, 3, 3)
            node = slot
            while node >= 0:
                v, j = self._node_local_fields(node, position)
                velocity += v
                gradient += j
                node = self.tree.node_parent[node]
            self.velocity[target] += velocity
            self.gradient[target] += gradient

    @ti.kernel
    def _prepare_target_paths(self, count: ti.i32):
        self.target_path_error[None] = 0
        for slot in range(count):
            length = self.tree.node_depth[slot] + 1
            self.target_path_length[slot] = length
            if 0 < length <= self.target_path_capacity:
                node = slot
                level = length - 1
                while node >= 0 and level >= 0:
                    self.target_path[level, slot] = node
                    node = self.tree.node_parent[node]
                    level -= 1
                if node >= 0 or level != -1:
                    ti.atomic_add(self.target_path_error[None], 1)
            else:
                ti.atomic_add(self.target_path_error[None], 1)

    @ti.kernel
    def _evaluate_near_lanes(self, inclusive: ti.template(), count: ti.i32):
        for work in range(count * _NEAR_LANES):
            # Lane-major ordering keeps neighbouring Morton targets adjacent
            # within a warp while independent source-list lanes expose enough
            # work even when streaming activates only a few target cells.
            lane, slot = work // count, work % count
            target = self.tree.sorted_indices[slot]
            position = self.tree.position[target]
            if self.block_mode[None] == 0:
                position = self._transform(position, 0)
            velocity, gradient = ti.Vector.zero(ti.f32, 3), ti.Matrix.zero(ti.f32, 3, 3)
            # This changes summation order across target ancestors only. The
            # per-node source list, left-first subtree walk and lane reduction
            # retain their original order and complete interaction coverage.
            for level in range(self.target_path_length[slot]):
                node = self.target_path[level, slot]
                pairs = self.near_count[node]
                first = inclusive[node] - pairs
                begin = first + (pairs * lane) // _NEAR_LANES
                end = first + (pairs * (lane + 1)) // _NEAR_LANES
                v, j = self._near_pairs_fields(begin, end, position)
                velocity += v
                gradient += j
            self.near_partial_velocity[target, lane] = velocity
            self.near_partial_gradient[target, lane] = gradient

    @ti.kernel
    def _reduce_near_lanes(self, count: ti.i32):
        for target in range(count):
            velocity, gradient = ti.Vector.zero(ti.f32, 3), ti.Matrix.zero(ti.f32, 3, 3)
            for lane in range(_NEAR_LANES):
                velocity += self.near_partial_velocity[target, lane]
                gradient += self.near_partial_gradient[target, lane]
            self.velocity[target] += velocity
            self.gradient[target] += gradient

    @ti.kernel
    def _clear_outputs(self, count: ti.i32):
        for target in range(count):
            self.velocity[target] = ti.Vector.zero(ti.f32, 3)
            self.gradient[target] = ti.Matrix.zero(ti.f32, 3, 3)

    @ti.kernel
    def _publish(
        self,
        velocity: ti.template(),
        gradient: ti.template(),
        background: ti.template(),
        count: ti.i32,
        output_start: ti.i32,
        write_velocity: ti.template(),
        write_gradient: ti.template(),
    ):
        for target in range(count):
            if ti.static(write_velocity):
                velocity[output_start + target] = self.velocity[target] + background[None]
            if ti.static(write_gradient):
                gradient[output_start + target] = self.gradient[target]

    def prepare_targets(self, target_position, target_count, *, target_start=0):
        """Build one immutable target tile, reusable for translations/reflections.

        The tile is copied into owned scratch.  Mutations to caller fields
        afterwards cannot accidentally change this prepared geometry.
        """
        count = int(target_count)
        start = int(target_start)
        if not 0 <= count <= self.max_targets or start < 0:
            raise ValueError("target count/start exceeds prepared target workspace")
        # A failed build must not leave an incompletely prepared tile eligible
        # for publication through evaluate_prepared/evaluate_image_block.
        self._prepared_count = 0
        if count:
            self._copy_target_slice(target_position, start, count)
            self.tree.build(self.query_position, self.zero_strength, self.zero_radius, count)
            self._prepare_leaves(count)
            self._prepare_target_paths(count)
            if int(self.target_path_error[None]):
                raise RuntimeError("target hierarchy exceeds the bounded ancestry scratch")
        self._prepared_count = count
        self.target_geometry_builds += 1

    def evaluate(self, target_position, target_velocity, target_gradient, target_count, background):
        """Evaluate current sources; publish outputs only after successful traversal."""
        self.prepare_targets(target_position, target_count)
        self.evaluate_prepared(target_velocity, target_gradient, background)

    def evaluate_prepared(
        self, target_velocity, target_gradient, background, *, shift=0.0, odd=False, output_start=0
    ):
        """Evaluate a rigid image of the prepared tile without rebuilding it.

        Coordinates follow ``z' = shift - z`` for odd reflection and
        ``z' = z - shift`` for translation.  Returned fields are derivatives
        with respect to these transformed coordinates; the slab wrapper owns
        vector/Jacobian parity and convergence-block accumulation.
        """
        count = self._prepared_count
        if target_velocity is None and target_gradient is None:
            raise ValueError("at least one target output is required")
        if not math.isfinite(shift) or output_start < 0:
            raise ValueError("invalid target transform or output offset")
        if count == 0:
            return
        self._set_image_transform(float(shift), int(bool(odd)))
        self._evaluate_prepared_fields(target_velocity, target_gradient, background, output_start)

    def evaluate_image_block(
        self, images, target_velocity, target_gradient, background, *, output_start=0
    ):
        """Sum a complete supplied image block into one local expansion per cell.

        ``images`` contains (shift, odd) pairs in the caller's existing shell
        block.  No image is dropped and no convergence decision is made here.
        Both returned fields are already transformed to physical coordinates;
        callers must not apply reflection parity a second time.
        """
        images = tuple(images)
        if not 1 <= len(images) <= self.max_images:
            raise ValueError("image block exceeds prepared image capacity")
        shifts = np.zeros(self.max_images, dtype=np.float32)
        odd_flags = np.zeros(self.max_images, dtype=np.int32)
        for index, (shift, odd) in enumerate(images):
            if not math.isfinite(shift):
                raise ValueError("image shift must be finite")
            shifts[index] = shift
            odd_flags[index] = int(bool(odd))
        self.image_shift.from_numpy(shifts)
        self.image_odd.from_numpy(odd_flags)
        self.image_count[None] = len(images)
        self.block_mode[None] = 1
        self._evaluate_prepared_fields(target_velocity, target_gradient, background, output_start)

    def _evaluate_prepared_fields(self, target_velocity, target_gradient, background, output_start):
        count = self._prepared_count
        if target_velocity is None and target_gradient is None:
            raise ValueError("at least one target output is required")
        if output_start < 0:
            raise ValueError("output offset must be non-negative")
        if not count:
            return
        leaf_count = int(self.leaf_count[None])
        node_count = 2 * count - 1
        image_count = int(self.image_count[None])
        started = perf_counter()
        self.last_diagnostics = {
            "target_count": count,
            "target_cells": leaf_count,
            "image_count": image_count,
            "m2l_pairs": 0,
            "near_cell_pairs": 0,
            "near_lanes": _NEAR_LANES,
            "direct_particle_pairs": 0,
            "monopole_target_pairs": 0,
            "monopole_cell_pairs": 0,
            "legacy_subtree_target_pairs": 0,
            "work_batches": 0,
            "subdivided_batches": 0,
            "peak_stored_pairs": 0,
            "pair_capacity": self.max_pairs,
        }
        self._clear_outputs(count)
        # Split work, never disposable fields, when a list fills. Already
        # completed subblocks remain private until the entire caller's block
        # succeeds; an irreducible decline publishes absolutely nothing.
        jobs = [(0, leaf_count, 0, image_count, -1)]
        while jobs:
            first_leaf, leaves, first_image, images, root = jobs.pop()
            with self._profile_phase("initialise"):
                self._initialise(node_count)
            with self._profile_phase("traversal"):
                self._walk_sources(first_leaf, leaves, first_image, images, root)
            error = int(self.error[None])
            m2l = int(self.m2l_count[None])
            near = int(self.near_pair_count[None])
            if error == 1:
                required = max(m2l, near, int(self.frontier_count[None]), leaves * images)
                self.last_diagnostics["subdivided_batches"] += 1
                if images > 1:
                    split = images // 2
                    jobs.extend([
                        (first_leaf, leaves, first_image + split, images - split, root),
                        (first_leaf, leaves, first_image, split, root),
                    ])
                elif leaves > 1:
                    split = leaves // 2
                    jobs.extend([
                        (first_leaf + split, leaves - split, first_image, images, -1),
                        (first_leaf, split, first_image, images, -1),
                    ])
                else:
                    if root < 0:
                        root = int(self.leaf_nodes[first_leaf])
                    if int(self.tree.node_particle_count[root]) > 1:
                        jobs.extend([
                            (0, 1, first_image, images, int(self.tree.node_right[root])),
                            (0, 1, first_image, images, int(self.tree.node_left[root])),
                        ])
                    elif self.block_mode[None]:
                        raise TargetBlockNotWorthwhile(
                            required, self.max_pairs, self.last_diagnostics
                        )
                    else:
                        raise TargetInteractionCapacityError(self.max_pairs, required)
                continue
            if error:
                raise RuntimeError("target FMM dual-tree traversal exceeded its depth bound")
            self.last_diagnostics["m2l_pairs"] += m2l
            self.last_diagnostics["near_cell_pairs"] += near
            self.last_diagnostics["direct_particle_pairs"] += int(self.direct_work[None])
            self.last_diagnostics["monopole_target_pairs"] += int(self.monopole_work[None])
            self.last_diagnostics["monopole_cell_pairs"] += int(self.monopole_pair_count[None])
            self.last_diagnostics["legacy_subtree_target_pairs"] += int(self.legacy_subtree_work[None])
            self.last_diagnostics["work_batches"] += 1
            self.last_diagnostics["peak_stored_pairs"] = max(
                self.last_diagnostics["peak_stored_pairs"], m2l, near
            )
            for start in range(0, m2l, self.batch_capacity):
                batch = min(self.batch_capacity, m2l - start)
                with self._profile_phase("derivative_cache"):
                    self._compute_derivatives(start, batch)
                with self._profile_phase("local_translation"):
                    self._translate(start, batch)
            if m2l:
                with self._profile_phase("local_evaluation"):
                    self._evaluate_local_particles(count)
            if near:
                with self._profile_phase("near_ordering"):
                    self._copy_counts(node_count)
                    inclusive, scratch = self.scan_a, self.scan_b
                    stride = 1
                    while stride < node_count:
                        self._scan_counts(inclusive, scratch, node_count, stride)
                        inclusive, scratch = scratch, inclusive
                        stride *= 2
                    self._initialise_cursor(inclusive, node_count)
                    self._order_near(near)
                with self._profile_phase("near_evaluation"):
                    self._evaluate_near_lanes(inclusive, count)
                    self._reduce_near_lanes(count)
        self._publish(
            self.velocity if target_velocity is None else target_velocity,
            self.gradient if target_gradient is None else target_gradient,
            background,
            count,
            int(output_start),
            target_velocity is not None,
            target_gradient is not None,
        )
        self.last_diagnostics["seconds"] = perf_counter() - started


__all__ = ["FMMTargetEvaluator", "TargetBlockNotWorthwhile", "TargetInteractionCapacityError"]
