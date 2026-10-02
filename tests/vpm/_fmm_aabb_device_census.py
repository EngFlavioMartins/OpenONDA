"""Unwired instrumentation of tighter exact-MAC bounds; no operator changes."""

from contextlib import contextmanager

import taichi as ti

from source.solvers.vpm.physics.induction.fmm import targets as target_module
from source.solvers.vpm.physics.induction.fmm.targets import FMMTargetEvaluator
from source.solvers.vpm.physics.induction.treecode.lbvh import _OwnedFields


@ti.data_oriented
class AABBCensusEvaluator(FMMTargetEvaluator):
    def __init__(self, *args, **kwargs):
        self._census_owner = None
        super().__init__(*args, **kwargs)
        try:
            fields = _OwnedFields()
            self._census_owner = fields
            self.census_cells = fields.scalar(dtype=ti.i64, shape=3)
            self.census_targets = fields.scalar(dtype=ti.i64, shape=3)
            self.census_errors = fields.scalar(dtype=ti.i64, shape=())
            fields.finalize()
        except BaseException:
            self.destroy()
            raise

    def destroy(self):
        owner, self._census_owner = self._census_owner, None
        if owner is not None:
            owner.destroy()
        super().destroy()

    @ti.func
    def _distance_endpoint_accept(self, source: ti.i32, square: ti.f32, distance: ti.f32):
        accepted = False
        if square > 0:
            diameter = 2.0 * self.source.tree.node_half_size[source]
            accepted = (
                distance > ti.max(1e-8, self.source.tree.node_avg_radius[source])
                and diameter * diameter / square < self.source.tree.theta_sq
                and self.source.tree._node_core_is_admissible(
                    source, distance, self.source.tree.node_max_radius[source]
                ) != 0
            )
        return accepted

    @ti.func
    def _aabb_classification(self, target: ti.i32, source: ti.i32, image: ti.i32):
        # Exactly the existing f32 inverse-query transform, applied to extrema;
        # monotonic rounding encloses ALL target-query displacement components.
        first = self._transform(self.tree._node_aabb_min[target], image) - self.source.tree.node_com[source]
        last = self._transform(self.tree._node_aabb_max[target], image) - self.source.tree.node_com[source]
        lower, upper = ti.min(first, last), ti.max(first, last)
        nearest = ti.max(ti.max(lower, -upper), 0.0)
        furthest = ti.max(ti.abs(lower), ti.abs(upper))
        tiny = ti.cast(1.1754943508222875e-38, ti.f32)
        low_sq = ti.max(0.0, nearest.dot(nearest) * (1 - 16 * 2**-24) - tiny)
        high_sq = furthest.dot(furthest) * (1 + 16 * 2**-24) + tiny
        low = ti.sqrt(low_sq) * (1 - 8 * 2**-24)
        high = ti.sqrt(high_sq) * (1 + 8 * 2**-24)
        result = 0  # MIXED/unknown; zero is also the safe nonfinite fallback.
        if low_sq >= 0 and high_sq >= 0 and high_sq < 3.4028234663852886e38:
            if self._distance_endpoint_accept(source, low_sq, low):
                result = 1  # Every point accepts this same legacy source node.
            elif not self._distance_endpoint_accept(source, high_sq, high):
                result = 2  # No point accepts; opening remains necessary.
        return result

    @ti.kernel
    def _record_aabb_decisions(self, pair_count: ti.i32):
        for pair in range(pair_count):
            if self.near_legacy[pair]:
                target, source, image = self.near_target[pair], self.near_source[pair], self.near_image[pair]
                classification = self._aabb_classification(target, source, image)
                count = self.tree.node_particle_count[target]
                ti.atomic_add(self.census_cells[classification], 1)
                ti.atomic_add(self.census_targets[classification], ti.cast(count, ti.i64))
                # Validate every newly classified point against the production
                # predicate. This is instrumentation work, not an optimized
                # runtime branch; original field execution is wholly unchanged.
                if classification != 0:
                    first = self.tree.node_particle_start[target]
                    for slot in range(first, first + count):
                        particle = self.tree.sorted_indices[slot]
                        query = self._transform(self.tree.position[particle], image)
                        accepted = self._legacy_accept(source, query - self.source.tree.node_com[source])
                        if (classification == 1 and not accepted) or (classification == 2 and accepted):
                            ti.atomic_add(self.census_errors[None], 1)

    def _walk_sources(self, *args, **kwargs):
        super()._walk_sources(*args, **kwargs)
        # Failed capacity attempts are discarded and replayed by the original
        # bounded scheduler. Count only successful disjoint batches.
        if not int(self.error[None]):
            self._record_aabb_decisions(int(self.near_pair_count[None]))

    def evaluate_image_block(self, *args, **kwargs):
        self.census_cells.fill(0)
        self.census_targets.fill(0)
        self.census_errors[None] = 0
        super().evaluate_image_block(*args, **kwargs)
        errors = int(self.census_errors[None])
        self.last_diagnostics["aabb_decision_census"] = {
            "categories": ["still_mixed", "proved_all", "proved_none"],
            "cell_jobs": self.census_cells.to_numpy().tolist(),
            "target_jobs": self.census_targets.to_numpy().tolist(),
            "pointwise_conservatism_errors": errors,
            "field_operator_changed": False,
            "target_local_radius_gate_changed": False,
            "none_is_not_a_speedup_claim": True,
        }
        if errors:
            raise AssertionError(f"AABB census made {errors} false pointwise classifications")


@contextmanager
def aabb_census_factory():
    from source.solvers.vpm.physics.induction import reuse_backends

    original = target_module.FMMTargetEvaluator
    if original is not FMMTargetEvaluator or reuse_backends.FMMTargetEvaluator is not original:
        raise RuntimeError("census requires the unmodified original target factory")
    target_module.FMMTargetEvaluator = AABBCensusEvaluator
    try:
        yield
    finally:
        target_module.FMMTargetEvaluator = original
