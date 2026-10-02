"""Unwired qualification-only split of monopole and heavy near-field kernels.

The ordered interaction lists, lane ranges and final deterministic reduction
are unchanged. Only grouping inside each lane changes: accepted monopoles are
summed first, then legacy-subtree/exact terms. Existing private lane buffers
are reused; no caller field is published before the entire image block passes.
This experiment is deliberately absent from production dispatch.
"""

from contextlib import contextmanager
from time import perf_counter

import taichi as ti

from source.solvers.vpm.physics.induction.fmm import targets as target_module
from source.solvers.vpm.physics.induction.fmm.targets import _NEAR_LANES, FMMTargetEvaluator


@ti.data_oriented
class SpecializedNearEvaluator(FMMTargetEvaluator):
    # Bind the original decorated kernel through Taichi's instance machinery;
    # direct class-level calls intentionally reject data-oriented instances.
    _evaluate_monolithic_near_lanes = FMMTargetEvaluator._evaluate_near_lanes

    @contextmanager
    def _prototype_phase(self, name):
        enabled = bool(getattr(self.source, "profile_passes", False))
        if enabled:
            ti.sync()
            started = perf_counter()
        yield
        if enabled:
            ti.sync()
            # Nested attribution is deliberately outside passes_seconds,
            # whose near_evaluation already includes both class kernels.
            phases = self.last_diagnostics["prototype_near_classes"].setdefault("seconds", {})
            phases[name] = phases.get(name, 0.0) + perf_counter() - started

    @ti.func
    def _class_pairs_fields(
        self, begin: ti.i32, end: ti.i32, position: ti.template(), monopoles: ti.template()
    ):
        velocity = ti.Vector.zero(ti.f32, 3)
        gradient = ti.Matrix.zero(ti.f32, 3, 3)
        for pair in range(begin, end):
            source = self.ordered_near_source[pair]
            if ti.static(monopoles):
                if source < 0:
                    v, j = self._monopole_fields(
                        -source - 1, position, self.ordered_near_image[pair]
                    )
                    velocity += v
                    gradient += j
            else:
                if source >= 0:
                    image = self.ordered_near_image[pair]
                    v, j = ti.Vector.zero(ti.f32, 3), ti.Matrix.zero(ti.f32, 3, 3)
                    if self.ordered_near_legacy[pair]:
                        v, j = self._legacy_subtree_fields(source, position, image)
                    else:
                        v, j = self._exact_node_fields(source, position, image)
                    velocity += v
                    gradient += j
        return velocity, gradient

    @ti.kernel
    def _evaluate_near_class_lanes(
        self, inclusive: ti.template(), count: ti.i32,
        monopoles: ti.template(), accumulate: ti.template(),
    ):
        for work in range(count * _NEAR_LANES):
            lane, slot = work // count, work % count
            target = self.tree.sorted_indices[slot]
            position = self.tree.position[target]
            if self.block_mode[None] == 0:
                position = self._transform(position, 0)
            velocity, gradient = ti.Vector.zero(ti.f32, 3), ti.Matrix.zero(ti.f32, 3, 3)
            node = slot
            while node >= 0:
                pairs = self.near_count[node]
                first = inclusive[node] - pairs
                begin = first + (pairs * lane) // _NEAR_LANES
                end = first + (pairs * (lane + 1)) // _NEAR_LANES
                v, j = self._class_pairs_fields(begin, end, position, monopoles)
                velocity += v
                gradient += j
                node = self.tree.node_parent[node]
            if ti.static(accumulate):
                velocity += self.near_partial_velocity[target, lane]
                gradient += self.near_partial_gradient[target, lane]
            self.near_partial_velocity[target, lane] = velocity
            self.near_partial_gradient[target, lane] = gradient

    def _evaluate_near_lanes(self, inclusive, count):
        monopoles = int(self.monopole_pair_count[None])
        total = int(self.near_pair_count[None])
        if not 0 <= monopoles <= total:
            raise RuntimeError("inconsistent private near-field class counts")
        classes = self.last_diagnostics.setdefault(
            "prototype_near_classes", {"monopole_passes": 0, "heavy_passes": 0}
        )
        if monopoles:
            with self._prototype_phase("monopole_lanes"):
                self._evaluate_near_class_lanes(inclusive, count, True, False)
            classes["monopole_passes"] += 1
        if total > monopoles or not total:
            with self._prototype_phase("heavy_lanes"):
                self._evaluate_near_class_lanes(inclusive, count, False, monopoles > 0)
            classes["heavy_passes"] += 1


@contextmanager
def specialized_near_factory():
    """Select this prototype in ONE isolated qualification process only.

    Enter before constructing an evaluator; no existing workspace is replaced.
    A native qualification driver can enter this context and invoke the normal
    checkpoint operator profiler, recording this file's hash alongside normal
    source hashes. Never use this context in a production solver or cache test:
    the exact-backend reuse whitelist correctly rejects this unqualified class.
    """
    # Import the real whitelist BEFORE replacing the factory. Otherwise its
    # first import inside this context could snapshot the experiment as the
    # standard class and accidentally certify it for exact-induction reuse.
    from source.solvers.vpm.physics.induction import reuse_backends

    previous = target_module.FMMTargetEvaluator
    if previous is not FMMTargetEvaluator or reuse_backends.FMMTargetEvaluator is not previous:
        raise RuntimeError("qualification requires the unmodified standard target factory")
    target_module.FMMTargetEvaluator = SpecializedNearEvaluator
    try:
        yield
    finally:
        target_module.FMMTargetEvaluator = previous


__all__ = ["SpecializedNearEvaluator", "specialized_near_factory"]
