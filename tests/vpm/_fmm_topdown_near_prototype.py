"""Unwired target-ancestry scheduling experiment, with unchanged source work.

Precompute target paths once per prepared geometry, then visit root-to-leaf.
Neighbouring Morton targets encounter common ancestors at the same loop depth
even when leaf depths differ. The same ordered source lists and source-lane
ranges are evaluated; only summation order across target ancestors changes.
No warp-intrinsic source traversal, opening criterion or local expansion changes.
"""

from contextlib import contextmanager
from time import perf_counter

import numpy as np
import taichi as ti

from source.solvers.vpm.physics.induction.fmm import targets as target_module
from source.solvers.vpm.physics.induction.fmm.targets import _NEAR_LANES, FMMTargetEvaluator


@ti.data_oriented
class TopdownNearEvaluator(FMMTargetEvaluator):
    # The optimized traversal is now production. Retain a thin instrumented
    # wrapper without allocating a second, hidden copy of the ancestry fields.
    _evaluate_topdown_near_lanes = FMMTargetEvaluator._evaluate_near_lanes

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.path_capacity = self.target_path_capacity
        self.path_error = self.target_path_error
        self.path_geometry_diagnostics = {}

    def prepare_targets(self, *args, **kwargs):
        started = perf_counter()
        super().prepare_targets(*args, **kwargs)
        elapsed = perf_counter() - started
        self.path_geometry_diagnostics = {
            "complete_target_geometry_prepare_seconds": elapsed,
            "path_payload_bytes": (self.path_capacity + 1) * self.max_targets * 4 + 4,
            "additional_prototype_field_bytes": 0,
            "source_partition_changed": False,
            "summation_order": "target-root-to-leaf, source-left-first, unchanged-source-lanes",
        }
        if bool(getattr(self.source, "profile_passes", False)):
            started = perf_counter()
            lengths = self.target_path_length.to_numpy()[:self._prepared_count]
            spreads = [int(np.ptp(lengths[i:i + 32])) for i in range(0, len(lengths), 32)]
            self.path_geometry_diagnostics.update(
                leaf_depth_min=int(lengths.min() - 1) if len(lengths) else 0,
                leaf_depth_max=int(lengths.max() - 1) if len(lengths) else 0,
                warp_leaf_depth_spreads=spreads,
                unequal_depth_warps=sum(value != 0 for value in spreads),
                diagnostic_readback_seconds=perf_counter() - started,
            )

    @ti.kernel
    def _evaluate_bottomup_near_lanes(self, inclusive: ti.template(), count: ti.i32):
        # Frozen original bottom-up oracle, not an alias to the now-optimized
        # production method. Source lists, lane ranges and kernels stay shared.
        for work in range(count * _NEAR_LANES):
            lane, slot = work // count, work % count
            target = self.tree.sorted_indices[slot]
            position = self.tree.position[target]
            if self.block_mode[None] == 0:
                position = self._transform(position, 0)
            velocity = ti.Vector.zero(ti.f32, 3)
            gradient = ti.Matrix.zero(ti.f32, 3, 3)
            node = slot
            while node >= 0:
                pairs = self.near_count[node]
                first = inclusive[node] - pairs
                begin = first + (pairs * lane) // _NEAR_LANES
                end = first + (pairs * (lane + 1)) // _NEAR_LANES
                v, j = self._near_pairs_fields(begin, end, position)
                velocity += v
                gradient += j
                node = self.tree.node_parent[node]
            self.near_partial_velocity[target, lane] = velocity
            self.near_partial_gradient[target, lane] = gradient

    def _evaluate_near_lanes(self, inclusive, count):
        self.last_diagnostics["topdown_ancestry"] = dict(self.path_geometry_diagnostics)
        self._evaluate_topdown_near_lanes(inclusive, count)


@contextmanager
def topdown_near_factory():
    from source.solvers.vpm.physics.induction import reuse_backends

    original = target_module.FMMTargetEvaluator
    if original is not FMMTargetEvaluator or reuse_backends.FMMTargetEvaluator is not original:
        raise RuntimeError("qualification requires the unmodified standard target factory")
    target_module.FMMTargetEvaluator = TopdownNearEvaluator
    try:
        yield
    finally:
        target_module.FMMTargetEvaluator = original


__all__ = ["TopdownNearEvaluator", "topdown_near_factory"]
