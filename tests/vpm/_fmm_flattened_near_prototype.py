"""Unwired register-pressure experiment against the frozen root-first solver.

One lane accumulator receives every terminal source contribution directly.
The root-first target path, ordered pair list, per-node lane ranges, left-first
source traversal, radial functions and inverse-query image arithmetic stay
unchanged. Floating-point parentheses change across source subtrees and target
ancestors; odd-image signs are applied per terminal rather than after its
subtree sum. There is no new approximation, acceptance rule or output owner.

This qualification module intentionally imports only the preserved numerical
source. It must never be imported into a production solver process.
"""

from contextlib import contextmanager
import importlib
from pathlib import Path
import sys

import taichi as ti

FROZEN_SOURCE_ROOT = (
    Path(__file__).resolve().parents[2]
    / "tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow/solution/restart-branches"
    / "performance-rootfirst-predictor-20261002-sP7XZi"
)


def assert_frozen_numerical_imports():
    for name, module in tuple(sys.modules.items()):
        filename = getattr(module, "__file__", None)
        if (
            name.startswith(("source.", "openonda.")) and filename
            and not Path(filename).resolve().is_relative_to(FROZEN_SOURCE_ROOT)
        ):
            raise RuntimeError(f"qualification imported live numerical source: {filename}")


assert_frozen_numerical_imports()
if not (FROZEN_SOURCE_ROOT / "source/solvers/vpm/physics/induction/fmm/targets.py").is_file():
    raise RuntimeError("the preserved root-first qualification source is missing")
sys.path.insert(0, str(FROZEN_SOURCE_ROOT))
target_module = importlib.import_module("source.solvers.vpm.physics.induction.fmm.targets")
FMMTargetEvaluator = target_module.FMMTargetEvaluator
_NEAR_LANES = target_module._NEAR_LANES
assert_frozen_numerical_imports()


@ti.data_oriented
class FlattenedNearEvaluator(FMMTargetEvaluator):
    _evaluate_grouped_near_lanes = FMMTargetEvaluator._evaluate_near_lanes

    @ti.kernel
    def _evaluate_flattened_near_lanes(self, inclusive: ti.template(), count: ti.i32):
        for work in range(count * _NEAR_LANES):
            lane, slot = work // count, work % count
            target = self.tree.sorted_indices[slot]
            position = self.tree.position[target]
            if self.block_mode[None] == 0:
                position = self._transform(position, 0)
            velocity = ti.Vector.zero(ti.f32, 3)
            gradient = ti.Matrix.zero(ti.f32, 3, 3)
            for level in range(self.target_path_length[slot]):
                target_node = self.target_path[level, slot]
                pairs = self.near_count[target_node]
                first = inclusive[target_node] - pairs
                begin = first + (pairs * lane) // _NEAR_LANES
                end = first + (pairs * (lane + 1)) // _NEAR_LANES
                for pair in range(begin, end):
                    source = self.ordered_near_source[pair]
                    image = self.ordered_near_image[pair]
                    query = position
                    if self.block_mode[None]:
                        query = self._transform(position, image)
                    if self.ordered_near_legacy[pair]:
                        # Identical bounded, stackless, left-first subtree
                        # walk; only its private subtotal has been removed.
                        root = source
                        node = root
                        while node >= 0:
                            accepted = self._legacy_accept(
                                node, query - self.source.tree.node_com[node]
                            )
                            if accepted or self.source.tree.node_is_leaf[node]:
                                v = ti.Vector.zero(ti.f32, 3)
                                j = ti.Matrix.zero(ti.f32, 3, 3)
                                if accepted:
                                    v, j = self._monopole_query_fields(node, query)
                                else:
                                    v, j = self._exact_query_node_fields(node, query)
                                v, j = self._physical_image_fields(v, j, image)
                                velocity += v
                                gradient += j
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
                    else:
                        v = ti.Vector.zero(ti.f32, 3)
                        j = ti.Matrix.zero(ti.f32, 3, 3)
                        if source < 0:
                            v, j = self._monopole_query_fields(-source - 1, query)
                        else:
                            v, j = self._exact_query_node_fields(source, query)
                        v, j = self._physical_image_fields(v, j, image)
                        velocity += v
                        gradient += j
            self.near_partial_velocity[target, lane] = velocity
            self.near_partial_gradient[target, lane] = gradient

    def _evaluate_near_lanes(self, inclusive, count):
        self._evaluate_flattened_near_lanes(inclusive, count)


@contextmanager
def flattened_near_factory():
    # Freeze the true-class whitelist before replacing the factory. The
    # experimental class remains ineligible for the exact-state cache.
    reuse = importlib.import_module("source.solvers.vpm.physics.induction.reuse_backends")
    previous = target_module.FMMTargetEvaluator
    if previous is not FMMTargetEvaluator or reuse.FMMTargetEvaluator is not previous:
        raise RuntimeError("qualification requires an unmodified frozen target factory")
    assert_frozen_numerical_imports()
    target_module.FMMTargetEvaluator = FlattenedNearEvaluator
    try:
        yield
    finally:
        target_module.FMMTargetEvaluator = previous


__all__ = [
    "FROZEN_SOURCE_ROOT", "FlattenedNearEvaluator", "assert_frozen_numerical_imports",
    "flattened_near_factory",
]
