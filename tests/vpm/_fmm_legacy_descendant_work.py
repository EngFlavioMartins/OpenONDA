"""Unwired pointwise work census for proposed source-expansion packets.

This evaluates no velocity, derivative, or polynomial.  It follows the exact
legacy point decision and its left-first stackless traversal.  Counts measure
the pointwise terminal representation, NOT necessarily the production packet
engine's cost: that engine can already share some descendant local expansions.
"""

import numpy as np
import taichi as ti

from source.solvers.vpm.physics.induction.treecode.lbvh import _OwnedFields

_METRICS = (
    "packets", "covered_target_jobs", "sampled_target_jobs", "node_visits",
    "accepted_monopole_terminals", "exact_leaf_terminals", "exact_particle_terms",
    "opened_internal_nodes", "parent_climbs", "root_accepted_target_jobs",
    "terminal_source_coverage",
)
_METRIC_COUNT = len(_METRICS)


@ti.data_oriented
class LegacyDescendantWorkCounter:
    """Bounded metadata-only accounting; any incomplete walk invalidates it."""

    def __init__(self, evaluator, *, target_stride=1, target_offset=0, max_visits_per_target=None):
        if int(target_stride) != target_stride or target_stride < 1:
            raise ValueError("target_stride must be a positive integer")
        if int(target_offset) != target_offset or not 0 <= target_offset < target_stride:
            raise ValueError("target_offset must be an integer within the sampling stride")
        self.evaluator = evaluator
        self.tree = evaluator.source.tree
        self.target_stride, self.target_offset = int(target_stride), int(target_offset)
        self.source_generation = evaluator.source.source_multipole_generation
        self.max_visits_per_target = (
            2 * int(evaluator.source.max_n_particles)
            if max_visits_per_target is None else int(max_visits_per_target)
        )
        if self.max_visits_per_target < 1:
            raise ValueError("max_visits_per_target must be positive")
        self.owner = _OwnedFields()
        try:
            self.counts = self.owner.scalar(dtype=ti.i64, shape=(4, _METRIC_COUNT))
            self.rejections = self.owner.scalar(dtype=ti.i64, shape=(4, 16))
            # 0 visit/depth cap, 1 source coverage, 2 point-decision mismatch.
            self.errors = self.owner.scalar(dtype=ti.i64, shape=3)
            self.owner.finalize()
            self.reset()
        except BaseException:
            self.destroy()
            raise

    def destroy(self):
        owner, self.owner = self.owner, None
        if owner is not None:
            owner.destroy()

    def reset(self):
        if self.owner is None:
            raise RuntimeError("work counter is closed")
        if self.evaluator.source.source_multipole_generation != self.source_generation:
            raise RuntimeError("source changed after work counter preparation")
        self.counts.fill(0)
        self.rejections.fill(0)
        self.errors.fill(0)

    @ti.func
    def _rejection_mask(self, node: ti.i32, displacement: ti.template()):
        """Reconstruct reasons, but NEVER use them instead of the real MAC."""
        radius_sq = displacement.dot(displacement)
        radius = ti.sqrt(radius_sq)
        diameter = 2.0 * self.tree.node_half_size[node]
        mask = 0
        if not (radius > ti.max(1e-8, self.tree.node_avg_radius[node])):
            mask += 1
        mac_ok = False
        if radius_sq > 0:
            mac_ok = diameter * diameter / radius_sq < self.tree.theta_sq
        if not mac_ok:
            mask += 2
        spread = self.tree.node_max_radius[node] - self.tree.node_min_radius[node]
        extent = self.tree.node_half_size[node] + (
            self.tree.node_com[node] - self.tree.node_centre[node]
        ).norm()
        outside = radius - extent > (
            self.tree.regularization_tail_cutoff[None] * self.tree.node_max_radius[node]
        )
        common = spread <= 1e-5 * ti.max(self.tree.node_avg_radius[node], 1e-12)
        if not (common or outside):
            mask += 4
        net = self.tree.node_net_vortex_strength[node]
        if net.dot(net) <= 1e-24:
            mask += 8
        return mask

    @ti.func
    def record(self, target: ti.i32, root: ti.i32, image: ti.i32, order: ti.i32):
        evaluator = self.evaluator
        first = evaluator.tree.node_particle_start[target]
        count = evaluator.tree.node_particle_count[target]
        ti.atomic_add(self.counts[order, 0], 1)
        ti.atomic_add(self.counts[order, 1], ti.cast(count, ti.i64))
        for slot in range(first, first + count):
            if slot % self.target_stride == self.target_offset:
                values = ti.Vector.zero(ti.i64, _METRIC_COUNT)
                values[2] = 1
                query = evaluator.tree.position[evaluator.tree.sorted_indices[slot]]
                if evaluator.block_mode[None]:
                    query = evaluator._transform(query, image)
                node = root
                complete = True
                while node >= 0:
                    if values[3] >= self.max_visits_per_target:
                        ti.atomic_add(self.errors[0], 1)
                        complete = False
                        break
                    values[3] += 1
                    displacement = query - self.tree.node_com[node]
                    accepted = evaluator._legacy_accept(node, displacement)
                    reasons = self._rejection_mask(node, displacement)
                    if accepted != (reasons == 0):
                        ti.atomic_add(self.errors[2], 1)
                    if not accepted:
                        ti.atomic_add(self.rejections[order, reasons], 1)
                    leaf = self.tree.node_is_leaf[node] != 0
                    if accepted or leaf:
                        if accepted:
                            values[4] += 1
                            if node == root:
                                values[9] += 1
                        else:
                            values[5] += 1
                            values[6] += ti.cast(self.tree.node_particle_count[node], ti.i64)
                        values[10] += ti.cast(self.tree.node_particle_count[node], ti.i64)
                        previous = node
                        node = -1
                        hops = 0
                        while previous != root:
                            if hops >= self.max_visits_per_target:
                                ti.atomic_add(self.errors[0], 1)
                                complete = False
                                break
                            parent = self.tree.node_parent[previous]
                            if parent < 0:
                                ti.atomic_add(self.errors[0], 1)
                                complete = False
                                break
                            values[8] += 1
                            hops += 1
                            if self.tree.node_left[parent] == previous:
                                node = self.tree.node_right[parent]
                                break
                            previous = parent
                    else:
                        values[7] += 1
                        node = self.tree.node_left[node]
                if complete and values[10] != ti.cast(self.tree.node_particle_count[root], ti.i64):
                    ti.atomic_add(self.errors[1], 1)
                for metric in ti.static(range(2, _METRIC_COUNT)):
                    ti.atomic_add(self.counts[order, metric], values[metric])

    def report(self):
        if self.owner is None:
            raise RuntimeError("work counter is closed")
        if self.evaluator.source.source_multipole_generation != self.source_generation:
            raise RuntimeError("source changed while counting legacy work")
        errors = self.errors.to_numpy()
        if np.any(errors):
            raise RuntimeError(f"incomplete or inconsistent legacy work census: {errors.tolist()}")
        counts = self.counts.to_numpy()
        return {
            "scope": "pointwise legacy descendants inside proposed replacement packets only",
            "sampling": {
                "target_stride": self.target_stride, "target_offset": self.target_offset,
                "exhaustive": self.target_stride == 1,
                "selection": "tile-local sorted target slot modulo stride",
                "population_extrapolation_performed": False,
            },
            "by_order": {
                str(order): {name: int(counts[index, metric]) for metric, name in enumerate(_METRICS)}
                for index, order in enumerate((3, 5, 7, 9))
            },
            "rejection_histogram_by_order": self.rejections.to_numpy().tolist(),
            "rejection_mask_bits": {
                "1": "distance does not exceed mean core or 1e-8",
                "2": "strict legacy geometric MAC rejects",
                "4": "mixed cores and not wholly outside regularization tail",
                "8": "net source strength squared <= 1e-24",
            },
            "structural_or_decision_errors": errors.tolist(),
            "fields_evaluated": False,
            "cost_warning": (
                "Exact pointwise counts, not measured production savings: the packet engine may "
                "already share some accepted descendant local expansions. Sampled counts are "
                "raw observations and are not extrapolated."
            ),
        }
