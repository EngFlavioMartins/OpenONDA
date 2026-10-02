"""Unwired device-tree opportunity census; this computes NO physical fields.

Potential admissions include source truncation and full core-tail charges and
reserve half the explicit conditioning budget for future arithmetic. They are
NOT runtime-qualified: the higher-order moment/translation implementation does
not exist yet. All runtime_admissible counts therefore remain zero. This census
is useful only for deciding whether that implementation could remove material
work. It never changes the solver, its MAC, tolerances, or image tail decisions.
"""

import math

import numpy as np
import taichi as ti

from source.solvers.vpm.physics.induction.treecode.lbvh import _OwnedFields
from tests.vpm._fmm_source_remainder_prototype import source_derivative_remainder

_ORDERS = (3, 5, 7, 9)


@ti.data_oriented
class SourceOrderFeasibilityCensus:
    def __init__(self, evaluator, *, relative_budget=16 * np.finfo(np.float32).eps):
        if not math.isfinite(relative_budget) or not 0 < relative_budget < 1:
            raise ValueError("require an explicit positive conditioning budget")
        self.evaluator = evaluator
        self.source = evaluator.source
        self.tree = self.source.tree
        if self.source.kernel_name not in {"GAUSSIAN", "WINCKELMANS"}:
            raise ValueError("census core certificates currently cover only Gaussian/Winckelmans")
        self.source_generation = self.source.source_multipole_generation
        if self.source_generation < 1:
            raise ValueError("source multipoles must be freshly prepared")
        self.relative_budget = float(relative_budget)
        internal = int(self.source._active_internal_count[None])
        leaves = int(self.source._active_leaf_count[None])
        nodes = np.concatenate((
            self.source._active_internal_nodes.to_numpy()[:internal],
            self.source._active_leaf_nodes.to_numpy()[:leaves],
        )).astype(np.int32)
        if not len(nodes):
            raise ValueError("source moments/schedule must already be prepared")
        mapping = np.full(self.source.max_nodes, -1, np.int32)
        mapping[nodes] = np.arange(len(nodes), dtype=np.int32)
        self.count = len(nodes)
        self.owner = _OwnedFields()
        try:
            fields = self.owner
            self.nodes = fields.scalar(dtype=ti.i32, shape=self.count)
            self.slot = fields.scalar(dtype=ti.i32, shape=self.source.max_nodes)
            self.radius = fields.scalar(dtype=ti.f64, shape=self.count)
            self.strength = fields.scalar(dtype=ti.f64, shape=self.count)
            self.moments = fields.scalar(dtype=ti.f64, shape=(self.count, len(_ORDERS)))
            self.coverage = fields.scalar(dtype=ti.i64, shape=6)
            self.target_jobs = fields.scalar(dtype=ti.i64, shape=6)
            self.cell_jobs = fields.scalar(dtype=ti.i64, shape=6)
            fields.finalize()
            self.nodes.from_numpy(nodes)
            self.slot.from_numpy(mapping)
            self._metadata()
        except BaseException:
            self.destroy()
            raise

    def destroy(self):
        owner, self.owner = self.owner, None
        if owner is not None:
            owner.destroy()

    @ti.kernel
    def _metadata(self):
        # Compact active-cell metadata only. No N*220 higher-order allocation.
        # Float64 summation is qualification overhead, not a proposed hot path.
        for slot in range(self.count):
            node = self.nodes[slot]
            centre = ti.cast(self.tree.node_centre[node], ti.f64)
            first = self.tree.node_particle_start[node]
            count = self.tree.node_particle_count[node]
            radius, strength = ti.cast(0.0, ti.f64), ti.cast(0.0, ti.f64)
            moments = ti.Vector.zero(ti.f64, 4)
            for index in range(first, first + count):
                particle = self.tree.sorted_indices[index]
                offset = ti.cast(self.tree.position[particle], ti.f64) - centre
                length = offset.norm()
                weight = ti.cast(self.tree.vortex_strength[particle], ti.f64).norm()
                radius = ti.max(radius, length)
                strength += weight
                for order in ti.static(range(4)):
                    moments[order] += weight * length ** (_ORDERS[order] + 1)
            inflation = 1.0 + 64.0 * count * 2.220446049250313e-16
            self.radius[slot] = radius * inflation + 32 * 2.220446049250313e-16 * centre.norm()
            self.strength[slot] = strength * inflation
            for order in ti.static(range(4)):
                self.moments[slot, order] = moments[order] * inflation

    @ti.func
    def _potential_order(self, target: ti.i32, source: ti.i32, image: ti.i32):
        result = -1
        slot = self.slot[source]
        if slot >= 0:
            evaluator = self.evaluator
            target_centre = evaluator._transform(evaluator.tree.node_centre[target], image)
            source_centre = self.tree.node_centre[source]
            radius = self.radius[slot]
            target_radius = ti.cast(evaluator.node_bound_radius[target], ti.f64)
            distance = (ti.cast(target_centre, ti.f64) - ti.cast(source_centre, ti.f64)).norm()
            # Include floating image transformation and rounded-centre scale.
            padding = 16 * 1.1920928955078125e-7 * (
                ti.cast(ti.abs(evaluator.tree.node_centre[target]).sum(), ti.f64)
                + ti.cast(ti.abs(source_centre).sum(), ti.f64)
                + ti.cast(ti.abs(evaluator.image_shift[image]), ti.f64)
                + distance + radius + target_radius
            )
            minimum = distance - target_radius - padding
            maximum = distance + target_radius + radius + padding
            nearest = minimum - radius
            core = ti.cast(self.tree.node_max_radius[source], ti.f64)
            # Singular expansion never substitutes a common-core monopole
            # while inside its regularized tail. This is source-only sigma.
            if nearest > evaluator.target_core_cutoff * core and nearest > 0 and maximum > 0:
                absolute = self.strength[slot]
                q_error = 1e-7 / (4 * math.pi)
                b_error = 3e-7 / (4 * math.pi)
                core_u = absolute * q_error / nearest**2
                core_j = absolute * (ti.static(math.sqrt(2)) * q_error + b_error) / nearest**3
                # The remaining half is RESERVED, not claimed as a proven
                # high-order moment/translation rounding certificate.
                budget_u = 0.5 * self.relative_budget * absolute / (4 * math.pi * maximum**2)
                budget_j = 0.5 * self.relative_budget * absolute / (4 * math.pi * maximum**3)
                q = radius / minimum
                for index in ti.static(range(4)):
                    if result < 0:
                        n = ti.static(_ORDERS[index] + 1)
                        errors = ti.Vector.zero(ti.f64, 2)
                        valid = True
                        for m in ti.static(range(1, 3)):
                            ratio = ti.static(math.sqrt((n + m + 1) * (2 * (n + m) + 1)) / (n + 1))
                            coefficient = ti.static(
                                source_derivative_remainder(_ORDERS[index], m, 1, 0, 1)
                            )
                            if q * ratio < 1:
                                errors[m - 1] = (
                                    coefficient * self.moments[slot, index]
                                    / minimum ** (n + m + 1) / (1 - q * ratio)
                                )
                            else:
                                valid = False
                        if valid and errors[0] + core_u <= budget_u and errors[1] + core_j <= budget_j:
                            result = index
        return result

    @ti.kernel
    def _clear(self):
        for index in range(6):
            self.coverage[index] = 0
            self.target_jobs[index] = 0
            self.cell_jobs[index] = 0

    @ti.kernel
    def _advance(
        self, source_in: ti.template(), target_in: ti.template(), image_in: ti.template(),
        source_out: ti.template(), target_out: ti.template(), image_out: ti.template(), count: ti.i32,
    ):
        evaluator = self.evaluator
        evaluator.frontier_count[None] = 0
        for work in range(count):
            if evaluator.error[None] == 0:
                source, target, image = source_in[work], target_in[work], image_in[work]
                flags = evaluator._admissibility(target, source, image)
                category = -1
                if flags[0]:
                    category = 0  # Existing ALL admission always wins unchanged.
                else:
                    order = self._potential_order(target, source, image)
                    if order >= 0:
                        category = order + 1
                    elif not flags[2] or (
                        self.tree.node_is_leaf[source] != 0
                        and evaluator.tree.node_particle_count[target] <= 32
                    ):
                        category = 5  # Unmodified mixed-subtree or exact path.
                if category >= 0:
                    targets = evaluator.tree.node_particle_count[target]
                    sources = self.tree.node_particle_count[source]
                    ti.atomic_add(self.cell_jobs[category], 1)
                    ti.atomic_add(self.target_jobs[category], ti.cast(targets, ti.i64))
                    ti.atomic_add(self.coverage[category], ti.cast(targets, ti.i64) * ti.cast(sources, ti.i64))
                else:
                    destination = ti.atomic_add(evaluator.frontier_count[None], 2)
                    if destination + 1 >= evaluator.max_pairs:
                        evaluator.error[None] = 1
                    else:
                        split_target = self.tree.node_is_leaf[source] != 0
                        if not split_target and evaluator.tree.node_particle_count[target] > 1:
                            split_target = (
                                evaluator.source_separation * self.tree.node_half_size[source]
                                < evaluator.tree.node_half_size[target]
                            )
                        for side in ti.static(range(2)):
                            source_out[destination + side] = source
                            target_out[destination + side] = target
                            image_out[destination + side] = image
                        if split_target:
                            target_out[destination] = evaluator.tree.node_left[target]
                            target_out[destination + 1] = evaluator.tree.node_right[target]
                        else:
                            source_out[destination] = self.tree.node_left[source]
                            source_out[destination + 1] = self.tree.node_right[source]

    def run(self, image_count):
        evaluator = self.evaluator
        if self.source.source_multipole_generation != self.source_generation:
            raise RuntimeError("source changed after census metadata preparation")
        leaves = int(evaluator.leaf_count[None])
        count = leaves * image_count
        if count > evaluator.max_pairs:
            raise RuntimeError("census initial frontier exceeds bounded workspace")
        self._clear()
        evaluator.error[None] = 0
        evaluator._initialise_frontier(0, leaves, 0, image_count, -1)
        current = (evaluator.frontier_source_a, evaluator.frontier_target_a, evaluator.frontier_image_a)
        following = (evaluator.frontier_source_b, evaluator.frontier_target_b, evaluator.frontier_image_b)
        levels = 0
        while count:
            self._advance(*current, *following, count)
            if int(evaluator.error[None]):
                raise RuntimeError("census frontier exceeds bounded workspace; no certificate published")
            count = int(evaluator.frontier_count[None])
            current, following = following, current
            levels += 1
            if levels > 256:
                raise RuntimeError("census traversal exceeded structural depth guard")
        coverage = self.coverage.to_numpy()
        target_jobs = self.target_jobs.to_numpy()
        basis_sizes = [math.comb(order + 3, 3) for order in _ORDERS]
        expected = int(self.tree.n_particles_total[None]) * evaluator._prepared_count * image_count
        if int(coverage.sum()) != expected:
            raise AssertionError("census source/target/image coverage is not complete and disjoint")
        return {
            "runtime_admissible_interactions": 0,
            "qualification": "opportunity only; remaining arithmetic budget is not certified",
            "relative_conditioning_budget": self.relative_budget,
            "remaining_arithmetic_budget_fraction": 0.5,
            "source_active_cells": self.count,
            "compact_metadata_bytes": self.count * (4 + 8 * 6) + self.source.max_nodes * 4,
            "categories": ["legacy_all", "potential_p3", "potential_p5", "potential_p7", "potential_p9", "unchanged_near"],
            "cell_jobs": self.cell_jobs.to_numpy().tolist(),
            "target_jobs": target_jobs.tolist(),
            "potential_m2p_basis_sizes": basis_sizes,
            "potential_m2p_coefficient_target_products": [
                int(target_jobs[index + 1]) * basis_sizes[index] for index in range(4)
            ],
            "cost_warning": (
                "Covered exact particle pairs are not legacy treecode operations. "
                "Coefficient-target products are a cost proxy, not measured speedup."
            ),
            "physical_source_target_coverage": coverage.tolist(),
            "expected_coverage": expected,
            "levels": levels,
        }
