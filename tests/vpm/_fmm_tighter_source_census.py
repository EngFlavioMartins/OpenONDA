"""Frozen-source, field-free comparison of historical and tighter tail bounds.

Both policies share the SAME active-cell metadata and 16eps conditioning-scale
budget, with half reserved for still-unproved arithmetic. The tighter policy
uses the analytic rank-one derivative sums and distance-dependent kernel-tail
envelopes. This is an opportunity census, not runtime source admission or an
image convergence test. Its A/d**k budget is the HISTORICAL census's absolute
strength scale, not a claim to equal the actual orientation-dependent direct
conditioning sum. No old numerical outputs or artifacts are modified.
"""

# ruff: noqa: I001 -- Admit immutable numerical source before all numerical imports.
from tests.vpm._fmm_flattened_near_prototype import assert_frozen_numerical_imports

import math

import numpy as np
import taichi as ti

from source.solvers.vpm.physics.induction.treecode.lbvh import _OwnedFields
from tests.vpm._fmm_legacy_descendant_work import LegacyDescendantWorkCounter
from tests.vpm._fmm_order_census_prototype import SourceOrderFeasibilityCensus, _ORDERS

assert_frozen_numerical_imports()


@ti.data_oriented
class TighterSourceOrderCensus(SourceOrderFeasibilityCensus):
    _historical_potential_order = SourceOrderFeasibilityCensus._potential_order

    def __init__(self, evaluator, *, target_stride=1, target_offset=0):
        self.work_counter = self._policy_owner = None
        super().__init__(evaluator)
        try:
            self._policy_owner = _OwnedFields()
            self.policy = self._policy_owner.scalar(dtype=ti.i32, shape=())
            # ti.cast(Python_float, f64) first rounds its literal at default_fp
            # in Taichi1.7.4. Host-loaded f64 constants preserve this reference
            # census's intended precision without changing the solver runtime.
            self.constants = self._policy_owner.scalar(dtype=ti.f64, shape=3)
            self._policy_owner.finalize()
            self.constants.from_numpy(np.array([1 / (4 * math.pi), math.pi, math.pi**1.5], np.float64))
            self.work_counter = LegacyDescendantWorkCounter(
                evaluator, target_stride=target_stride, target_offset=target_offset
            )
        except BaseException:
            self.destroy()
            raise

    def destroy(self):
        counter, self.work_counter = self.work_counter, None
        if counter is not None:
            counter.destroy()
        owner, self._policy_owner = self._policy_owner, None
        if owner is not None:
            owner.destroy()
        super().destroy()

    @ti.func
    def _finite_nonnegative(self, value: ti.f64):
        return value >= 0 and not ti.math.isnan(value) and not ti.math.isinf(value)

    @ti.func
    def _actual_core_defect(self, nearest: ti.f64, maximum: ti.f64, absolute: ti.f64):
        c = self.constants[0]
        rho = nearest / maximum
        tail, density = ti.cast(0.0, ti.f64), ti.cast(0.0, ti.f64)
        if ti.static(self.source.kernel_name == "GAUSSIAN"):
            tail = c * ti.min(1.0, ti.exp(-rho * rho) * (2 * rho + 1 / rho) / ti.sqrt(self.constants[1]))
            sigma = ti.min(maximum, nearest * ti.sqrt(ti.cast(2.0, ti.f64) / 3))
            density = ti.exp(-(nearest / sigma)**2) / (self.constants[2] * sigma**3)
        else:
            if rho >= 1:
                t = 1 / rho**2
                root = ti.sqrt(1 + t)
                tail = c * t**2 * (root**2 + root + 2 - 0.5 / (root + 1)) / ((root + 1) * root**5)
            else:
                root = ti.sqrt(1 + rho**2)
                tail = c * (1 - rho**3 * (rho**2 + 2.5) / root**5)
            sigma = ti.min(maximum, 2 * nearest / ti.sqrt(ti.cast(3.0, ti.f64)))
            base = nearest**2 + sigma**2
            density = 7.5 * c * sigma**4 / (base**3 * ti.sqrt(base))
        return ti.Vector([
            absolute * tail / nearest**2,
            ti.sqrt(ti.cast(2.0, ti.f64)) * absolute * (2 * tail / nearest**3 + density),
        ])

    @ti.func
    def _potential_order(self, target: ti.i32, source: ti.i32, image: ti.i32):
        result = -1
        if self.policy[None] == 0:
            result = self._historical_potential_order(target, source, image)
        else:
            slot = self.slot[source]
            if slot >= 0:
                evaluator = self.evaluator
                target_centre = evaluator._transform(evaluator.tree.node_centre[target], image)
                source_centre = self.tree.node_centre[source]
                radius = self.radius[slot]
                target_radius = ti.cast(evaluator.node_bound_radius[target], ti.f64)
                distance = (ti.cast(target_centre, ti.f64) - ti.cast(source_centre, ti.f64)).norm()
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
                if (nearest > 0 and maximum > 0 and core > 0
                        and self._finite_nonnegative(distance)
                        and self._finite_nonnegative(radius)
                        and self._finite_nonnegative(target_radius)
                        and self._finite_nonnegative(padding)
                        and self._finite_nonnegative(minimum)
                        and self._finite_nonnegative(maximum)
                        and self._finite_nonnegative(core)):
                    absolute = self.strength[slot]
                    core_error = self._actual_core_defect(nearest, core, absolute)
                    budget_u = 0.5 * self.relative_budget * absolute * self.constants[0] / maximum**2
                    budget_j = 0.5 * self.relative_budget * absolute * self.constants[0] / maximum**3
                    q = radius / minimum
                    inverse = 1 / (1 - q)
                    for index in ti.static(range(4)):
                        if result < 0:
                            n = ti.static(_ORDERS[index] + 1)
                            series_u = ti.sqrt(ti.cast(2.0, ti.f64)) * ((n + 1) * inverse + q * inverse**2)
                            series_j = 2 * (
                                (n + 1) * (n + 2) * inverse
                                + (2 * n + 3) * q * inverse**2
                                + q * (1 + q) * inverse**3
                            )
                            error_u = self.moments[slot, index] * series_u * self.constants[0] / minimum**(n + 2)
                            error_j = self.moments[slot, index] * series_j * self.constants[0] / minimum**(n + 3)
                            # An inf<=inf comparison is not a certificate.
                            # Explicitly decline nonfinite or underflowed budgets;
                            # even valid f64 evaluations are NOT interval bounds.
                            if (self._finite_nonnegative(absolute)
                                    and self._finite_nonnegative(self.moments[slot, index])
                                    and self._finite_nonnegative(core_error[0])
                                    and self._finite_nonnegative(core_error[1])
                                    and self._finite_nonnegative(budget_u) and budget_u > 0
                                    and self._finite_nonnegative(budget_j) and budget_j > 0
                                    and self._finite_nonnegative(error_u)
                                    and self._finite_nonnegative(error_j)
                                    and error_u + core_error[0] <= budget_u
                                    and error_j + core_error[1] <= budget_j):
                                result = index
        return result

    @ti.kernel
    def _advance(
        self, source_in: ti.template(), target_in: ti.template(), image_in: ti.template(),
        source_out: ti.template(), target_out: ti.template(), image_out: ti.template(), count: ti.i32,
    ):
        # Historical coverage traversal is copied unchanged; only the terminal
        # instrumentation hook below is new. No interaction fields are solved.
        evaluator = self.evaluator
        evaluator.frontier_count[None] = 0
        for work in range(count):
            if evaluator.error[None] == 0:
                source, target, image = source_in[work], target_in[work], image_in[work]
                flags = evaluator._admissibility(target, source, image)
                category = -1
                if flags[0]:
                    category = 0
                else:
                    order = self._potential_order(target, source, image)
                    if order >= 0:
                        category = order + 1
                    elif not flags[2] or (
                        self.tree.node_is_leaf[source] != 0
                        and evaluator.tree.node_particle_count[target] <= 32
                    ):
                        category = 5
                if category >= 0:
                    targets = evaluator.tree.node_particle_count[target]
                    sources = self.tree.node_particle_count[source]
                    ti.atomic_add(self.cell_jobs[category], 1)
                    ti.atomic_add(self.target_jobs[category], ti.cast(targets, ti.i64))
                    ti.atomic_add(self.coverage[category], ti.cast(targets, ti.i64) * ti.cast(sources, ti.i64))
                    if 1 <= category <= 4:
                        self.work_counter.record(target, source, image, category - 1)
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

    def run(self, image_count, *, policy="tight"):
        if policy not in ("historical", "tight"):
            raise ValueError("unknown source error-bound policy")
        self.policy[None] = int(policy == "tight")
        self.work_counter.reset()
        result = super().run(image_count)
        result.update(
            source_bound_policy=policy,
            metadata_shared_between_policies=True,
            legacy_descendant_work=self.work_counter.report(),
            source_bound=("rank-one geometric derivatives" if policy == "tight" else "historical Frobenius tensor majorant"),
            core_bound=("distance-dependent G/W singular defect" if policy == "tight" else "historical fixed 1e-7 radial coefficient bounds"),
            image_series_convergence_certified=False,
            arithmetic_and_local_errors_certified=False,
            conditioning_scale_is_actual_direct_pair_norm=False,
        )
        return result
