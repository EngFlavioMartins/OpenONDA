"""Unwired target-Taylor admission census; computes no velocity or gradient.

All source decisions and bounded work partitions come from the frozen p7
operator. Only its already-accepted, nonlocal monopole packets are inspected.
The hypothetical p12/.1 policy is a cost question, not an adopted numerical
method: these counters prove neither floating-point accuracy nor convergence.
"""

import math

import numpy as np
import taichi as ti

from tests.vpm._fmm_flattened_near_prototype import (
    FMMTargetEvaluator,
    assert_frozen_numerical_imports,
    target_module,
)


def taylor_coefficient(order, derivative, ratio):
    """Dimensionless exact-arithmetic absolute monopole remainder majorant.

    Multiply by |Gamma|/(4*pi*R**(derivative+1)). This bound excludes
    regularization and all coefficient/translation/evaluation roundoff.
    """
    if order < derivative or derivative not in (1, 2) or not 0 <= ratio < 1:
        raise ValueError("invalid Taylor remainder arguments")
    n = order - derivative + 1
    coefficient = math.sqrt(math.factorial(2 * (order + 1)) / 2 ** (order + 1)) / math.factorial(n)
    growth = math.sqrt((order + 2) * (2 * order + 3)) / (n + 1)
    if growth * ratio >= 1:
        return math.inf
    return coefficient * ratio**n / (1 - growth * ratio)


_P12_CONSTANTS = tuple(
    (
        math.sqrt(math.factorial(26) / 2**13) / math.factorial(13 - d),
        math.sqrt(14 * 27) / (14 - d),
    )
    for d in (1, 2)
)
_COUNTERS = (
    "old_monopole_packets", "old_monopole_target_terms",
    "promotable_packets", "promotable_target_terms",
    "promoted_local_nodes_per_successful_batch", "promoted_local_target_evaluations_per_batch",
    "blocked_by_regularization_tail", "blocked_by_target_ratio",
    "partition_classification_errors",
    "promoted_nodes_already_have_p7_locals", "promoted_target_evals_already_have_p7_locals",
    "old_local_nodes_per_successful_batch", "old_local_target_evaluations_per_batch",
)


def local_evaluation_costs(old_targets, promoted_targets, overlap_targets):
    """Coefficient visits, including old-only cells in the global-order case."""
    if not 0 <= overlap_targets <= min(old_targets, promoted_targets):
        raise ValueError("invalid old/promoted local overlap")
    return {
        "p12_selective_additional_local_target_coefficient_visits": (
            455 * promoted_targets - 120 * overlap_targets
        ),
        "p12_global_additional_local_target_coefficient_visits": (
            335 * old_targets + 455 * (promoted_targets - overlap_targets)
        ),
    }


@ti.data_oriented
class TargetOrderCensusEvaluator(FMMTargetEvaluator):
    def __init__(self, *args, **kwargs):
        self._census_owner = None
        super().__init__(*args, **kwargs)
        try:
            fields = target_module._OwnedFields()
            self._census_owner = fields
            self.census_count = fields.scalar(dtype=ti.i64, shape=len(_COUNTERS))
            self.census_bound = fields.scalar(dtype=ti.f64, shape=2)
            self.census_local_seen = fields.scalar(dtype=ti.i32, shape=2 * self.max_targets)
            fields.finalize()
        except BaseException:
            self.destroy()
            raise

    def destroy(self):
        owner, self._census_owner = self._census_owner, None
        try:
            if owner is not None:
                owner.destroy()
        finally:
            super().destroy()

    @ti.func
    def _hypothetical_target_gate(self, target: ti.i32, source: ti.i32, image: ti.i32):
        transformed = self._transform(self.tree.node_centre[target], image)
        displacement = transformed - self.source.tree.node_com[source]
        distance = displacement.norm()
        radius = self.node_bound_radius[target]
        extent = self.source.tree.node_half_size[source] + (
            self.source.tree.node_com[source] - self.source.tree.node_centre[source]
        ).norm()
        core = self.source.tree.node_max_radius[source]
        # Preserve the current production coordinate-scale padding and tail
        # test verbatim; the mathematical ratio uses f64 centre separation.
        padding = 8 * 1.1920928955078125e-7 * (
            distance + radius + extent + core
            + ti.abs(self.tree.node_centre[target]).sum()
            + ti.abs(self.source.tree.node_com[source]).sum()
            + ti.abs(self.image_shift[image])
        )
        minimum_distance = ti.max(0.0, distance - radius - padding)
        outside_tail = minimum_distance > extent + self.target_core_cutoff * core
        separation = (
            ti.cast(transformed, ti.f64) - ti.cast(self.source.tree.node_com[source], ti.f64)
        ).norm()
        ratio = ti.cast(1.0, ti.f64)
        if separation > 0.0:
            ratio = (ti.cast(radius, ti.f64) + ti.cast(padding, ti.f64)) / separation
        return outside_tail, ratio, separation

    @ti.kernel
    def _record_target_candidates(self, count: ti.i32, node_count: ti.i32):
        for node in range(node_count):
            if self.local_present[node]:
                ti.atomic_add(self.census_count[11], 1)
                ti.atomic_add(self.census_count[12], ti.cast(self.tree.node_particle_count[node], ti.i64))
        for pair in range(count):
            if not self.near_legacy[pair] and self.near_source[pair] < 0:
                target = self.near_target[pair]
                source = -self.near_source[pair] - 1
                image = self.near_image[pair]
                particles = ti.cast(self.tree.node_particle_count[target], ti.i64)
                ti.atomic_add(self.census_count[0], 1)
                ti.atomic_add(self.census_count[1], particles)
                accepted = self._admissibility(target, source, image)
                if not accepted[0] or accepted[1]:
                    ti.atomic_add(self.census_count[8], 1)
                outside_tail, ratio, distance = self._hypothetical_target_gate(target, source, image)
                if not outside_tail:
                    ti.atomic_add(self.census_count[6], 1)
                elif ratio <= 0.1:
                    ti.atomic_add(self.census_count[2], 1)
                    ti.atomic_add(self.census_count[3], particles)
                    previous = ti.atomic_or(self.census_local_seen[target], 1)
                    if previous == 0:
                        ti.atomic_add(self.census_count[4], 1)
                        ti.atomic_add(self.census_count[5], particles)
                        if self.local_present[target]:
                            ti.atomic_add(self.census_count[9], 1)
                            ti.atomic_add(self.census_count[10], particles)
                    strength = ti.cast(self.source.tree.node_net_vortex_strength[source], ti.f64).norm()
                    for index in ti.static(range(2)):
                        derivative = ti.static(index + 1)
                        n = ti.static(13 - derivative)
                        coefficient, growth = ti.static(_P12_CONSTANTS[index])
                        value = coefficient * ratio**n / (1.0 - growth * ratio)
                        # Summed over all covered targets: a field-L1 majorant
                        # for these promoted packets, NOT a relative-error or
                        # per-target cancellation-aware accuracy certificate.
                        error = strength * value / (4 * math.pi * distance**(derivative + 1))
                        ti.atomic_add(self.census_bound[index], ti.cast(particles, ti.f64) * error)
                else:
                    ti.atomic_add(self.census_count[7], 1)

    def census_image_block(self, images):
        """Walk the original bounded partition; never compute/publish fields."""
        assert_frozen_numerical_imports()
        count = self._prepared_count
        images = tuple(images)
        if count < 1 or not images or len(images) > self.max_images:
            raise ValueError("census requires prepared targets and a bounded nonempty image block")
        shifts = np.zeros(self.max_images, np.float32)
        odd = np.zeros(self.max_images, np.int32)
        for index, (shift, reflected) in enumerate(images):
            if not math.isfinite(shift):
                raise ValueError("image shift must be finite")
            shifts[index], odd[index] = shift, bool(reflected)
        self.image_shift.from_numpy(shifts)
        self.image_odd.from_numpy(odd)
        self.image_count[None] = len(images)
        self.block_mode[None] = 1
        self.census_count.fill(0)
        self.census_bound.fill(0)
        leaves = int(self.leaf_count[None])
        jobs = [(0, leaves, 0, len(images), -1)]
        report = {
            "target_count": count, "image_count": len(images), "successful_batches": 0,
            "discarded_capacity_attempts": 0, "old_m2l_packets": 0,
            "legacy_subtree_target_jobs_unchanged": 0, "direct_particle_terms_unchanged": 0,
            "peak_hypothetical_m2l_packets": 0, "pair_capacity": self.max_pairs,
            "hypothetical_m2l_capacity_exceeding_batches": 0,
        }
        while jobs:
            first_leaf, leaves, first_image, image_count, root = jobs.pop()
            self._initialise(2 * count - 1)
            self._walk_sources(first_leaf, leaves, first_image, image_count, root)
            error = int(self.error[None])
            if error == 1:
                report["discarded_capacity_attempts"] += 1
                if image_count > 1:
                    split = image_count // 2
                    jobs.extend([
                        (first_leaf, leaves, first_image + split, image_count - split, root),
                        (first_leaf, leaves, first_image, split, root),
                    ])
                elif leaves > 1:
                    split = leaves // 2
                    jobs.extend([
                        (first_leaf + split, leaves - split, first_image, image_count, -1),
                        (first_leaf, split, first_image, image_count, -1),
                    ])
                else:
                    if root < 0:
                        root = int(self.leaf_nodes[first_leaf])
                    if int(self.tree.node_particle_count[root]) <= 1:
                        raise RuntimeError("census hit an irreducible original interaction capacity")
                    jobs.extend([
                        (0, 1, first_image, image_count, int(self.tree.node_right[root])),
                        (0, 1, first_image, image_count, int(self.tree.node_left[root])),
                    ])
                continue
            if error:
                raise RuntimeError(f"original target traversal failed: {error}")
            before = int(self.census_count[2])
            self.census_local_seen.fill(0)
            self._record_target_candidates(int(self.near_pair_count[None]), 2 * count - 1)
            added = int(self.census_count[2]) - before
            old_m2l = int(self.m2l_count[None])
            report["successful_batches"] += 1
            report["old_m2l_packets"] += old_m2l
            report["legacy_subtree_target_jobs_unchanged"] += int(self.legacy_subtree_work[None])
            report["direct_particle_terms_unchanged"] += int(self.direct_work[None])
            report["peak_hypothetical_m2l_packets"] = max(
                report["peak_hypothetical_m2l_packets"], old_m2l + added
            )
            report["hypothetical_m2l_capacity_exceeding_batches"] += int(old_m2l + added > self.max_pairs)
        report.update(zip(_COUNTERS, self.census_count.to_numpy().tolist(), strict=True))
        if report["partition_classification_errors"]:
            raise AssertionError("census packet does not match frozen ALL/nonlocal classification")
        report.update(
            p7_old_translation_vector_coefficients=120 * report["old_m2l_packets"],
            p12_added_translation_vector_coefficients=455 * report["promotable_packets"],
            p12_global_order_extra_old_translation_coefficients=335 * report["old_m2l_packets"],
            p12_promoted_local_target_coefficient_visits=455 * report["promoted_local_target_evaluations_per_batch"],
            exact_arithmetic_promoted_field_l1_taylor_bounds=self.census_bound.to_numpy().tolist(),
            source_partition_changed=False, fields_evaluated=False, outputs_published=False,
            floating_point_and_total_error_certificate=False, tail_convergence_tested=False,
        )
        report.update(local_evaluation_costs(
            report["old_local_target_evaluations_per_batch"],
            report["promoted_local_target_evaluations_per_batch"],
            report["promoted_target_evals_already_have_p7_locals"],
        ))
        return report
