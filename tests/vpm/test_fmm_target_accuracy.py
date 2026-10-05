"""Adversarial target-local accuracy against pointwise field evaluation.

These cases deliberately make the pointwise monopole accurate, so its error
cannot hide a new off-centre local-polynomial truncation error. The acceptance
envelope is the pointwise error plus a float32 summation allowance, not the
looser general FMM qualification ceiling.
"""

import numpy as np
import pytest
import taichi as ti

from source.solvers.vpm.kernels.base import make_vortex_kernel
from source.solvers.vpm.physics.induction.fmm.targets import FMMTargetEvaluator
from tests.vpm.test_fmm_device import _DeviceFMMHarness


def _adversarial_cases(kernel_name):
    rng = np.random.default_rng(981031)
    line = np.zeros((24, 3), dtype=np.float32)
    line[:, 2] = np.linspace(5.1, 7.1, len(line), dtype=np.float32)
    yield (
        "one_source_wide_offcentre_leaf",
        np.zeros((1, 3), dtype=np.float32),
        np.array([[0.0, 1.0, 0.0]], dtype=np.float32),
        np.array([0.002], dtype=np.float32),
        line,
    )
    # Same dimensionless operator at small physical units. The true order-8
    # derivative is finite here, but unnormalised r^-17 intermediates are not.
    for length_scale in (np.float32(1e-4), np.float32(1e-5)):
        yield (
            f"small_units_point_source_{length_scale:g}",
            np.zeros((1, 3), dtype=np.float32),
            np.array([[0.0, length_scale**2, 0.0]], dtype=np.float32),
            np.array([0.002 * length_scale], dtype=np.float32),
            line * length_scale,
        )

    count = 128  # Deliberately bypass a small-source direct fallback.
    position = rng.uniform(-0.001, 0.001, (count, 3)).astype(np.float32)
    strength = np.zeros_like(position)
    strength[:, 1] = 1.0 / count
    core = rng.uniform(0.002, 0.008, count).astype(np.float32)
    yield "pointlike_large_cloud", position, strength, core, line

    cancelled_position = position.copy()
    cancelled_position[: count // 2, 2] -= 0.08
    cancelled_position[count // 2 :, 2] += 0.08
    cancelled_strength = strength.copy()
    cancelled_strength[count // 2 :] *= -1
    yield (
        "cancelled_large_cloud",
        cancelled_position,
        cancelled_strength,
        core,
        line,
    )

    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1
    yield (
        "rotated_cancelled_cloud",
        (cancelled_position @ rotation.T).astype(np.float32),
        (cancelled_strength @ rotation.T).astype(np.float32),
        core,
        (line @ rotation.T).astype(np.float32),
    )

    # Every aligned binary source block has zero net strength; moments of
    # degree zero through three vanish, while the fourth moment does not.
    # The pointwise monopole traversal opens these cancelled cells. A geometric
    # p=3 source MAC alone must not silently discard the surviving field.
    fourth_order_position = np.zeros((128, 3), dtype=np.float32)
    fourth_order_position[:, 2] = np.repeat(np.arange(-15, 16, 2) / 16, 8)
    fourth_order_strength = np.zeros_like(fourth_order_position)
    fourth_order_strength[:, 1] = np.repeat(
        [(-1.0) ** index.bit_count() / 8 for index in range(16)], 8
    )
    narrow_targets = np.zeros_like(line)
    narrow_targets[:, 2] = np.linspace(5.699, 5.701, len(line), dtype=np.float32)
    yield (
        "cancelled_zero_through_third_moment",
        fourth_order_position,
        fourth_order_strength,
        np.full(128, 0.002, dtype=np.float32),
        narrow_targets,
    )

    mixed_targets = line.copy()
    mixed_targets[:8] = position[:8]
    yield "mixed_coincident_and_far_targets", position, strength, core, mixed_targets

    # A pointlike common-core cloud is evaluated as the exact regularized
    # monopole by pointwise traversal. A singular expansion must not substitute
    # the looser self-FMM 1e-5 tail criterion for that target-field accuracy.
    tail_radius = np.float32(0.2)
    loose_cutoff = max(make_vortex_kernel(kernel_name).dimensionless_tail_cutoffs(1e-5, 1e-5))
    tail_targets = np.zeros_like(line)
    tail_targets[:, 2] = np.linspace(1.001, 1.0011, len(line)) * loose_cutoff * tail_radius
    yield (
        "common_core_at_self_tail_boundary",
        np.zeros((128, 3), dtype=np.float32),
        strength,
        np.full(128, tail_radius, dtype=np.float32),
        tail_targets,
    )


def _exact_fields_and_roundoff(kernel_name, position, strength, core, targets):
    kernel = make_vortex_kernel(kernel_name)
    displacement = targets[:, None, :].astype(np.float64) - position[None, :, :].astype(np.float64)
    strengths = strength[None, :, :].astype(np.float64)
    radii = core[None, :].astype(np.float64)
    # Passing the same source radius as both arguments implements source-only
    # core regularisation, not the particle-stage mean of source/target cores.
    pair_fields = (
        kernel.velocity_pair(displacement, strengths, radii, radii),
        kernel.gradient_pair(displacement, strengths, radii, radii),
    )
    exact = []
    roundoff = []
    for pairs in pair_fields:
        field = pairs.sum(axis=1, dtype=np.float64)
        absolute_sum = np.abs(pairs).sum(axis=1, dtype=np.float64)
        exact.append(field)
        # Conditioning-aware allowance remains absolute when net fields
        # cancel; it is approximately two ppm for a single noncancelled term.
        roundoff.append(
            16
            * np.finfo(np.float32).eps
            * np.linalg.norm(absolute_sum.reshape(len(targets), -1), axis=1)
        )
    return exact, roundoff


@pytest.mark.parametrize("kernel_name", ("GAUSSIAN", "WINCKELMANS"))
def test_target_local_error_preserves_pointwise_envelope_at_cell_extremes(kernel_name):
    harness = _DeviceFMMHarness(capacity=128, kernel_name=kernel_name)
    harness.induction._ensure_workspace(128)
    workspace = harness.induction.workspace
    target_count = 24
    query = ti.Vector.field(3, ti.f32, shape=target_count)
    previous_velocity = ti.Vector.field(3, ti.f32, shape=target_count)
    previous_gradient = ti.Matrix.field(3, 3, ti.f32, shape=target_count)
    new_velocity = ti.Vector.field(3, ti.f32, shape=target_count)
    new_gradient = ti.Matrix.field(3, 3, ti.f32, shape=target_count)
    zero = harness.induction.physics._zero_velocity
    backend = FMMTargetEvaluator(workspace, target_count)
    failures = []
    try:
        for label, position, strength, core, targets in _adversarial_cases(kernel_name):
            # Includes source-prefix shrink/growth and same-field mutations.
            harness.evaluate(position, strength, core)
            query.from_numpy(targets)
            workspace.tree.compute_external_target_fields(
                query,
                previous_velocity,
                previous_gradient,
                zero,
                target_count,
                workspace.target_batch_capacity,
                write_velocity=True,
                write_gradient=True,
            )
            backend.evaluate(query, new_velocity, new_gradient, target_count, zero)
            exact, roundoff = _exact_fields_and_roundoff(
                kernel_name, position, strength, core, targets
            )
            for quantity, new, old, reference, allowance in zip(
                ("velocity", "Jacobian"),
                (new_velocity.to_numpy(), new_gradient.to_numpy()),
                (previous_velocity.to_numpy(), previous_gradient.to_numpy()),
                exact,
                roundoff,
                strict=True,
            ):
                new_error = np.linalg.norm((new - reference).reshape(target_count, -1), axis=1)
                old_error = np.linalg.norm((old - reference).reshape(target_count, -1), axis=1)
                norm = max(np.linalg.norm(reference), np.finfo(np.float64).tiny)
                metrics = {
                    "case": label,
                    "quantity": quantity,
                    "new_relative_l2": float(np.linalg.norm(new_error) / norm),
                    "old_relative_l2": float(np.linalg.norm(old_error) / norm),
                    "roundoff_relative_l2": float(np.linalg.norm(allowance) / norm),
                    "m2l_pairs": backend.last_diagnostics["m2l_pairs"],
                    "target_cells": backend.last_diagnostics["target_cells"],
                }
                checks = (
                    np.linalg.norm(new_error)
                    <= np.linalg.norm(old_error) + np.linalg.norm(allowance),
                    np.percentile(new_error, 95)
                    <= np.percentile(old_error, 95) + np.percentile(allowance, 95),
                    np.max(new_error) <= np.max(old_error) + np.max(allowance),
                )
                if not all(checks):
                    failures.append(metrics)
    finally:
        backend.destroy()
    assert not failures, failures
