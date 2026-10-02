"""Source-to-local arbitrary-target qualification against exact blob sums."""

import numpy as np
import pytest
import taichi as ti

from source.solvers.vpm.kernels.base import make_vortex_kernel
from source.solvers.vpm.physics.base import PhysicsBase
from source.solvers.vpm.physics.induction.fmm import FMMInduction
from source.solvers.vpm.physics.induction.fmm.targets import (
    _DERIVATIVE_INDICES,
    FMMTargetEvaluator,
    TargetBlockNotWorthwhile,
)
from source.solvers.vpm.physics.induction.slip_slab import SlipSlabInduction
from tests.vpm.test_fmm_device import _DeviceFMMHarness
from tests.vpm.test_fmm_target_accuracy import _exact_fields_and_roundoff


@ti.kernel
def _far_axis_derivative(backend: ti.template(), output: ti.template(), index: ti.i32):
    output[None] = backend._inverse_r_derivative(ti.Vector([0.0, 0.0, 5000.0]), index)


def test_highest_order_derivative_avoids_far_image_intermediate_underflow():
    import math

    harness = _DeviceFMMHarness(capacity=1)
    backend = FMMTargetEvaluator(harness.induction.workspace, 1)
    output = ti.field(ti.f32, shape=())
    order = max(sum(alpha) for alpha in _DERIVATIVE_INDICES)
    _far_axis_derivative(backend, output, _DERIVATIVE_INDICES.index((0, 0, order)))
    expected = (-1)**order * math.factorial(order) / (4 * math.pi * 5000.0**(order + 1))
    assert abs(float(output[None])) > 0.0
    np.testing.assert_allclose(float(output[None]), expected, rtol=3e-5, atol=0.0)
    backend.destroy()


@ti.kernel
def _angular_derivative_oracle(backend: ti.template(), output: ti.template(), direction: ti.template()):
    for index in range(len(_DERIVATIVE_INDICES)):
        output[index] = backend._angular_derivative(direction[None], index)


def test_cartesian_recurrence_matches_independent_contraction_table():
    import math

    harness = _DeviceFMMHarness(capacity=1)
    harness.evaluate(
        np.zeros((1, 3), np.float32),
        np.array([[0.0, 0.01, 0.0]], np.float32),
        np.array([0.1], np.float32),
    )
    direction = ti.Vector.field(3, ti.f32, shape=())
    oracle = ti.field(ti.f32, shape=len(_DERIVATIVE_INDICES))
    for vector in ([0.0, 0.0, 1.0], [1, 2, 3], [-2, 0.1, 0.7]):
        unit = np.asarray(vector, dtype=np.float32)
        unit /= np.linalg.norm(unit)
        targets = np.tile(10 * unit, (2, 1))
        targets[:, 0] += np.array([-0.001, 0.001], dtype=np.float32)
        backend, _, _ = _query(harness, targets)
        direction[None] = unit
        _angular_derivative_oracle(backend, oracle, direction)
        scale = np.array(
            [math.prod(range(1, 2 * sum(alpha), 2)) / (4 * math.pi) for alpha in _DERIVATIVE_INDICES]
        )
        error = np.abs(backend.derivatives.to_numpy()[0] - oracle.to_numpy())
        assert np.max(error / scale) < 3e-5
        backend.destroy()


def test_bounded_block_decline_does_not_publish_any_output():
    rng = np.random.default_rng(51)
    harness = _DeviceFMMHarness(capacity=64)
    position = rng.normal(0, 1, (64, 3)).astype(np.float32)
    harness.evaluate(position, rng.normal(0, 0.01, (64, 3)).astype(np.float32), np.full(64, 0.03, np.float32))
    query = ti.Vector.field(3, ti.f32, shape=64)
    velocity = ti.Vector.field(3, ti.f32, shape=64)
    gradient = ti.Matrix.field(3, 3, ti.f32, shape=64)
    query.from_numpy(position)
    velocity.fill(17.0)
    gradient.fill(19.0)
    backend = FMMTargetEvaluator(harness.induction.workspace, 64, max_pairs=1)
    backend.prepare_targets(query, 64)
    with pytest.raises(TargetBlockNotWorthwhile):
        backend.evaluate_image_block([(0.0, False)], velocity, gradient, harness.induction.physics._zero_velocity)
    np.testing.assert_array_equal(velocity.to_numpy(), 17.0)
    np.testing.assert_array_equal(gradient.to_numpy(), 19.0)
    assert backend.last_diagnostics["m2l_pairs"] <= 128
    backend.destroy()


def test_streamed_image_subblocks_match_single_batch_without_partial_publication():
    rng = np.random.default_rng(775)
    harness = _DeviceFMMHarness(capacity=16)
    position = rng.normal(0, 0.08, (16, 3)).astype(np.float32)
    harness.evaluate(
        position, rng.normal(0, 0.01, (16, 3)).astype(np.float32), np.full(16, 0.03, np.float32)
    )
    query = ti.Vector.field(3, ti.f32, shape=7)
    query.from_numpy(rng.normal(0, 0.2, (7, 3)).astype(np.float32))
    results = []
    # A true source-tree split needs a two-child frontier; use a small but
    # feasible budget here. The separate decline test covers capacity one.
    for capacity in (4096, 16):
        backend = FMMTargetEvaluator(harness.induction.workspace, 7, max_pairs=capacity)
        velocity = ti.Vector.field(3, ti.f32, shape=7)
        gradient = ti.Matrix.field(3, 3, ti.f32, shape=7)
        backend.prepare_targets(query, 7)
        backend.evaluate_image_block(
            [(0.0, False), (1.0, True), (7.0, False)],
            velocity, gradient, harness.induction.physics._zero_velocity,
        )
        results.append((velocity.to_numpy(), gradient.to_numpy()))
        if capacity == 16:
            assert backend.last_diagnostics["work_batches"] > 1
            assert backend.last_diagnostics["peak_stored_pairs"] <= capacity
        assert backend.estimated_memory_bytes() > 0
        backend.destroy()
    for streamed, full in zip(results[1], results[0], strict=True):
        np.testing.assert_allclose(streamed, full, rtol=3e-6, atol=2e-7)


def test_exact_near_work_refines_target_cells_and_profiles_separate_passes():
    rng = np.random.default_rng(932)
    position = rng.uniform(-0.1, 0.1, (64, 3)).astype(np.float32)
    strength = rng.normal(0, 0.01, (64, 3)).astype(np.float32)
    # Every target lies inside every core, so source acceptance is uniformly
    # false and this exercises explicit P2P, not mixed legacy-subtree jobs.
    core = np.full(64, 1.0, dtype=np.float32)
    targets = rng.uniform(-0.1, 0.1, (128, 3)).astype(np.float32)
    harness = _DeviceFMMHarness(capacity=64)
    harness.evaluate(position, strength, core)
    harness.induction.workspace.profile_passes = True
    backend, velocity, gradient = _query(harness, targets)
    near = int(backend.near_pair_count[None])
    exact = (backend.near_source.to_numpy()[:near] >= 0) & (
        backend.near_legacy.to_numpy()[:near] == 0
    )
    near_nodes = backend.near_target.to_numpy()[:near][exact]
    assert near > 0
    assert np.max(backend.tree.node_particle_count.to_numpy()[near_nodes]) <= 32
    assert {"traversal", "near_ordering", "near_evaluation"} <= set(
        backend.last_diagnostics["passes_seconds"]
    )
    for actual, exact in zip((velocity, gradient), _oracle("GAUSSIAN", position, strength, core, targets), strict=True):
        np.testing.assert_allclose(actual, exact, rtol=3e-5, atol=3e-6)
    backend.destroy()


def test_single_source_single_target_preserves_root_only_tree():
    position = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    strength = np.array([[0.0, 0.01, 0.0]], dtype=np.float32)
    core = np.array([0.1], dtype=np.float32)
    targets = np.array([[0.2, -0.1, 0.05]], dtype=np.float32)
    harness = _DeviceFMMHarness(capacity=1)
    harness.evaluate(position, strength, core)
    backend, velocity, gradient = _query(harness, targets)
    expected = _oracle("GAUSSIAN", position, strength, core, targets)
    for actual, reference in zip((velocity, gradient), expected, strict=True):
        np.testing.assert_allclose(actual, reference, rtol=2e-5, atol=1e-7)
    backend.destroy()


def _oracle(kernel_name, position, strength, core, targets):
    kernel = make_vortex_kernel(kernel_name)
    difference = targets[:, None, :] - position[None, :, :]
    velocity = kernel.velocity_pair(
        difference, strength[None, :, :], core[None, :], core[None, :]
    ).sum(axis=1)
    gradient = kernel.gradient_pair(
        difference, strength[None, :, :], core[None, :], core[None, :]
    ).sum(axis=1)
    return velocity, gradient


def _query(harness, targets):
    count = len(targets)
    query = ti.Vector.field(3, ti.f32, shape=count)
    velocity = ti.Vector.field(3, ti.f32, shape=count)
    gradient = ti.Matrix.field(3, 3, ti.f32, shape=count)
    query.from_numpy(np.asarray(targets, dtype=np.float32))
    backend = FMMTargetEvaluator(harness.induction.workspace, count)
    backend.evaluate(query, velocity, gradient, count, harness.induction.physics._zero_velocity)
    return backend, velocity.to_numpy(), gradient.to_numpy()


def _legacy_query(harness, targets):
    """Call the preserved strict monopole traversal, independent of dispatch."""
    count = len(targets)
    query = ti.Vector.field(3, ti.f32, shape=count)
    velocity = ti.Vector.field(3, ti.f32, shape=count)
    gradient = ti.Matrix.field(3, 3, ti.f32, shape=count)
    query.from_numpy(np.asarray(targets, dtype=np.float32))
    workspace = harness.induction.workspace
    workspace.tree.compute_external_target_fields(
        query,
        velocity,
        gradient,
        harness.induction.physics._zero_velocity,
        count,
        workspace.target_batch_capacity,
        write_velocity=True,
        write_gradient=True,
    )
    return velocity.to_numpy(), gradient.to_numpy()


@pytest.mark.parametrize(
    "kernel_name", ["GAUSSIAN", "WINCKELMANS", "HIGH_ORDER_GAUSSIAN", "SUPER_GAUSSIAN"]
)
def test_target_fmm_preserves_exact_source_only_core_near_fields(kernel_name):
    rng = np.random.default_rng(1208)
    position = rng.uniform(-0.2, 0.2, (16, 3)).astype(np.float32)
    strength = rng.normal(0, 0.02, (16, 3)).astype(np.float32)
    core = rng.uniform(0.08, 0.25, 16).astype(np.float32)
    targets = np.vstack(
        (position[:2], position[:2] + 0.01, rng.uniform(-0.3, 0.3, (16, 3)))
    ).astype(np.float32)
    harness = _DeviceFMMHarness(capacity=16, kernel_name=kernel_name)
    harness.evaluate(position, strength, core)
    backend, velocity, gradient = _query(harness, targets)
    expected_velocity, expected_gradient = _oracle(kernel_name, position, strength, core, targets)
    np.testing.assert_allclose(velocity, expected_velocity, rtol=4e-5, atol=1e-6)
    np.testing.assert_allclose(gradient, expected_gradient, rtol=4e-5, atol=2e-5)
    assert backend.last_diagnostics["m2l_pairs"] == 0
    backend.destroy()


@pytest.mark.parametrize("kernel_name", ["HIGH_ORDER_GAUSSIAN", "SUPER_GAUSSIAN"])
def test_internal_regularised_monopoles_keep_the_selected_radial_kernel(kernel_name):
    position = np.zeros((1, 3), dtype=np.float32)
    strength = np.array([[0.0, 0.01, 0.0]], dtype=np.float32)
    core = np.array([0.1], dtype=np.float32)
    targets = np.zeros((17, 3), dtype=np.float32)
    targets[:, 2] = np.linspace(0.2, 0.4, len(targets))
    harness = _DeviceFMMHarness(capacity=1, kernel_name=kernel_name)
    harness.evaluate(position, strength, core)
    backend, velocity, gradient = _query(harness, targets)
    assert backend.last_diagnostics["monopole_target_pairs"] > 0
    for actual, expected in zip(
        (velocity, gradient), _oracle(kernel_name, position, strength, core, targets), strict=True
    ):
        np.testing.assert_allclose(actual, expected, rtol=4e-6, atol=2e-7)
    backend.destroy()


def test_complete_image_block_matches_independent_reflected_source_sum():
    rng = np.random.default_rng(667)
    position = rng.uniform(-0.25, 0.25, (64, 3)).astype(np.float32)
    strength = rng.normal(0, 0.01, (64, 3)).astype(np.float32)
    core = rng.uniform(0.03, 0.08, 64).astype(np.float32)
    targets = rng.uniform(-0.3, 0.3, (64, 3)).astype(np.float32)
    harness = _DeviceFMMHarness(capacity=64)
    harness.evaluate(position, strength, core)
    query = ti.Vector.field(3, ti.f32, shape=64)
    velocity = ti.Vector.field(3, ti.f32, shape=64)
    gradient = ti.Matrix.field(3, 3, ti.f32, shape=64)
    query.from_numpy(targets)
    backend = FMMTargetEvaluator(harness.induction.workspace, 64)
    backend.prepare_targets(query, 64)
    for images in (((-0.96, True),), ((-8.64, False), (-9.6, True), (8.64, False), (7.68, True))):
        expected_velocity = np.zeros((64, 3))
        expected_gradient = np.zeros((64, 3, 3))
        summed_velocity = np.zeros((64, 3))
        summed_gradient = np.zeros((64, 3, 3))
        old_velocity = np.zeros((64, 3))
        old_gradient = np.zeros((64, 3, 3))
        rounding = [np.zeros(64), np.zeros(64)]
        for shift, odd in images:
            image_position = position.copy()
            image_strength = strength.copy()
            image_position[:, 2] = shift - position[:, 2] if odd else position[:, 2] + shift
            if odd:
                image_strength[:, :2] *= -1
            v, j = _oracle("GAUSSIAN", image_position, image_strength, core, targets)
            _, image_rounding = _exact_fields_and_roundoff(
                "GAUSSIAN", image_position, image_strength, core, targets
            )
            for index in (0, 1):
                rounding[index] += image_rounding[index]
            expected_velocity += v
            expected_gradient += j
            transformed_target = targets.copy()
            transformed_target[:, 2] = shift - targets[:, 2] if odd else targets[:, 2] - shift
            old_v, old_j = _legacy_query(harness, transformed_target)
            if odd:
                old_v[:, 2] *= -1
                old_j[:, :2, 2] *= -1
                old_j[:, 2, :2] *= -1
            old_velocity += old_v
            old_gradient += old_j
            backend.evaluate_prepared(
                velocity, gradient, harness.induction.physics._zero_velocity, shift=shift, odd=odd
            )
            v, j = velocity.to_numpy(), gradient.to_numpy()
            if odd:
                v[:, 2] *= -1
                j[:, :2, 2] *= -1
                j[:, 2, :2] *= -1
            summed_velocity += v
            summed_gradient += j
        backend.evaluate_image_block(
            images, velocity, gradient, harness.induction.physics._zero_velocity
        )
        for new, serial, exact, old, roundoff in zip(
            (velocity.to_numpy(), gradient.to_numpy()),
            (summed_velocity, summed_gradient),
            (expected_velocity, expected_gradient),
            (old_velocity, old_gradient),
            rounding,
            strict=True,
        ):
            np.testing.assert_allclose(new, serial, rtol=3e-4, atol=2e-7)
            error = np.linalg.norm(new - exact) / max(np.linalg.norm(exact), 1e-12)
            old_error = np.linalg.norm(old - exact) / max(np.linalg.norm(exact), 1e-12)
            allowance = np.linalg.norm(roundoff) / max(np.linalg.norm(exact), 1e-12)
            assert error <= old_error + allowance, (images, error, old_error, allowance)
        assert backend.target_geometry_builds == 1
    backend.destroy()


@pytest.mark.parametrize("configuration", ["far", "mixed", "cancelled"])
def test_target_fmm_compares_actual_error_to_existing_target_path(configuration):
    rng = np.random.default_rng(901)
    position = rng.uniform(-0.5, 0.5, (128, 3)).astype(np.float32)
    strength = rng.normal(0, 0.01, (128, 3)).astype(np.float32)
    if configuration == "cancelled":
        strength[1::2] = -strength[::2]
    core = rng.uniform(0.02, 0.08, 128).astype(np.float32)
    targets = rng.uniform(-0.4, 0.4, (96, 3)).astype(np.float32)
    targets[:, 2] += 8.0
    if configuration == "mixed":
        targets[:32, 2] -= 8.0
    harness = _DeviceFMMHarness(capacity=128)
    harness.evaluate(position, strength, core)
    backend, velocity, gradient = _query(harness, targets)
    expected = _oracle("GAUSSIAN", position, strength, core, targets)
    _, rounding = _exact_fields_and_roundoff("GAUSSIAN", position, strength, core, targets)
    old = _legacy_query(harness, targets)
    for label, new, reference, previous, roundoff in zip(
        ("velocity", "gradient"), (velocity, gradient), expected, old, rounding, strict=True
    ):
        scale = max(np.linalg.norm(reference), np.finfo(np.float32).eps)
        error = np.linalg.norm(new - reference) / scale
        old_error = np.linalg.norm(previous - reference) / scale
        allowance = np.linalg.norm(roundoff) / scale
        assert error <= old_error + allowance, (label, configuration, error, old_error, allowance)
    if configuration != "mixed":
        assert (
            backend.last_diagnostics["m2l_pairs"]
            + backend.last_diagnostics["monopole_cell_pairs"]
            + backend.last_diagnostics["legacy_subtree_target_pairs"]
        ) > 0
    assert backend.last_diagnostics["target_cells"] < len(targets)
    backend.destroy()


def test_prepared_target_reflections_preserve_full_jacobian_and_geometry():
    rng = np.random.default_rng(444)
    position = rng.uniform(-0.3, 0.3, (64, 3)).astype(np.float32)
    strength = rng.normal(0, 0.01, (64, 3)).astype(np.float32)
    core = rng.uniform(0.02, 0.08, 64).astype(np.float32)
    targets = rng.uniform(-0.4, 0.4, (64, 3)).astype(np.float32)
    harness = _DeviceFMMHarness(capacity=64)
    harness.evaluate(position, strength, core)
    query = ti.Vector.field(3, ti.f32, shape=64)
    velocity = ti.Vector.field(3, ti.f32, shape=128)
    gradient = ti.Matrix.field(3, 3, ti.f32, shape=128)
    query.from_numpy(targets)
    backend = FMMTargetEvaluator(harness.induction.workspace, 64)
    backend.prepare_targets(query, 64)
    # The owned immutable tile, not caller field identity, guarantees reuse.
    query.fill(100.0)
    for shift, odd in ((-0.96, True), (8.64, False), (10.56, True)):
        transformed = targets.copy()
        transformed[:, 2] = shift - targets[:, 2] if odd else targets[:, 2] - shift
        backend.evaluate_prepared(
            velocity,
            gradient,
            harness.induction.physics._zero_velocity,
            shift=shift,
            odd=odd,
            output_start=64,
        )
        expected = _oracle("GAUSSIAN", position, strength, core, transformed)
        previous = _legacy_query(harness, transformed)
        kernel = make_vortex_kernel("GAUSSIAN")
        displacement = transformed[:, None, :] - position[None, :, :]
        pairs = (
            kernel.velocity_pair(displacement, strength[None, :, :], core[None, :], core[None, :]),
            kernel.gradient_pair(displacement, strength[None, :, :], core[None, :], core[None, :]),
        )
        for result, reference, old, pair in zip(
            (velocity.to_numpy()[64:], gradient.to_numpy()[64:]), expected, previous, pairs, strict=True
        ):
            conditioning = np.linalg.norm(pair.reshape(64, 64, -1), axis=2).sum(axis=1)
            allowance = 16 * np.finfo(np.float32).eps * np.linalg.norm(conditioning)
            error, old_error = np.linalg.norm(result - reference), np.linalg.norm(old - reference)
            assert error <= old_error + allowance, (shift, odd, error, old_error, allowance)
        assert backend.target_geometry_builds == 1
    backend.destroy()


class _LegacyImageInduction:
    # A plain proxy is intentional: Taichi's data-oriented __getattribute__
    # binds property getters from the decorated parent, ignoring a subclass's
    # override. That would silently compare the accelerated path to itself.
    supports_image_blocks = False

    def __init__(self):
        object.__setattr__(self, "_base", FMMInduction())

    def __getattr__(self, name):
        return getattr(self._base, name)

    def __setattr__(self, name, value):
        setattr(self._base, name, value)

    def bind(self, physics, *, kernel=None):
        self._base.bind(physics, kernel=kernel)
        return self


def _slab_oracle(kernel_name, position, strength, core, target, shell, *, stage):
    kernel = make_vortex_kernel(kernel_name)
    outputs = [np.zeros((len(target), 3)), np.zeros((len(target), 3, 3))]
    conditioning = [np.zeros(len(target)), np.zeros(len(target))]
    for k in range(-shell, shell + 1):
        for odd in (False, True):
            image_position, image_strength = position.copy(), strength.copy()
            image_position[:, 2] = (
                (-0.96 + 1.92 * k - position[:, 2]) if odd else (position[:, 2] + 1.92 * k)
            )
            if odd:
                image_strength[:, :2] *= -1
            difference = target[:, None, :] - image_position[None, :, :]
            target_core = core[:, None] if stage and k == 0 and not odd else core[None, :]
            fields = (
                kernel.velocity_pair(
                    difference, image_strength[None, :, :], target_core, core[None, :]
                ),
                kernel.gradient_pair(
                    difference, image_strength[None, :, :], target_core, core[None, :]
                ),
            )
            for index, field in enumerate(fields):
                outputs[index] += field.sum(axis=1)
                conditioning[index] += np.linalg.norm(
                    field.reshape(len(target), len(position), -1), axis=2
                ).sum(axis=1)
    return outputs, conditioning


@pytest.mark.parametrize("kernel_name", ["GAUSSIAN", "WINCKELMANS"])
def test_full_slab_blocks_preserve_tail_stage_rates_and_target_operator(kernel_name):
    rng = np.random.default_rng(776)
    n = 6
    position = rng.uniform(-0.3, 0.3, (n, 3)).astype(np.float32)
    strength = rng.normal(0, 0.01, (n, 3)).astype(np.float32)
    core = rng.uniform(0.04, 0.12, n).astype(np.float32)
    targets = np.array(
        [[0.1, -0.1, -0.48], [0.1, -0.1, 0.48], [0.04, 0.05, 0.03]], dtype=np.float32
    )
    # Initialise Taichi without relying on another test's execution order.
    _DeviceFMMHarness(capacity=1)
    x = ti.Vector.field(3, ti.f32, shape=n)
    gamma = ti.Vector.field(3, ti.f32, shape=n)
    radius = ti.field(ti.f32, shape=n)
    query = ti.Vector.field(3, ti.f32, shape=3)
    x.from_numpy(position)
    gamma.from_numpy(strength)
    radius.from_numpy(core)
    query.from_numpy(targets)
    records = []
    for backend in (_LegacyImageInduction, FMMInduction):
        physics = PhysicsBase(kernel_name, n, ti.f32, max_evaluation_points=7)
        slab = SlipSlabInduction(
            backend(), z_min=-0.48, z_max=0.48, tail_tolerance=1e-4, max_shells=129
        ).bind(physics)
        assert slab.base.supports_image_blocks == (backend is FMMInduction)
        velocity = ti.Vector.field(3, ti.f32, shape=n)
        gradient = ti.Matrix.field(3, 3, ti.f32, shape=n)
        rate = ti.Vector.field(3, ti.f32, shape=n)
        item = {}
        for mode in ("DIRECT", "TRANSPOSED", "MIXED"):
            slab.stretching_scheme = slab.base.stretching_scheme = mode
            slab.base._stretching_mode = {"DIRECT": 0, "TRANSPOSED": 1, "MIXED": 2}[mode]
            slab.evaluate_stage(
                position=x,
                vortex_strength=gamma,
                core_radius=radius,
                count=n,
                velocity_out=velocity,
                vortex_strength_rate_out=rate,
                velocity_gradient_out=gradient,
            )
            item[mode] = (
                velocity.to_numpy(),
                gradient.to_numpy(),
                rate.to_numpy(),
                dict(slab.last_tail),
            )
        slab.evaluate_targets(
            target_position=query,
            source_position=x,
            source_vortex_strength=gamma,
            source_core_radius=radius,
            target_velocity=velocity,
            target_velocity_gradient=gradient,
            target_count=3,
            source_count=n,
            include_freestream=False,
            background_velocity=physics._zero_velocity,
        )
        item["targets"] = (
            velocity.to_numpy()[:3],
            gradient.to_numpy()[:3],
            None,
            dict(slab.last_tail),
        )
        records.append(item)
    for name in ("DIRECT", "TRANSPOSED", "MIXED", "targets"):
        old, new = records[0][name], records[1][name]
        assert old[3]["shell"] == new[3]["shell"]
        assert new[3]["relative"] <= 1e-4
        exact, conditioning = _slab_oracle(
            kernel_name,
            position,
            strength,
            core,
            targets if name == "targets" else position,
            new[3]["shell"],
            stage=name != "targets",
        )
        for index in (0, 1):
            previous_error = np.linalg.norm(old[index] - exact[index])
            current_error = np.linalg.norm(new[index] - exact[index])
            allowance = 16 * np.finfo(np.float32).eps * np.linalg.norm(conditioning[index])
            assert current_error <= previous_error + allowance, (
                kernel_name,
                name,
                index,
                current_error,
                previous_error,
                allowance,
            )
        if name != "targets":
            j = new[1]
            expected_rate = (
                np.einsum("nij,nj->ni", j, strength)
                if name == "DIRECT"
                else np.einsum("nji,nj->ni", j, strength)
            )
            if name == "MIXED":
                expected_rate = 0.5 * (
                    np.einsum("nij,nj->ni", j, strength) + np.einsum("nji,nj->ni", j, strength)
                )
            np.testing.assert_allclose(new[2], expected_rate, rtol=2e-5, atol=2e-7)
        assert new[3]["target_local_evaluations"] < old[3]["target_local_evaluations"]
