"""Pure execution checks for complete-block image fast-path fallback.

The numerical target expansion has independent direct-sum qualification. Here
NumPy stand-ins isolate the slab's control flow, without JIT or device work:
declining one tile must retain every original image, parity, tail decision and
stretching contribution, including when later tiles use the fast path.
"""

from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from source.solvers.vpm.physics.induction import slip_slab
from source.solvers.vpm.physics.induction.fmm.targets import (
    FMMTargetEvaluator,
    TargetBlockNotWorthwhile,
    TargetInteractionCapacityError,
)


class _Array:
    def __init__(self, shape):
        self.values = np.zeros(shape)

    def from_numpy(self, values):
        self.values[...] = values

    def __getitem__(self, key):
        return self.values[key]

    def __setitem__(self, key, value):
        self.values[key] = value


def _fields(points):
    """A smooth, rapidly decaying field with all reflection-sensitive terms."""
    x, y, z = np.asarray(points).T
    weight = np.exp(-0.15 * z * z)
    velocity = weight[:, None] * np.column_stack((1 + x + z, 2 + y - z, 3 + x * z))
    gradient = weight[:, None, None] * (np.arange(1, 10).reshape(1, 3, 3) + z[:, None, None] * 0.1)
    return velocity, gradient


def _physical_image(points, shift, odd):
    # Both production paths upload shifts to f32 device fields.
    shift = np.float32(shift)
    transformed = points.copy()
    transformed[:, 2] = shift - points[:, 2] if odd else points[:, 2] - shift
    velocity, gradient = _fields(transformed)
    if odd:
        velocity[:, 2] *= -1
        gradient[:, :2, 2] *= -1
        gradient[:, 2, :2] *= -1
    return velocity, gradient


@pytest.fixture
def numpy_slab(monkeypatch):
    def zero(velocity, gradient, count):
        velocity[:count] = 0
        gradient[:count] = 0

    def maxima(velocity, gradient, max_velocity, max_gradient, count):
        max_velocity[None] = np.linalg.norm(velocity[:count], axis=1).max()
        max_gradient[None] = np.linalg.norm(gradient[:count].reshape(count, 9), axis=1).max()

    def build(original, transformed, shifts, odds, start, points, images):
        for image in range(images):
            query = original[start : start + points].copy()
            query[:, 2] = (
                shifts[image] - query[:, 2] if odds[image] else query[:, 2] - shifts[image]
            )
            transformed[image * points : (image + 1) * points] = query

    def add_fields(
        v,
        j,
        strength,
        velocity,
        rate,
        gradient,
        block_v,
        block_j,
        start,
        count,
        mode,
        rate_enabled,
        has_v,
        has_j,
        is_stage,
    ):
        target = slice(start, start + count)
        if has_v:
            velocity[target] += v
        if has_j:
            gradient[target] += j
        block_v[target] += v
        block_j[target] += j
        if is_stage and rate_enabled:
            direct = np.einsum("nij,nj->ni", j, strength[target])
            transposed = np.einsum("nji,nj->ni", j, strength[target])
            rate[target] += (
                direct if mode == 0 else transposed if mode == 1 else (direct + transposed) / 2
            )

    def add_block(image_v, image_j, *arguments):
        count = arguments[7]
        add_fields(image_v[:count], image_j[:count], *arguments)

    def add_reflected(
        image_v,
        image_j,
        _shifts,
        odds,
        strength,
        velocity,
        rate,
        gradient,
        block_v,
        block_j,
        start,
        points,
        images,
        *flags,
    ):
        summed_v = np.zeros((points, 3))
        summed_j = np.zeros((points, 3, 3))
        for image in range(images):
            v = image_v[image * points : (image + 1) * points].copy()
            j = image_j[image * points : (image + 1) * points].copy()
            if odds[image]:
                v[:, 2] *= -1
                j[:, :2, 2] *= -1
                j[:, 2, :2] *= -1
            summed_v += v
            summed_j += j
        add_fields(
            summed_v,
            summed_j,
            strength,
            velocity,
            rate,
            gradient,
            block_v,
            block_j,
            start,
            points,
            *flags,
        )

    monkeypatch.setattr(slip_slab, "_zero_shell", zero)
    monkeypatch.setattr(slip_slab, "_shell_maxima", maxima)
    monkeypatch.setattr(slip_slab, "_build_reflected_targets", build)
    monkeypatch.setattr(slip_slab, "_add_image_block_results", add_block)
    monkeypatch.setattr(slip_slab, "_add_reflected_results", add_reflected)

    def make(fast, mode="DIRECT", fatal=None):
        calls = []
        slab = object.__new__(slip_slab.SlipSlabInduction)
        slab.z_min, slab.z_max = -0.48, 0.48
        slab.max_shells, slab.tail_tolerance = 65, 1e-4
        slab.velocity_scale = slab.gradient_scale = 1.0
        slab.stretching_scheme = mode
        slab.physics = SimpleNamespace(
            particle_kernel="WINCKELMANS",
            max_evaluation_points=4,
            accumulator_dtype=slip_slab.ti.f32,
            _zero_velocity=np.zeros(3),
        )
        slab._image_velocity = np.zeros((4, 3))
        slab._image_gradient = np.zeros((4, 3, 3))
        slab._query_position = np.zeros((4, 3))
        slab._image_shifts, slab._image_odd = _Array(64), _Array(64)
        slab._block_velocity = np.zeros((9, 3))
        slab._block_gradient = np.zeros((9, 3, 3))
        slab._max_shell_velocity, slab._max_shell_gradient = {None: 0.0}, {None: 0.0}

        def evaluate_targets(**args):
            count = args["target_count"]
            calls.append(("pointwise", count))
            v, j = _fields(args["target_position"][:count])
            args["target_velocity"][:count] = v
            args["target_velocity_gradient"][:count] = j

        def evaluate_block(**args):
            if fatal is not None:
                raise fatal
            images = args["images"]
            start, count = args["target_start"], args["target_count"]
            # Decline the first dense block for every tile, then one tile in
            # the next block. Subsequent blocks must still try acceleration.
            decline = len(images) == 1 or (start == 4 and max(abs(s) for s, _ in images) < 3)
            calls.append(("declined" if decline else "accepted", start, tuple(images)))
            if decline:
                raise TargetBlockNotWorthwhile(99, 64, {"reason": "conditions-test"})
            points = args["target_position"][start : start + count]
            v, j = np.zeros((count, 3)), np.zeros((count, 3, 3))
            for shift, odd in images:
                image_v, image_j = _physical_image(points, shift, odd)
                v += image_v
                j += image_j
            args["target_velocity"][:count] = v
            args["target_velocity_gradient"][:count] = j

        slab.base = SimpleNamespace(
            supports_image_blocks=fast,
            evaluate_image_block=evaluate_block,
            evaluate_targets=evaluate_targets,
            fixed_source_targets=lambda *a, **kw: nullcontext(),
        )
        return slab, calls

    return make


def _run(slab, *, outputs="both", stage=False):
    points = np.column_stack(
        (np.linspace(-0.2, 0.2, 9), np.linspace(0.3, -0.1, 9), np.linspace(-0.4, 0.4, 9))
    )
    strength = np.arange(27, dtype=float).reshape(9, 3) / 30
    velocity, gradient, rate = (
        np.full((9, 3), 17.0),
        np.full((9, 3, 3), 19.0),
        np.full((9, 3), 23.0),
    )
    slab._images(
        points,
        strength,
        np.full(9, 0.1),
        points,
        9,
        9,
        None if outputs == "gradient" else velocity,
        None if outputs == "velocity" else gradient,
        stage_strength=strength if stage else None,
        stage_rate=rate if stage else None,
        rate_enabled=stage,
    )
    return velocity, gradient, rate


@pytest.mark.parametrize(
    "mode,outputs,stage",
    [
        ("DIRECT", "both", True),
        ("TRANSPOSED", "both", True),
        ("MIXED", "both", True),
        ("DIRECT", "velocity", False),
        ("DIRECT", "gradient", False),
    ],
)
def test_mixed_declined_and_accepted_tiles_preserve_complete_pointwise_operator(
    numpy_slab, mode, outputs, stage
):
    pointwise, _ = numpy_slab(False, mode)
    hybrid, calls = numpy_slab(True, mode)
    before = _run(pointwise, outputs=outputs, stage=stage)
    after = _run(hybrid, outputs=outputs, stage=stage)
    for expected, actual in zip(before, after, strict=True):
        np.testing.assert_allclose(actual, expected, rtol=2e-8, atol=2e-8)
    for key in ("shell", "block_start", "relative", "velocity", "gradient", "target_evaluations"):
        np.testing.assert_allclose(
            hybrid.last_tail[key], pointwise.last_tail[key], rtol=2e-8, atol=2e-8
        )
    assert len(hybrid.last_tail["declined_blocks"]) == 4
    assert calls[0][0] == "declined"
    assert any(call[0] == "accepted" and len(call[2]) >= 8 for call in calls)
    assert (
        hybrid.last_tail["target_local_evaluations"]
        < pointwise.last_tail["target_local_evaluations"]
    )


@pytest.mark.parametrize(
    "fatal", [RuntimeError("incomplete traversal"), TargetInteractionCapacityError(4, 5)]
)
def test_non_cost_failures_are_not_swallowed_as_pointwise_fallback(numpy_slab, fatal):
    slab, calls = numpy_slab(True, fatal=fatal)
    with pytest.raises(type(fatal), match=str(fatal)):
        _run(slab)
    assert not calls


def _streaming_rig(*, fail_last=False):
    """Exercise the real Python scheduler, replacing only its device kernels."""
    rig = SimpleNamespace(
        _prepared_count=4,
        leaf_count={None: 2},
        leaf_nodes=np.array([4, 5]),
        image_count={None: 4},
        block_mode={None: 1},
        error={None: 0},
        m2l_count={None: 0},
        near_pair_count={None: 0},
        direct_work={None: 0},
        monopole_work={None: 0},
        monopole_pair_count={None: 0},
        subtree_work={None: 0},
        frontier_count={None: 0},
        max_pairs=1,
        batch_capacity=1,
        scan_a=object(),
        scan_b=object(),
        tree=SimpleNamespace(
            node_particle_count=np.array([1, 1, 1, 1, 2, 2, 4]),
            node_left=np.array([-1, -1, -1, -1, 0, 2, 4]),
            node_right=np.array([-1, -1, -1, -1, 1, 3, 5]),
        ),
        velocity=np.zeros((4, 3)),
        gradient=np.zeros((4, 3, 3)),
        visits=np.zeros((4, 4), dtype=int),
        publications=0,
    )
    descendants = {0: [0], 1: [1], 2: [2], 3: [3], 4: [0, 1], 5: [2, 3]}

    def clear(count):
        rig.velocity[:count] = 0
        rig.gradient[:count] = 0

    def initialise(_nodes):
        rig.job = []

    def walk(first_leaf, leaves, first_image, images, root):
        targets = (
            descendants[root]
            if root >= 0
            else [
                target
                for leaf in range(first_leaf, first_leaf + leaves)
                for target in descendants[rig.leaf_nodes[leaf]]
            ]
        )
        rig.job = [
            (target, image)
            for target in targets
            for image in range(first_image, first_image + images)
        ]
        required = len(rig.job)
        if fail_last and (3, 3) in rig.job:
            required += 1
        rig.error[None] = int(required > rig.max_pairs)
        if rig.error[None]:
            rig.m2l_count[None] = min(required, rig.max_pairs + 1)
            rig.near_pair_count[None] = 0
        else:
            # Alternate far and near jobs so both independently accumulating
            # device phases are exercised by the real execution.
            rig.m2l_count[None] = sum(image % 2 == 0 for _, image in rig.job)
            rig.near_pair_count[None] = sum(image % 2 == 1 for _, image in rig.job)

    def evaluate(*, near):
        for target, image in rig.job:
            if (image % 2 == 1) != near:
                continue
            rig.visits[target, image] += 1
            rig.velocity[target] += image + 1
            rig.gradient[target] += 2 * (image + 1)

    def publish(velocity, gradient, background, count, offset, has_v, has_j):
        rig.publications += 1
        if has_v:
            velocity[offset : offset + count] = rig.velocity[:count] + background
        if has_j:
            gradient[offset : offset + count] = rig.gradient[:count]

    rig._clear_outputs, rig._initialise, rig._walk_sources = clear, initialise, walk
    rig._evaluate_local_particles = lambda count: evaluate(near=False)
    rig._evaluate_near_lanes = lambda inclusive, count: None
    rig._reduce_near_lanes = lambda count: evaluate(near=True)
    rig._profile_phase = lambda name: nullcontext()
    rig._publish = publish
    for method in (
        "_compute_derivatives",
        "_translate",
        "_copy_counts",
        "_scan_counts",
        "_initialise_cursor",
        "_order_near",
    ):
        setattr(rig, method, lambda *args: None)
    return rig


def test_streaming_scheduler_partitions_images_cells_and_subtrees_exactly_once():
    rig = _streaming_rig()
    velocity, gradient = np.full((4, 3), 17.0), np.full((4, 3, 3), 19.0)
    FMMTargetEvaluator._evaluate_prepared_fields(rig, velocity, gradient, np.ones(3), 0)
    np.testing.assert_array_equal(rig.visits, 1)
    np.testing.assert_array_equal(velocity, 11)
    np.testing.assert_array_equal(gradient, 20)
    assert rig.publications == 1
    assert rig.last_diagnostics["work_batches"] == 16
    assert rig.last_diagnostics["peak_stored_pairs"] == 1
    assert rig.last_diagnostics["m2l_pairs"] == 8
    assert rig.last_diagnostics["near_cell_pairs"] == 8


def test_irreducible_late_streaming_decline_discards_all_private_partial_results():
    rig = _streaming_rig(fail_last=True)
    velocity, gradient = np.full((4, 3), 17.0), np.full((4, 3, 3), 19.0)
    with pytest.raises(TargetBlockNotWorthwhile):
        FMMTargetEvaluator._evaluate_prepared_fields(rig, velocity, gradient, np.ones(3), 0)
    assert rig.visits.sum() == 15  # Many completed subjobs preceded the failure.
    assert rig.publications == 0
    np.testing.assert_array_equal(velocity, 17)
    np.testing.assert_array_equal(gradient, 19)
