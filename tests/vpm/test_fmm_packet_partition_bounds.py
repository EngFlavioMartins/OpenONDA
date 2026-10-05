"""Packet decisions must never change the pointwise source partition.

These tests only build tiny hierarchies and evaluate admissibility. They do
not compile the expensive FMM field kernels or use a GPU.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import taichi as ti

from source.solvers.vpm.physics.induction.fmm.targets import FMMTargetEvaluator
from source.solvers.vpm.physics.induction.treecode.lbvh import TaichiTreecode, _DeviceFields


@ti.kernel
def _classify_roots(evaluator: ti.template(), packet: ti.template(), points: ti.template()):
    target_root = evaluator.tree._root[None]
    source_root = evaluator.source.tree._root[None]
    packet[None] = evaluator._admissibility(target_root, source_root, 0)
    for point in range(2):
        query = evaluator._transform(evaluator.tree.position[point], 0)
        displacement = query - evaluator.source.tree.node_com[source_root]
        squared = displacement.dot(displacement)
        distance = ti.sqrt(squared)
        diameter = 2.0 * evaluator.source.tree.node_half_size[source_root]
        # Deliberately spell out the LBVH condition independently of the
        # packet implementation's _pointwise_accept helper.
        points[point] = ti.cast(
            distance > ti.max(1e-8, evaluator.source.tree.node_avg_radius[source_root])
            and diameter * diameter / squared < evaluator.source.tree.theta_sq
            and evaluator.source.tree._node_core_is_admissible(
                source_root, distance, evaluator.source.tree.node_max_radius[source_root]
            )
            != 0,
            ti.i32,
        )


@ti.kernel
def _compare_image_pair_fields(
    evaluator: ti.template(), velocity: ti.template(), gradient: ti.template()
):
    root = evaluator.source.tree._root[None]
    evaluator.block_mode[None] = 1
    for point in range(2):
        position = evaluator.tree.position[point]
        query = evaluator._transform(position, 0)
        displacement = query - evaluator.source.tree.node_com[root]
        distance = displacement.norm()
        radius = evaluator.source.tree.node_avg_radius[root]
        velocity[0, point], gradient[0, point] = evaluator._monopole_fields(root, position, 0)
        velocity[1, point] = evaluator.source.tree._far_velocity_node(
            root, displacement, distance, radius
        )
        gradient[1, point] = evaluator.source.tree._far_gradient_node(
            root, displacement, distance, radius
        )
        velocity[2, point], gradient[2, point] = evaluator._exact_node_fields(root, position, 0)
        velocity[3, point] = evaluator.source.tree._target_leaf_velocity_sum(root, query)
        gradient[3, point] = evaluator.source.tree._target_leaf_gradient_sum(root, query)
        if evaluator.image_odd[0]:
            for reference in ti.static((1, 3)):
                velocity[reference, point][2] = -velocity[reference, point][2]
                gradient[reference, point][0, 2] = -gradient[reference, point][0, 2]
                gradient[reference, point][1, 2] = -gradient[reference, point][1, 2]
                gradient[reference, point][2, 0] = -gradient[reference, point][2, 0]
                gradient[reference, point][2, 1] = -gradient[reference, point][2, 1]


@pytest.fixture(scope="module")
def classifier():
    if ti.lang.impl.get_runtime().prog is None:
        ti.init(arch=ti.cpu, offline_cache=False, cpu_max_num_threads=2)
    tree = TaichiTreecode(
        max_n_particles=2,
        max_nodes=4,
        max_leaf_size=1,
        theta=0.1,
        kernel_type="GAUSSIAN",
        multipole_order=1,
        hierarchy_only=True,
        max_evaluation_points=1,
    )
    workspace = _DeviceFields()
    position = workspace.vector(3, dtype=ti.f32, shape=2)
    strength = workspace.vector(3, dtype=ti.f32, shape=2)
    radius = workspace.scalar(dtype=ti.f32, shape=2)
    query = workspace.vector(3, dtype=ti.f32, shape=2)
    packet = workspace.vector(3, dtype=ti.i32, shape=())
    points = workspace.scalar(dtype=ti.i32, shape=2)
    image_velocity = workspace.vector(3, dtype=ti.f32, shape=(4, 2))
    image_gradient = workspace.matrix(3, 3, dtype=ti.f32, shape=(4, 2))
    workspace.finalize()
    source = SimpleNamespace(
        tree=tree,
        kernel_name="GAUSSIAN",
        velocity_tail_cutoff=5.0,
        gradient_tail_cutoff=5.0,
        radial_factors=tree._gaussian_radial_factors,
    )
    evaluator = FMMTargetEvaluator(source, 2)

    def evaluate(source_position, targets, cores, *, cancelled=False, shift=0.0, odd=False):
        values = np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        if cancelled:
            values[1] *= -1
        position.from_numpy(np.asarray(source_position, dtype=np.float32))
        strength.from_numpy(values)
        radius.from_numpy(np.asarray(cores, dtype=np.float32))
        tree.build(position, strength, radius, 2)
        query.from_numpy(np.asarray(targets, dtype=np.float32))
        evaluator.prepare_targets(query, 2)
        evaluator._set_image_transform(float(shift), int(odd))
        _classify_roots(evaluator, packet, points)
        return packet.to_numpy().astype(bool), points.to_numpy().astype(bool)

    def image_fields():
        _compare_image_pair_fields(evaluator, image_velocity, image_gradient)
        return image_velocity.to_numpy(), image_gradient.to_numpy()

    try:
        yield evaluate, tree, image_fields
    finally:
        evaluator.destroy()
        tree.destroy()
        workspace.destroy()


@pytest.mark.parametrize("failure_mode", ["false_all", "false_none"])
@pytest.mark.parametrize("image", [(0.0, False), (100000000.0, False), (100000000.0, True)])
def test_translated_midpoint_roundoff_cannot_change_partition(classifier, failure_mode, image):
    evaluate, _tree, _image_fields = classifier
    shift, odd = image
    targets = [[100000000.0, 4.0, shift], [100000008.0, 4.0, shift]]
    if failure_mode == "false_all":
        sources = [[100000088.0, 0.0, 0.0], [100000088.0, 8.0, 0.0]]
    else:
        sources = [[99999928.0, 24.0, 0.0], [99999928.0, 32.0, 0.0]]
    packet, pointwise = evaluate(sources, targets, [1.0, 1.0], shift=shift, odd=odd)
    assert pointwise.any() and not pointwise.all(), pointwise
    assert not packet[0], "ALL must not merge a pointwise-rejected source node"
    assert not packet[2], "NONE must not open a pointwise-accepted source node"


@pytest.mark.parametrize("boundary", ["strict_mac", "mean_core", "mixed_core_tail"])
def test_float32_acceptance_boundaries_remain_mixed(classifier, boundary):
    evaluate, tree, _image_fields = classifier
    sources = np.zeros((2, 3), dtype=np.float32)
    if boundary == "strict_mac":
        sources[:, 1] = [-4, 4]
        cores = np.ones(2, dtype=np.float32)
        threshold = np.float32(80.0)
    elif boundary == "mean_core":
        cores = np.full(2, 0.2, dtype=np.float32)
        threshold = cores[0]
    else:
        cores = np.array([0.1, 0.2], dtype=np.float32)
        threshold = np.float32(float(tree.regularization_tail_cutoff[None]) * float(cores[1]))
    low, high = threshold, threshold
    for _ in range(4):
        low = np.nextafter(low, np.float32(-np.inf))
        high = np.nextafter(high, np.float32(np.inf))
    targets = np.zeros((2, 3), dtype=np.float32)
    targets[:, 0] = [low, high]
    packet, pointwise = evaluate(sources, targets, cores)
    assert pointwise.any() and not pointwise.all(), (boundary, pointwise)
    assert not packet[0] and not packet[2], (boundary, packet)


def test_cancelled_common_core_packet_is_not_accepted(classifier):
    evaluate, _tree, _image_fields = classifier
    packet, pointwise = evaluate(
        [[0, -1, 0], [0, 1, 0]], [[100, 0, 0], [101, 0, 0]], [0.1, 0.1], cancelled=True
    )
    assert not pointwise.any()
    assert not packet[0] and packet[2]


@pytest.mark.parametrize("odd", [False, True])
def test_image_near_fields_preserve_inverse_query_rounding(classifier, odd):
    evaluate, _tree, image_fields = classifier
    source_z = -4.0 if odd else 4.0
    evaluate(
        [[0, 0, source_z], [0, 0, source_z]],
        [[0, 0, 100000008.0], [0, 0, 100000016.0]],
        [1.0, 1.0],
        shift=100000000.0,
        odd=odd,
    )
    # Pointwise evaluation computes (target_z-shift)-source_z, not
    # target_z-f32(source_z+shift). The latter loses a four-unit displacement.
    velocity, gradient = image_fields()
    assert np.linalg.norm(velocity[1]) > 0
    for actual, reference in ((0, 1), (2, 3)):
        np.testing.assert_allclose(velocity[actual], velocity[reference], rtol=2e-6, atol=1e-10)
        np.testing.assert_allclose(gradient[actual], gradient[reference], rtol=2e-6, atol=1e-10)
