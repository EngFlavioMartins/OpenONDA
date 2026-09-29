"""Target count never grows the source tree or its fixed query scratch."""

import numpy as np
import pytest
import taichi as ti

from source.solvers.vpm.physics.base import PhysicsBase
from source.solvers.vpm.physics.induction.treecode.evaluator import TreecodeInduction


@pytest.fixture(autouse=True)
def runtime():
    ti.reset()
    ti.init(arch=ti.cpu, default_fp=ti.f32, cpu_max_num_threads=2, offline_cache=False)
    yield
    ti.reset()


def test_tree_allocates_declared_source_ceiling_once(monkeypatch):
    import source.solvers.vpm.physics.induction.treecode.lbvh as lbvh

    allocations = []

    class FakeTree:
        def __init__(self, **kwargs):
            allocations.append(kwargs)
            self.theta = kwargs["theta"]

        def set_kernel_type(self, _value):
            pass

        def set_multipole_order(self, _value):
            pass

        def set_sort_particle_targets(self, _value):
            pass

    monkeypatch.setattr(lbvh, "TaichiTreecode", FakeTree)
    physics = PhysicsBase.__new__(PhysicsBase)
    physics.max_n_particles = 1_000_000
    physics.max_evaluation_points = 7
    physics._treecode = None
    physics._treecode_max_particles = 0
    physics.particle_kernel = "GAUSSIAN"
    physics.treecode_multipole_order = 3
    physics.treecode_sort_particle_targets = False
    physics.treecode_traversal_block_dim = 128
    first = physics._get_or_create_treecode(2, 0.3)
    assert physics._get_or_create_treecode(750_000, 0.3) is first
    assert len(allocations) == 1
    assert allocations[0]["max_n_particles"] == 1_000_000
    assert allocations[0]["max_nodes"] == 2_000_000
    assert allocations[0]["max_evaluation_points"] == 7
    with pytest.raises(ValueError, match="declared max_n_particles"):
        physics._get_or_create_treecode(1_000_001, 0.3)


def test_native_tree_target_batches_preserve_velocity_gradient_and_tree_identity():
    source_count = 2
    targets = np.array(
        [[0.02 * i, 0.03 * (i % 3), 0.1 + 0.01 * i] for i in range(8)],
        dtype=np.float32,
    )
    position = ti.Vector.field(3, ti.f32, shape=source_count)
    strength = ti.Vector.field(3, ti.f32, shape=source_count)
    radius = ti.field(ti.f32, shape=source_count)
    background = ti.Vector.field(3, ti.f32, shape=())
    position.from_numpy(np.array([[0, 0, 0], [0.2, -0.1, 0.1]], dtype=np.float32))
    strength.from_numpy(np.array([[0.03, 0.01, -0.02], [-0.01, 0.02, 0.03]], dtype=np.float32))
    radius.from_numpy(np.array([0.08, 0.1], dtype=np.float32))
    background[None] = [0.2, -0.1, 0.05]

    class Cloud:
        def __len__(self):
            return source_count

    cloud = Cloud()
    cloud.position = position
    cloud.vortex_strength = strength
    cloud.core_radius = radius
    cloud.velocity_background = background
    cloud.state_revision = 1

    reference = PhysicsBase("GAUSSIAN", source_count, ti.f32, max_evaluation_points=3)
    expected_velocity = reference.compute_target_velocity(cloud, targets)
    expected_gradient = reference.compute_target_velocity_gradient(cloud, targets)
    expected_vorticity = np.concatenate(
        [reference.compute_target_vorticity(cloud, targets[i : i + 2]) for i in range(0, 8, 2)]
    )
    np.testing.assert_allclose(
        reference.compute_target_vorticity(cloud, targets), expected_vorticity, rtol=0, atol=0
    )
    expected_transport = np.concatenate(
        [
            reference.compute_transport_target_velocity(cloud, targets[i : i + 2], 0.09)
            for i in range(0, 8, 2)
        ]
    )
    np.testing.assert_allclose(
        reference.compute_transport_target_velocity(cloud, targets, 0.09),
        expected_transport,
        rtol=0,
        atol=0,
    )
    np.testing.assert_allclose(
        reference.compute_velocities_from_arrays(
            position.to_numpy(), strength.to_numpy(), radius.to_numpy(), targets
        ),
        np.concatenate(
            [
                reference.compute_velocities_from_arrays(
                    position.to_numpy(), strength.to_numpy(), radius.to_numpy(), targets[i : i + 2]
                )
                for i in range(0, 8, 2)
            ]
        ),
        rtol=0,
        atol=0,
    )

    physics = PhysicsBase("GAUSSIAN", source_count, ti.f32, max_evaluation_points=3)
    physics.induction = TreecodeInduction(theta=0.3, multipole_order=3).bind(physics)
    velocity, gradient = physics.compute_target_velocity_and_gradients_hierarchical(
        cloud, targets, theta=0.3
    )
    tree = physics._treecode
    assert tree is not None
    np.testing.assert_allclose(velocity, expected_velocity, rtol=2e-4, atol=2e-5)
    np.testing.assert_allclose(gradient, expected_gradient, rtol=2e-4, atol=2e-5)
    assert physics.compute_target_velocity_hierarchical(cloud, targets, theta=0.3).shape == (8, 3)
    assert physics.compute_target_velocity_gradient_hierarchical(
        cloud, targets, theta=0.3
    ).shape == (8, 9)
    np.testing.assert_allclose(
        physics.compute_target_velocity(cloud, targets), expected_velocity, rtol=2e-4, atol=2e-5
    )
    np.testing.assert_allclose(
        physics.compute_target_velocity_gradient(cloud, targets),
        expected_gradient,
        rtol=2e-4,
        atol=2e-5,
    )
    destination = ti.Vector.field(3, ti.f32, shape=len(targets))
    assert physics.compute_target_velocity(cloud, targets, target_velocity=destination) is None
    np.testing.assert_allclose(destination.to_numpy(), expected_velocity, rtol=2e-4, atol=2e-5)
    assert physics._treecode is tree
    assert tree.max_n_particles == 8192
    assert tree.max_evaluation_points == 3


def test_native_two_source_hundred_thousand_target_query_uses_fixed_scratch():
    source_count = 2
    physics = PhysicsBase("GAUSSIAN", source_count, ti.f32)
    position = ti.Vector.field(3, ti.f32, shape=source_count)
    strength = ti.Vector.field(3, ti.f32, shape=source_count)
    radius = ti.field(ti.f32, shape=source_count)
    background = ti.Vector.field(3, ti.f32, shape=())
    position.from_numpy(np.array([[0, 0, 0], [0.2, 0, 0]], dtype=np.float32))
    strength.from_numpy(np.array([[0, 0, 0.03], [0, 0.01, 0]], dtype=np.float32))
    radius.fill(0.08)
    background[None] = [0.1, 0, 0]

    class Cloud:
        def __len__(self):
            return source_count

    cloud = Cloud()
    cloud.position = position
    cloud.vortex_strength = strength
    cloud.core_radius = radius
    cloud.velocity_background = background
    target = np.array([[0.05, 0.1, 0.2]], dtype=np.float32)
    expected = physics.compute_target_velocity(cloud, target)
    targets = np.repeat(target, 100_001, axis=0)
    actual = physics.compute_target_velocity(cloud, targets)
    assert actual.shape == (100_001, 3)
    np.testing.assert_allclose(actual, np.repeat(expected, len(targets), axis=0), rtol=0, atol=0)
    assert physics._target_field_size == 65_536


def test_empty_source_target_field_receives_background_without_query_allocation():
    physics = PhysicsBase("GAUSSIAN", 2, ti.f32, max_evaluation_points=3)

    class EmptyCloud:
        def __len__(self):
            return 0

    cloud = EmptyCloud()
    cloud.velocity_background = ti.Vector.field(3, ti.f32, shape=())
    cloud.velocity_background[None] = [0.1, -0.2, 0.3]
    targets = np.zeros((8, 3), dtype=np.float32)
    destination = ti.Vector.field(3, ti.f32, shape=len(targets))
    assert physics.compute_target_velocity(cloud, targets, target_velocity=destination) is None
    np.testing.assert_allclose(
        destination.to_numpy(), np.tile([0.1, -0.2, 0.3], (len(targets), 1)), atol=1e-7
    )
    assert physics.compute_target_velocity(cloud, targets[:0], target_velocity=destination) is None
