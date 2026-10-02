"""One configured backend call for paired velocity/Jacobian target queries."""

from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import taichi as ti

from source.solvers.vpm.core.solver import VPMSolver
from source.solvers.vpm.physics.base import PhysicsBase


class _Cloud:
    def __init__(self, count):
        self.count = count
        self.position = ti.Vector.field(3, ti.f32, shape=max(1, count))
        self.vortex_strength = ti.Vector.field(3, ti.f32, shape=max(1, count))
        self.core_radius = ti.field(ti.f32, shape=max(1, count))
        self.velocity_background = ti.Vector.field(3, ti.f32, shape=())
        self.velocity_background[None] = [0.2, -0.1, 0.05]

    def __len__(self):
        return self.count

    def velocity_background_cpu(self):
        return np.asarray(self.velocity_background[None], dtype=np.float32)


@pytest.fixture
def runtime():
    ti.reset()
    ti.init(arch=ti.cpu, default_fp=ti.f32, offline_cache=False, cpu_max_num_threads=2)
    yield
    ti.reset()


def _backend(name, physics):
    from source.solvers.vpm.physics.induction.direct import DirectInduction
    from source.solvers.vpm.physics.induction.planar import PlanarInduction
    from source.solvers.vpm.physics.induction.slip_slab import SlipSlabInduction

    if "fmm" in name:
        from source.solvers.vpm.physics.induction.fmm import FMMInduction

        base = FMMInduction()
    else:
        base = DirectInduction()
    if name.startswith("slab"):
        base = SlipSlabInduction(base, z_min=-0.5, z_max=0.5, max_shells=129, tail_tolerance=1e-4)
    elif name == "planar":
        base = PlanarInduction()
    physics.induction = base.bind(physics)
    physics.velocity_method = "DIRECT" if name == "direct" else name.upper()
    return physics.induction


@pytest.mark.parametrize("name", ["direct", "planar", "slab-direct", "fmm", "slab-fmm"])
def test_fused_backend_fields_and_mixed_trace_match_separate_calls(runtime, name):
    cloud = _Cloud(4)
    positions = np.array([[0, 0, 0], [0.2, -0.1, 0], [-0.15, 0.2, 0], [0.1, 0.1, 0]], np.float32)
    strengths = np.array(
        [[0.01, 0.02, 0.03], [-0.02, 0.01, 0.01], [0.01, -0.01, 0.02], [0.02, 0.01, -0.03]],
        np.float32,
    )
    if name == "planar":
        strengths[:, :2] = 0
        cloud.velocity_background[None] = [0.2, -0.1, 0.0]
    cloud.position.from_numpy(positions)
    cloud.vortex_strength.from_numpy(strengths)
    cloud.core_radius.from_numpy(np.array([0.08, 0.12, 0.2, 0.1], np.float32))
    physics = PhysicsBase("GAUSSIAN", 4, ti.f32, max_evaluation_points=3)
    backend = _backend(name, physics)
    original = backend.evaluate_targets
    calls = []

    def counted(**kwargs):
        calls.append(
            (
                kwargs["target_count"],
                kwargs["target_velocity"] is not None,
                kwargs["target_velocity_gradient"] is not None,
            )
        )
        return original(**kwargs)

    backend.evaluate_targets = counted
    target_field = physics.target_position
    targets = np.array([[0.02 * i, -0.05 + 0.03 * (i % 3), 0.07] for i in range(8)], np.float32)
    for mutated in (False, True):
        if mutated:
            # Reuse all source objects and target storage, without relying on
            # object identity to indicate that their numerical values changed.
            cloud.position.from_numpy(positions + np.array([0.025, -0.01, 0], np.float32))
            cloud.vortex_strength.from_numpy(strengths * np.float32(1.3))
            cloud.core_radius.from_numpy(np.array([0.1, 0.09, 0.13, 0.14], np.float32))
            targets[:, 0] += 0.04
        for include_freestream in (False, True):
            calls.clear()
            expected_velocity = physics.compute_target_velocity(
                cloud, targets, include_freestream=include_freestream
            )
            expected_gradient = physics.compute_target_velocity_gradient(cloud, targets)
            assert len(calls) == 6
            calls.clear()
            velocity, gradient = physics.compute_target_velocity_and_gradients_consistent(
                cloud, targets, include_freestream=include_freestream
            )
            assert calls == [(3, True, True), (3, True, True), (2, True, True)]
            assert velocity.shape == (8, 3) and gradient.shape == (8, 9)
            assert velocity.dtype == gradient.dtype == np.float32
            assert physics.target_position is target_field and physics._target_field_size == 3
            np.testing.assert_allclose(velocity, expected_velocity, rtol=2e-6, atol=3e-7)
            np.testing.assert_allclose(gradient, expected_gradient, rtol=2e-6, atol=8e-7)
        normals = np.tile(np.array([1.0, 2.0, -1.0]), (len(targets), 1))
        unit = normals / np.linalg.norm(normals, axis=1)[:, None]
        normal_gradient = np.einsum("fij,fj->fi", expected_gradient.reshape(-1, 3, 3), unit)
        expected_trace = (
            normal_gradient - np.einsum("fi,fi->f", normal_gradient, unit)[:, None] * unit
        )
        owner = SimpleNamespace(
            physics=physics,
            particles=cloud,
            vlm_solver=None,
            n_sources=0,
            _body_induced_fn=None,
            _add_target_velocity_corrections=lambda points, velocity, **kwargs: velocity,
        )
        velocity, trace = VPMSolver.compute_velocity_and_tangential_normal_gradient_at_points(
            owner, targets, normals, particle_spacing=0.04
        )
        np.testing.assert_allclose(velocity, expected_velocity, rtol=2e-6, atol=3e-7)
        np.testing.assert_allclose(trace, expected_trace, rtol=2e-6, atol=8e-7)


def test_fused_empty_and_background_contracts_do_not_dispatch(runtime):
    physics = PhysicsBase("GAUSSIAN", 4, ti.f32, max_evaluation_points=3)
    physics.velocity_method = "FMM"

    def forbidden(**kwargs):
        raise AssertionError("Empty sources or targets must not invoke the backend")

    physics.induction = SimpleNamespace(evaluate_targets=forbidden)
    cloud = _Cloud(0)
    for target_count in (0, 7):
        for include in (False, True):
            velocity, gradient = physics.compute_target_velocity_and_gradients_consistent(
                cloud, np.zeros((target_count, 3)), include_freestream=include
            )
            expected = np.zeros((target_count, 3), dtype=np.float32)
            if include:
                expected += cloud.velocity_background_cpu()
            np.testing.assert_array_equal(velocity, expected)
            np.testing.assert_array_equal(gradient, np.zeros((target_count, 9)))
    cloud.count = 1
    velocity, gradient = physics.compute_target_velocity_and_gradients_consistent(
        cloud, np.zeros((0, 3))
    )
    assert velocity.shape == (0, 3) and gradient.shape == (0, 9)


def test_treecode_fused_route_and_direct_fallback_select_requested_backend():
    particles, points = object(), np.zeros((2, 3))
    result = (np.ones((2, 3)), np.ones((2, 9)))
    calls = []

    def hierarchical(*args, **kwargs):
        calls.append((args, kwargs))
        return result

    owner = SimpleNamespace(
        velocity_method="TREECODE",
        velocity_theta=0.25,
        compute_target_velocity_and_gradients_hierarchical=hierarchical,
    )
    method = MethodType(PhysicsBase.compute_target_velocity_and_gradients_consistent, owner)
    assert method(particles, points, include_freestream=False) is result
    assert len(calls) == 1
    assert calls[0][0] == (particles, points)
    assert calls[0][1] == {"theta": 0.25, "include_freestream": False}
    owner.velocity_method = "DIRECT"
    owner.compute_target_velocity = lambda *args, **kwargs: result[0]
    owner.compute_target_velocity_gradient = lambda *args, **kwargs: result[1]
    actual = method(particles, points)
    assert actual[0] is result[0] and actual[1] is result[1]
