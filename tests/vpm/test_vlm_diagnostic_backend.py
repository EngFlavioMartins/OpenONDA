"""Scheduled diagnostics retain their physical filter and do not alter output clocks."""

import numpy as np
import pytest
import taichi as ti

from source.solvers.vpm.boundary_elements.vlm.config import VLMSetup, VLMSurfaceSetup
from source.solvers.vpm.physics.base import PhysicsBase
from source.solvers.vpm.physics.induction.direct import DirectInduction
from source.solvers.vpm.physics.induction.treecode.evaluator import TreecodeInduction


@pytest.fixture(autouse=True)
def runtime():
    ti.reset()
    ti.init(arch=ti.cpu, default_fp=ti.f32, cpu_max_num_threads=2, offline_cache=False)
    yield
    ti.reset()


@pytest.mark.parametrize("backend", [DirectInduction, TreecodeInduction])
def test_uniform_transport_filter_backend_matches_direct_and_retains_sources(backend):
    """Heterogeneous source cores still use (source+target)/2, even through a tree."""
    rng = np.random.default_rng(813)
    count, targets = 173, rng.normal(size=(51, 3))
    physics = PhysicsBase(
        particle_kernel="GAUSSIAN",
        max_n_particles=count,
        max_evaluation_points=51,
        accumulator_dtype=ti.f32,
    )
    physics.induction = (
        backend(theta=0.3, multipole_order=3) if backend is TreecodeInduction else backend()
    ).bind(physics)
    position = ti.Vector.field(3, ti.f32, shape=count)
    strength = ti.Vector.field(3, ti.f32, shape=count)
    radius = ti.field(ti.f32, shape=count)
    background = ti.Vector.field(3, ti.f32, shape=())
    position.from_numpy(rng.normal(size=(count, 3)))
    strength.from_numpy(rng.normal(size=(count, 3)))
    radius.from_numpy(rng.uniform(0.04, 0.6, count))
    background[None] = [0.1, 0.2, -0.3]

    class Cloud:
        def __len__(self):
            return count

    cloud = Cloud()
    cloud.position, cloud.vortex_strength, cloud.core_radius, cloud.velocity_background = (
        position,
        strength,
        radius,
        background,
    )
    before = [field.to_numpy().copy() for field in (position, strength, radius)]
    for core in (0.1, 0.5):
        for include in (False, True):
            expected = physics.compute_transport_target_velocity(
                cloud, targets, core, include_freestream=include
            )
            actual = physics.compute_transport_target_velocity(
                cloud, targets, core, include_freestream=include, use_induction_backend=True
            )
            error = np.linalg.norm(actual - expected) / np.linalg.norm(expected)
            assert error < (3e-6 if backend is DirectInduction else 1e-3)
    for field, old in zip((position, strength, radius), before, strict=True):
        np.testing.assert_array_equal(field.to_numpy(), old)
    assert "uniform_target_core" in physics.transport_target_operator_label(
        use_induction_backend=True
    )


@pytest.mark.parametrize("backend", [DirectInduction, TreecodeInduction])
def test_uniform_transport_filter_empty_cloud_retains_background(backend):
    """An emptied cloud must not retain sources from the preceding tree query."""
    physics = PhysicsBase(
        particle_kernel="GAUSSIAN",
        max_n_particles=4,
        max_evaluation_points=4,
        accumulator_dtype=ti.f32,
    )
    physics.induction = backend().bind(physics)

    class Cloud:
        count = 1

        def __len__(self):
            return self.count

    cloud = Cloud()
    cloud.position = ti.Vector.field(3, ti.f32, shape=4)
    cloud.vortex_strength = ti.Vector.field(3, ti.f32, shape=4)
    cloud.core_radius = ti.field(ti.f32, shape=4)
    cloud.velocity_background = ti.Vector.field(3, ti.f32, shape=())
    cloud.core_radius.fill(0.2)
    cloud.vortex_strength[0] = [0, 0, 1]
    cloud.velocity_background[None] = [1, 2, 3]
    targets = np.array([[0.1, 0, 0], [0, 0.1, 0]])
    physics.compute_transport_target_velocity(cloud, targets, 0.2, use_induction_backend=True)
    cloud.count = 0
    for include in (True, False):
        actual = physics.compute_transport_target_velocity(
            cloud, targets, 0.2, use_induction_backend=True, include_freestream=include
        )
        expected = np.broadcast_to([1, 2, 3] if include else [0, 0, 0], (2, 3))
        np.testing.assert_allclose(actual, expected, atol=1e-7)


@pytest.mark.parametrize(
    "field,value",
    [
        ("density", np.nan),
        ("density", np.inf),
        ("kinematic_viscosity", np.nan),
        ("kinematic_viscosity", np.inf),
        ("freestream_velocity", (0, 0, np.nan)),
        ("dtype", "f16"),
        ("max_n_panels", True),
        ("max_n_panels", 1.5),
        ("linear_solver", "unknown"),
    ],
)
def test_invalid_physical_configuration_fails_before_solver_allocation(field, value):
    """Nonfinite physics and invalid allocation/solver choices cannot reach kernels."""
    with pytest.raises(ValueError):
        VLMSetup(surfaces=(VLMSurfaceSetup({}),), **{field: value})
