"""Traversal scheduling changes execution order, never source arithmetic."""

import numpy as np
import pytest
import taichi as ti

from source.solvers.vpm.physics.base import PhysicsBase
from source.solvers.vpm.physics.induction.treecode.evaluator import TreecodeInduction


@pytest.mark.parametrize("kernel", ["GAUSSIAN", "WINCKELMANS"])
def test_sorted_stage_targets_preserve_every_field_and_source(kernel):
    """Random source order and heterogeneous cores exercise nontrivial target permutations."""
    ti.reset()
    ti.init(arch=ti.cpu, default_fp=ti.f32, cpu_max_num_threads=2, offline_cache=False)
    try:
        rng = np.random.default_rng(427)
        count = 71
        physics = PhysicsBase(
            particle_kernel=kernel, max_n_particles=count, accumulator_dtype=ti.f32
        )
        position, strength, velocity, rate = [
            ti.Vector.field(3, ti.f32, shape=count) for _ in range(4)
        ]
        radius = ti.field(ti.f32, shape=count)
        gradient = ti.Matrix.field(3, 3, ti.f32, shape=count)
        position.from_numpy(rng.normal(size=(count, 3)))
        strength.from_numpy(rng.normal(size=(count, 3)))
        radius.from_numpy(rng.uniform(0.1, 0.5, count))
        inputs = [field.to_numpy().copy() for field in (position, strength, radius)]
        results = []
        for sorted_targets, block in ((False, 128), (True, 32)):
            backend = TreecodeInduction(
                theta=0.3,
                multipole_order=3,
                sort_particle_targets=sorted_targets,
                traversal_block_dim=block,
            )
            copy = backend.build()
            assert copy.sort_particle_targets == sorted_targets
            assert copy.traversal_block_dim == block
            backend.bind(physics)
            backend.evaluate_stage(
                position=position,
                vortex_strength=strength,
                core_radius=radius,
                count=count,
                velocity_out=velocity,
                vortex_strength_rate_out=rate,
                velocity_gradient_out=gradient,
            )
            results.append([field.to_numpy().copy() for field in (velocity, rate, gradient)])
        for original, sorted_result in zip(*results, strict=True):
            np.testing.assert_array_equal(original, sorted_result)
        for field, original in zip((position, strength, radius), inputs, strict=True):
            np.testing.assert_array_equal(field.to_numpy(), original)
    finally:
        ti.reset()


@pytest.mark.parametrize(
    "settings",
    [
        {"sort_particle_targets": 1},
        {"traversal_block_dim": True},
        {"traversal_block_dim": -1},
        {"traversal_block_dim": 3.5},
    ],
)
def test_public_schedule_rejects_ambiguous_controls(settings):
    """Do not silently truncate or coerce malformed scheduling parameters."""
    with pytest.raises(ValueError, match="treecode"):
        TreecodeInduction(**settings)
