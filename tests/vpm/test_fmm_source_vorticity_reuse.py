"""Backup vorticity reuses FMM storage and rebuilds accepted source contents."""

import numpy as np
import pytest
import taichi as ti

from source.solvers.vpm.physics.base import PhysicsBase
from source.solvers.vpm.physics.induction.fmm import FMMInduction
from source.solvers.vpm.physics.induction.slip_slab import SlipSlabInduction
from source.solvers.vpm.physics.induction.treecode.lbvh import TaichiTreecode


@pytest.fixture(scope="module", autouse=True)
def cpu_runtime():
    ti.init(arch=ti.cpu, cpu_max_num_threads=1, offline_cache=False)
    yield
    ti.reset()


class _Particles:
    def __init__(self, capacity):
        self.count = 2048
        self.position = ti.Vector.field(3, ti.f32, shape=capacity)
        self.vortex_strength = ti.Vector.field(3, ti.f32, shape=capacity)
        self.core_radius = ti.field(ti.f32, shape=capacity)
        self.vorticity = ti.Vector.field(3, ti.f32, shape=capacity)

    def __len__(self):
        return self.count


@pytest.mark.parametrize("wrapped", [False, True], ids=["direct_fmm", "slip_slab_fmm"])
def test_backup_vorticity_reuses_hierarchy_with_changed_sources_and_count_growth(wrapped):
    capacity = 2067
    rng = np.random.default_rng(741)
    position = rng.uniform([-0.9, -0.9, 0.001], [0.9, 0.9, 0.999], (capacity, 3)).astype(np.float32)
    position[:3] = [0.01, 0.02, 0.001]  # duplicate Morton keys near a slip wall
    position[3] = [0.01, 0.02, 0.999]
    strength = rng.normal(0.0, 0.001, (capacity, 3)).astype(np.float32)
    radius = rng.uniform(0.004, 0.03, capacity).astype(np.float32)
    radius[:4] = [0.01, 0.08, 0.2, 0.12]  # unequal pair-mean radii and wide cores
    particles = _Particles(capacity)
    physics = PhysicsBase("GAUSSIAN", capacity, ti.f32, max_evaluation_points=37)
    base = FMMInduction()
    backend = SlipSlabInduction(base, z_min=0.0, z_max=1.0) if wrapped else base
    physics.induction = backend.bind(physics)
    reference = TaichiTreecode(
        max_n_particles=capacity,
        max_nodes=2 * capacity,
        kernel_type="GAUSSIAN",
        hierarchy_only=True,
        max_evaluation_points=37,
    )
    reference_output = ti.Vector.field(3, ti.f32, shape=capacity)
    try:
        particles.position.from_numpy(position)
        particles.vortex_strength.from_numpy(strength)
        particles.core_radius.from_numpy(radius)
        # Model the source hierarchy already present from normal induction.
        base._ensure_workspace(particles.count)
        base.workspace.tree.build(
            particles.position, particles.vortex_strength, particles.core_radius, particles.count
        )
        previous = None
        for change in ("initial", "accepted_contents", "growth", "repeat"):
            # Normal induction can leave valid multipoles for this old source
            # state. The diagnostic rebuild must invalidate them as well.
            base.workspace.prepare_source_multipoles(particles.count)
            base._source_moments_ready = True
            if change == "accepted_contents":
                position[:, :2] *= 0.63
                position[:, 2] = 0.002 + 0.996 * position[:, 2]
                strength *= -0.71
                radius *= 1.04
                particles.position.from_numpy(position)
                particles.vortex_strength.from_numpy(strength)
                particles.core_radius.from_numpy(radius)
            elif change == "growth":
                particles.count = 2059
            workspace, tree = base.workspace, base.workspace.tree
            reference.build(
                particles.position,
                particles.vortex_strength,
                particles.core_radius,
                particles.count,
            )
            reference_output.fill(19.0)
            reference.compute_gaussian_particle_vorticity(reference_output, particles.count)
            expected = reference_output.to_numpy()
            particles.vorticity.fill(19.0)
            physics.compute_vorticities(particles)
            actual = particles.vorticity.to_numpy()
            # The independent hierarchy sums physical sources only, including
            # self terms. Slab images must not enter this stored diagnostic.
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(actual[particles.count :], 19.0)
            assert physics._vorticity_tree is None
            assert base._fixed_source_key is None
            assert not base._source_moments_ready
            assert base.workspace.tree._built_n == particles.count
            if change != "growth":
                assert base.workspace is workspace and base.workspace.tree is tree
            else:
                assert base.workspace.max_n_particles >= particles.count
                assert tree._device_fields.tree is None
            if change == "accepted_contents":
                assert not np.array_equal(actual[: particles.count], previous[: particles.count])
            previous = actual.copy()
    finally:
        reference.destroy()
        if wrapped:
            backend.close_mesh_session()
        base._release_target_workspace()
        if base.workspace is not None:
            base.workspace.destroy()
