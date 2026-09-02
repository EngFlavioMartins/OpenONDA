"""Structural and numerical tests for the VPM FMM hierarchy components."""

import numpy as np
import taichi as ti

from source.solvers.vpm.kernels.base import make_vortex_kernel
from source.solvers.vpm.physics.induction.direct import DirectInduction
from source.solvers.vpm.physics.induction.fmm import FMMTree, interaction_lists
from source.solvers.vpm.physics.induction.fmm.local_expansions import l2l, l2p, m2l
from source.solvers.vpm.physics.induction.fmm.multipoles import m2m, p2m
from source.solvers.vpm.physics.induction.fmm.near_field import p2p_velocity


class _Field:
    def __init__(self, values):
        self._values = np.asarray(values)

    def to_numpy(self):
        return self._values


def test_fmm_tree_owns_deterministic_stage_geometry_and_core_metadata():
    position = np.array([[-1.0, 0.0, 0.0], [-0.8, 0.0, 0.0], [0.8, 0.0, 0.0], [1.0, 0.0, 0.0]])
    strength = np.arange(12, dtype=float).reshape(4, 3)
    core_radius = np.array([0.1, 0.2, 0.3, 0.4])
    tree = FMMTree(leaf_capacity=1)

    tree.build(_Field(position), _Field(strength), _Field(core_radius), 4)

    assert len(tree.cells) == 4
    assert sum(len(cell.indices) for cell in tree.cells) == 4
    assert max(cell.max_core_radius for cell in tree.cells) == 0.4
    assert len(interaction_lists(tree, tolerance=1.0e-3)) == 16


def test_multipole_and_local_translations_preserve_leading_coefficients():
    position = np.array([[0.0, 0.0, 0.0], [0.2, -0.1, 0.3]])
    strength = np.array([[0.4, -0.2, 0.1], [-0.1, 0.3, 0.5]])
    child_a = p2m(position[:1], strength[:1], np.zeros(3))
    child_b = p2m(position[1:], strength[1:], np.array([0.2, -0.1, 0.3]))
    parent = m2m([child_a, child_b], [np.zeros(3), np.array([0.2, -0.1, 0.3])], np.zeros(3))

    np.testing.assert_allclose(parent["circulation"], strength.sum(axis=0))
    local = m2l(parent, np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(l2p(l2l(local, np.array([0.1, 0.0, 0.0]))), parent["circulation"])


def test_near_field_p2p_uses_the_shared_radial_kernel_and_excludes_self_pairs():
    kernel = make_vortex_kernel("GAUSSIAN")
    position = np.array([[0.0, 0.0, 0.0], [0.4, 0.0, 0.0]])
    strength = np.array([[0.0, 0.2, 0.0], [0.0, -0.1, 0.0]])
    core_radius = np.array([0.1, 0.3])

    actual = p2p_velocity(kernel, position, position, strength, core_radius, core_radius)
    expected = kernel.velocity_pair(
        position[0] - position[1], strength[1], core_radius[0], core_radius[1]
    )

    np.testing.assert_allclose(actual[0], expected)
    np.testing.assert_allclose(
        actual[1],
        kernel.velocity_pair(
            position[1] - position[0], strength[0], core_radius[1], core_radius[0]
        ),
    )


def test_fmm_stage_velocity_and_rate_share_the_supplied_temporary_state(tmp_path):
    from openonda.vpm import Backup, FMMInduction, Numerics, ViscousConfig, VPMCase, VPMSolver

    rng = np.random.default_rng(20260901)
    position = rng.uniform(-1.0, 1.0, size=(64, 3)).astype(np.float32)
    strength = rng.normal(scale=0.01, size=(64, 3)).astype(np.float32)
    radius = rng.uniform(0.05, 0.15, size=64).astype(np.float32)
    solver = VPMSolver(
        VPMCase(
            directory=tmp_path,
            backup=Backup(0),
            numerics=Numerics(
                compute_device="CPU",
                max_n_particles=64,
                max_evaluation_points=64,
                induction=FMMInduction(tolerance=1.0e-3, leaf_capacity=1),
                viscous=ViscousConfig.inviscid(particle_spacing=0.2),
                verbose=False,
            ),
        )
    )
    solver.add_vortex_particles(
        position=position,
        velocity=np.zeros_like(position),
        vortex_strength=strength,
        core_radius=radius,
        particle_volume=np.full(64, 0.008, dtype=np.float32),
        kinematic_viscosity=np.zeros(64, dtype=np.float32),
    )
    direct = DirectInduction(solver.physics, max_n_particles=64)
    velocity_fmm = ti.Vector.field(3, dtype=ti.f32, shape=(64,))
    rate_fmm = ti.Vector.field(3, dtype=ti.f32, shape=(64,))
    velocity_direct = ti.Vector.field(3, dtype=ti.f32, shape=(64,))
    rate_direct = ti.Vector.field(3, dtype=ti.f32, shape=(64,))
    solver.induction.evaluate_stage(
        position=solver.particles.position,
        vortex_strength=solver.particles.vortex_strength,
        core_radius=solver.particles.core_radius,
        count=64,
        velocity_out=velocity_fmm,
        vortex_strength_rate_out=rate_fmm,
    )
    direct.evaluate_stage(
        position=solver.particles.position,
        vortex_strength=solver.particles.vortex_strength,
        core_radius=solver.particles.core_radius,
        count=64,
        velocity_out=velocity_direct,
        vortex_strength_rate_out=rate_direct,
    )

    reference = velocity_direct.to_numpy()
    error = np.linalg.norm(velocity_fmm.to_numpy() - reference) / np.linalg.norm(reference)
    assert solver.induction.diagnostics.m2l_interactions > 0
    assert error < 2.0e-2
    rate_error = np.linalg.norm(rate_fmm.to_numpy() - rate_direct.to_numpy()) / np.linalg.norm(
        rate_direct.to_numpy()
    )
    assert rate_error < 5.0e-2
    assert solver.induction.diagnostics.hierarchical_strength_rates == 1
    assert solver.induction.diagnostics.direct_strength_rate_fallbacks == 0


def test_fmm_velocity_error_decreases_with_tighter_requested_tolerance(tmp_path):
    from openonda.vpm import Backup, FMMInduction, Numerics, ViscousConfig, VPMCase, VPMSolver

    rng = np.random.default_rng(20260902)
    count = 48
    position = rng.uniform(-1.0, 1.0, size=(count, 3)).astype(np.float32)
    strength = rng.normal(scale=0.01, size=(count, 3)).astype(np.float32)
    radius = np.full(count, 0.08, dtype=np.float32)
    solver = VPMSolver(
        VPMCase(
            directory=tmp_path,
            backup=Backup(0),
            numerics=Numerics(
                compute_device="CPU",
                max_n_particles=count,
                max_evaluation_points=count,
                viscous=ViscousConfig.inviscid(particle_spacing=0.2),
                verbose=False,
            ),
        )
    )
    solver.add_vortex_particles(
        position=position,
        velocity=np.zeros_like(position),
        vortex_strength=strength,
        core_radius=radius,
        particle_volume=np.full(count, 0.008, dtype=np.float32),
        kinematic_viscosity=np.zeros(count, dtype=np.float32),
    )
    direct = DirectInduction(solver.physics, max_n_particles=count)
    reference = ti.Vector.field(3, dtype=ti.f32, shape=(count,))
    reference_rate = ti.Vector.field(3, dtype=ti.f32, shape=(count,))
    direct.evaluate_stage(
        position=solver.particles.position,
        vortex_strength=solver.particles.vortex_strength,
        core_radius=solver.particles.core_radius,
        count=count,
        velocity_out=reference,
        vortex_strength_rate_out=reference_rate,
    )
    errors = []
    for tolerance in (1.0e-2, 1.0e-3, 1.0e-4):
        induction = FMMInduction(
            solver.physics, tolerance=tolerance, max_n_particles=count, leaf_capacity=1
        )
        velocity = ti.Vector.field(3, dtype=ti.f32, shape=(count,))
        rate = ti.Vector.field(3, dtype=ti.f32, shape=(count,))
        induction.evaluate_stage(
            position=solver.particles.position,
            vortex_strength=solver.particles.vortex_strength,
            core_radius=solver.particles.core_radius,
            count=count,
            velocity_out=velocity,
            vortex_strength_rate_out=rate,
        )
        errors.append(
            np.linalg.norm(velocity.to_numpy() - reference.to_numpy())
            / np.linalg.norm(reference.to_numpy())
        )

    assert errors[0] >= errors[1] >= errors[2]
