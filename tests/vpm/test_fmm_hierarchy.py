"""Structural and numerical tests for the VPM FMM hierarchy components."""

import numpy as np
import pytest
import taichi as ti

from source.solvers.vpm.kernels.base import make_vortex_kernel
from source.solvers.vpm.physics.induction.direct import DirectInduction
from source.solvers.vpm.physics.induction.fmm import FMMInduction, FMMTree, interaction_lists
from source.solvers.vpm.physics.induction.fmm.local_expansions import l2l, l2p, m2l
from source.solvers.vpm.physics.induction.fmm.multipoles import m2m, p2m
from source.solvers.vpm.physics.induction.fmm.near_field import p2p_velocity
from source.solvers.vpm.physics.induction.fmm.reference import HostFMMReference
from source.solvers.vpm.physics.induction.treecode import TreecodeInduction


class _Field:
    def __init__(self, values):
        self._values = np.asarray(values)

    def to_numpy(self):
        return self._values


class _HostPhysics:
    """Minimal transfer surface for qualifying the host FMM without Taichi fields."""

    max_n_particles = 16
    np_dtype = np.float64

    @staticmethod
    def _download_vector_field(values, count):
        return np.asarray(values[:count], dtype=np.float64).copy()

    @staticmethod
    def _download_scalar_field(values, count):
        return np.asarray(values[:count], dtype=np.float64).copy()

    @staticmethod
    def _upload_vector_array(values, output, count):
        output[:count] = values

    @staticmethod
    def _upload_matrix_array(values, output, count):
        output[:count] = values

    @staticmethod
    def _zero_vec3_field(output, count):
        output[:count] = 0.0


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
    translated = l2l(local, np.array([0.1, 0.0, 0.0]))
    np.testing.assert_allclose(
        l2p(translated),
        local["value"] + local["gradient"] @ np.array([0.1, 0.0, 0.0]),
    )


def test_near_field_p2p_uses_the_shared_radial_kernel_and_excludes_self_pairs():
    kernel = make_vortex_kernel("GAUSSIAN")
    position = np.array([[0.0, 0.0, 0.0], [0.4, 0.0, 0.0]])
    strength = np.array([[0.0, 0.2, 0.0], [0.0, -0.1, 0.0]])
    core_radius = np.array([0.1, 0.3])

    actual = p2p_velocity(
        kernel,
        position,
        position,
        strength,
        core_radius,
        core_radius,
        exclude_self=True,
    )
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


def test_near_field_p2p_preserves_matching_indices_for_distinct_sets():
    kernel = make_vortex_kernel("GAUSSIAN")
    target_position = np.array([[0.0, 0.0, 0.0], [0.4, 0.0, 0.0]])
    source_position = np.array([[0.1, 0.1, 0.0], [0.5, 0.1, 0.0]])
    source_strength = np.array([[0.0, 0.2, 0.0], [0.0, -0.1, 0.0]])
    target_core = np.array([0.1, 0.3])
    source_core = np.array([0.2, 0.4])

    actual = p2p_velocity(
        kernel,
        target_position,
        source_position,
        source_strength,
        target_core,
        source_core,
    )
    displacement = target_position[:, None, :] - source_position[None, :, :]
    expected = kernel.velocity_pair(
        displacement,
        source_strength[None, :, :],
        target_core[:, None],
        source_core[None, :],
    ).sum(axis=1)

    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("induction_type", (DirectInduction, TreecodeInduction, FMMInduction))
@pytest.mark.parametrize("scheme", ("DIRECT", "TRANSPOSED", "MIXED"))
def test_backends_evaluate_the_selected_stretching_on_the_supplied_stage(
    tmp_path, monkeypatch, induction_type, scheme
):
    from openonda.vpm import Backup, Numerics, ViscousConfig, VPMCase, VPMSolver

    rng = np.random.default_rng(20260901)
    count = 64
    position = rng.normal(scale=0.08, size=(count, 3)).astype(np.float32)
    position[:32, 0] -= 4.0
    position[32:, 0] += 4.0
    strength = rng.normal(scale=0.01, size=(count, 3)).astype(np.float32)
    radius = rng.uniform(0.008, 0.016, size=count).astype(np.float32)
    solver = VPMSolver(
        VPMCase(
            directory=tmp_path,
            backup=Backup(0),
            numerics=Numerics(
                compute_device="CPU",
                max_n_particles=count,
                max_evaluation_points=count,
                induction=induction_type(stretching_scheme=scheme),
                viscous=ViscousConfig.inviscid(particle_spacing=0.2),
                verbose=False,
            ),
        )
    )
    # Independent stage buffers ensure that induction never reads accepted state.
    stage_position = ti.Vector.field(3, dtype=ti.f32, shape=count)
    stage_strength = ti.Vector.field(3, dtype=ti.f32, shape=count)
    stage_radius = ti.field(dtype=ti.f32, shape=count)
    stage_position.from_numpy(position)
    stage_strength.from_numpy(strength)
    stage_radius.from_numpy(radius)
    velocity = ti.Vector.field(3, dtype=ti.f32, shape=count)
    rate = ti.Vector.field(3, dtype=ti.f32, shape=count)
    gradient = ti.Matrix.field(3, 3, dtype=ti.f32, shape=count)

    if induction_type is not DirectInduction:

        def forbid_direct_fallback(*args, **kwargs):
            raise AssertionError("accelerated stretching fell back to direct summation")

        for name in (
            "compute_velocity_and_stretching_rate_kernel",
            "compute_stretching_rate_batch_kernel",
            "compute_stretching_rate_kernel",
            "compute_velocity_gradients_kernel",
        ):
            monkeypatch.setattr(solver.physics, name, forbid_direct_fallback)

    # Analytical pair Jacobians are independently checked against finite
    # differences in test_vortex_kernel_contract, and use unequal pair cores.
    kernel = make_vortex_kernel("GAUSSIAN")
    displacement = position[:, None, :].astype(float) - position[None, :, :]
    expected_velocity = kernel.velocity_pair(
        displacement, strength[None, :, :], radius[:, None], radius[None, :]
    ).sum(axis=1)
    expected_gradient = kernel.gradient_pair(
        displacement, strength[None, :, :], radius[:, None], radius[None, :]
    ).sum(axis=1)
    operator = {
        "DIRECT": expected_gradient,
        "TRANSPOSED": expected_gradient.swapaxes(1, 2),
        "MIXED": 0.5 * (expected_gradient + expected_gradient.swapaxes(1, 2)),
    }[scheme]
    expected_rate = np.einsum("nij,nj->ni", operator, strength)
    rate_tolerance = 3.0e-5 if induction_type is DirectInduction else 1.5e-2
    stage = {
        "position": stage_position,
        "vortex_strength": stage_strength,
        "core_radius": stage_radius,
        "count": count,
        "velocity_out": velocity,
        "vortex_strength_rate_out": rate,
    }
    try:
        # Optional gradient output must not select a different rate formula.
        for with_gradient in (False, True):
            solver.induction.evaluate_stage(
                **stage, velocity_gradient_out=gradient if with_gradient else None
            )
            velocity_error = np.linalg.norm(velocity.to_numpy() - expected_velocity)
            assert velocity_error / np.linalg.norm(expected_velocity) < 5.0e-3
            rate_error = np.linalg.norm(rate.to_numpy() - expected_rate)
            assert rate_error / np.linalg.norm(expected_rate) < rate_tolerance
            if with_gradient:
                error = np.linalg.norm(gradient.to_numpy() - expected_gradient)
                assert error / np.linalg.norm(expected_gradient) < 1.5e-2
        if induction_type is FMMInduction:
            diagnostics = solver.induction.diagnostics
            assert diagnostics.m2l_interactions > 0
            assert diagnostics.hierarchical_strength_rates == 2
            assert diagnostics.direct_strength_rate_fallbacks == 0
            assert diagnostics.host_particle_transfers == 0
            assert diagnostics.hierarchy_builds == 2
            assert diagnostics.stretching_scheme == scheme
            if scheme == "TRANSPOSED":
                assert diagnostics.last_relative_rate_defect <= 1.0e-3
        solver.induction.evaluate_stage(**stage, strength_rate_enabled=False)
        np.testing.assert_array_equal(rate.to_numpy(), 0.0)
    finally:
        solver.close()


@pytest.mark.parametrize("scheme", ("DIRECT", "TRANSPOSED", "MIXED"))
def test_private_host_fmm_reference_qualifies_all_radial_kernels(scheme):
    rng = np.random.default_rng(20260903)
    count = 16
    position = rng.uniform(-1.0, 1.0, size=(count, 3))
    strength = rng.normal(scale=0.01, size=(count, 3))
    radius = rng.uniform(0.08, 0.15, size=count)
    physics = _HostPhysics()

    for name in ("GAUSSIAN", "HIGH_ORDER_GAUSSIAN", "SUPER_GAUSSIAN", "WINCKELMANS"):
        induction = HostFMMReference(
            physics,
            kernel=make_vortex_kernel(name),
            tolerance=1.0e-3,
            max_n_particles=count,
            leaf_capacity=1,
            stretching_scheme=scheme,
        )
        velocity = np.zeros((count, 3), dtype=np.float64)
        gradient = np.zeros((count, 3, 3), dtype=np.float64)
        rate = np.zeros((count, 3), dtype=np.float64)
        induction.evaluate_stage(
            position=position,
            vortex_strength=strength,
            core_radius=radius,
            count=count,
            velocity_out=velocity,
            vortex_strength_rate_out=rate,
            velocity_gradient_out=gradient,
        )

        kernel = make_vortex_kernel(name)
        displacement = position[:, None, :] - position[None, :, :]
        expected_velocity = kernel.velocity_pair(
            displacement,
            strength[None, :, :],
            radius[:, None],
            radius[None, :],
        )
        expected_gradient = kernel.gradient_pair(
            displacement,
            strength[None, :, :],
            radius[:, None],
            radius[None, :],
        )
        diagonal = np.arange(count)
        expected_velocity[diagonal, diagonal] = 0.0
        expected_gradient[diagonal, diagonal] = 0.0
        expected_velocity = expected_velocity.sum(axis=1)
        expected_gradient = expected_gradient.sum(axis=1)
        operator = {
            "DIRECT": expected_gradient,
            "TRANSPOSED": expected_gradient.swapaxes(1, 2),
            "MIXED": 0.5 * (expected_gradient + expected_gradient.swapaxes(1, 2)),
        }[scheme]
        expected_rate = np.einsum("nij,nj->ni", operator, strength)

        velocity_error = np.linalg.norm(velocity - expected_velocity) / np.linalg.norm(
            expected_velocity
        )
        gradient_error = np.linalg.norm(gradient - expected_gradient) / np.linalg.norm(
            expected_gradient
        )
        rate_error = np.linalg.norm(rate - expected_rate) / np.linalg.norm(expected_rate)

        assert velocity_error < 3.0e-2
        assert gradient_error < 3.0e-2
        assert rate_error < 5.0e-2
        assert induction.diagnostics.m2l_interactions > 0
        assert induction.diagnostics.l2l_operations > 0
        assert induction.diagnostics.nonzero_l2l_operations > 0
        assert induction.diagnostics.direct_strength_rate_fallbacks == 0
        assert induction.diagnostics.stretching_scheme == scheme
        assert induction.diagnostics.last_strength_rate_norm > 0.0
        assert induction.diagnostics.last_relative_rate_defect >= 0.0
