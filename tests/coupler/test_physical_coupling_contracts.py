"""Physical contracts for absolute FVM-state replacement in the overlap."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial import cKDTree

from source.coupler.boundary import advance_fvm_substeps
from source.coupler.interpolation import FVMVelocityInterpolator
from source.coupler.vorticity_transfer import (
    VorticityTransfer,
    replace_particles_from_fvm,
    replacement_eta,
)

BOX = np.array([-0.5, 0.5, -0.5, 0.5, -0.5, 0.5])


class _Particles:
    def __init__(self, position: np.ndarray, vortex_strength: np.ndarray, capacity: int = 100):
        self.position = np.asarray(position, dtype=np.float64).reshape(-1, 3).copy()
        self.vortex_strength = np.asarray(vortex_strength, dtype=np.float64).reshape(-1, 3).copy()
        self.capacity = capacity
        self.n_particles_total = len(self.position)

    def position_cpu(self):
        return self.position.copy()

    def vortex_strength_cpu(self):
        return self.vortex_strength.copy()


class _VPM:
    np_dtype = np.float64

    def __init__(self, position: np.ndarray, vortex_strength: np.ndarray, capacity: int = 100):
        self.particles = _Particles(position, vortex_strength, capacity)
        self.added_fields: list[dict] = []

    def update_particle_vortex_strength(self, mask, increment):
        self.particles.vortex_strength[np.asarray(mask, dtype=bool)] += np.asarray(increment)

    def remove_particles(self, particle_indices=None, remove_all=False):
        if remove_all:
            keep = np.zeros(self.particles.n_particles_total, dtype=bool)
        else:
            keep = np.ones(self.particles.n_particles_total, dtype=bool)
            keep[np.asarray(particle_indices, dtype=np.int64)] = False
        self.particles.position = self.particles.position[keep]
        self.particles.vortex_strength = self.particles.vortex_strength[keep]
        self.particles.n_particles_total = len(self.particles.position)

    def add_vortex_particles(self, **fields):
        self.added_fields.append({name: np.asarray(value).copy() for name, value in fields.items()})
        self.particles.position = np.concatenate(
            [self.particles.position, np.asarray(fields["position"], dtype=np.float64)]
        )
        self.particles.vortex_strength = np.concatenate(
            [
                self.particles.vortex_strength,
                np.asarray(fields["vortex_strength"], dtype=np.float64),
            ]
        )
        self.particles.n_particles_total = len(self.particles.position)


def _replace(
    vpm: _VPM,
    fvm_position: np.ndarray,
    fvm_volume: np.ndarray,
    fvm_vorticity: np.ndarray,
    *,
    blend_width: float = 0.0,
    solid_mask: np.ndarray | None = None,
):
    return replace_particles_from_fvm(
        vpm,
        transfer_box=BOX,
        eta_blend_width=blend_width,
        fvm_position=fvm_position,
        fvm_cell_volume=fvm_volume,
        fvm_vorticity=fvm_vorticity,
        core_radius_ratio=1.25,
        kinematic_viscosity=1.0e-3,
        fvm_solid_mask=solid_mask,
    )


def test_eta_zero_width_is_hard_ownership_and_positive_width_is_c1_blend():
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.25, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.6, 0.0, 0.0],
        ]
    )
    np.testing.assert_array_equal(replacement_eta(points, BOX, 0.0), [1.0, 1.0, 1.0, 0.0])
    np.testing.assert_allclose(replacement_eta(points, BOX, 0.5), [1.0, 0.5, 0.0, 0.0])

    epsilon = 1.0e-7
    near_boundary = np.array([[0.5 - epsilon, 0.0, 0.0]])
    near_core = np.array([[epsilon, 0.0, 0.0]])
    assert replacement_eta(near_boundary, BOX, 0.5)[0] < 2.0e-12
    assert 1.0 - replacement_eta(near_core, BOX, 0.5)[0] < 2.0e-12


def test_hard_replacement_removes_inner_particles_injects_cell_circulation_and_preserves_outer():
    outer_position = np.array([[0.8, 0.0, 0.0], [-0.7, 0.1, 0.0]])
    outer_strength = np.array([[0.1, 0.2, 0.3], [-0.2, 0.1, 0.0]])
    vpm = _VPM(
        np.vstack(([0.0, 0.0, 0.0], outer_position)),
        np.vstack(([9.0, 9.0, 9.0], outer_strength)),
    )
    fvm_position = np.array([[0.0, 0.0, 0.0], [0.9, 0.0, 0.0]])
    volume = np.array([0.125, 0.125])
    vorticity = np.array([[0.0, 4.0, -2.0], [7.0, 7.0, 7.0]])

    result = _replace(vpm, fvm_position, volume, vorticity)

    assert result.n_particles_before == 3
    assert result.n_particles_removed == 1
    assert result.n_particles_blended == 0
    assert result.n_particles_injected == 1
    assert result.n_particles_after == 3
    np.testing.assert_array_equal(vpm.particles.position[:2], outer_position)
    np.testing.assert_array_equal(vpm.particles.vortex_strength[:2], outer_strength)
    np.testing.assert_array_equal(vpm.particles.position[2], fvm_position[0])
    np.testing.assert_allclose(vpm.particles.vortex_strength[2], volume[0] * vorticity[0])
    np.testing.assert_allclose(vpm.added_fields[-1]["particle_volume"], [volume[0]])
    np.testing.assert_allclose(vpm.added_fields[-1]["core_radius"], [1.25 * np.cbrt(volume[0])])


def test_identical_hard_replacement_is_an_exact_fixed_point_for_one_hundred_cycles():
    outer_position = np.array([[0.8, 0.0, 0.0]])
    outer_strength = np.array([[0.1, -0.2, 0.3]])
    vpm = _VPM(outer_position, outer_strength)
    fvm_position = np.array([[-0.2, 0.0, 0.0], [0.2, 0.0, 0.0]])
    volume = np.array([0.01, 0.02])
    vorticity = np.array([[2.0, -1.0, 0.5], [-0.5, 3.0, 1.0]])

    expected_position = None
    expected_strength = None
    for cycle in range(100):
        result = _replace(vpm, fvm_position, volume, vorticity)
        if cycle == 0:
            expected_position = vpm.particles.position.copy()
            expected_strength = vpm.particles.vortex_strength.copy()
        else:
            np.testing.assert_array_equal(vpm.particles.position, expected_position)
            np.testing.assert_array_equal(vpm.particles.vortex_strength, expected_strength)
            np.testing.assert_allclose(result.state_change_vortex_strength_net, 0.0, atol=1.0e-18)
        np.testing.assert_array_equal(vpm.particles.position[0], outer_position[0])
        np.testing.assert_array_equal(vpm.particles.vortex_strength[0], outer_strength[0])


def test_eta_blend_is_a_state_partition_not_an_additive_defect():
    # At x=0.25 the distance to the x+ face is 0.25, so a 0.5-m ramp has eta=0.5.
    position = np.array([[0.25, 0.0, 0.0], [0.8, 0.0, 0.0]])
    target_strength = np.array([[0.0, 2.0, 0.0], [0.3, 0.1, -0.2]])
    vpm = _VPM(position, target_strength)
    volume = np.array([0.25])
    vorticity = np.array([[0.0, 8.0, 0.0]])

    result = _replace(
        vpm,
        position[:1],
        volume,
        vorticity,
        blend_width=0.5,
    )

    assert result.n_particles_removed == 0
    assert result.n_particles_blended == 1
    assert result.n_particles_injected == 1
    # The retained half and injected half sum to the original target state.
    np.testing.assert_allclose(vpm.particles.vortex_strength[0], [0.0, 1.0, 0.0])
    np.testing.assert_allclose(vpm.particles.vortex_strength[2], [0.0, 1.0, 0.0])
    np.testing.assert_allclose(
        vpm.particles.vortex_strength[[0, 2]].sum(axis=0),
        target_strength[0],
    )
    np.testing.assert_array_equal(vpm.particles.vortex_strength[1], target_strength[1])


def test_solid_fvm_cells_are_not_injected():
    vpm = _VPM(np.empty((0, 3)), np.empty((0, 3)))
    result = _replace(
        vpm,
        np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]]),
        np.array([0.01, 0.01]),
        np.array([[4.0, 0.0, 0.0], [0.0, 5.0, 0.0]]),
        solid_mask=np.array([True, False]),
    )
    assert result.n_particles_injected == 1
    np.testing.assert_array_equal(vpm.particles.position, [[0.2, 0.0, 0.0]])


def test_capacity_failure_is_checked_before_particle_mutation():
    position = np.array([[0.8, 0.0, 0.0]])
    strength = np.array([[0.1, 0.2, 0.3]])
    vpm = _VPM(position, strength, capacity=1)
    before_position = vpm.particles.position.copy()
    before_strength = vpm.particles.vortex_strength.copy()

    with pytest.raises(RuntimeError, match="exceeding the VPM capacity"):
        _replace(
            vpm,
            np.array([[-0.2, 0.0, 0.0], [0.2, 0.0, 0.0]]),
            np.array([0.01, 0.01]),
            np.ones((2, 3)),
        )
    np.testing.assert_array_equal(vpm.particles.position, before_position)
    np.testing.assert_array_equal(vpm.particles.vortex_strength, before_strength)


def test_fvm_gradient_curl_matches_cell_vorticity_convention():
    gradient = np.zeros((2, 3, 3))
    gradient[0, 1, 2] = 3.0
    gradient[0, 2, 1] = -2.0
    gradient[1, 2, 0] = 4.0
    gradient[1, 0, 2] = 1.5
    np.testing.assert_array_equal(
        VorticityTransfer._vorticity_from_gradient(gradient),
        [[5.0, 0.0, 0.0], [0.0, 2.5, 0.0]],
    )


def _quadratic_velocity(position):
    x, y, z = np.asarray(position).T
    velocity = np.column_stack(
        (
            x**2 + 0.3 * y * z,
            y**2 - 0.2 * x * z,
            z**2 + 0.1 * x * y,
        )
    )
    gradient = np.empty((len(position), 3, 3))
    gradient[:, 0, :] = np.column_stack((2.0 * x, -0.2 * z, 0.1 * y))
    gradient[:, 1, :] = np.column_stack((0.3 * z, 2.0 * y, 0.1 * x))
    gradient[:, 2, :] = np.column_stack((0.3 * y, -0.2 * x, 2.0 * z))
    return velocity, gradient


def _graded_interpolation_error(node_count):
    uniform = np.linspace(-1.0, 1.0, node_count)
    graded = np.sign(uniform) * np.abs(uniform) ** 1.5
    mesh = np.meshgrid(graded, uniform, graded, indexing="ij")
    cell_centre = np.column_stack([component.ravel() for component in mesh])
    velocity, gradient = _quadratic_velocity(cell_centre)
    target_axis = np.linspace(-0.78, 0.78, 11)
    target_mesh = np.meshgrid(target_axis, target_axis, target_axis, indexing="ij")
    target = np.column_stack([component.ravel() for component in target_mesh])
    expected, _ = _quadratic_velocity(target)
    sampled = FVMVelocityInterpolator(
        cell_centre,
        cKDTree(cell_centre),
        neighbour_count=4,
    ).sample(target, velocity, gradient)
    return float(np.linalg.norm(sampled - expected) / np.linalg.norm(expected))


def test_fvm_velocity_interpolation_is_affine_exact_and_second_order_on_graded_meshes():
    axis = np.linspace(-1.0, 1.0, 7)
    mesh = np.meshgrid(axis, axis, axis, indexing="ij")
    cell_centre = np.column_stack([component.ravel() for component in mesh])
    gradient_matrix = np.array([[0.2, -0.1, 0.3], [0.4, 0.1, -0.2], [-0.3, 0.2, 0.15]])
    offset = np.array([0.8, -0.2, 0.1])
    velocity = offset + cell_centre @ gradient_matrix
    gradient = np.broadcast_to(gradient_matrix, (len(cell_centre), 3, 3)).copy()
    target = np.array([[-0.73, 0.18, 0.44], [0.02, -0.31, 0.11], [0.81, 0.62, -0.58]])
    interpolator = FVMVelocityInterpolator(
        cell_centre,
        cKDTree(cell_centre),
        neighbour_count=4,
    )
    np.testing.assert_allclose(
        interpolator.sample(target, velocity, gradient),
        offset + target @ gradient_matrix,
        rtol=0.0,
        atol=2.0e-15,
    )

    errors = np.array([_graded_interpolation_error(n) for n in (7, 13, 25)])
    orders = np.log2(errors[:-1] / errors[1:])
    assert np.all(orders > 1.8), (errors, orders)


def test_vorticity_mixed_substeps_use_both_time_endpoints(monkeypatch):
    coupler = SimpleNamespace(
        n_fvm_substeps=4,
        fvm_time_step_size=0.005,
        freestream_velocity=np.array([1.0, 0.0, 0.0]),
        setup=SimpleNamespace(boundary_condition_mode="vorticity_mixed"),
    )
    recorded = []

    def record_step(
        _coupler,
        _patch,
        velocity,
        pressure_gradient=None,
        normal_velocity=None,
        tangential_gradient=None,
    ):
        recorded.append((velocity.copy(), normal_velocity.copy(), tangential_gradient.copy()))

    monkeypatch.setattr("source.coupler.boundary.apply_fvm_boundary", record_step)
    face_centre = np.zeros((3, 3))
    face_normal = np.tile([1.0, 0.0, 0.0], (3, 1))
    face_area = np.ones(3)
    previous_velocity = np.tile([1.0, 0.0, 0.0], (3, 1))
    next_velocity = np.tile([1.0, 0.4, 0.0], (3, 1))
    previous_normal_velocity = np.full(3, 0.2)
    next_normal_velocity = np.full(3, 0.6)
    previous_tangential_gradient = np.zeros((3, 3))
    next_tangential_gradient = np.full((3, 3), 0.8)

    advance_fvm_substeps(
        coupler,
        "numericalBoundary",
        face_centre,
        face_normal,
        face_area,
        previous_velocity,
        next_velocity,
        previous_normal_velocity=previous_normal_velocity,
        next_normal_velocity=next_normal_velocity,
        previous_tangential_gradient=previous_tangential_gradient,
        next_tangential_gradient=next_tangential_gradient,
    )

    for values, alpha in zip(recorded, (0.25, 0.5, 0.75, 1.0), strict=True):
        np.testing.assert_allclose(
            values[0], (1.0 - alpha) * previous_velocity + alpha * next_velocity
        )
        np.testing.assert_allclose(
            values[1],
            (1.0 - alpha) * previous_normal_velocity + alpha * next_normal_velocity,
        )
        np.testing.assert_allclose(values[2], alpha * next_tangential_gradient)
