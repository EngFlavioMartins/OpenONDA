"""Physical contracts for absolute FVM-state replacement in the overlap."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial import cKDTree
from scipy.special import erf

from source.coupler.boundary import advance_fvm_substeps
from source.coupler.interpolation import FVMVelocityInterpolator
from source.coupler.lattice_transfer import evaluate_gaussian_vorticity
from source.coupler.vorticity_transfer import (
    VorticityTransfer,
    replace_particles_from_fvm,
    replace_particles_from_lattice_blend,
    replacement_eta,
)

BOX = np.array([-0.5, 0.5, -0.5, 0.5, -0.5, 0.5])


class _Particles:
    def __init__(self, position: np.ndarray, vortex_strength: np.ndarray, capacity: int = 100):
        self.position = np.asarray(position, dtype=np.float64).reshape(-1, 3).copy()
        self.vortex_strength = np.asarray(vortex_strength, dtype=np.float64).reshape(-1, 3).copy()
        self.capacity = capacity
        self.n_particles_total = len(self.position)
        index = np.arange(self.n_particles_total, dtype=np.float64)
        self.velocity = np.column_stack((index, -index, 0.5 * index))
        self.core_radius = 0.1 + 0.01 * index
        self.particle_volume = 0.001 + 0.001 * index
        self.kinematic_viscosity = 0.01 + 0.001 * index
        self.eddy_viscosity = 0.02 + 0.001 * index
        self.group_id = np.arange(self.n_particles_total, dtype=np.int32)
        self.zone_id = (10 + np.arange(self.n_particles_total)).astype(np.int32)
        self.velocity_gradient = np.broadcast_to(np.eye(3), (self.n_particles_total, 3, 3)).copy()
        self.strain_rate = np.broadcast_to(2.0 * np.eye(3), (self.n_particles_total, 3, 3)).copy()

    def position_cpu(self):
        return self.position.copy()

    def vortex_strength_cpu(self):
        return self.vortex_strength.copy()

    def velocity_cpu(self):
        return self.velocity.copy()

    def core_radius_cpu(self):
        return self.core_radius.copy()

    def particle_volume_cpu(self):
        return self.particle_volume.copy()

    def kinematic_viscosity_cpu(self):
        return self.kinematic_viscosity.copy()

    def eddy_viscosity_cpu(self):
        return self.eddy_viscosity.copy()

    def group_id_cpu(self):
        return self.group_id.copy()

    def zone_id_cpu(self):
        return self.zone_id.copy()

    def velocity_gradient_cpu(self):
        return self.velocity_gradient.copy()

    def strain_rate_cpu(self):
        return self.strain_rate.copy()


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
        for name in (
            "position",
            "vortex_strength",
            "velocity",
            "core_radius",
            "particle_volume",
            "kinematic_viscosity",
            "eddy_viscosity",
            "group_id",
            "zone_id",
            "velocity_gradient",
            "strain_rate",
        ):
            setattr(self.particles, name, getattr(self.particles, name)[keep])
        self.particles.n_particles_total = len(self.particles.position)

    def add_vortex_particles(self, **fields):
        self.added_fields.append({name: np.asarray(value).copy() for name, value in fields.items()})
        count = len(fields["position"])
        defaults = {
            "velocity": np.zeros((count, 3)),
            "core_radius": np.zeros(count),
            "particle_volume": np.zeros(count),
            "kinematic_viscosity": np.zeros(count),
            "eddy_viscosity": np.zeros(count),
            "group_id": np.zeros(count, dtype=np.int32),
            "zone_id": np.zeros(count, dtype=np.int32),
            "velocity_gradient": np.zeros((count, 3, 3)),
            "strain_rate": np.zeros((count, 3, 3)),
        }
        for name in (
            "position",
            "vortex_strength",
            "velocity",
            "core_radius",
            "particle_volume",
            "kinematic_viscosity",
            "eddy_viscosity",
            "group_id",
            "zone_id",
            "velocity_gradient",
            "strain_rate",
        ):
            value = fields.get(name, defaults.get(name))
            setattr(
                self.particles,
                name,
                np.concatenate((getattr(self.particles, name), np.asarray(value))),
            )
        self.particles.n_particles_total = len(self.particles.position)

    def replace_vortex_particles(self, *, report_removal=True, **fields):
        del report_removal
        for name in (
            "position",
            "vortex_strength",
            "velocity",
            "core_radius",
            "particle_volume",
            "kinematic_viscosity",
            "eddy_viscosity",
            "group_id",
            "zone_id",
            "velocity_gradient",
            "strain_rate",
        ):
            setattr(self.particles, name, np.asarray(fields[name]).copy())
        self.particles.n_particles_total = len(self.particles.position)


class _FailingVPM(_VPM):
    def __init__(self, *args, fail_at: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.fail_at = fail_at

    def update_particle_vortex_strength(self, mask, increment):
        super().update_particle_vortex_strength(mask, increment)
        if self.fail_at == "update_particle_vortex_strength":
            raise RuntimeError("injected update failure")

    def remove_particles(self, *args, **kwargs):
        super().remove_particles(*args, **kwargs)
        if self.fail_at == "remove_particles":
            raise RuntimeError("injected removal failure")

    def add_vortex_particles(self, **fields):
        super().add_vortex_particles(**fields)
        if self.fail_at == "add_vortex_particles":
            raise RuntimeError("injected add failure")


class _Float32VPM(_VPM):
    np_dtype = np.float32


class _CountMismatchingVPM(_VPM):
    def add_vortex_particles(self, **fields):
        super().add_vortex_particles(**fields)
        self.particles.n_particles_total += 1


def _particle_state(particles: _Particles) -> dict[str, np.ndarray | int]:
    return {
        "position": particles.position.copy(),
        "vortex_strength": particles.vortex_strength.copy(),
        "velocity": particles.velocity.copy(),
        "core_radius": particles.core_radius.copy(),
        "particle_volume": particles.particle_volume.copy(),
        "kinematic_viscosity": particles.kinematic_viscosity.copy(),
        "eddy_viscosity": particles.eddy_viscosity.copy(),
        "group_id": particles.group_id.copy(),
        "zone_id": particles.zone_id.copy(),
        "velocity_gradient": particles.velocity_gradient.copy(),
        "strain_rate": particles.strain_rate.copy(),
        "n_particles_total": particles.n_particles_total,
    }


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


def _replace_lattice(
    vpm: _VPM,
    fvm_position: np.ndarray,
    fvm_volume: np.ndarray,
    fvm_vorticity: np.ndarray,
    *,
    blend_width: float = 0.25,
    spacing: float = 0.125,
    solid_contains=None,
):
    return replace_particles_from_lattice_blend(
        vpm,
        transfer_box=BOX,
        eta_blend_width=blend_width,
        fvm_position=fvm_position,
        fvm_cell_volume=fvm_volume,
        fvm_vorticity=fvm_vorticity,
        lattice_anchor=np.zeros(3),
        particle_spacing=spacing,
        core_radius_ratio=1.25,
        kinematic_viscosity=1.0e-3,
        solid_contains=solid_contains,
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


def test_common_lattice_blend_is_an_exact_fixed_point_for_matching_states():
    h = 0.125
    fvm_position = np.array([[0.375, 0.0, 0.0]])
    volume = np.array([h**3])
    vorticity = np.array([[0.0, 8.0, -4.0]])
    target_strength = volume[:, None] * vorticity
    outer_position = np.array([[0.75, 0.0, 0.0]])
    outer_strength = np.array([[0.3, -0.2, 0.1]])
    vpm = _VPM(
        np.vstack((fvm_position, outer_position)),
        np.vstack((target_strength, outer_strength)),
    )
    outer_core_radius = vpm.particles.core_radius[1:2].copy()

    for _ in range(20):
        # The FVM state is the same physical state: its donor includes the
        # continuous tail of the persistent outer VPM Gaussian.
        fvm_state = vorticity + evaluate_gaussian_vorticity(
            fvm_position,
            outer_position,
            outer_strength,
            outer_core_radius,
        )
        result = _replace_lattice(
            vpm,
            fvm_position,
            volume,
            fvm_state,
            blend_width=0.25,
            spacing=h,
        )
        by_position = {
            tuple(position): strength
            for position, strength in zip(
                vpm.particles.position,
                vpm.particles.vortex_strength,
                strict=True,
            )
        }
        np.testing.assert_allclose(by_position[tuple(fvm_position[0])], target_strength[0])
        np.testing.assert_array_equal(by_position[tuple(outer_position[0])], outer_strength[0])
        np.testing.assert_allclose(result.state_change_vortex_strength_net, 0.0, atol=1.0e-18)
        assert result.transfer_method == "common_m4_lattice_blend"
        assert len(by_position) == vpm.particles.n_particles_total


def test_manufactured_convecting_release_crosses_interface_and_survives_next_handoff():
    h = 0.125
    fvm_position = np.array([[0.375, 0.0, 0.0]])
    volume = np.array([h**3])
    vorticity = np.array([[0.0, 16.0, 0.0]])
    vpm = _VPM(np.empty((0, 3)), np.empty((0, 3)))

    _replace_lattice(vpm, fvm_position, volume, vorticity, blend_width=0.25, spacing=h)
    assert vpm.particles.position.shape == (1, 3)
    released_strength = vpm.particles.vortex_strength[0].copy()
    vpm.particles.position[:, 0] += 0.15
    assert vpm.particles.position[0, 0] > BOX[1]

    _replace_lattice(vpm, fvm_position, volume, vorticity, blend_width=0.25, spacing=h)
    outside = vpm.particles.position[:, 0] > BOX[1]
    np.testing.assert_allclose(
        vpm.particles.vortex_strength[outside].sum(axis=0),
        released_strength,
        atol=1.0e-18,
    )
    _replace_lattice(vpm, fvm_position, volume, vorticity, blend_width=0.25, spacing=h)
    outside = vpm.particles.position[:, 0] > BOX[1]
    np.testing.assert_allclose(
        vpm.particles.vortex_strength[outside].sum(axis=0),
        released_strength,
        atol=1.0e-18,
    )


def test_blend_merges_release_support_into_a_retained_regular_node_without_duplicates():
    h = 0.125
    source_position = np.array([[0.4375, 0.0, 0.0]])
    source_strength = np.array([[0.0, 2.0, 0.0]])
    boundary_position = np.array([[BOX[1], 0.0, 0.0]])
    boundary_strength = np.array([[0.0, 0.3, 0.0]])
    vpm = _VPM(
        np.vstack((source_position, boundary_position)),
        np.vstack((source_strength, boundary_strength)),
    )

    _replace_lattice(
        vpm,
        source_position,
        np.array([1.0]),
        source_strength
        + evaluate_gaussian_vorticity(
            source_position,
            boundary_position,
            boundary_strength,
            vpm.particles.core_radius[1:],
        ),
        blend_width=0.25,
        spacing=h,
    )

    matches = np.all(np.isclose(vpm.particles.position, boundary_position[0]), axis=1)
    assert matches.sum() == 1
    m4_weight = 0.5625
    np.testing.assert_allclose(
        vpm.particles.vortex_strength[matches][0],
        boundary_strength[0] + m4_weight * source_strength[0],
        atol=1.0e-15,
    )
    assert len(np.unique(vpm.particles.position, axis=0)) == vpm.particles.n_particles_total


def test_hard_release_support_adds_to_persistent_outer_node():
    h = 0.125
    source_position = np.array([[0.4375, 0.0, 0.0]])
    source_strength = np.array([[0.0, 2.0, 0.0]])
    persistent_position = np.array([[0.625, 0.0, 0.0]])
    persistent_strength = np.array([[0.0, 0.3, 0.0]])
    vpm = _VPM(
        np.vstack((source_position, persistent_position)),
        np.vstack((source_strength, persistent_strength)),
    )

    _replace_lattice(
        vpm,
        source_position,
        np.array([1.0]),
        source_strength
        + evaluate_gaussian_vorticity(
            source_position,
            persistent_position,
            persistent_strength,
            vpm.particles.core_radius[1:],
        ),
        blend_width=0.0,
        spacing=h,
    )

    matches = np.all(np.isclose(vpm.particles.position, persistent_position[0]), axis=1)
    assert matches.sum() == 1
    np.testing.assert_allclose(
        vpm.particles.vortex_strength[matches][0],
        persistent_strength[0] - 0.0625 * source_strength[0],
        atol=1.0e-15,
    )


def test_hard_release_recognizes_f32_lattice_nodes_at_large_coordinates():
    h = 0.03
    source_position = np.array([[9999.99, 0.0, 0.0]], dtype=np.float32)
    persistent_position = np.array([[10000.02, 0.0, 0.0]], dtype=np.float32)
    vpm = _Float32VPM(persistent_position, np.array([[0.0, 0.3, 0.0]]))

    replace_particles_from_lattice_blend(
        vpm,
        transfer_box=np.array([9999.9, 10000.0, -0.5, 0.5, -0.5, 0.5]),
        eta_blend_width=0.0,
        fvm_position=source_position,
        fvm_cell_volume=np.array([1.0]),
        fvm_vorticity=np.array([[0.0, 2.0, 0.0]]),
        lattice_anchor=np.zeros(3),
        particle_spacing=h,
        core_radius_ratio=1.0,
        kinematic_viscosity=1.0e-3,
    )

    close_to_persistent = (
        np.max(np.abs(vpm.particles.position - persistent_position.astype(np.float64)), axis=1)
        <= 2.0e-3
    )
    assert close_to_persistent.sum() == 1


def test_lattice_blend_redistributes_solid_targets_without_losing_moments():
    h = 0.125
    vpm = _VPM(np.empty((0, 3)), np.empty((0, 3)))

    def target_solid(points):
        return np.isclose(np.asarray(points)[:, 0], h)

    result = _replace_lattice(
        vpm,
        np.array([[0.5 * h, 0.0, 0.0]]),
        np.array([h**3]),
        np.array([[0.0, 4.0, 0.0]]),
        blend_width=0.25,
        spacing=h,
        solid_contains=target_solid,
    )
    assert not np.any(np.isclose(vpm.particles.position[:, 0], h))
    assert result.excluded_solid_target_nodes > 0
    np.testing.assert_allclose(result.excluded_solid_vortex_strength_net, 0.0, atol=1.0e-15)
    assert np.linalg.norm(result.redistributed_solid_vortex_strength_net) > 0.0
    expected_strength = h**3 * np.array([0.0, 4.0, 0.0])
    np.testing.assert_allclose(
        vpm.particles.vortex_strength.sum(axis=0), expected_strength, atol=1.0e-14
    )
    np.testing.assert_allclose(
        vpm.particles.position.T @ vpm.particles.vortex_strength,
        np.outer(np.array([0.5 * h, 0.0, 0.0]), expected_strength),
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        np.einsum(
            "ni,nj,nk->ijk",
            vpm.particles.position,
            vpm.particles.position,
            vpm.particles.vortex_strength,
        ),
        np.einsum(
            "i,j,k->ijk",
            np.array([0.5 * h, 0.0, 0.0]),
            np.array([0.5 * h, 0.0, 0.0]),
            expected_strength,
        ),
        atol=2.0e-14,
    )


def _linear_impulse(position: np.ndarray, strength: np.ndarray) -> np.ndarray:
    return 0.5 * np.cross(position, strength).sum(axis=0)


def _gaussian_angular_impulse(
    position: np.ndarray,
    strength: np.ndarray,
    core_radius: float,
) -> np.ndarray:
    angular_correction = 1.5
    return np.cross(position, np.cross(position, strength)).sum(axis=0) / 3.0 - (
        2.0 / 9.0
    ) * angular_correction * core_radius**2 * strength.sum(axis=0)


def _gaussian_divergence(
    point: np.ndarray,
    position: np.ndarray,
    strength: np.ndarray,
    core_radius: np.ndarray,
) -> np.ndarray:
    evaluation_point = np.asarray(point, dtype=np.float64).reshape(-1, 3)
    source_position = np.asarray(position, dtype=np.float64).reshape(-1, 3)
    source_strength = np.asarray(strength, dtype=np.float64).reshape(-1, 3)
    radius = np.asarray(core_radius, dtype=np.float64).reshape(-1)
    displacement = evaluation_point[:, None, :] - source_position[None, :, :]
    sigma = radius[None, :, None]
    zeta = (
        np.pi ** (-1.5) * np.exp(-np.sum((displacement / sigma) ** 2, axis=2)) / sigma[..., 0] ** 3
    )
    return np.sum(
        -2.0 * np.einsum("npi,pi->np", displacement, source_strength) * zeta / sigma[..., 0] ** 2,
        axis=1,
    )


def _gaussian_velocity(
    point: np.ndarray,
    position: np.ndarray,
    strength: np.ndarray,
    core_radius: np.ndarray,
) -> np.ndarray:
    evaluation_point = np.asarray(point, dtype=np.float64).reshape(-1, 3)
    source_position = np.asarray(position, dtype=np.float64).reshape(-1, 3)
    source_strength = np.asarray(strength, dtype=np.float64).reshape(-1, 3)
    radius = np.asarray(core_radius, dtype=np.float64).reshape(-1)
    displacement = evaluation_point[:, None, :] - source_position[None, :, :]
    radius_sq = np.einsum("npi,npi->np", displacement, displacement)
    density = np.sqrt(radius_sq) / radius[None, :]
    q = (erf(density) - 2.0 / np.sqrt(np.pi) * density * np.exp(-(density**2))) / (4.0 * np.pi)
    scale = np.divide(
        q,
        radius_sq * np.sqrt(radius_sq),
        out=np.zeros_like(q),
        where=radius_sq > 0.0,
    )
    return -np.sum(scale[..., None] * np.cross(displacement, source_strength[None, :, :]), axis=1)


def _gaussian_tail_case(
    *,
    distance_in_h: float,
    blend_width: float,
) -> tuple[_VPM, np.ndarray, np.ndarray, np.ndarray, np.ndarray, object]:
    h = 0.125
    sigma = 1.25 * h
    fvm_spacing = h / 4.0
    axis = np.arange(BOX[0] + 0.5 * fvm_spacing, BOX[1], fvm_spacing)
    fvm_position = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    particle_position = np.array([[BOX[1] + distance_in_h * h, 0.0, 0.0]])
    particle_strength = np.array([[0.0, 1.0, 0.0]])
    core_radius = np.array([sigma])
    fvm_vorticity = evaluate_gaussian_vorticity(
        fvm_position,
        particle_position,
        particle_strength,
        core_radius,
    )
    vpm = _VPM(particle_position, particle_strength, capacity=100_000)
    vpm.particles.core_radius[:] = core_radius
    sample_axis = (
        np.linspace(0.25, 0.8, 17),
        np.linspace(-0.22, 0.22, 13),
        np.linspace(-0.2, 0.2, 11),
    )
    sample_position = np.stack(np.meshgrid(*sample_axis, indexing="ij"), axis=-1).reshape(-1, 3)
    result = _replace_lattice(
        vpm,
        fvm_position,
        np.full(len(fvm_position), fvm_spacing**3),
        fvm_vorticity,
        blend_width=blend_width,
        spacing=h,
    )
    return vpm, particle_position, particle_strength, core_radius, sample_position, result


@pytest.mark.parametrize("distance_in_h", [0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0])
@pytest.mark.parametrize("blend_width_in_h", [0.0, 3.0])
def test_persistent_gaussian_tail_residual_is_a_continuous_field_fixed_point(
    distance_in_h,
    blend_width_in_h,
):
    h = 0.125
    vpm, before_position, before_strength, before_core, sample_position, result = (
        _gaussian_tail_case(
            distance_in_h=distance_in_h,
            blend_width=blend_width_in_h * h,
        )
    )
    before_omega = evaluate_gaussian_vorticity(
        sample_position, before_position, before_strength, before_core
    )
    before_velocity = _gaussian_velocity(
        sample_position, before_position, before_strength, before_core
    )
    before_divergence = _gaussian_divergence(
        sample_position, before_position, before_strength, before_core
    )
    after_position = vpm.particles.position
    after_strength = vpm.particles.vortex_strength
    after_core = vpm.particles.core_radius
    after_omega = evaluate_gaussian_vorticity(
        sample_position, after_position, after_strength, after_core
    )
    after_velocity = _gaussian_velocity(sample_position, after_position, after_strength, after_core)
    after_divergence = _gaussian_divergence(
        sample_position, after_position, after_strength, after_core
    )

    assert result.n_particles_before == result.n_particles_after == 1
    assert result.n_particles_removed == result.n_particles_injected == 0
    assert result.persistent_fraction_rms == pytest.approx(1.0, abs=2.0e-15)
    assert result.persistent_fraction_max == pytest.approx(1.0, abs=2.0e-15)
    np.testing.assert_array_equal(after_position, before_position)
    np.testing.assert_array_equal(after_strength, before_strength)
    np.testing.assert_array_equal(after_core, before_core)
    np.testing.assert_allclose(after_omega, before_omega, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(after_velocity, before_velocity, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(after_divergence, before_divergence, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        _linear_impulse(after_position, after_strength),
        _linear_impulse(before_position, before_strength),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        _gaussian_angular_impulse(after_position, after_strength, before_core[0]),
        _gaussian_angular_impulse(before_position, before_strength, before_core[0]),
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.xfail(
    strict=True,
    reason="R1 removes persistent tails, but a nonzero replaced Gaussian still exceeds "
    "the continuous-field projection tolerance after FVM-to-M4 transfer",
)
@pytest.mark.parametrize("blend_width_in_h", [0.0, 3.0])
def test_mixed_persistent_and_replaced_gaussian_state_is_a_continuous_fixed_point(
    blend_width_in_h,
):
    """Release gate for the full R+Q identity at dense off-lattice points."""
    h = 0.125
    sigma = 1.25 * h
    fvm_spacing = h / 4.0
    axis = np.arange(BOX[0] + 0.5 * fvm_spacing, BOX[1], fvm_spacing)
    fvm_position = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    position = np.array([[-0.1, 0.02, -0.01], [BOX[1] + 0.25 * h, 0.03, -0.04]])
    strength = np.array([[0.0, 0.5, 0.2], [0.0, 1.0, 0.0]])
    core_radius = np.full(2, sigma)
    fvm_vorticity = evaluate_gaussian_vorticity(
        fvm_position,
        position,
        strength,
        core_radius,
    )
    vpm = _VPM(position, strength, capacity=100_000)
    vpm.particles.core_radius[:] = core_radius
    sample_axis = (
        np.linspace(-0.45, 0.8, 20),
        np.linspace(-0.3, 0.3, 13),
        np.linspace(-0.3, 0.3, 13),
    )
    sample_position = np.stack(np.meshgrid(*sample_axis, indexing="ij"), axis=-1).reshape(-1, 3)
    before = evaluate_gaussian_vorticity(sample_position, position, strength, core_radius)

    _replace_lattice(
        vpm,
        fvm_position,
        np.full(len(fvm_position), fvm_spacing**3),
        fvm_vorticity,
        blend_width=blend_width_in_h * h,
        spacing=h,
    )

    after = evaluate_gaussian_vorticity(
        sample_position,
        vpm.particles.position,
        vpm.particles.vortex_strength,
        vpm.particles.core_radius,
    )
    relative_rms_error = np.sqrt(np.mean((after - before) ** 2)) / np.sqrt(np.mean(before**2))
    assert relative_rms_error <= 5.0e-3


@pytest.mark.parametrize("distance_in_h", [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0])
@pytest.mark.parametrize("geometry", ["plane", "edge", "corner"])
def test_solid_redistribution_preserves_solver_moments_and_is_well_conditioned(
    distance_in_h,
    geometry,
):
    h = 0.125
    distance = distance_in_h * h
    if geometry == "plane":
        donor_position = np.array([[-distance, 0.0, 0.0]])
    elif geometry == "edge":
        donor_position = np.array([[-distance, -distance, 0.0]])
    else:
        donor_position = np.array([[-distance, -distance, -distance]])

    def solid_contains(points):
        query = np.asarray(points)
        if geometry == "plane":
            return query[:, 0] > 0.0
        if geometry == "edge":
            return (query[:, 0] > 0.0) & (query[:, 1] > 0.0)
        return np.all(query > 0.0, axis=1)

    donor_strength = np.array([[0.2, -0.3, 0.5]])
    vpm = _VPM(np.empty((0, 3)), np.empty((0, 3)))

    result = _replace_lattice(
        vpm,
        donor_position,
        np.array([1.0]),
        donor_strength,
        blend_width=0.0,
        spacing=h,
        solid_contains=solid_contains,
    )

    final_position = vpm.particles.position
    final_strength = vpm.particles.vortex_strength
    assert not np.any(solid_contains(final_position))
    np.testing.assert_allclose(final_strength.sum(axis=0), donor_strength[0], atol=3.0e-14)
    np.testing.assert_allclose(
        _linear_impulse(final_position, final_strength),
        _linear_impulse(donor_position, donor_strength),
        atol=3.0e-14,
    )
    np.testing.assert_allclose(
        _gaussian_angular_impulse(final_position, final_strength, 1.25 * h),
        _gaussian_angular_impulse(donor_position, donor_strength, 1.25 * h),
        atol=5.0e-14,
    )
    audit = result.solid_redistribution_audit
    if distance_in_h < 1.0:
        assert len(audit.condition_numbers) > 0
        assert np.max(audit.condition_numbers) < 1.0e6
        assert np.max(audit.max_abs_weights) < 8.0
        assert np.max(audit.weight_l1) < 16.0
    else:
        assert len(audit.condition_numbers) == 0


def test_lattice_blend_capacity_failure_is_atomic():
    h = 0.125
    position = np.array([[0.75, 0.0, 0.0]])
    strength = np.array([[0.1, 0.2, 0.3]])
    vpm = _VPM(position, strength, capacity=1)
    before_position = vpm.particles.position.copy()
    before_strength = vpm.particles.vortex_strength.copy()

    with pytest.raises(RuntimeError, match="exceeding the VPM capacity"):
        _replace_lattice(
            vpm,
            np.array([[-0.25, 0.0, 0.0], [0.25, 0.0, 0.0]]),
            np.array([h**3, h**3]),
            np.array([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]),
            blend_width=0.25,
            spacing=h,
        )
    np.testing.assert_array_equal(vpm.particles.position, before_position)
    np.testing.assert_array_equal(vpm.particles.vortex_strength, before_strength)


@pytest.mark.parametrize(
    "failure",
    ["update_particle_vortex_strength", "remove_particles", "add_vortex_particles"],
)
def test_particle_mutation_failures_roll_back_every_particle_field(failure):
    if failure == "update_particle_vortex_strength":
        vpm = _FailingVPM(
            np.array([[0.25, 0.0, 0.0]]),
            np.array([[0.0, 2.0, 0.0]]),
            fail_at=failure,
        )
        replace_arguments = (
            np.array([[0.25, 0.0, 0.0]]),
            np.array([0.25]),
            np.array([[0.0, 8.0, 0.0]]),
        )
        blend_width = 0.5
    else:
        vpm = _FailingVPM(
            np.array([[0.0 if failure == "remove_particles" else 0.8, 0.0, 0.0]]),
            np.array([[0.0, 2.0, 0.0]]),
            fail_at=failure,
        )
        replace_arguments = (
            np.array([[0.0, 0.0, 0.0]]),
            np.array([0.25]),
            np.array([[0.0, 8.0, 0.0]]),
        )
        blend_width = 0.0
    before = _particle_state(vpm.particles)

    with pytest.raises(RuntimeError, match="injected"):
        _replace(vpm, *replace_arguments, blend_width=blend_width)

    after = _particle_state(vpm.particles)
    assert after.keys() == before.keys()
    for name in before:
        if isinstance(before[name], np.ndarray):
            np.testing.assert_array_equal(after[name], before[name])
        else:
            assert after[name] == before[name]


@pytest.mark.parametrize(
    "failure",
    ["update_particle_vortex_strength", "remove_particles", "add_vortex_particles"],
)
def test_lattice_particle_mutation_failures_roll_back_every_particle_field(failure):
    h = 0.125
    if failure == "update_particle_vortex_strength":
        vpm = _FailingVPM(
            np.array([[0.4375, 0.0, 0.0], [0.625, 0.0, 0.0]]),
            np.array([[0.0, 2.0, 0.0], [0.0, 0.3, 0.0]]),
            fail_at=failure,
        )
        blend_width = 0.0
        source_position = np.array([[0.4375, 0.0, 0.0]])
    elif failure == "remove_particles":
        vpm = _FailingVPM(
            np.array([[0.0, 0.0, 0.0]]),
            np.array([[0.0, 2.0, 0.0]]),
            fail_at=failure,
        )
        blend_width = 0.0
        source_position = np.array([[0.0, 0.0, 0.0]])
    else:
        vpm = _FailingVPM(
            np.array([[0.8, 0.0, 0.0]]),
            np.array([[0.0, 2.0, 0.0]]),
            fail_at=failure,
        )
        blend_width = 0.0
        source_position = np.array([[0.0, 0.0, 0.0]])
    before = _particle_state(vpm.particles)

    with pytest.raises(RuntimeError, match="injected"):
        _replace_lattice(
            vpm,
            source_position,
            np.array([h**3]),
            np.array([[0.0, 8.0, 0.0]]),
            blend_width=blend_width,
            spacing=h,
        )

    after = _particle_state(vpm.particles)
    for name in before:
        if isinstance(before[name], np.ndarray):
            np.testing.assert_array_equal(after[name], before[name])
        else:
            assert after[name] == before[name]


def test_post_mutation_count_failure_rolls_back_every_particle_field():
    vpm = _CountMismatchingVPM(
        np.array([[0.8, 0.0, 0.0]]),
        np.array([[0.0, 2.0, 0.0]]),
    )
    before = _particle_state(vpm.particles)

    with pytest.raises(RuntimeError, match="count after replacement"):
        _replace(
            vpm,
            np.array([[0.0, 0.0, 0.0]]),
            np.array([0.25]),
            np.array([[0.0, 8.0, 0.0]]),
        )

    after = _particle_state(vpm.particles)
    for name in before:
        if isinstance(before[name], np.ndarray):
            np.testing.assert_array_equal(after[name], before[name])
        else:
            assert after[name] == before[name]


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
