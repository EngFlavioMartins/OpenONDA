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
    replace_particles_from_lattice_blend,
    replacement_eta,
    required_renewal_buffer_length,
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


def _canonical_vortex_state(vpm: _VPM) -> tuple[np.ndarray, np.ndarray]:
    """Return position/strength ordered by position for renewal comparisons."""
    position = vpm.particles.position.copy()
    strength = vpm.particles.vortex_strength.copy()
    order = np.lexsort((position[:, 2], position[:, 1], position[:, 0]))
    return position[order], strength[order]


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

    for _ in range(20):
        result = _replace_lattice(
            vpm,
            fvm_position,
            volume,
            vorticity,
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


@pytest.mark.parametrize(
    ("blend_width", "persistent_x"),
    [(0.0, 0.625), (0.25, BOX[1])],
    ids=("hard", "c1-blend"),
)
def test_repeated_phase_shifted_m4_renewal_is_idempotent_across_release_support(
    blend_width: float,
    persistent_x: float,
):
    """An unchanged absolute handoff must not re-add the same M4 support.

    The half-lattice donor is the phase occurring at the cube's downstream
    interface.  Its complete M4' stencil crosses the ownership face and
    overlaps a persistent VPM node.  Once the first handoff has formed the
    common-lattice state, repeating that same handoff without advancing either
    solver must be an exact fixed point.
    """
    h = 0.125
    source_position = np.array([[0.4375, 0.0, 0.0]])
    source_strength = np.array([[0.0, 2.0, 0.0]])
    persistent_position = np.array([[persistent_x, 0.0, 0.0]])
    persistent_strength = np.array([[0.0, 0.3, 0.0]])
    vpm = _VPM(
        np.vstack((source_position, persistent_position)),
        np.vstack((source_strength, persistent_strength)),
        capacity=1_000,
    )

    _replace_lattice(
        vpm,
        source_position,
        np.array([1.0]),
        source_strength,
        blend_width=blend_width,
        spacing=h,
    )
    expected_position, expected_strength = _canonical_vortex_state(vpm)
    expected_count = vpm.particles.n_particles_total
    expected_max_strength = np.linalg.norm(expected_strength, axis=1).max()

    for _ in range(20):
        result = _replace_lattice(
            vpm,
            source_position,
            np.array([1.0]),
            source_strength,
            blend_width=blend_width,
            spacing=h,
        )
        actual_position, actual_strength = _canonical_vortex_state(vpm)

        assert result.n_particles_after == expected_count
        np.testing.assert_array_equal(actual_position, expected_position)
        np.testing.assert_allclose(actual_strength, expected_strength, rtol=0.0, atol=2.0e-15)
        assert np.linalg.norm(actual_strength, axis=1).max() <= expected_max_strength + 2.0e-15
        np.testing.assert_allclose(result.state_change_vortex_strength_net, 0.0, atol=2.0e-15)


def test_renewal_buffer_covers_release_travel_plus_complete_m4_support():
    h = 0.03125
    coupling_time_step = 0.01

    buffer_length = required_renewal_buffer_length(
        [1.0, 0.0, 0.0],
        coupling_time_step,
        h,
    )

    assert buffer_length == pytest.approx(1.5 * coupling_time_step + 2.0 * h)
    assert buffer_length == pytest.approx(0.0775)
    assert required_renewal_buffer_length([0.0, 0.0, 0.0], coupling_time_step, h) == pytest.approx(
        2.0 * h
    )


def test_sub_h_release_is_coalesced_in_the_renewal_belt_without_strength_blowup():
    h = 0.125
    displacement = 0.32 * h
    fvm_position = np.array([[0.4375, 0.0, 0.0]])
    source_strength = np.array([[0.0, 2.0, 0.0]])
    vpm = _VPM(np.empty((0, 3)), np.empty((0, 3)), capacity=10_000)
    release_band_counts: list[int] = []
    maximum_strengths: list[float] = []

    for _ in range(30):
        _replace_lattice(
            vpm,
            fvm_position,
            np.ones(1),
            source_strength,
            blend_width=2.0 * h,
            spacing=h,
        )
        position = vpm.particles.position
        strength = vpm.particles.vortex_strength
        release_band = (position[:, 0] > BOX[1]) & (position[:, 0] <= BOX[1] + 2.0 * h)
        release_band_counts.append(int(np.count_nonzero(release_band)))
        maximum_strengths.append(float(np.linalg.norm(strength, axis=1).max()))
        assert len(np.unique(position, axis=0)) == len(position)
        vpm.particles.position[:, 0] += displacement

    # Although one-step travel is much smaller than h, the fixed belt has at
    # most one particle per active node.  Older packets may leave as physical
    # free wake, but they cannot stack at sub-h spacing inside the handoff.
    assert max(release_band_counts) <= 2
    assert max(maximum_strengths) <= 0.5625 * np.linalg.norm(source_strength[0]) + 2.0e-15


def test_blend_reconciles_release_support_as_an_absolute_regular_node_without_duplicates():
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
        source_strength,
        blend_width=0.25,
        spacing=h,
    )

    matches = np.all(np.isclose(vpm.particles.position, boundary_position[0]), axis=1)
    assert matches.sum() == 1
    m4_weight = 0.5625
    np.testing.assert_allclose(
        vpm.particles.vortex_strength[matches][0],
        m4_weight * source_strength[0],
        atol=1.0e-15,
    )
    assert len(np.unique(vpm.particles.position, axis=0)) == vpm.particles.n_particles_total


def test_hard_release_support_overwrites_managed_outer_node_absolutely():
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
        source_strength,
        blend_width=0.0,
        spacing=h,
    )

    matches = np.all(np.isclose(vpm.particles.position, persistent_position[0]), axis=1)
    assert matches.sum() == 1
    np.testing.assert_allclose(
        vpm.particles.vortex_strength[matches][0],
        -0.0625 * source_strength[0],
        atol=1.0e-15,
    )


def test_zero_regular_support_guard_source_is_removed_instead_of_accumulating():
    """A represented guard node with a zero absolute target cannot remain stale."""
    h = 0.125
    guard_position = np.array([[0.875, 0.0, 0.0]])
    vpm = _VPM(guard_position, np.zeros((1, 3)))

    result = _replace_lattice(
        vpm,
        np.array([[0.0, 0.0, 0.0]]),
        np.array([h**3]),
        np.zeros((1, 3)),
        blend_width=0.25,
        spacing=h,
    )

    assert result.n_particles_removed == 1
    assert result.n_particles_injected == 0
    assert vpm.particles.n_particles_total == 0


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


def test_lattice_blend_excludes_solid_targets_with_an_explicit_budget():
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
    assert result.excluded_solid_active_nodes > 0
    assert result.excluded_solid_vortex_strength_l1 > 0.0
    expected_strength = h**3 * np.array([0.0, 4.0, 0.0])
    np.testing.assert_allclose(
        vpm.particles.vortex_strength.sum(axis=0) + result.excluded_solid_vortex_strength_net,
        expected_strength,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        vpm.particles.position.T @ vpm.particles.vortex_strength
        + result.excluded_solid_first_moment,
        np.outer(np.array([0.5 * h, 0.0, 0.0]), expected_strength),
        atol=1.0e-14,
    )


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
            np.array([[0.1, 0.0, 0.0]]),
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
