"""Manufactured checks for the isolated conservative flux-release candidate."""

from __future__ import annotations

import numpy as np
import pytest

from source.coupler.flux_handoff import FluxReleaseHandoff, vorticity_transport_flux
from source.coupler.lattice_transfer import evaluate_gaussian_vorticity


def test_vorticity_transport_flux_has_the_conservative_viscous_sign():
    velocity = np.array([[2.0, 0.5, -0.25]])
    vorticity = np.array([[0.25, -3.0, 1.5]])
    normal = np.array([[1.0, 0.0, 0.0]])
    normal_gradient = np.array([[4.0, -2.0, 0.5]])

    actual = vorticity_transport_flux(
        velocity,
        vorticity,
        normal,
        normal_gradient,
        kinematic_viscosity=0.125,
    )
    expected = 2.0 * vorticity - 0.25 * velocity - 0.125 * normal_gradient
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


def test_nonzero_diffusive_flux_is_included_in_the_emitted_circulation_budget():
    handoff = FluxReleaseHandoff(particle_spacing=1.0, core_radius=1.0)
    normal = np.array([[1.0, 0.0, 0.0]])
    velocity = np.array([[1.0, 0.0, 0.0]])
    vorticity = np.array([[0.0, 2.0, 0.0]])
    normal_gradient = np.array([[0.0, 4.0, 0.0]])
    flux = vorticity_transport_flux(
        velocity,
        vorticity,
        normal,
        normal_gradient,
        kinematic_viscosity=0.25,
    )

    batch = handoff.advance(
        slot_id=np.array([[0, 0, 0]]),
        slot_position=np.zeros((1, 3)),
        slot_normal=normal,
        patch_area=np.ones(1),
        vorticity_flux=flux,
        normal_velocity=np.ones(1),
        transport_velocity=velocity,
        time_step_size=1.0,
    )

    # Convective flux contributes +2 and outward diffusive flux contributes
    # -nu*d_n(omega)=-1, so the emitted vector circulation is exactly +1.
    np.testing.assert_allclose(batch.vortex_strength, [[0.0, 1.0, 0.0]])
    np.testing.assert_allclose(batch.conservation_error, 0.0, atol=2.0e-15)


def test_convective_flux_vector_cannot_copy_an_oblique_vortex_line_direction():
    """Expose the geometric limit of mapping contracted flux directly to Gamma."""
    normal = np.array([[1.0, 0.0, 0.0]])
    velocity = np.array([[2.0, 0.0, 0.0]])
    vorticity = np.array([[1.0, 1.0, 0.0]]) / np.sqrt(2.0)
    flux = vorticity_transport_flux(velocity, vorticity, normal)

    # The inviscid contracted vorticity flux is identically tangent to a
    # surface: n dot (u_n*omega - omega_n*u) == 0.  A particle strength is a
    # volume-integrated vorticity vector, so equating the two cannot preserve
    # the direction of a vortex line with a nonzero normal component.
    np.testing.assert_allclose(np.einsum("ij,ij->i", flux, normal), 0.0, atol=2.0e-15)
    handoff = FluxReleaseHandoff(particle_spacing=1.0, core_radius=1.0)
    batch = handoff.advance(
        slot_id=np.array([[0, 0, 0]]),
        slot_position=np.zeros((1, 3)),
        slot_normal=normal,
        patch_area=np.ones(1),
        vorticity_flux=flux,
        normal_velocity=np.array([2.0]),
        transport_velocity=velocity,
        time_step_size=0.5,
    )
    physical_packet_strength = vorticity  # omega * area * U*dt, with area*U*dt=1
    np.testing.assert_allclose(batch.vortex_strength[:, 0], 0.0, atol=2.0e-15)
    assert not np.allclose(batch.vortex_strength, physical_packet_strength)
    direction_cosine = float(
        np.dot(batch.vortex_strength[0], physical_packet_strength[0])
        / (np.linalg.norm(batch.vortex_strength[0]) * np.linalg.norm(physical_packet_strength[0]))
    )
    assert direction_cosine == pytest.approx(1.0 / np.sqrt(2.0), abs=2.0e-15)
    # F1 still closes its contracted-flux budget exactly; conservation of that
    # budget alone is therefore insufficient to certify the physical field.
    np.testing.assert_allclose(batch.conservation_error, 0.0, atol=2.0e-15)


def _planar_patch_inputs(
    *,
    h: float,
    normal_speed: float,
    vorticity: np.ndarray,
):
    return {
        "slot_id": np.array([[0, 0, 0]], dtype=np.int64),
        "slot_position": np.zeros((1, 3)),
        "slot_normal": np.array([[1.0, 0.0, 0.0]]),
        "patch_area": np.array([h**2]),
        "vorticity_flux": normal_speed * np.asarray(vorticity, dtype=np.float64).reshape(1, 3),
        "normal_velocity": np.array([normal_speed]),
    }


def test_flux_reservoir_uses_transport_distance_not_coupling_frequency():
    h = 0.03125
    time_step = 0.01
    transport_ratio = 0.317
    normal_speed = transport_ratio * h / time_step
    vorticity = np.array([0.0, 2.0, -0.5])
    handoff = FluxReleaseHandoff(particle_spacing=h, core_radius=h)
    patch = _planar_patch_inputs(h=h, normal_speed=normal_speed, vorticity=vorticity)
    existing = np.empty((0, 3))
    emitted_strength: list[np.ndarray] = []
    emission_step: list[int] = []

    for step in range(20):
        existing[:, 0] += normal_speed * time_step
        batch = handoff.advance(**patch, time_step_size=time_step, existing_position=existing)
        if len(batch.position):
            emitted_strength.append(batch.vortex_strength)
            emission_step.extend([step + 1] * len(batch.position))
            assert batch.min_new_existing_separation >= h
            assert np.all(batch.neighbour_count_within_2sigma >= 0)
        existing = np.vstack((existing, batch.position))
        np.testing.assert_allclose(batch.conservation_error, 0.0, atol=2.0e-18)

    position = existing
    strength = np.vstack(emitted_strength)
    assert emission_step == [4, 7, 10, 13, 16, 19]
    np.testing.assert_allclose(
        np.diff(position[:, 0]),
        -h,
        rtol=0.0,
        atol=2.0e-15,
    )
    np.testing.assert_allclose(
        strength,
        np.broadcast_to(vorticity * h**3, strength.shape),
        rtol=0.0,
        atol=3.0e-20,
    )
    pending = handoff.slot_status()[0].pending_strength
    np.testing.assert_allclose(
        handoff.emitted_vortex_strength_total + pending,
        handoff.outward_flux_total,
        atol=2.0e-18,
    )


def test_release_slot_holds_circulation_when_geometry_is_unsafe_then_releases_it():
    h = 1.0
    handoff = FluxReleaseHandoff(
        particle_spacing=h,
        core_radius=h,
        max_pending_age=0.5,
    )
    patch = _planar_patch_inputs(
        h=h,
        normal_speed=1.0,
        vorticity=np.array([0.0, 1.0, 0.0]),
    )
    blocked = handoff.advance(
        **patch,
        time_step_size=1.0,
        existing_position=np.array([[0.5, 0.0, 0.0]]),
    )
    assert len(blocked.position) == 0
    assert blocked.held_slot_ids == ((0, 0, 0),)
    assert blocked.trapped_slot_ids == ((0, 0, 0),)
    np.testing.assert_allclose(blocked.pending_vortex_strength_net, [0.0, 1.0, 0.0])

    released = handoff.advance(
        **patch,
        time_step_size=1.0,
        existing_position=np.empty((0, 3)),
    )
    assert len(released.position) == 1
    np.testing.assert_allclose(released.vortex_strength.sum(axis=0), [0.0, 1.0, 0.0])
    assert released.held_slot_ids == ((0, 0, 0),)
    np.testing.assert_allclose(released.pending_vortex_strength_net, [0.0, 1.0, 0.0])
    np.testing.assert_allclose(released.conservation_error, 0.0, atol=2.0e-15)


def test_duplicate_surface_patches_accumulate_into_one_global_release_slot():
    handoff = FluxReleaseHandoff(particle_spacing=1.0, core_radius=1.0)
    batch = handoff.advance(
        slot_id=np.array([[2, -1, 3], [2, -1, 3]], dtype=np.int64),
        slot_position=np.zeros((2, 3)),
        slot_normal=np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        patch_area=np.array([0.5, 0.5]),
        vorticity_flux=np.array([[0.0, 2.0, 0.0], [0.0, 2.0, 0.0]]),
        normal_velocity=np.array([1.0, 1.0]),
        time_step_size=1.0,
    )
    assert len(batch.position) == 1
    np.testing.assert_array_equal(batch.slot_id, [[2, -1, 3]])
    np.testing.assert_allclose(batch.vortex_strength, [[0.0, 2.0, 0.0]])
    assert batch.outward_area_fraction == 1.0
    np.testing.assert_allclose(batch.conservation_error, 0.0, atol=2.0e-15)


def test_inward_patches_are_reported_but_not_emitted_by_a_one_way_handoff():
    handoff = FluxReleaseHandoff(particle_spacing=1.0, core_radius=1.0)
    batch = handoff.advance(
        slot_id=np.array([[0, 0, 0], [1, 0, 0]], dtype=np.int64),
        slot_position=np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        slot_normal=np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        patch_area=np.array([1.0, 1.0]),
        vorticity_flux=np.array([[0.0, 2.0, 0.0], [0.0, 3.0, 0.0]]),
        normal_velocity=np.array([1.0, -1.0]),
        time_step_size=1.0,
    )
    assert len(batch.position) == 1
    assert batch.outward_area_fraction == 0.5
    assert batch.inward_area_fraction == 0.5
    np.testing.assert_allclose(batch.inward_flux_vortex_strength_increment, [0.0, 3.0, 0.0])
    np.testing.assert_allclose(batch.outward_flux_vortex_strength_increment, [0.0, 2.0, 0.0])


def test_birth_spacing_is_invariant_under_a_coupling_timestep_sweep():
    h = 1.0
    vorticity = np.array([0.0, 1.0, 0.0])
    for transport_ratio in (0.1, 0.25, 0.317, 0.5, 1.0, 1.5):
        handoff = FluxReleaseHandoff(particle_spacing=h, core_radius=h)
        patch = _planar_patch_inputs(h=h, normal_speed=1.0, vorticity=vorticity)
        time_step = transport_ratio
        existing = np.empty((0, 3))
        strength: list[np.ndarray] = []
        for _ in range(int(np.ceil(12.0 / transport_ratio))):
            existing[:, 0] += time_step
            batch = handoff.advance(**patch, time_step_size=time_step, existing_position=existing)
            existing = np.vstack((existing, batch.position))
            strength.append(batch.vortex_strength)
            assert batch.min_new_new_separation >= h - 3.0e-14
            assert batch.min_new_existing_separation >= h - 3.0e-14
            if len(batch.position):
                assert np.all(batch.nearest_neighbour_distance_over_spacing >= 1.0 - 3.0e-14)
                np.testing.assert_allclose(batch.core_radius_over_spacing, 1.0)
            np.testing.assert_allclose(batch.conservation_error, 0.0, atol=1.0e-12)
        assert len(existing) >= 12
        ordered = np.sort(existing[:, 0])
        np.testing.assert_allclose(np.diff(ordered), h, rtol=0.0, atol=3.0e-14)
        np.testing.assert_allclose(
            np.vstack(strength),
            np.broadcast_to(vorticity, (len(existing), 3)),
            rtol=0.0,
            atol=3.0e-14,
        )


@pytest.mark.parametrize("crossing_angle_degrees", [0.0, 30.0, 45.0, 60.0, 75.0])
def test_oblique_transport_advects_an_interpolated_birth_to_the_interval_endpoint(
    crossing_angle_degrees,
):
    handoff = FluxReleaseHandoff(particle_spacing=1.0, core_radius=1.0)
    tangential_speed = np.tan(np.deg2rad(crossing_angle_degrees))
    patch = {
        "slot_id": np.array([[0, 0, 0]]),
        "slot_position": np.zeros((1, 3)),
        "slot_normal": np.array([[1.0, 0.0, 0.0]]),
        "patch_area": np.ones(1),
        "vorticity_flux": np.array([[0.0, 1.0, 0.0]]),
        "normal_velocity": np.ones(1),
        "transport_velocity": np.array([[1.0, tangential_speed, 0.0]]),
    }

    first = handoff.advance(**patch, time_step_size=0.6)
    second = handoff.advance(**patch, time_step_size=0.6)

    assert len(first.position) == 0
    # The crossing occurs 0.4 s into the second interval, leaving 0.2 s of
    # full oblique convection before the end-of-interval insertion state.
    np.testing.assert_allclose(
        second.position,
        [[0.2, 0.2 * tangential_speed, 0.0]],
        rtol=0.0,
        atol=2.0e-15,
    )
    np.testing.assert_allclose(second.vortex_strength, [[0.0, 1.0, 0.0]])


def test_oblique_transport_rejects_an_inconsistent_declared_normal_speed():
    handoff = FluxReleaseHandoff(particle_spacing=1.0, core_radius=1.0)
    with pytest.raises(ValueError, match="normal component"):
        handoff.advance(
            slot_id=np.array([[0, 0, 0]]),
            slot_position=np.zeros((1, 3)),
            slot_normal=np.array([[1.0, 0.0, 0.0]]),
            patch_area=np.ones(1),
            vorticity_flux=np.array([[0.0, 1.0, 0.0]]),
            normal_velocity=np.ones(1),
            transport_velocity=np.array([[0.9, 2.0, 0.0]]),
            time_step_size=1.0,
        )


def test_steady_resolved_vorticity_sheet_matches_fvm_reference_at_check_surface():
    """F1's first physical-field test: a resolved convective sheet at Udt/h=.317."""
    h = 0.03125
    time_step = 0.01
    transport_ratio = 0.317
    normal_speed = transport_ratio * h / time_step
    width = 8.0 * h
    tangent_index = np.arange(-14, 15)
    j, k = np.meshgrid(tangent_index, tangent_index, indexing="ij")
    slot_position = np.column_stack((np.zeros(j.size), h * j.ravel(), h * k.ravel()))
    slot_id = np.column_stack((np.zeros(j.size, dtype=np.int64), j.ravel(), k.ravel()))
    slot_normal = np.tile([1.0, 0.0, 0.0], (len(slot_position), 1))
    source_vorticity = np.column_stack(
        (
            np.zeros(len(slot_position)),
            np.exp(-((slot_position[:, 2] / width) ** 2)),
            np.zeros(len(slot_position)),
        )
    )
    flux = vorticity_transport_flux(
        np.tile([normal_speed, 0.0, 0.0], (len(slot_position), 1)),
        source_vorticity,
        slot_normal,
    )
    handoff = FluxReleaseHandoff(particle_spacing=h, core_radius=h)
    existing_position = np.empty((0, 3))
    emitted_strength: list[np.ndarray] = []
    for _ in range(25):
        existing_position[:, 0] += normal_speed * time_step
        batch = handoff.advance(
            slot_id=slot_id,
            slot_position=slot_position,
            slot_normal=slot_normal,
            patch_area=np.full(len(slot_position), h**2),
            vorticity_flux=flux,
            normal_velocity=np.full(len(slot_position), normal_speed),
            time_step_size=time_step,
            existing_position=existing_position,
        )
        assert not batch.held_slot_ids
        np.testing.assert_allclose(batch.conservation_error, 0.0, atol=1.0e-14)
        existing_position = np.vstack((existing_position, batch.position))
        emitted_strength.append(batch.vortex_strength)

    y, z = np.meshgrid(
        np.linspace(-3.0 * h, 3.0 * h, 9),
        np.linspace(-3.0 * h, 3.0 * h, 9),
        indexing="ij",
    )
    check_position = np.column_stack((np.full(y.size, 4.0 * h), y.ravel(), z.ravel()))
    vpm_vorticity = evaluate_gaussian_vorticity(
        check_position,
        existing_position,
        np.vstack(emitted_strength),
        np.full(len(existing_position), h),
    )
    fvm_vorticity = np.column_stack(
        (
            np.zeros(len(check_position)),
            np.exp(-((check_position[:, 2] / width) ** 2)),
            np.zeros(len(check_position)),
        )
    )
    relative_rms = np.sqrt(np.mean((vpm_vorticity - fvm_vorticity) ** 2)) / np.sqrt(
        np.mean(fvm_vorticity**2)
    )
    assert relative_rms < 0.01
    assert len(existing_position) == 7 * len(slot_position)
