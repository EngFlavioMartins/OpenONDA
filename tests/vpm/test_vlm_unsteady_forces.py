"""Analytical pressure-patch force/moment checks, independent of KJ loads."""

from dataclasses import replace

from _flat_plate_geometry import create_flat_plate
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import taichi as ti

import openonda.vpm as vpm
from source.solvers.vpm.boundary_elements.vlm.solver.influence import compute_panel_force_coupled
from source.solvers.vpm.boundary_elements.vlm.solver.loading_distribution import (
    VLMLoadingDistribution,
)
from source.solvers.vpm.boundary_elements.vlm.solver.unsteady import add_unsteady_pressure_loads
from source.solvers.vpm.boundary_elements.vlm.solver.vlm_solver import VLMSolver


@pytest.fixture(autouse=True)
def runtime():
    ti.reset()
    ti.init(arch=ti.cpu, default_fp=ti.f64, offline_cache=False, cpu_max_num_threads=2)
    yield
    ti.reset()


def plate():
    geometry = create_flat_plate(
        chord=1,
        span=2,
        angle_of_attack_degrees=0,
        n_chordwise_panels=3,
        n_spanwise_panels=2,
    )
    solver = VLMSolver(
        vpm.VLMSetup(
            surfaces=(vpm.VLMSurfaceSetup(geometry, name="plate"),),
            dtype="f64",
            force=vpm.ForceConfig.kutta_joukowski(unsteady=True),
        )
    )
    solver.generate_mesh()
    return solver


def single_panel_plate():
    geometry = create_flat_plate(
        chord=1,
        span=1,
        angle_of_attack_degrees=0,
        n_chordwise_panels=1,
        n_spanwise_panels=1,
    )
    solver = VLMSolver(
        vpm.VLMSetup(
            surfaces=(vpm.VLMSurfaceSetup(geometry, name="single_panel"),),
            dtype="f64",
            force=vpm.ForceConfig.kutta_joukowski(unsteady=True),
        )
    )
    solver.generate_mesh()
    return solver


def set_circulation(solver, gamma, old_fraction=0.4):
    lattice = solver.lattice
    padded = np.zeros(lattice.max_n_panels)
    padded[: lattice.n_panels] = old_fraction * gamma
    lattice.circulation.from_numpy(padded)
    solver._compute_cumulative_circulation_cpu()
    lattice.save_old_circulation()
    padded[: lattice.n_panels] = gamma
    lattice.circulation.from_numpy(padded)
    solver._compute_cumulative_circulation_cpu()


def pressure_load(solver, density, dt):
    lattice = solver.lattice
    lattice.panel_force.fill(0)
    add_unsteady_pressure_loads(
        lattice.panel_corner_position,
        lattice.vortex_point_position,
        lattice.circulation,
        lattice.circulation_old,
        lattice.cumulative_circulation,
        lattice.cumulative_circulation_old,
        lattice.panel_force,
        lattice.unsteady_panel_force,
        lattice.panel_moment_correction,
        lattice.unsteady_pressure_jump_coefficient,
        lattice.n_panels,
        density / dt,
        2 / (density * 10**2),
    )
    solver._force_density = density
    solver._solved = True


def _triangle_area(a, b, c):
    """Return an oriented triangle area vector independently of Taichi."""
    return 0.5 * np.cross(b - a, c - a)


def _pressure_area_vectors(corners, vortex):
    """Return the fore/aft physical-panel area vectors for one horseshoe."""
    a, b = vortex[1], vortex[2]
    p, q, r, s = corners
    fore = _triangle_area(p, a, b) + _triangle_area(p, b, q)
    aft = _triangle_area(a, s, r) + _triangle_area(a, r, b)
    return fore, aft


def _set_bound_velocity_fields(solver, fluid, kinematic):
    lattice = solver.lattice
    velocity = np.zeros((lattice.max_n_panels, 3), dtype=lattice.np_dtype)
    motion = np.zeros_like(velocity)
    velocity[: lattice.n_panels] = fluid
    motion[: lattice.n_panels] = kinematic
    lattice.bound_vortex_velocity.from_numpy(velocity)
    lattice.bound_kinematic_velocity.from_numpy(motion)


def _kutta_joukowski_load(solver, density):
    lattice = solver.lattice
    lattice.panel_force.fill(0.0)
    compute_panel_force_coupled(
        lattice.bound_vortex_velocity,
        lattice.vortex_point_position,
        lattice.circulation,
        lattice.circulation_old,
        lattice.bound_kinematic_velocity,
        lattice.panel_force,
        lattice.n_panels,
        density,
        0,
    )
    return lattice.get_forces().copy()


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("rotated", [False, True])
def test_patch_partition_matches_whole_downstream_areas_and_centroid_moments(sign, rotated):
    solver = plate()
    lattice = solver.lattice
    n = lattice.n_panels
    corners = lattice.panel_corner_position.to_numpy()[:n]
    points = lattice.bound_vortex_midpoint.to_numpy()[:n]
    width = corners[:, 1, 1] - corners[:, 0, 1]
    gamma = sign * np.sign(width) * (1.0 + points[:, 0] + 0.2 * points[:, 1] ** 2)
    set_circulation(solver, gamma)
    density, dt = 1.3, 0.2
    # Independently integrate each increment's entire downstream rectangle.
    jump_rate = 0.6 * gamma / dt
    patch_forces = np.zeros((n, 3))
    patch_forces[:, 2] = density * jump_rate * width * (1 - points[:, 0])
    patch_centres = points.copy()
    patch_centres[:, 0] = (1 + points[:, 0]) / 2
    force = patch_forces.sum(axis=0)
    moment = np.cross(patch_centres, patch_forces).sum(axis=0)
    rotation = Rotation.from_rotvec([0.7, -0.4, 0.2]).as_matrix() if rotated else np.eye(3)
    shift = np.array([2.3, -1.1, 0.9])
    lattice.rotate_translate_panels(rotation, np.zeros(3), shift)
    pressure_load(solver, density, dt)
    values = solver.compute_forces(density, np.array([10.0, 0.0, 0.0]))
    expected_force = rotation @ force
    expected_moment = rotation @ moment + np.cross(shift, expected_force)
    np.testing.assert_allclose([values[f"force_{a}"] for a in "xyz"], expected_force, atol=1e-12)
    reference_moment = expected_moment - np.cross(
        solver.aircraft.refs["reference_point"], expected_force
    )
    np.testing.assert_allclose([values[f"moment_{a}"] for a in "xyz"], reference_moment, atol=1e-12)
    # Quarter-chord follows the geometry; its moment must ignore a world shift.
    moment_c4 = rotation @ (moment - np.cross([0.25, 0, 0], force))
    denominators = values["dynamic_pressure"] * values["reference_area"] * np.array([2.0, 1.0, 2.0])
    reported_c4 = np.array(
        [values[f"{a}_moment_coefficient_quarter_chord"] for a in ("rolling", "pitching", "yawing")]
    )
    np.testing.assert_allclose(reported_c4 * denominators, moment_c4, atol=1e-12)
    for multiplier in (1.0, 2.5):
        surface = solver.compute_per_surface_forces(
            density * multiplier, np.array([10.0, 0.0, 0.0])
        )["plate"]
        np.testing.assert_allclose(
            [surface[f"moment_{a}"] for a in "xyz"], multiplier * expected_moment, atol=1e-12
        )
        total = solver.compute_forces(density * multiplier, np.array([10.0, 0.0, 0.0]))
        np.testing.assert_allclose(
            [total[f"unsteady_force_{a}"] for a in "xyz"], multiplier * expected_force, atol=1e-12
        )


def test_manufactured_panel_separates_pressure_ramp_from_rigid_motion():
    """Check signed pressure construction and keep rigid-motion KJ separate.

    The pressure fixture is a fixed-geometry circulation ramp on the active
    half-panel.  The second half-panel is zeroed so the independent area
    integration is a one-panel calculation.  The same panel is then rotated
    with constant circulation: the unsteady circulation-rate term must be a
    no-op, while the independently prescribed relative velocity still enters
    Kutta--Joukowski with one density factor.
    """
    solver = single_panel_plate()
    lattice = solver.lattice
    n = lattice.n_panels
    gamma = np.zeros(n)
    gamma[0] = 1.0
    set_circulation(solver, gamma, old_fraction=0.4)
    density, dt = 1.3, 0.2
    pressure_load(solver, density, dt)

    corners = lattice.panel_corner_position.to_numpy()[:n]
    vortex = lattice.vortex_point_position.to_numpy()[:n]
    fore, aft = _pressure_area_vectors(corners[0], vortex[0])
    current_gamma = lattice.circulation.to_numpy()[:n]
    old_gamma = lattice.circulation_old.to_numpy()[:n]
    current_cumulative = lattice.cumulative_circulation.to_numpy()[:n]
    old_cumulative = lattice.cumulative_circulation_old.to_numpy()[:n]
    expected_pressure = np.zeros(3)
    for i in range(n):
        delta_cumulative = current_cumulative[i] - old_cumulative[i]
        delta_gamma = current_gamma[i] - old_gamma[i]
        panel_fore, panel_aft = _pressure_area_vectors(corners[i], vortex[i])
        expected_pressure += (
            density
            / dt
            * ((delta_cumulative - delta_gamma) * panel_fore + delta_cumulative * panel_aft)
        )
    np.testing.assert_allclose(
        lattice.unsteady_panel_force.to_numpy()[:n].sum(axis=0), expected_pressure, atol=1e-12
    )
    # The active one-panel geometry has b=0.5, c=1, so the aft area is 3/8 m².
    np.testing.assert_allclose(fore, [0.0, 0.0, 0.125], atol=1e-12)
    np.testing.assert_allclose(aft, [0.0, 0.0, 0.375], atol=1e-12)
    np.testing.assert_allclose(expected_pressure, [0.0, 0.0, 1.4625], atol=1e-12)

    # Rigidly rotate the same panel with constant circulation.  The pressure
    # term is a circulation-rate derivative and must remain zero; the relative
    # velocity force is checked independently under the same frame rotation.
    set_circulation(solver, gamma, old_fraction=1.0)
    rotation = Rotation.from_rotvec([0.23, -0.31, 0.17]).as_matrix()
    fluid = np.array([0.4, -0.2, 0.7])
    kinematic = np.array([-0.3, 0.6, 0.1])
    _set_bound_velocity_fields(
        solver,
        np.tile(fluid, (n, 1)),
        np.tile(kinematic, (n, 1)),
    )
    baseline_force = _kutta_joukowski_load(solver, density)[0]
    lattice.rotate_translate_panels(
        rotation,
        origin=np.array([0.1, -0.2, 0.3]),
        displacement=np.array([0.7, -0.4, 0.2]),
        start_idx=0,
        end_idx=1,
    )
    _set_bound_velocity_fields(
        solver,
        np.tile(rotation @ fluid, (n, 1)),
        np.tile(rotation @ kinematic, (n, 1)),
    )
    rotated_force = _kutta_joukowski_load(solver, density)[0]
    np.testing.assert_allclose(rotated_force, rotation @ baseline_force, atol=1e-12)
    pressure_load(solver, density, dt)
    np.testing.assert_allclose(lattice.unsteady_panel_force.to_numpy()[:n], 0.0, atol=1e-12)
    # Re-add the separately constructed KJ force and verify the public total
    # force path reports that same vector without a hidden density multiplier.
    rotated_force = _kutta_joukowski_load(solver, density)
    solver._force_density = density
    solver._solved = True
    total = solver.compute_forces(density, np.array([1.0, 0.0, 0.0]))
    np.testing.assert_allclose(
        [total[f"force_{axis}"] for axis in "xyz"], rotated_force.sum(axis=0), atol=1e-12
    )


def test_upstream_jump_loads_downstream_panels_and_power_at_pressure_centres():
    solver = plate()
    lattice = solver.lattice
    n = lattice.n_panels
    points = lattice.bound_vortex_midpoint.to_numpy()[:n]
    corners = lattice.panel_corner_position.to_numpy()[:n]
    width = corners[:, 1, 1] - corners[:, 0, 1]
    leading = lattice.is_leading_edge.to_numpy()[:n].astype(bool)
    set_circulation(solver, np.where(leading, 2 * np.sign(width), 0.0), old_fraction=0.0)
    pressure_load(solver, 1.0, 0.1)
    dx = corners[:, 3, 0] - corners[:, 0, 0]
    loaded_dx = np.where(leading, 0.75 * dx, dx)
    expected = 20 * np.abs(width) * loaded_dx
    np.testing.assert_allclose(lattice.get_forces()[:, 2], expected, atol=1e-12)
    # The same actual panel force is available from native loading samples.
    tables = VLMLoadingDistribution.extract_distributions(
        solver, "plate", np.array([10.0, 0.0, 0.0]), 1.0
    )
    assert tables["chordwise"]["unsteady_force_z"].sum() == pytest.approx(expected.sum())
    motion = vpm.RotatingVLM(angular_speed=2.0, axis=[0, 1, 0], rotation_centre=[0.2, 0, 0])
    aircraft, _ = solver.surfaces["plate"]
    solver.surfaces["plate"] = (aircraft, motion)
    values = solver.compute_per_surface_forces(1.0, np.array([10.0, 0.0, 0.0]))["plate"]
    centre_x = np.where(
        leading, (points[:, 0] + corners[:, 3, 0]) / 2, (corners[:, 0, 0] + corners[:, 3, 0]) / 2
    )
    expected_moment_y = -np.sum((centre_x - 0.2) * expected)
    assert values["moment_y"] == pytest.approx(expected_moment_y)
    assert values["rotational_power"] == pytest.approx(2 * expected_moment_y)


def test_postprocess_adds_only_pressure_term_and_constant_circulation_is_noop():
    solver = plate()
    lattice = solver.lattice
    corners = lattice.panel_corner_position.to_numpy()[: lattice.n_panels]
    gamma = np.sign(corners[:, 1, 1] - corners[:, 0, 1])
    set_circulation(solver, gamma)
    incident = np.tile([10.0, 0.0, 0.0], (lattice.n_panels, 1))
    solver.compute_postprocess(incident, np.array([10.0, 0.0, 0.0]), 1.4, 0.1)
    total = lattice.get_forces().copy()
    pressure = lattice.unsteady_panel_force.to_numpy()[: lattice.n_panels]
    assert np.linalg.norm(pressure) > 1
    solver.force = replace(solver.force, unsteady=False)
    solver.compute_postprocess(incident, np.array([10.0, 0.0, 0.0]), 1.4)
    np.testing.assert_allclose(total - lattice.get_forces(), pressure, atol=1e-12)
    assert not lattice.panel_moment_correction.to_numpy().any()
    assert not lattice.unsteady_pressure_jump_coefficient.to_numpy().any()
    solver.force = replace(solver.force, unsteady=True)
    set_circulation(solver, gamma, old_fraction=1.0)
    solver.compute_postprocess(incident, np.array([10.0, 0.0, 0.0]), 1.4, 0.1)
    assert not lattice.unsteady_panel_force.to_numpy().any()
    assert not lattice.panel_moment_correction.to_numpy().any()
    for dt in (None, 0.0, -0.1, float("nan")):
        with pytest.raises(ValueError, match="positive time_step_size"):
            solver.compute_postprocess(incident, np.array([10.0, 0.0, 0.0]), 1.4, dt)
