"""Finite-surface collision classification for non-mutating diagnostics."""

from types import SimpleNamespace

from _flat_plate_geometry import create_flat_plate
import numpy as np
import pytest
import taichi as ti

from source.solvers.vpm.boundary_elements.vlm.config import VLMSetup, VLMSurfaceSetup
from source.solvers.vpm.boundary_elements.vlm.kernels.collision import (
    SURFACE_COLLISION_EVENT_CORE_OVERLAP,
    SURFACE_COLLISION_EVENT_INTERSECTION,
    SURFACE_COLLISION_EVENT_SIDE_BYPASS,
    classify_finite_surface_segment,
    classify_moving_finite_surface_segment,
    detect_surface_collision_events_kernel,
    detect_surface_collisions_kernel,
)
from source.solvers.vpm.boundary_elements.vlm.solver.vlm_solver import VLMSolver
from source.solvers.vpm.coupling.stepper import CouplingStepper


@pytest.fixture(scope="module", autouse=True)
def cpu_runtime():
    ti.reset()
    ti.init(arch=ti.cpu, default_fp=ti.f64, offline_cache=False, cpu_max_num_threads=2)
    yield
    ti.reset()


def _run_point_collisions(points, normal):
    n = len(points)
    if n == 0:
        return np.array([], dtype=np.int32)

    corners = ti.Vector.field(3, dtype=ti.f64, shape=(1, 4))
    normals = ti.Vector.field(3, dtype=ti.f64, shape=1)
    tags = ti.field(dtype=ti.i32, shape=16)
    positions = ti.Vector.field(3, dtype=ti.f64, shape=16)

    corners.from_numpy(
        np.array(
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            ],
            dtype=np.float64,
        )
    )
    normals.from_numpy(np.array([normal], dtype=np.float64))
    points_np = np.asarray(points, dtype=np.float64)
    positions_buffer = np.zeros((16, 3), dtype=np.float64)
    positions_buffer[:n, :] = points_np
    positions.from_numpy(positions_buffer)
    tags.fill(0)

    detect_surface_collisions_kernel(
        positions,
        tags,
        corners,
        normals,
        n,
        1,
        0.0015,
    )

    return tags.to_numpy()[:n]


def _run_event_classification(start, end, radius):
    n = len(start)
    if n == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32)

    corners = ti.Vector.field(3, dtype=ti.f64, shape=(1, 4))
    normals = ti.Vector.field(3, dtype=ti.f64, shape=1)
    start_field = ti.Vector.field(3, dtype=ti.f64, shape=16)
    end_field = ti.Vector.field(3, dtype=ti.f64, shape=16)
    radius_field = ti.field(dtype=ti.f64, shape=16)
    event = ti.field(dtype=ti.i32, shape=16)
    panel = ti.field(dtype=ti.i32, shape=16)

    corners.from_numpy(
        np.array(
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            ],
            dtype=np.float64,
        )
    )
    normals.from_numpy(np.array([[0.0, 0.0, 1.0]], dtype=np.float64))
    start_np = np.zeros((16, 3), dtype=np.float64)
    end_np = np.zeros((16, 3), dtype=np.float64)
    radius_np = np.zeros(16, dtype=np.float64)
    start_np[:n] = np.asarray(start, dtype=np.float64)
    end_np[:n] = np.asarray(end, dtype=np.float64)
    radius_np[:n] = np.asarray(radius, dtype=np.float64)
    start_field.from_numpy(start_np)
    end_field.from_numpy(end_np)
    radius_field.from_numpy(radius_np)

    event.fill(0)
    panel.fill(-1)
    detect_surface_collision_events_kernel(
        start_field,
        end_field,
        radius_field,
        event,
        panel,
        corners,
        normals,
        n,
        1,
        0.001,
        1.0,
    )
    return event.to_numpy()[:n], panel.to_numpy()[:n]


def test_collision_classifier_is_invariant_to_panel_normal_scale():
    point = np.array([[0.5, 0.5, -0.0010]], dtype=np.float64)
    scaled = _run_point_collisions(point, normal=(0.0, 0.0, 2.0))
    unit = _run_point_collisions(point, normal=(0.0, 0.0, 1.0))
    np.testing.assert_array_equal(scaled, unit)
    np.testing.assert_array_equal(scaled, np.array([1], dtype=np.int32))


def test_collision_classifier_accepts_reverse_normal_winding():
    point = np.array([[0.5, 0.5, -0.0010]], dtype=np.float64)
    reverse = _run_point_collisions(point, normal=(0.0, 0.0, -1.0))
    np.testing.assert_array_equal(reverse, np.array([1], dtype=np.int32))


def test_surface_event_kernel_classifies_intersection_bypass_and_core_overlap():
    start = np.array(
        [
            [0.5, 0.5, 0.010],
            [-0.2, 0.5, 0.010],
            [0.5, 1.05, 0.0025],
        ],
        dtype=np.float64,
    )
    end = np.array(
        [
            [0.5, 0.5, -0.010],
            [-0.2, 0.5, -0.010],
            [0.5, 1.05, 0.0030],
        ],
        dtype=np.float64,
    )
    radius = np.array([0.0, 0.0, 0.0015], dtype=np.float64)

    event, panel = _run_event_classification(start, end, radius)
    np.testing.assert_array_equal(panel, np.array([0, 0, 0], dtype=np.int32))
    np.testing.assert_array_equal(
        event,
        np.array(
            [
                SURFACE_COLLISION_EVENT_INTERSECTION,
                SURFACE_COLLISION_EVENT_SIDE_BYPASS,
                SURFACE_COLLISION_EVENT_CORE_OVERLAP,
            ],
            dtype=np.int32,
        ),
    )


def _host_panel():
    return (
        np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float64,
        ),
        np.array([0.0, 0.0, 1.0], dtype=np.float64),
    )


def test_host_surface_observer_requires_finite_interior_intersection():
    corners, normal = _host_panel()
    result = classify_finite_surface_segment(
        np.array([0.5, 0.5, 0.01]),
        np.array([0.5, 0.5, -0.01]),
        0.0,
        corners,
        normal,
        tolerance=1.0e-4,
    )

    assert result["event"] == SURFACE_COLLISION_EVENT_INTERSECTION
    np.testing.assert_allclose(result["position"], [0.5, 0.5, 0.0])


def test_host_surface_observer_distinguishes_side_bypass_and_core_overlap():
    corners, normal = _host_panel()
    bypass = classify_finite_surface_segment(
        np.array([-0.2, 0.5, 0.01]),
        np.array([-0.2, 0.5, -0.01]),
        0.0,
        corners,
        normal,
        tolerance=1.0e-4,
    )
    overlap = classify_finite_surface_segment(
        np.array([0.5, 1.0005, 0.0005]),
        np.array([0.5, 1.0005, 0.0005]),
        0.0015,
        corners,
        normal,
        tolerance=1.0e-4,
    )

    assert bypass["event"] == SURFACE_COLLISION_EVENT_SIDE_BYPASS
    assert overlap["event"] == SURFACE_COLLISION_EVENT_CORE_OVERLAP
    assert bypass["distance"] > overlap["distance"]


def test_host_surface_observer_detects_core_overlap_over_face_interior():
    corners, normal = _host_panel()
    result = classify_finite_surface_segment(
        np.array([0.5, 0.5, 5.0e-4]),
        np.array([0.5, 0.5, 5.0e-4]),
        1.0e-3,
        corners,
        normal,
        tolerance=1.0e-5,
    )
    assert result["event"] == SURFACE_COLLISION_EVENT_CORE_OVERLAP
    np.testing.assert_allclose(result["distance"], 5.0e-4, atol=1.0e-12)


def test_moving_surface_observer_uses_relative_motion_not_midpoint_plane():
    corners, normal = _host_panel()
    translated = corners + np.array([0.0, 0.0, 0.2])
    result = classify_moving_finite_surface_segment(
        np.array([0.5, 0.5, 0.1]),
        np.array([0.5, 0.5, 0.1]),
        0.0,
        corners,
        translated,
        normal,
        normal,
        tolerance=1.0e-5,
    )
    assert result["event"] == SURFACE_COLLISION_EVENT_INTERSECTION
    np.testing.assert_allclose(result["position"], [0.5, 0.5, 0.1], atol=1.0e-12)


def test_moving_surface_observer_handles_rigid_panel_rotation():
    corners, normal = _host_panel()
    centre = np.array([0.5, 0.5, 0.0])
    angle = np.pi / 2.0
    rotation = np.array(
        [
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ]
    )
    rotated = (corners - centre) @ rotation.T + centre
    rotated_normal = rotation @ normal
    result = classify_moving_finite_surface_segment(
        centre + np.array([0.0, 0.0, 0.1]),
        centre + np.array([0.0, 0.0, 0.1]),
        0.0,
        corners,
        rotated,
        normal,
        rotated_normal,
        tolerance=1.0e-5,
    )
    assert result["event"] == SURFACE_COLLISION_EVENT_INTERSECTION
    np.testing.assert_allclose(result["position"], centre + [0.0, 0.0, 0.1])


def test_host_surface_observer_reports_endpoint_on_surface_and_warped_panel():
    corners, normal = _host_panel()
    endpoint = classify_finite_surface_segment(
        np.array([0.5, 0.5, 0.0]),
        np.array([0.5, 0.5, 0.02]),
        0.0,
        corners,
        normal,
        tolerance=1.0e-5,
    )
    assert endpoint["event"] == SURFACE_COLLISION_EVENT_INTERSECTION
    warped = corners.copy()
    warped[2, 2] = 0.05
    warped_result = classify_finite_surface_segment(
        np.array([0.5, 0.5, 0.02]),
        np.array([0.5, 0.5, -0.02]),
        0.0,
        warped,
        normal,
        tolerance=1.0e-4,
    )
    assert warped_result["event"] in {
        SURFACE_COLLISION_EVENT_INTERSECTION,
        SURFACE_COLLISION_EVENT_SIDE_BYPASS,
    }


def test_host_surface_observer_is_invariant_to_translation_normal_scale_and_winding():
    corners, normal = _host_panel()
    translated = corners + np.array([3.0, -2.0, 0.4])
    start = np.array([3.25, -1.75, 0.41])
    end = np.array([3.25, -1.75, 0.39])
    forward = classify_finite_surface_segment(
        start,
        end,
        0.0,
        translated,
        2.0 * normal,
        tolerance=1.0e-4,
    )
    reverse = classify_finite_surface_segment(
        start,
        end,
        0.0,
        translated[[0, 3, 2, 1]],
        -normal,
        tolerance=1.0e-4,
    )

    assert forward["event"] == SURFACE_COLLISION_EVENT_INTERSECTION
    assert reverse["event"] == SURFACE_COLLISION_EVENT_INTERSECTION
    np.testing.assert_allclose(forward["position"], reverse["position"])


def test_host_surface_observer_rejects_degenerate_panels():
    corners, normal = _host_panel()
    corners[2] = corners[1] + np.array([0.5, 0.0, 0.0])
    corners[3] = corners[0]
    with pytest.raises(ValueError, match="degenerate panel"):
        classify_finite_surface_segment(
            np.array([0.5, 0.5, 0.01]),
            np.array([0.5, 0.5, -0.01]),
            0.0,
            corners,
            normal,
            tolerance=1.0e-4,
        )


def test_surface_observer_is_pure_and_runs_before_any_particle_topology_change():
    solver = VLMSolver(
        VLMSetup(
            surfaces=(
                VLMSurfaceSetup(
                    create_flat_plate(
                        chord=1.0,
                        span=1.0,
                        n_chordwise_panels=1,
                        n_spanwise_panels=1,
                    ),
                    name="receiving_surface",
                ),
            ),
            dtype="f64",
            freestream_velocity=(1.0, 0.0, 0.0),
        )
    )
    solver.generate_mesh()
    lattice = solver.lattice
    corners_before = lattice.panel_corner_position.to_numpy().copy()
    circulation_before = lattice.circulation.to_numpy().copy()
    cumulative_before = lattice.cumulative_circulation.to_numpy().copy()
    transported_before = solver._transported_bound.to_numpy().copy()
    centre = lattice.collocation_point.to_numpy()[0]
    normal = lattice.normal.to_numpy()[0]

    class Particles:
        def __init__(self):
            self.position = np.array([centre + 0.1 * normal], dtype=np.float64)
            self.end_position = np.array([centre - 0.1 * normal], dtype=np.float64)
            self.n_particles_total = 1

        def position_cpu(self, use_cache=False):
            del use_cache
            return self.position.copy()

        def vortex_strength_cpu(self, use_cache=False):
            del use_cache
            return np.array([[0.0, 0.2, 0.1]], dtype=np.float64)

        def core_radius_cpu(self, use_cache=False):
            del use_cache
            return np.array([1.0e-3], dtype=np.float64)

        def group_id_cpu(self, use_cache=False):
            del use_cache
            return np.array([7], dtype=np.int32)

    particles = Particles()
    solver._snapshot_pre_transport_positions(particles, time=0.0, time_step_size=0.02)
    particles.position = particles.end_position.copy()
    result = solver.observe_surface_interaction(particles)

    assert result["n_events"] == 1
    assert result["events"][0]["provenance"] == "active_particle_index"
    assert result["events"][0]["surface"] == "receiving_surface"
    np.testing.assert_array_equal(lattice.panel_corner_position.to_numpy(), corners_before)
    np.testing.assert_array_equal(lattice.circulation.to_numpy(), circulation_before)
    np.testing.assert_array_equal(lattice.cumulative_circulation.to_numpy(), cumulative_before)
    np.testing.assert_array_equal(solver._transported_bound.to_numpy(), transported_before)
    np.testing.assert_array_equal(particles.position, particles.end_position)


def test_coupling_stepper_runs_coupling_when_wake_release_is_disabled():
    calls = []

    def fake_advance_coupled(particles, physics, config, time_step_size, step, time, release_wake):
        calls.append((time_step_size, release_wake))
        if not release_wake:
            return None
        return {
            "_gpu_transfer_ready": True,
            "vertex_position": np.zeros((0, 3)),
            "velocity": np.zeros((0, 3)),
            "vortex_strength": np.zeros((0, 3)),
            "core_radius": np.zeros(0),
            "particle_volume": np.zeros(0),
        }

    added = []

    solver = SimpleNamespace(
        vlm_solver=SimpleNamespace(advance_coupled=fake_advance_coupled),
        particles=SimpleNamespace(),
        physics=SimpleNamespace(),
        setup=SimpleNamespace(),
        stepper=SimpleNamespace(time=0.25, step=3),
        _release_wake_particles=False,
        _release_interval=0.4,
        time_step_size=0.5,
        add_vortex_particles=lambda **kwargs: added.append(kwargs),
    )

    stepper = CouplingStepper(solver)
    stepper.advance_vlm(0.2)

    assert calls == [(0.4, False)]
    assert added == []

    solver._release_wake_particles = True
    stepper.advance_vlm(0.2)

    assert calls[-1] == (0.4, True)
    assert len(added) == 1
