"""Finite-surface collision classification for non-mutating diagnostics."""

from types import SimpleNamespace

import numpy as np
import pytest
import taichi as ti

from source.solvers.vpm.boundary_elements.vlm.kernels.collision import (
    SURFACE_COLLISION_EVENT_CORE_OVERLAP,
    SURFACE_COLLISION_EVENT_INTERSECTION,
    SURFACE_COLLISION_EVENT_SIDE_BYPASS,
    detect_surface_collision_events_kernel,
    detect_surface_collisions_kernel,
)
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
