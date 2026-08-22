"""Boundary and time-coordinate properties of the minimal coupler."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from source.coupler.boundary import (
    advance_fvm_substeps,
    boundary_flux_tolerance,
    evaluate_vpm_boundary,
    evaluate_vpm_velocity,
    initialize_vpm_boundary_history,
    tangential_normal_velocity_gradient,
)
from source.coupler.reporting import compute_diagnostics
from source.coupler.solver import FVMVPMCoupler
from source.coupler.vorticity_transfer import TransferResult, vortex_strength_from_velocity_trace
from source.solvers.FVM.fields.diagnostics import compute_vorticity
from source.solvers.FVM.mesh.cartesian import structured_box
from source.solvers.FVM.mesh.geometry import compute_mesh_geometry


def _cube_face_quadrature(nside: int = 6):
    edge = np.linspace(-1.0, 1.0, nside + 1)
    centre = 0.5 * (edge[:-1] + edge[1:])
    area = (2.0 / nside) ** 2
    points, normals = [], []
    for axis in range(3):
        for sign in (-1.0, 1.0):
            a, b = np.meshgrid(centre, centre, indexing="ij")
            face = np.zeros((nside * nside, 3))
            face[:, axis] = sign
            face[:, (axis + 1) % 3] = a.ravel()
            face[:, (axis + 2) % 3] = b.ravel()
            normal = np.zeros_like(face)
            normal[:, axis] = sign
            points.append(face)
            normals.append(normal)
    return np.vstack(points), np.vstack(normals), np.full(6 * nside * nside, area)


class _Particles:
    n_particles = 0


def test_constant_vpm_bc_is_reproduced_exactly_without_particles():
    class _VPM:
        particles = _Particles()

        @staticmethod
        def compute_target_velocities(points, **kwargs):
            return np.tile([1.0, -0.25, 0.5], (len(points), 1))

    freestream = np.array([1.0, -0.25, 0.5])
    centres, normals, areas = _cube_face_quadrature()
    velocity, diagnostics = evaluate_vpm_velocity(
        _VPM(),
        centres,
        normals,
        areas,
        freestream_velocity=freestream,
        fvm_box=np.array([-1.0, 1.0] * 3),
        particle_spacing=0.05,
    )
    np.testing.assert_allclose(velocity, np.tile(freestream, (len(centres), 1)), atol=1e-15)
    assert diagnostics["raw_relative"] < 1.0e-14
    assert diagnostics["corrected_mismatch"] < 1.0e-14


def test_flux_projection_accepts_only_discretization_scale_residual():
    centres, normals, areas = _cube_face_quadrature(nside=5)
    box = np.array([-1.0, 1.0] * 3)
    h = 0.05
    tolerance = boundary_flux_tolerance(h, box)

    class _VPM:
        particles = _Particles()

    base = np.tile([1.0, 0.0, 0.0], (len(centres), 1))
    small = base + 0.25 * tolerance * normals
    corrected, diagnostics = evaluate_vpm_velocity(
        _VPM(),
        centres,
        normals,
        areas,
        freestream_velocity=np.array([1.0, 0.0, 0.0]),
        fvm_box=box,
        particle_spacing=h,
        evaluated_velocity=small,
    )
    assert diagnostics["raw_relative"] < diagnostics["acceptance_limit"]
    assert abs(np.dot(np.einsum("ij,ij->i", corrected, normals), areas)) < 1.0e-13

    large = base + 4.0 * tolerance * normals
    with pytest.raises(RuntimeError, match="physically significant net flux"):
        evaluate_vpm_velocity(
            _VPM(),
            centres,
            normals,
            areas,
            freestream_velocity=np.array([1.0, 0.0, 0.0]),
            fvm_box=box,
            particle_spacing=h,
            evaluated_velocity=large,
        )


def _time_coupler(vpm):
    coupler = object.__new__(FVMVPMCoupler)
    coupler._is_master = True
    coupler.vpm_solver = vpm
    coupler._velocity_bc_prev = None
    coupler._pressure_gradient_bc_prev = None
    coupler._pressure_gradient_bc_next = None
    coupler.freestream_velocity = np.array([1.0, 0.0, 0.0])
    coupler.fvm_box = np.array([-1.0, 1.0] * 3)
    coupler.setup = SimpleNamespace(boundary_condition_mode="dirichlet", vpm_particle_spacing=0.05)
    return coupler


def test_first_interval_uses_distinct_initial_and_next_boundary_states():
    class _VPM:
        particles = _Particles()
        time = 0.0

        def compute_target_velocities(self, points, **kwargs):
            return np.tile([1.0, self.time, 0.0], (len(points), 1))

    vpm = _VPM()
    coupler = _time_coupler(vpm)
    geometry = _cube_face_quadrature(nside=2)
    initialize_vpm_boundary_history(coupler, *geometry)
    initial = coupler._velocity_bc_prev.copy()
    vpm.time = 0.2
    previous, next_value, *_ = evaluate_vpm_boundary(coupler, *geometry)

    np.testing.assert_array_equal(previous, initial)
    np.testing.assert_allclose(previous[:, 1], 0.0)
    np.testing.assert_allclose(next_value[:, 1], 0.2)


def test_subcycling_linearly_interpolates_both_mixed_trace_fields(monkeypatch):
    coupler = object.__new__(FVMVPMCoupler)
    coupler.fvm_substeps = 4
    coupler.fvm_time_step_size = 0.025
    coupler.freestream_velocity = np.array([1.0, 0.0, 0.0])
    coupler.setup = SimpleNamespace(boundary_condition_mode="vorticity_mixed")
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
    centres, normals, areas = _cube_face_quadrature(nside=2)
    u_prev = np.tile([1.0, 0.0, 0.0], (len(centres), 1))
    u_next = np.tile([1.0, 0.4, 0.0], (len(centres), 1))
    normal_prev = np.einsum("ij,ij->i", u_prev, normals)
    normal_next = np.einsum("ij,ij->i", u_next, normals)
    gradient_prev = np.zeros_like(normals)
    gradient_next = np.full_like(normals, 0.8)

    advance_fvm_substeps(
        coupler,
        "numericalBoundary",
        centres,
        normals,
        areas,
        u_prev,
        u_next,
        normal_velocity_prev=normal_prev,
        normal_velocity_next=normal_next,
        tangential_gradient_prev=gradient_prev,
        tangential_gradient_next=gradient_next,
    )

    for values, alpha in zip(recorded, (0.25, 0.5, 0.75, 1.0), strict=True):
        np.testing.assert_allclose(values[0], (1.0 - alpha) * u_prev + alpha * u_next)
        np.testing.assert_allclose(values[1], (1.0 - alpha) * normal_prev + alpha * normal_next)
        np.testing.assert_allclose(values[2], alpha * gradient_next)


def test_vorticity_mixed_builds_normal_and_tangential_trace():
    jacobian = np.array([[0.2, -0.3, 0.4], [0.5, -0.1, 0.2], [-0.2, 0.6, -0.1]])
    jacobian[2, 2] = -jacobian[0, 0] - jacobian[1, 1]
    offset = np.array([1.0, -0.2, 0.1])

    class _VPM:
        particles = _Particles()

        @staticmethod
        def compute_complete_target_velocity_and_tangential_normal_gradient(
            points, normals, *, particle_spacing
        ):
            velocity = np.asarray(points) @ jacobian.T + offset
            gradient = np.tile(jacobian, (len(points), 1, 1))
            return velocity, tangential_normal_velocity_gradient(gradient, normals)

    coupler = object.__new__(FVMVPMCoupler)
    coupler._is_master = True
    coupler.vpm_solver = _VPM()
    coupler._velocity_bc_prev = None
    coupler._normal_velocity_bc_prev = None
    coupler._normal_velocity_bc_next = None
    coupler._tangential_gradient_bc_prev = None
    coupler._tangential_gradient_bc_next = None
    coupler._pressure_gradient_bc_prev = None
    coupler._pressure_gradient_bc_next = None
    coupler.freestream_velocity = offset
    coupler.fvm_box = np.array([-1.0, 1.0] * 3)
    coupler.setup = SimpleNamespace(
        boundary_condition_mode="vorticity_mixed", vpm_particle_spacing=0.04
    )
    centres, normals, areas = _cube_face_quadrature(nside=3)
    evaluate_vpm_boundary(coupler, centres, normals, areas)

    velocity = centres @ jacobian.T + offset
    np.testing.assert_allclose(
        coupler._normal_velocity_bc_next,
        np.einsum("ij,ij->i", velocity, normals),
        atol=1e-14,
    )


def test_velocity_trace_recovers_affine_curl_exactly():
    rng = np.random.default_rng(4)
    positions = rng.uniform(-1.0, 1.0, size=(20, 3))
    h = 0.08
    gradient = np.array([[0.2, -0.3, 0.4], [0.7, -0.1, 0.5], [-0.2, 0.6, 0.3]])

    def velocity(points):
        return np.asarray(points) @ gradient.T + np.array([0.4, -0.2, 0.1])

    circulation = vortex_strength_from_velocity_trace(positions, h, velocity)
    curl = np.array(
        [
            gradient[2, 1] - gradient[1, 2],
            gradient[0, 2] - gradient[2, 0],
            gradient[1, 0] - gradient[0, 1],
        ]
    )
    np.testing.assert_allclose(circulation, np.tile(curl * h**3, (len(positions), 1)))


def test_vorticity_sign_matches_the_fvm_curl_convention():
    mesh = structured_box(3, 3, 3, lx=2.0, ly=2.0, lz=2.0)
    geometry = compute_mesh_geometry(mesh)
    n_cells = mesh["n_cells"]
    n_boundary = mesh["n_faces"] - mesh["n_interior_faces"]
    velocity = np.zeros((n_cells + n_boundary, 3))
    centres = geometry["cell_centroids"][:n_cells]
    velocity[:n_cells, 0] = -centres[:, 1]
    velocity[:n_cells, 1] = centres[:, 0]
    for patch in mesh["boundary"]:
        patch["velocity_type"] = "zeroGradient"
        for local in range(patch["n_faces"]):
            face = patch["start_face"] + local
            ghost = n_cells + face - mesh["n_interior_faces"]
            point = geometry["face_centroids"][face]
            velocity[ghost, :2] = (-point[1], point[0])
    vorticity = compute_vorticity(velocity, mesh, geometry)[:n_cells]
    np.testing.assert_allclose(vorticity, np.tile([0.0, 0.0, 2.0], (n_cells, 1)), atol=5.0e-15)


def test_diagnostics_use_literal_transfer_and_boundary_names():
    result = TransferResult(
        pos=np.zeros((2, 3)),
        circ=np.ones((2, 3)),
        vol=np.ones(2),
        rad=np.ones(2),
        n_existing=3,
        n_support=5,
        correction_vortex_strength_l1=6.0,
        correction_vortex_strength_net=np.array([2.0, 2.0, 2.0]),
    )
    coupler = object.__new__(FVMVPMCoupler)
    coupler.fvm_substeps = 2
    coupler.vorticity_transfer = None
    coupler.pressure_reference = None
    coupler._last_transfer_result = result
    coupler._last_vpm_bc_flux_diagnostics = {
        "raw_mismatch": 1.0e-8,
        "raw_relative": 2.0e-9,
        "acceptance_limit": 1.0e-4,
        "applied_correction": 1.0e-9,
        "corrected_mismatch": 0.0,
    }
    diagnostics = compute_diagnostics(coupler)
    assert set(diagnostics) >= {"vpm_bc_flux", "transfer", "vortex_line_closure"}
    assert diagnostics["transfer"]["n_added"] == 2
    assert "flux_ratio" not in str(diagnostics)


def test_restart_api_remains_available_for_transfer_round_trips():
    assert callable(FVMVPMCoupler.save_state)
    assert callable(FVMVPMCoupler.load_state)
