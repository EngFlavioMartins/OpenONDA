"""Transfer-operator certification for the native FVM--VPM coupler (M9)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from source.coupler.boundary import (
    advance_fvm_substeps,
    evaluate_vpm_boundary,
    evaluate_vpm_velocity,
    project_normal_velocity,
    project_solenoidal_velocity,
    resynchronize_vpm_boundary,
    tangential_normal_velocity_gradient,
)
from source.coupler.reporting import compute_diagnostics
from source.coupler.solver import FVMVPMCoupler
from source.coupler.vorticity_transfer import (
    circulation_from_velocity_trace,
    continuous_transfer,
)
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


def test_constant_vpm_bc_is_reproduced_exactly_without_particles():
    class _Particles:
        number_of_particles = 0

    class _VPM:
        particles = _Particles()

        @staticmethod
        def compute_target_velocities(points, **kwargs):
            return np.tile([1.0, -0.25, 0.5], (len(points), 1))

    freestream_velocity = np.array([1.0, -0.25, 0.5])
    centres, normals, areas = _cube_face_quadrature()
    u_bc, diagnostics = evaluate_vpm_velocity(
        _VPM(),
        centres,
        normals,
        areas,
        freestream_velocity=freestream_velocity,
        fvm_box=np.full(6, 10.0),
    )
    np.testing.assert_allclose(u_bc, np.tile(freestream_velocity, (len(centres), 1)), atol=1e-15)
    assert max(diagnostics.values()) < 1.0e-14


def test_body_potential_is_retained_before_particle_injection():
    class _Particles:
        number_of_particles = 0

    class _VPM:
        particles = _Particles()

        @staticmethod
        def compute_target_velocities(points, **kwargs):
            return np.tile([0.9, 0.0, 0.0], (len(points), 1))

    centres, normals, areas = _cube_face_quadrature()
    u_bc, _ = evaluate_vpm_velocity(
        _VPM(),
        centres,
        normals,
        areas,
        freestream_velocity=np.array([1.0, 0.0, 0.0]),
        fvm_box=np.full(6, 10.0),
    )

    np.testing.assert_allclose(
        u_bc,
        np.tile([0.9, 0.0, 0.0], (len(centres), 1)),
        atol=1e-15,
    )


def test_blending_zone_and_vpm_bc_share_one_target_evaluation():
    class _Particles:
        number_of_particles = 12

    class _VPM:
        particles = _Particles()

        def __init__(self):
            self.calls = []

        def compute_target_velocities(self, points, **kwargs):
            self.calls.append(np.asarray(points).copy())
            return np.tile([1.0, 0.0, 0.0], (len(points), 1))

    class _BlendingZone:
        active_cell_centres = np.array([[0.0, -0.5, 0.0], [0.0, 0.5, 0.0]])

        def __init__(self):
            self.active_velocity = None

        def update_target(self, active_velocity=None):
            self.active_velocity = active_velocity

    coupler = object.__new__(FVMVPMCoupler)
    coupler._is_master = True
    coupler.vpm = _VPM()
    coupler.blending = _BlendingZone()
    coupler._u_bc_prev = None
    coupler.freestream_velocity = np.array([1.0, 0.0, 0.0])
    coupler.fvm_box = np.array([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
    coupler.setup = SimpleNamespace(vpm_bc_mode="dirichlet")

    centres, normals, areas = _cube_face_quadrature(nside=3)
    previous, u_bc, *_timings = evaluate_vpm_boundary(coupler, centres, normals, areas)

    assert len(coupler.vpm.calls) == 1
    np.testing.assert_array_equal(
        coupler.vpm.calls[0],
        np.concatenate((coupler.blending.active_cell_centres, centres), axis=0),
    )
    np.testing.assert_array_equal(
        coupler.blending.active_velocity,
        np.tile([1.0, 0.0, 0.0], (2, 1)),
    )
    np.testing.assert_allclose(previous, u_bc)


def test_pressure_vpm_bc_uses_the_same_body_complete_velocity_as_dirichlet_data():
    class _Particles:
        number_of_particles = 12

    class _VPM:
        particles = _Particles()

        def __init__(self):
            self.pressure_kwargs = None

        @staticmethod
        def compute_target_velocities(points, **kwargs):
            return np.tile([0.7, 0.0, 0.0], (len(points), 1))

        def compute_target_pressure_gradients(self, points, **kwargs):
            self.pressure_kwargs = kwargs
            result = {"grad_p": np.zeros((len(points), 3))}
            velocity = np.tile([1.0, 0.0, 0.0], (len(points), 1))
            return result, velocity

    class _BlendingZone:
        active_cell_centres = np.empty((0, 3))

        @staticmethod
        def update_target(_active_velocity=None):
            pass

    coupler = object.__new__(FVMVPMCoupler)
    coupler._is_master = True
    coupler.vpm = _VPM()
    coupler.blending = _BlendingZone()
    coupler._u_bc_prev = None
    coupler._pressure_gradient_bc_prev = None
    coupler._pressure_gradient_bc_next = None
    coupler._pressure_velocity_snapshot = None
    coupler.freestream_velocity = np.array([1.0, 0.0, 0.0])
    coupler.fvm_box = np.array([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
    coupler.vpm_time_step_size = 0.05
    coupler.rho = 1.0
    coupler.nu = 1.0e-3
    coupler.setup = SimpleNamespace(
        vpm_bc_mode="pressure_gradient",
        vpm_particle_spacing=0.04,
    )

    centres, normals, areas = _cube_face_quadrature(nside=3)
    previous, u_bc, *_timings = evaluate_vpm_boundary(coupler, centres, normals, areas)

    assert coupler.vpm.pressure_kwargs["include_body"] is True
    expected = np.tile([1.0, 0.0, 0.0], (len(centres), 1))
    np.testing.assert_allclose(previous, expected)
    np.testing.assert_allclose(u_bc, expected)


def test_vorticity_mixed_transfer_builds_normal_and_tangential_gradient_trace():
    jacobian = np.array(
        [[0.2, -0.3, 0.4], [0.5, -0.1, 0.2], [-0.2, 0.6, -0.1]]
    )  # trace-free, so the linear field is flux-compatible on the closed box
    body_jacobian = np.array(
        [[-0.03, 0.04, 0.02], [0.01, 0.02, -0.05], [0.02, 0.01, 0.01]]
    )  # trace-free body correction
    total_jacobian = jacobian + body_jacobian
    offset = np.array([1.0, -0.2, 0.1])

    class _Particles:
        number_of_particles = 4

    class _VPM:
        particles = _Particles()

        @staticmethod
        def _body_induced_fn(points):
            return np.asarray(points) @ body_jacobian.T

        @staticmethod
        def compute_target_velocities(points, **kwargs):
            return np.asarray(points) @ total_jacobian.T + offset

        @staticmethod
        def compute_complete_target_velocity_gradients(points, *, particle_spacing):
            return np.tile(total_jacobian, (len(points), 1, 1))

        @staticmethod
        def compute_complete_target_velocity_and_gradients(points, *, particle_spacing):
            return (
                _VPM.compute_target_velocities(points),
                _VPM.compute_complete_target_velocity_gradients(points, particle_spacing=0.04),
            )

        @staticmethod
        def compute_complete_target_velocity_and_tangential_normal_gradient(
            points, normals, *, particle_spacing
        ):
            gradient = _VPM.compute_complete_target_velocity_gradients(
                points, particle_spacing=0.04
            )
            return (
                _VPM.compute_target_velocities(points),
                tangential_normal_velocity_gradient(gradient, normals),
            )

    class _BlendingZone:
        active_cell_centres = np.empty((0, 3))

        @staticmethod
        def update_target(_active_velocity=None):
            pass

    coupler = object.__new__(FVMVPMCoupler)
    coupler._is_master = True
    coupler.vpm = _VPM()
    coupler.blending = _BlendingZone()
    coupler._u_bc_prev = None
    coupler._normal_velocity_bc_prev = None
    coupler._normal_velocity_bc_next = None
    coupler._tangential_gradient_bc_prev = None
    coupler._tangential_gradient_bc_next = None
    coupler._pressure_gradient_bc_prev = None
    coupler._pressure_gradient_bc_next = None
    coupler.freestream_velocity = offset
    coupler.fvm_box = np.array([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
    coupler.setup = SimpleNamespace(vpm_bc_mode="vorticity_mixed", vpm_particle_spacing=0.04)

    centres, normals, areas = _cube_face_quadrature(nside=3)
    evaluate_vpm_boundary(coupler, centres, normals, areas)

    velocity = _VPM.compute_target_velocities(centres)
    expected_normal = np.einsum("ij,ij->i", velocity, normals)
    d_u_dn = np.einsum("ij,fj->fi", total_jacobian, normals)
    expected_tangent = d_u_dn - np.einsum("fi,fi->f", d_u_dn, normals)[:, None] * normals
    np.testing.assert_allclose(coupler._normal_velocity_bc_next, expected_normal, atol=1e-14)
    np.testing.assert_allclose(coupler._tangential_gradient_bc_next, expected_tangent)
    np.testing.assert_array_equal(
        coupler._normal_velocity_bc_prev, coupler._normal_velocity_bc_next
    )
    np.testing.assert_array_equal(
        coupler._tangential_gradient_bc_prev, coupler._tangential_gradient_bc_next
    )


def test_post_transfer_resync_refreshes_velocity_and_pressure_snapshots_together():
    class _VPM:
        @staticmethod
        def compute_target_velocities(points, **kwargs):
            return np.column_stack(
                (1.0 + 0.1 * points[:, 0], 0.05 * points[:, 1], 0.05 * points[:, 2])
            )

    class _BlendingZone:
        active_cell_centres = np.array([[0.0, 0.0, 0.0]])

        def __init__(self):
            self.endpoint = None

        def update_endpoint(self, values=None):
            self.endpoint = values

    coupler = object.__new__(FVMVPMCoupler)
    coupler._is_master = True
    coupler.vpm = _VPM()
    coupler.blending = _BlendingZone()
    coupler.setup = SimpleNamespace(
        bc_resync_after_transfer=True,
        vpm_bc_mode="pressure_gradient",
    )
    coupler.freestream_velocity = np.array([1.0, 0.0, 0.0])
    centres, normals, areas = _cube_face_quadrature(nside=2)
    coupler._u_bc_prev = np.zeros_like(centres)
    coupler._pressure_velocity_snapshot = np.zeros_like(centres)

    resynchronize_vpm_boundary(coupler, centres, normals, areas)

    expected = _VPM.compute_target_velocities(centres)
    np.testing.assert_allclose(coupler._u_bc_prev, expected)
    np.testing.assert_allclose(coupler._pressure_velocity_snapshot, expected)
    np.testing.assert_allclose(coupler.blending.endpoint, [[1.0, 0.0, 0.0]])


def test_post_transfer_resync_refreshes_both_mixed_trace_fields():
    jacobian = np.array([[0.1, -0.2, 0.3], [0.4, -0.05, 0.2], [-0.1, 0.5, -0.05]])

    class _VPM:
        @staticmethod
        def compute_target_velocities(points, **kwargs):
            return np.asarray(points) @ jacobian.T + np.array([1.0, 0.0, 0.0])

        @staticmethod
        def compute_complete_target_velocity_gradients(points, *, particle_spacing):
            return np.tile(jacobian, (len(points), 1, 1))

        @staticmethod
        def compute_complete_target_velocity_and_gradients(points, *, particle_spacing):
            return (
                _VPM.compute_target_velocities(points),
                _VPM.compute_complete_target_velocity_gradients(points, particle_spacing=0.04),
            )

        @staticmethod
        def compute_complete_target_velocity_and_tangential_normal_gradient(
            points, normals, *, particle_spacing
        ):
            gradient = _VPM.compute_complete_target_velocity_gradients(
                points, particle_spacing=0.04
            )
            return (
                _VPM.compute_target_velocities(points),
                tangential_normal_velocity_gradient(gradient, normals),
            )

    class _BlendingZone:
        active_cell_centres = np.empty((0, 3))

        @staticmethod
        def update_endpoint(_values=None):
            pass

    coupler = object.__new__(FVMVPMCoupler)
    coupler._is_master = True
    coupler.vpm = _VPM()
    coupler.blending = _BlendingZone()
    coupler.setup = SimpleNamespace(
        bc_resync_after_transfer=True,
        vpm_bc_mode="vorticity_mixed",
        vpm_particle_spacing=0.04,
    )
    coupler.freestream_velocity = np.array([1.0, 0.0, 0.0])
    centres, normals, areas = _cube_face_quadrature(nside=2)
    coupler._u_bc_prev = np.zeros_like(centres)
    coupler._normal_velocity_bc_prev = np.zeros(len(centres))
    coupler._tangential_gradient_bc_prev = np.zeros_like(centres)

    resynchronize_vpm_boundary(coupler, centres, normals, areas)

    expected_velocity = project_solenoidal_velocity(
        _VPM.compute_target_velocities(centres), normals, areas
    )
    expected_normal = np.einsum("ij,ij->i", expected_velocity, normals)
    expected_tangent = tangential_normal_velocity_gradient(
        _VPM.compute_complete_target_velocity_gradients(centres, particle_spacing=0.04), normals
    )
    np.testing.assert_allclose(coupler._u_bc_prev, expected_velocity)
    np.testing.assert_allclose(coupler._normal_velocity_bc_prev, expected_normal)
    np.testing.assert_allclose(coupler._tangential_gradient_bc_prev, expected_tangent)


def test_zero_target_evaluation_fails_before_blending_zone_mutation():
    class _Particles:
        number_of_particles = 12

    class _VPM:
        particles = _Particles()

        @staticmethod
        def compute_target_velocities(points, **kwargs):
            return np.zeros((len(points), 3))

    class _BlendingZone:
        active_cell_centres = np.array([[0.0, 0.0, 0.0]])

        def __init__(self):
            self.was_updated = False

        def update_target(self, active_velocity=None):
            self.was_updated = True

    coupler = object.__new__(FVMVPMCoupler)
    coupler._is_master = True
    coupler.vpm = _VPM()
    coupler.blending = _BlendingZone()
    coupler._u_bc_prev = None
    coupler.freestream_velocity = np.array([1.0, 0.0, 0.0])
    coupler.setup = SimpleNamespace(vpm_bc_mode="dirichlet")

    centres, normals, areas = _cube_face_quadrature(nside=2)
    with pytest.raises(RuntimeError, match="identically zero field"):
        evaluate_vpm_boundary(coupler, centres, normals, areas)
    assert not coupler.blending.was_updated


def test_velocity_trace_recovers_linear_field_curl_exactly():
    rng = np.random.default_rng(4)
    positions = rng.uniform(-1.0, 1.0, size=(20, 3))
    h = 0.08
    gradient = np.array(
        [
            [0.2, -0.3, 0.4],
            [0.7, -0.1, 0.5],
            [-0.2, 0.6, 0.3],
        ]
    )

    def velocity(points):
        return np.asarray(points) @ gradient + np.array([0.4, -0.2, 0.1])

    circulation = circulation_from_velocity_trace(positions, h, velocity)
    curl = np.array(
        [
            gradient[1, 2] - gradient[2, 1],
            gradient[2, 0] - gradient[0, 2],
            gradient[0, 1] - gradient[1, 0],
        ]
    )
    np.testing.assert_allclose(circulation, np.tile(curl * h**3, (len(positions), 1)))


def test_velocity_trace_uses_the_no_slip_body_face():
    h = 0.05
    position = np.array([[0.5 + 0.5 * h, 0.0, 0.0]])

    def velocity(points):
        points = np.asarray(points)
        result = np.zeros_like(points)
        fluid = points[:, 0] > 0.5
        result[fluid, 1] = points[fluid, 0] - 0.5
        return result

    circulation = circulation_from_velocity_trace(position, h, velocity)
    np.testing.assert_allclose(circulation, [[0.0, 0.0, h**3]])


def test_direct_circulation_target_bypasses_cell_remeshing():
    box = np.array([-0.5, 0.5, -0.5, 0.5, -0.5, 0.5])
    h = 0.1
    target = np.array([0.0, 0.0, 2.0 * h**3])
    result = continuous_transfer(
        np.zeros((0, 3)),
        np.zeros((0, 3)),
        box,
        h,
        circulation_at_node=lambda points: np.tile(target, (len(points), 1)),
        mesh_weight_at_node=lambda points: np.ones(len(points)),
        overlap_zone_ramp_width=h,
        transfer_buffer_length=h,
        transfer_prune_threshold_abs=0.0,
        lattice_anchor=np.array([-0.45, -0.45, -0.45]),
    )
    core = np.all(np.abs(result.pos) < 0.3, axis=1)
    relative = np.linalg.norm(result.circ[core] - target, axis=1) / np.linalg.norm(target)
    assert relative.max() < 0.12


def test_vorticity_sign_matches_the_fvm_curl_convention():
    mesh = structured_box(3, 3, 3, lx=2.0, ly=2.0, lz=2.0)
    geometry = compute_mesh_geometry(mesh)
    n_cells = mesh["n_elements"]
    n_boundary = mesh["n_faces"] - mesh["n_interior_faces"]
    U = np.zeros((n_cells + n_boundary, 3))

    centres = geometry["element_centroids"][:n_cells]
    U[:n_cells, 0] = -centres[:, 1]  # U=(-y, x, 0), curl = +2 z
    U[:n_cells, 1] = centres[:, 0]
    for patch in mesh["boundary"]:
        start, count = patch["startFace"], patch["nFaces"]
        for local in range(count):
            face = start + local
            ghost = n_cells + face - mesh["n_interior_faces"]
            point = geometry["face_centroids"][face]
            U[ghost, :2] = (-point[1], point[0])
            patch["bc_type_velocity"] = "zeroGradient"

    vorticity = compute_vorticity(U, mesh, geometry)[:n_cells]
    np.testing.assert_allclose(vorticity, np.tile([0.0, 0.0, 2.0], (n_cells, 1)), atol=1e-12)


def test_projection_is_flux_free_and_subcycle_ratio_is_strict():
    centres, normals, areas = _cube_face_quadrature(nside=8)
    rng = np.random.default_rng(7)
    raw = rng.normal(size=centres.shape)
    projected = project_solenoidal_velocity(raw, normals, areas)
    flux = float(np.dot(np.einsum("ij,ij->i", projected, normals), areas))
    assert abs(flux) < 1e-12 * float(np.sum(areas))

    assert FVMVPMCoupler._derive_fvm_substeps(0.15, 0.05) == 3
    assert FVMVPMCoupler._derive_coupling_step_count(0.15, 0.05) == 3
    with pytest.raises(ValueError, match="integer multiple"):
        FVMVPMCoupler._derive_fvm_substeps(0.14, 0.05)
    assert FVMVPMCoupler._derive_coupling_step_count(0.14, 0.05) == 3  # round(2.8)


def test_vorticity_mixed_subcycling_interpolates_and_reprojects_both_fields(
    monkeypatch,
):
    class _Blending:
        @staticmethod
        def push_target(_alpha):
            pass

    coupler = object.__new__(FVMVPMCoupler)
    coupler.fvm_substeps = 2
    coupler.fvm_time_step_size = 0.05
    coupler.freestream_velocity = np.array([1.0, 0.0, 0.0])
    coupler.setup = SimpleNamespace(vpm_bc_mode="vorticity_mixed")
    coupler.blending = _Blending()
    recorded = []

    def record_step(
        _coupler,
        patch,
        velocity,
        pressure_gradient=None,
        normal_velocity=None,
        tangential_gradient=None,
    ):
        recorded.append((normal_velocity.copy(), tangential_gradient.copy()))

    monkeypatch.setattr("source.coupler.boundary.apply_fvm_boundary", record_step)
    centres, normals, areas = _cube_face_quadrature(nside=2)
    u_prev = np.tile([1.0, 0.0, 0.0], (len(centres), 1))
    u_next = u_prev.copy()
    normal_prev = normals @ np.array([1.0, 0.0, 0.0])
    normal_next = normal_prev + np.linspace(-0.2, 0.3, len(normal_prev))
    gradient_prev = np.zeros_like(normals)
    gradient_next = np.arange(normals.size, dtype=float).reshape(-1, 3) / normals.size

    advance_fvm_substeps(
        coupler,
        "cut",
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

    assert len(recorded) == 2
    for index, alpha in enumerate((0.5, 1.0)):
        expected_normal = project_normal_velocity(
            (1.0 - alpha) * normal_prev + alpha * normal_next, areas
        )
        np.testing.assert_allclose(recorded[index][0], expected_normal)
        np.testing.assert_allclose(recorded[index][1], alpha * gradient_next)
        assert abs(np.dot(recorded[index][0], areas)) < 1.0e-14


def test_correction_diagnostics_expose_raw_applied_and_corrected_mismatch():
    box = np.array([-0.5, 0.5, -0.5, 0.5, -0.5, 0.5])
    h = 0.1
    rng = np.random.default_rng(2)
    positions = rng.uniform(-0.2, 0.2, size=(20, 3))
    circulation = np.zeros((20, 3))
    circulation[:, 2] = np.linspace(1.0e-4, 2.0e-3, 20)
    result = continuous_transfer(
        positions,
        circulation,
        box,
        h,
        circulation_at_node=lambda points: np.zeros((len(points), 3)),
        transfer_buffer_length=0.1,
        transfer_prune_threshold_abs=5.0e-4,
    )
    coupler = object.__new__(FVMVPMCoupler)
    coupler.fvm_substeps = 3
    coupler.transfer = None
    coupler.pressure_reference = None
    coupler._last_transfer_result = result
    coupler._last_vpm_bc_flux_diagnostics = {
        "raw_mismatch": 0.2,
        "applied_correction": 0.05,
        "corrected_mismatch": 1.0e-12,
    }
    diagnostics = compute_diagnostics(coupler)

    assert diagnostics["n_fvm_substeps"] == 3
    for section in ("conservation", "vpm_bc_flux", "transfer"):
        assert section in diagnostics
    for values in diagnostics["conservation"].values():
        assert set(values) == {"circulation", "linear_impulse", "angular_impulse"}
        assert np.isfinite(list(values.values())).all()
    assert np.isfinite(list(diagnostics["vpm_bc_flux"].values())).all()
    assert np.isfinite(list(diagnostics["transfer"].values())).all()
    assert diagnostics["transfer"]["n_pruned"] == result.n_pruned
    assert diagnostics["transfer"]["cfl"] == result.cfl


def test_deferred_transfer_diagnostics_are_marked_unmeasured():
    result = continuous_transfer(
        np.zeros((0, 3)),
        np.zeros((0, 3)),
        np.array([-0.5, 0.5, -0.5, 0.5, -0.5, 0.5]),
        0.1,
        circulation_at_node=lambda points: np.zeros((len(points), 3)),
        compute_diagnostics=False,
    )
    coupler = object.__new__(FVMVPMCoupler)
    coupler.fvm_substeps = 1
    coupler.transfer = None
    coupler.pressure_reference = None
    coupler._last_transfer_result = result
    coupler._last_vpm_bc_flux_diagnostics = {
        "raw_mismatch": 0.0,
        "applied_correction": 0.0,
        "corrected_mismatch": 0.0,
    }

    diagnostics = compute_diagnostics(coupler)
    transfer = diagnostics["transfer"]
    assert transfer["diagnostics_evaluated"] is False
    for name in (
        "flux_ratio",
        "transfer_in_band_residual",
        "transfer_pre_prune_residual",
        "transfer_out_of_band_fraction",
        "transfer_max_amplification",
    ):
        assert transfer[name] is None
    assert diagnostics["spectral_band_ratio"] is None
    assert "null" in json.dumps(diagnostics)


def test_restart_api_remains_available_for_transfer_round_trips():
    """The end-to-end round trip is exercised by test_fvm_vpm_smoke; retain
    this explicit API gate beside the other transfer-operator tests."""
    assert callable(FVMVPMCoupler.save_state)
    assert callable(FVMVPMCoupler.load_state)
