"""Transfer-operator certification for the native FVM--VPM coupler (M9)."""

from __future__ import annotations

import numpy as np
import pytest

from source.coupler.core.helpers.continuous_overlap import (
    circulation_from_velocity_trace,
    continuous_handoff,
)
from source.coupler.core.solver import FVMVPMCoupler
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


def test_constant_donor_is_reproduced_exactly_without_particles():
    class _Particles:
        number_of_particles = 0

    class _VPM:
        particles = _Particles()

    coupler = object.__new__(FVMVPMCoupler)
    coupler.vpm = _VPM()
    coupler.u_inf = np.array([1.0, -0.25, 0.5])
    coupler._last_omega_donor = None

    centres, normals, areas = _cube_face_quadrature()
    donor = coupler._donor_velocity(centres, normals, areas)
    np.testing.assert_array_equal(donor, np.tile(coupler.u_inf, (len(centres), 1)))
    assert coupler._last_omega_donor.shape == donor.shape
    assert coupler._last_donor_flux_diagnostics == {
        "raw_mismatch": 0.0,
        "applied_correction": 0.0,
        "corrected_mismatch": 0.0,
    }


def test_body_potential_is_retained_before_particle_injection():
    class _Particles:
        number_of_particles = 0

    class _VPM:
        particles = _Particles()
        _body_induced_fn = object()
        num_sources = 0

        @staticmethod
        def compute_target_velocities(points, include_freestream=True):
            return np.tile([0.9, 0.0, 0.0], (len(points), 1))

    coupler = object.__new__(FVMVPMCoupler)
    coupler.vpm = _VPM()
    coupler.u_inf = np.array([1.0, 0.0, 0.0])
    coupler._last_omega_donor = None
    coupler._log_outflow_deficit = lambda *_: None

    centres, normals, areas = _cube_face_quadrature()
    donor = coupler._donor_velocity(centres, normals, areas)

    np.testing.assert_allclose(
        donor,
        np.tile([0.9, 0.0, 0.0], (len(centres), 1)),
        atol=1e-15,
    )


def test_normal_panel_scope_changes_only_donor_normal_velocity():
    class _Particles:
        number_of_particles = 0

    class _Panel:
        coupling_scope = "normal"

        @staticmethod
        def compute_induced_velocity(points):
            return np.tile([0.2, 0.3, 0.4], (len(points), 1))

    class _VPM:
        particles = _Particles()
        panel_solver = _Panel()
        _body_induced_fn = None
        num_sources = 0

        @staticmethod
        def compute_target_velocities(points, include_freestream=True):
            return np.tile([1.0, 0.0, 0.0], (len(points), 1))

    coupler = object.__new__(FVMVPMCoupler)
    coupler.vpm = _VPM()
    coupler.u_inf = np.array([1.0, 0.0, 0.0])
    coupler._last_omega_donor = None
    coupler._log_outflow_deficit = lambda *_: None

    centres, normals, areas = _cube_face_quadrature()
    donor = coupler._donor_velocity(centres, normals, areas)
    correction = donor - coupler.u_inf

    expected = np.sum(np.array([0.2, 0.3, 0.4]) * normals, axis=1)[:, None] * normals
    np.testing.assert_allclose(correction, expected, atol=1e-15)


def test_linear_fvm_vorticity_field_is_reproduced_on_the_injection_lattice():
    """The FVM-to-particle path preserves a linear manufactured field where
    eta is one; this catches a transposed component or lattice-phase error."""
    box = np.array([-0.5, 0.5, -0.5, 0.5, -0.5, 0.5])
    h = 0.1

    def omega(points):
        return np.column_stack(
            [1.0 + 0.2 * points[:, 0], -0.1 + 0.1 * points[:, 1], 0.3 * points[:, 2]]
        )

    result = continuous_handoff(
        np.zeros((0, 3)),
        np.zeros((0, 3)),
        box,
        h,
        omega_at_node=omega,
        inside_mesh_at_node=lambda points: np.ones(len(points), dtype=bool),
        ramp_width=0.1,
        dead_zone=0.0,
        buffer_length=0.1,
        threshold_abs=0.0,
    )
    core = np.all(np.abs(result.pos) < 0.3, axis=1)
    np.testing.assert_allclose(result.circ[core] / h**3, omega(result.pos[core]), atol=1e-14)


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
    result = continuous_handoff(
        np.zeros((0, 3)),
        np.zeros((0, 3)),
        box,
        h,
        circulation_at_node=lambda points: np.tile(target, (len(points), 1)),
        inside_mesh_at_node=lambda points: np.ones(len(points), dtype=bool),
        ramp_width=h,
        buffer_length=h,
        threshold_abs=0.0,
        lattice_anchor=np.array([-0.45, -0.45, -0.45]),
    )
    core = np.all(np.abs(result.pos) < 0.3, axis=1)
    np.testing.assert_allclose(result.circ[core], np.tile(target, (core.sum(), 1)))


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
            patch["bc_type_U"] = "zeroGradient"

    vorticity = compute_vorticity(U, mesh, geometry)[:n_cells]
    np.testing.assert_allclose(vorticity, np.tile([0.0, 0.0, 2.0], (n_cells, 1)), atol=1e-12)


def test_projection_is_flux_free_and_subcycle_ratio_is_strict():
    centres, normals, areas = _cube_face_quadrature(nside=8)
    rng = np.random.default_rng(7)
    raw = rng.normal(size=centres.shape)
    projected = FVMVPMCoupler._project_to_solenoidal(raw, normals, areas)
    flux = float(np.dot(np.einsum("ij,ij->i", projected, normals), areas))
    assert abs(flux) < 1e-12 * float(np.sum(areas))

    assert FVMVPMCoupler._derive_period_multiplier(0.15, 0.05) == 3
    assert FVMVPMCoupler._derive_coupling_step_count(0.15, 0.05) == 3
    with pytest.raises(ValueError, match="integer multiple"):
        FVMVPMCoupler._derive_period_multiplier(0.14, 0.05)
    assert FVMVPMCoupler._derive_coupling_step_count(0.14, 0.05) == 3  # round(2.8)


def test_correction_diagnostics_expose_raw_applied_and_corrected_mismatch():
    box = np.array([-0.5, 0.5, -0.5, 0.5, -0.5, 0.5])
    h = 0.1
    rng = np.random.default_rng(2)
    positions = rng.uniform(-0.2, 0.2, size=(20, 3))
    circulation = np.zeros((20, 3))
    circulation[:, 2] = np.linspace(1.0e-4, 2.0e-3, 20)
    result = continuous_handoff(
        positions,
        circulation,
        box,
        h,
        buffer_length=0.1,
        threshold_abs=5.0e-4,
    )
    coupler = object.__new__(FVMVPMCoupler)
    coupler.period_multiplier = 3
    coupler._last_handoff_result = result
    coupler._last_donor_flux_diagnostics = {
        "raw_mismatch": 0.2,
        "applied_correction": 0.05,
        "corrected_mismatch": 1.0e-12,
    }
    diagnostics = coupler.compute_diagnostics()

    assert diagnostics["period_multiplier"] == 3
    for section in ("conservation", "donor_flux"):
        assert section in diagnostics
    for values in diagnostics["conservation"].values():
        assert set(values) == {"circulation", "linear_impulse", "angular_impulse"}
        assert np.isfinite(list(values.values())).all()
    assert np.isfinite(list(diagnostics["donor_flux"].values())).all()


def test_restart_api_remains_available_for_transfer_round_trips():
    """The end-to-end round trip is exercised by test_fvm_vpm_smoke; retain
    this explicit API gate beside the other transfer-operator tests."""
    assert callable(FVMVPMCoupler.save_state)
    assert callable(FVMVPMCoupler.load_state)
