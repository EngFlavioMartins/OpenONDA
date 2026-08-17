"""Transfer-operator certification for the native FVM--VPM coupler (M9)."""

from __future__ import annotations

from types import SimpleNamespace

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

        @staticmethod
        def compute_target_velocities(points, **kwargs):
            return np.tile([1.0, -0.25, 0.5], (len(points), 1))

    coupler = object.__new__(FVMVPMCoupler)
    coupler.vpm = _VPM()
    coupler.u_inf = np.array([1.0, -0.25, 0.5])
    coupler._log_outflow_deficit = lambda *_: None

    centres, normals, areas = _cube_face_quadrature()
    donor = coupler._donor_velocity(centres, normals, areas)
    np.testing.assert_allclose(donor, np.tile(coupler.u_inf, (len(centres), 1)), atol=1e-15)
    assert max(coupler._last_donor_flux_diagnostics.values()) < 1.0e-14


def test_body_potential_is_retained_before_particle_injection():
    class _Particles:
        number_of_particles = 0

    class _VPM:
        particles = _Particles()
        _body_induced_fn = object()
        num_sources = 0

        @staticmethod
        def compute_target_velocities(points, **kwargs):
            return np.tile([0.9, 0.0, 0.0], (len(points), 1))

    coupler = object.__new__(FVMVPMCoupler)
    coupler.vpm = _VPM()
    coupler.u_inf = np.array([1.0, 0.0, 0.0])
    coupler._log_outflow_deficit = lambda *_: None

    centres, normals, areas = _cube_face_quadrature()
    donor = coupler._donor_velocity(centres, normals, areas)

    np.testing.assert_allclose(
        donor,
        np.tile([0.9, 0.0, 0.0], (len(centres), 1)),
        atol=1e-15,
    )


def test_fringe_and_donor_share_one_target_evaluation():
    class _Particles:
        number_of_particles = 12

    class _VPM:
        particles = _Particles()

        def __init__(self):
            self.calls = []

        def compute_target_velocities(self, points, **kwargs):
            self.calls.append(np.asarray(points).copy())
            return np.tile([1.0, 0.0, 0.0], (len(points), 1))

    class _Fringe:
        active_cell_centres = np.array([[0.0, -0.5, 0.0], [0.0, 0.5, 0.0]])

        def __init__(self):
            self.active_velocity = None

        def update_target(self, active_velocity=None):
            self.active_velocity = active_velocity

    coupler = object.__new__(FVMVPMCoupler)
    coupler._is_master = True
    coupler.vpm = _VPM()
    coupler.fringe = _Fringe()
    coupler._u_bc_prev = None
    coupler.u_inf = np.array([1.0, 0.0, 0.0])
    coupler.config = SimpleNamespace(donor_boundary_mode="dirichlet")
    coupler._log_outflow_deficit = lambda *_: None

    centres, normals, areas = _cube_face_quadrature(nside=3)
    previous, donor, *_timings = coupler._transfer_vpm_to_fvm(centres, normals, areas)

    assert len(coupler.vpm.calls) == 1
    np.testing.assert_array_equal(
        coupler.vpm.calls[0],
        np.concatenate((coupler.fringe.active_cell_centres, centres), axis=0),
    )
    np.testing.assert_array_equal(
        coupler.fringe.active_velocity,
        np.tile([1.0, 0.0, 0.0], (2, 1)),
    )
    np.testing.assert_allclose(previous, donor)


def test_zero_target_evaluation_fails_before_fringe_mutation():
    class _Particles:
        number_of_particles = 12

    class _VPM:
        particles = _Particles()

        @staticmethod
        def compute_target_velocities(points, **kwargs):
            return np.zeros((len(points), 3))

    class _Fringe:
        active_cell_centres = np.array([[0.0, 0.0, 0.0]])

        def __init__(self):
            self.was_updated = False

        def update_target(self, active_velocity=None):
            self.was_updated = True

    coupler = object.__new__(FVMVPMCoupler)
    coupler._is_master = True
    coupler.vpm = _VPM()
    coupler.fringe = _Fringe()
    coupler._u_bc_prev = None
    coupler.u_inf = np.array([1.0, 0.0, 0.0])

    centres, normals, areas = _cube_face_quadrature(nside=2)
    with pytest.raises(RuntimeError, match="identically zero field"):
        coupler._transfer_vpm_to_fvm(centres, normals, areas)
    assert not coupler.fringe.was_updated


def test_particle_fingerprint_detects_read_only_phase_mutation():
    class _Particles:
        number_of_particles = 2

    class _VPM:
        particles = _Particles()
        particles_positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        particles_circulation = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
        particles_radii = np.ones(2)
        particles_volumes = np.ones(2)

    coupler = object.__new__(FVMVPMCoupler)
    coupler._is_master = True
    coupler.vpm = _VPM()
    expected = coupler._vpm_particle_fingerprint()

    coupler._assert_vpm_particle_fingerprint(expected, "test phase")
    coupler.vpm.particles_circulation[1, 2] = -2.0
    with pytest.raises(RuntimeError, match="changed during read-only test phase"):
        coupler._assert_vpm_particle_fingerprint(expected, "test phase")


def test_particle_validation_rejects_zeroed_backend_fields():
    class _Particles:
        number_of_particles = 2

    class _VPM:
        particles = _Particles()
        particles_positions = np.zeros((2, 3))
        particles_circulation = np.zeros((2, 3))
        particles_radii = np.zeros(2)
        particles_volumes = np.zeros(2)

    coupler = object.__new__(FVMVPMCoupler)
    coupler._is_master = True
    coupler.vpm = _VPM()
    with pytest.raises(RuntimeError, match="radii and volumes"):
        coupler._vpm_particle_fingerprint(validate=True)


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
        mesh_weight_at_node=lambda points: np.ones(len(points)),
        ramp_width=h,
        buffer_length=h,
        threshold_abs=0.0,
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
        circulation_at_node=lambda points: np.zeros((len(points), 3)),
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
    for section in ("conservation", "donor_flux", "handoff"):
        assert section in diagnostics
    for values in diagnostics["conservation"].values():
        assert set(values) == {"circulation", "linear_impulse", "angular_impulse"}
        assert np.isfinite(list(values.values())).all()
    assert np.isfinite(list(diagnostics["donor_flux"].values())).all()
    assert np.isfinite(list(diagnostics["handoff"].values())).all()
    assert diagnostics["handoff"]["n_pruned"] == result.n_pruned
    assert diagnostics["handoff"]["cfl"] == result.cfl


def test_restart_api_remains_available_for_transfer_round_trips():
    """The end-to-end round trip is exercised by test_fvm_vpm_smoke; retain
    this explicit API gate beside the other transfer-operator tests."""
    assert callable(FVMVPMCoupler.save_state)
    assert callable(FVMVPMCoupler.load_state)
