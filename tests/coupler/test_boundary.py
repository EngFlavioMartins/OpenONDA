"""Boundary-trace flux acceptance tests."""

from types import SimpleNamespace

import numpy as np
import pytest

from source.coupler.boundary import boundary_flux_tolerance, evaluate_vpm_velocity


def test_boundary_flux_tolerance_follows_trace_resolution() -> None:
    box = np.array([-1.5, 1.5, -1.5, 1.5, -0.475, 0.475])

    assert boundary_flux_tolerance(0.05, box) == pytest.approx((0.05 / 0.95) ** 2)
    assert boundary_flux_tolerance(0.2, box) == pytest.approx(1.0e-2)


def test_boundary_flux_projection_accepts_resolution_scale_residual() -> None:
    face_centre = np.array([[-1.5, 0.0, 0.0], [1.5, 0.0, 0.0]])
    face_normal = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    face_area = np.ones(2)
    velocity = np.array([[1.0, 0.0, 0.0], [1.003, 0.0, 0.0]])
    vpm = SimpleNamespace(particles=SimpleNamespace(n_particles_total=2))

    corrected, diagnostics = evaluate_vpm_velocity(
        vpm,
        face_centre,
        face_normal,
        face_area,
        freestream_velocity=np.array([1.0, 0.0, 0.0]),
        fvm_box=np.array([-1.5, 1.5, -1.5, 1.5, -0.475, 0.475]),
        particle_spacing=0.05,
        evaluated_velocity=velocity,
    )

    assert diagnostics["raw_relative"] < diagnostics["acceptance_limit"]
    np.testing.assert_allclose(
        np.dot(np.einsum("ij,ij->i", corrected, face_normal), face_area), 0.0, atol=1.0e-14
    )


def test_boundary_flux_above_hard_ceiling_is_rejected() -> None:
    face_centre = np.array([[-1.5, 0.0, 0.0], [1.5, 0.0, 0.0]])
    face_normal = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    face_area = np.ones(2)
    velocity = np.array([[1.0, 0.0, 0.0], [1.021, 0.0, 0.0]])
    vpm = SimpleNamespace(particles=SimpleNamespace(n_particles_total=2))

    with pytest.raises(RuntimeError, match="physically significant net flux"):
        evaluate_vpm_velocity(
            vpm,
            face_centre,
            face_normal,
            face_area,
            freestream_velocity=np.array([1.0, 0.0, 0.0]),
            fvm_box=np.array([-1.5, 1.5, -1.5, 1.5, -0.475, 0.475]),
            particle_spacing=0.05,
            evaluated_velocity=velocity,
        )
