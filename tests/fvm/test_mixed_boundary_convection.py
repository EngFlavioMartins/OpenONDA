"""Convection must retain the cell dependence of a directional mixed trace."""

import numpy as np
import pytest

from source.solvers.fvm.assemble.convection import assemble_convection_term_boundary
from source.solvers.fvm.fields.mixed_velocity_boundary import (
    reconstruct_normal_velocity_tangential_gradient,
)


def assembly(owner, normal, distance, un, gt, flux, component):
    faces = len(owner)
    boundary = reconstruct_normal_velocity_tangential_gradient(owner, normal, distance, un, gt)
    field = np.concatenate((owner[:, component], boundary[:, component]))
    return assemble_convection_term_boundary(
        field,
        flux,
        {"start_face": 0, "n_faces": faces, "velocity_type": "normalValueTangentialGradient"},
        {"n_cells": faces, "n_interior_faces": 0, "owners": np.arange(faces)},
        {"face_area_vector": normal},
        component=component,
    )


@pytest.mark.parametrize("include_axis_faces", [False, True])
def test_matrix_flux_matches_the_boundary_condition_after_cell_velocity_changes(include_axis_faces):
    rng = np.random.default_rng(612)
    count = 31
    normal = rng.normal(size=(count, 3))
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    if include_axis_faces:
        normal[:6] = np.concatenate((np.eye(3), -np.eye(3)))
    owner = rng.normal(size=(count, 3))
    distance = rng.uniform(0.01, 0.2, count)
    un = rng.normal(size=count)
    gt = rng.normal(size=(count, 3))
    gt -= np.einsum("ij,ij->i", gt, normal)[:, None] * normal
    flux = un * rng.uniform(0.01, 0.1, count)
    assert np.any(flux > 0) and np.any(flux < 0)

    for component in range(3):
        terms = assembly(owner, normal, distance, un, gt, flux, component)
        changed = owner.copy()
        # A finite change, not a differential perturbation of the implementation.
        changed[:, component] += rng.normal(size=count)
        physical_face = reconstruct_normal_velocity_tangential_gradient(
            changed, normal, distance, un, gt
        )
        predicted_flux = terms["flux_cf"] * changed[:, component] + terms["flux_vf"]
        np.testing.assert_allclose(
            predicted_flux, flux * physical_face[:, component], rtol=2e-14, atol=2e-16
        )
        original_face = reconstruct_normal_velocity_tangential_gradient(
            owner, normal, distance, un, gt
        )
        np.testing.assert_allclose(
            terms["flux_tf"], flux * original_face[:, component], rtol=2e-14, atol=2e-16
        )


def test_implicit_tangential_outflow_obeys_backward_euler_momentum_balance():
    # A full vector in a 3D control volume, with an x-normal mixed outflow.
    # Tangential flux is F*u_t. Prescribed inflow/source enters through rhs.
    owner = np.array([[1.0, 2.0, -3.0]])
    normal = np.array([[1.0, 0.0, 0.0]])
    distance, un, gt, flux = np.array([0.5]), np.array([1.0]), np.zeros((1, 3)), np.array([1.0])
    volume, dt = 1.0, 0.75
    for component in (1, 2):
        terms = assembly(owner, normal, distance, un, gt, flux, component)
        incoming = 0.4 * owner[0, component]
        actual = (volume / dt * owner[0, component] + incoming - terms["flux_vf"][0]) / (
            volume / dt + terms["flux_cf"][0]
        )
        expected = (volume / dt * owner[0, component] + incoming) / (volume / dt + flux[0])
        assert actual == pytest.approx(expected, rel=2e-15)
        # The old explicit boundary was a forward-Euler outflow contribution
        # inside an otherwise backward-Euler momentum equation.
        old_explicit = owner[0, component] + dt / volume * (
            incoming - flux[0] * owner[0, component]
        )
        assert abs(old_explicit - expected) > 0.1


def test_mixed_convection_requires_the_velocity_component():
    with pytest.raises(ValueError, match="component index"):
        assemble_convection_term_boundary(
            np.ones(2),
            np.ones(1),
            {"start_face": 0, "n_faces": 1, "velocity_type": "normalValueTangentialGradient"},
            {"n_cells": 1, "n_interior_faces": 0, "owners": np.array([0])},
            {"face_area_vector": np.array([[1.0, 0.0, 0.0]])},
        )
