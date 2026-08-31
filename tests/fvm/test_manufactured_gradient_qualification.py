"""Manufactured-solution qualification for FVM gradient reconstruction."""

from __future__ import annotations

import numpy as np

from source.solvers.fvm.fields.gradients import compute_lsq_gradient
from source.solvers.fvm.mesh.geometry import compute_mesh_geometry
from source.solvers.fvm.mesh.rectilinear import box_mesh_3d


def _gradient_error(n_cells_per_axis: int) -> float:
    axis = np.linspace(0.0, 2.0 * np.pi, n_cells_per_axis + 1)
    mesh = box_mesh_3d(axis, axis, axis)
    geometry = compute_mesh_geometry(mesh, gradient_scheme="lsq")
    centres = np.asarray(geometry["cell_centre"])
    face_centres = np.asarray(geometry["face_centre"])
    n_cells = int(mesh["n_cells"])
    n_boundary = int(mesh["n_faces"] - mesh["n_interior_faces"])
    scalar = np.empty(n_cells + n_boundary)
    scalar[:n_cells] = np.sin(centres[:, 0]) * np.cos(centres[:, 1]) * np.sin(centres[:, 2])
    boundary_centres = face_centres[int(mesh["n_interior_faces"]) :]
    scalar[n_cells:] = (
        np.sin(boundary_centres[:, 0])
        * np.cos(boundary_centres[:, 1])
        * np.sin(boundary_centres[:, 2])
    )
    expected = np.column_stack(
        (
            np.cos(centres[:, 0]) * np.cos(centres[:, 1]) * np.sin(centres[:, 2]),
            -np.sin(centres[:, 0]) * np.sin(centres[:, 1]) * np.sin(centres[:, 2]),
            np.sin(centres[:, 0]) * np.cos(centres[:, 1]) * np.cos(centres[:, 2]),
        )
    )
    actual = compute_lsq_gradient(scalar, mesh, geometry)[:, :, 0]
    spacing = 2.0 * np.pi / n_cells_per_axis
    interior = np.all((centres > spacing) & (centres < 2.0 * np.pi - spacing), axis=1)
    return float(
        np.linalg.norm(actual[:n_cells][interior] - expected[interior])
        / np.linalg.norm(expected[interior])
    )


def test_lsq_gradient_has_second_order_spatial_convergence(record_property):
    """Claim: LSQ reconstruction is second order for a smooth periodic field.

    The manufactured scalar is sin(x)cos(y)sin(z) on [0,2pi]^3 with an analytic
    gradient.  Relative discrete L2 error is measured on interior cells for
    uniform 8³, 16³, and 32³ meshes.  The CPU/f64 reconstruction is
    deterministic and has no time step or random seed.  Requiring order 1.8
    allows the expected asymptotic O(h²) behavior a 10% pre-asymptotic margin.
    """
    errors = [_gradient_error(n) for n in (8, 16, 32)]
    orders = [float(np.log2(errors[index] / errors[index + 1])) for index in (0, 1)]
    record_property("n8_relative_l2", errors[0])
    record_property("n16_relative_l2", errors[1])
    record_property("n32_relative_l2", errors[2])
    record_property("minimum_observed_order", min(orders))
    record_property("precision", "f64")
    record_property("backend", "CPU")
    assert errors[0] > errors[1] > errors[2]
    assert min(orders) > 1.8
