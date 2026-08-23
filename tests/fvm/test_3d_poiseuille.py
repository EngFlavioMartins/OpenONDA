"""Mesh-converged laminar square-duct verification."""

from __future__ import annotations

import numpy as np
import pytest

from source.solvers.fvm.assemble import diffusion, matrix_assembly
from source.solvers.fvm.fields.gradients import compute_lsq_gradient
from source.solvers.fvm.mesh.geometry import compute_mesh_geometry
from source.solvers.fvm.solve.linear_interface import solve_linear_system

from ._structured_mesh import structured_box


def square_duct_velocity(y, z, *, half_width=0.5, pressure_gradient=1.0, terms=101):
    """Return the fully developed axial velocity for a square duct.

    ``pressure_gradient`` is ``-dp/dx / mu``. Coordinates span
    ``[-half_width, half_width]`` in both cross-stream directions.
    """
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    velocity = np.zeros(np.broadcast_shapes(y.shape, z.shape), dtype=np.float64)
    for mode in range(1, terms, 2):
        sign = (-1) ** ((mode - 1) // 2)
        wall_factor = 1.0 - np.cosh(mode * np.pi * z / (2.0 * half_width)) / np.cosh(
            mode * np.pi / 2.0
        )
        velocity += sign * wall_factor * np.cos(mode * np.pi * y / (2.0 * half_width)) / mode**3
    return 16.0 * half_width**2 * pressure_gradient * velocity / np.pi**3


def square_duct_bulk_velocity(*, half_width=0.5, pressure_gradient=1.0, terms=401):
    """Return the analytical cross-section-mean velocity."""
    odd_modes = np.arange(1, terms, 2, dtype=np.float64)
    series = np.sum(
        (1.0 - 2.0 * np.tanh(odd_modes * np.pi / 2.0) / (odd_modes * np.pi)) / odd_modes**4
    )
    return 32.0 * half_width**2 * pressure_gradient * series / np.pi**4


def _solve_duct(cross_stream_cells):
    mesh = structured_box(2, cross_stream_cells, cross_stream_cells, lx=0.25)
    geometry = compute_mesh_geometry(mesh, gradient_scheme="lsq")
    n_cells = mesh["n_cells"]
    n_interior = mesh["n_interior_faces"]
    field = np.zeros(n_cells + mesh["n_faces"] - n_interior)

    for patch in mesh["boundary"]:
        patch["boundary_condition_type"] = "fixedValue"
        faces = np.arange(patch["start_face"], patch["start_face"] + patch["n_faces"])
        ghosts = n_cells + faces - n_interior
        centres = geometry["face_centre"][faces]
        field[ghosts] = square_duct_velocity(centres[:, 1] - 0.5, centres[:, 2] - 0.5)

    volumes = geometry["cell_volume"]
    for _ in range(80):
        gradient = compute_lsq_gradient(field, mesh, geometry)
        flux = diffusion.assemble_diffusion_term(
            field, gradient, np.ones(n_cells), mesh, geometry, mesh["boundary"]
        )
        matrix = matrix_assembly.assemble_matrix_from_fluxes_vectorized(flux, mesh)
        rhs = matrix_assembly.assemble_rhs_from_fluxes_vectorized(flux, mesh) + volumes
        solution = solve_linear_system(matrix, rhs, method="spsolve", equation_type="scalar")
        change = np.linalg.norm(solution - field[:n_cells]) / max(np.linalg.norm(solution), 1e-30)
        field[:n_cells] = 0.8 * solution + 0.2 * field[:n_cells]
        if change < 1e-12:
            break
    else:
        raise AssertionError("Square-duct non-orthogonal iteration did not converge")

    centres = geometry["cell_centre"]
    exact = square_duct_velocity(centres[:, 1] - 0.5, centres[:, 2] - 0.5)
    profile_error = np.sqrt(np.sum(volumes * (field[:n_cells] - exact) ** 2) / np.sum(volumes))
    numerical_bulk = np.sum(volumes * field[:n_cells]) / np.sum(volumes)
    exact_bulk = square_duct_bulk_velocity()
    pressure_drop_error = abs(exact_bulk / numerical_bulk - 1.0)
    return profile_error, pressure_drop_error


@pytest.mark.verification
@pytest.mark.slow
def test_square_duct_profile_converges_over_three_mesh_levels():
    levels = np.asarray((16, 32, 64), dtype=float)
    errors = np.asarray([_solve_duct(int(level)) for level in levels])
    profile_errors = errors[:, 0]
    pressure_drop_errors = errors[:, 1]
    profile_order = np.polyfit(np.log(1.0 / levels), np.log(profile_errors), 1)[0]
    pressure_drop_order = np.polyfit(np.log(1.0 / levels), np.log(pressure_drop_errors), 1)[0]

    assert np.all(np.diff(profile_errors) < 0.0), (
        f"non-monotone square-duct profile errors: {profile_errors}"
    )
    assert np.all(np.diff(pressure_drop_errors) < 0.0), (
        f"non-monotone square-duct pressure-drop errors: {pressure_drop_errors}"
    )
    assert profile_order >= 1.8, (
        f"square-duct profile order {profile_order:.3f}; errors={profile_errors}"
    )
    assert pressure_drop_order >= 1.8, (
        f"square-duct pressure-drop order {pressure_drop_order:.3f}; errors={pressure_drop_errors}"
    )
