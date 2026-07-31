"""Three-level analytical 3D Poisson verification on unstructured cell families."""

import numpy as np
import pytest

from source.solvers.FVM.assemble import diffusion, matrix_assembly
from source.solvers.FVM.fields.gradients import compute_lsq_gradient
from source.solvers.FVM.mesh.geometry import compute_mesh_geometry
from source.solvers.FVM.solve.linear_interface import solve_linear_system

from ._polyhedral_mesh import split_prism_box


def _exact(points):
    return np.prod(np.sin(np.pi * points), axis=1)


def _set_dirichlet_ghosts(field, mesh, geo):
    n_cells = mesh["n_elements"]
    n_internal = mesh["n_interior_faces"]
    for patch in mesh["boundary"]:
        patch["bc_type"] = "fixedValue"
        faces = np.arange(patch["startFace"], patch["startFace"] + patch["nFaces"])
        ghosts = n_cells + faces - n_internal
        field[ghosts] = _exact(geo["face_centroids"][faces])


def _solve_poisson(mesh):
    geo = compute_mesh_geometry(mesh, gradient_scheme="lsq")
    n_cells = mesh["n_elements"]
    n_total = n_cells + mesh["n_faces"] - mesh["n_interior_faces"]
    field = np.zeros(n_total)
    _set_dirichlet_ghosts(field, mesh, geo)
    source = 3.0 * np.pi**2 * _exact(geo["element_centroids"])
    volumes = geo["element_volumes"]

    for _ in range(80):
        gradient = compute_lsq_gradient(field, mesh, geo)
        flux = diffusion.assemble_diffusion_term(
            field, gradient, np.ones(n_cells), mesh, geo, mesh["boundary"]
        )
        matrix = matrix_assembly.assemble_matrix_from_fluxes_vectorized(flux, mesh)
        rhs = matrix_assembly.assemble_rhs_from_fluxes_vectorized(flux, mesh) + source * volumes
        solution = solve_linear_system(matrix, rhs, method="spsolve", equation_type="scalar")
        change = np.linalg.norm(solution - field[:n_cells]) / max(np.linalg.norm(solution), 1e-30)
        field[:n_cells] = 0.7 * solution + 0.3 * field[:n_cells]
        if change < 1e-11:
            break
    else:
        raise AssertionError("Non-orthogonal Poisson iteration did not converge")

    error = field[:n_cells] - _exact(geo["element_centroids"])
    return np.sqrt(np.sum(volumes * error**2) / np.sum(volumes))


def _tetrahedral_box(size):
    gmsh = pytest.importorskip("gmsh", reason="Gmsh FVM test dependency is not installed")
    gmsh.initialize()
    try:
        gmsh.model.add("poisson_tet")
        gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), size)
        gmsh.model.mesh.generate(3)
        from source.solvers.FVM.mesh.gmsh_importer import GmshImporter

        return GmshImporter().get_mesh_data()
    finally:
        gmsh.finalize()


@pytest.mark.slow
@pytest.mark.parametrize(
    ("family", "mesh_factory", "sizes", "minimum_order"),
    [
        ("tet", _tetrahedral_box, (0.5, 0.25, 0.125), 0.7),
        ("prism", lambda n: split_prism_box(n), (2, 4, 8), 0.7),
        ("mixed", lambda n: split_prism_box(n, mixed=True), (2, 4, 8), 0.7),
    ],
)
def test_poisson_mms_converges_over_three_levels(family, mesh_factory, sizes, minimum_order):
    errors = [_solve_poisson(mesh_factory(size)) for size in sizes]
    spacing = np.asarray(sizes, dtype=float)
    if family != "tet":
        spacing = 1.0 / spacing
    observed_order = np.polyfit(np.log(spacing), np.log(errors), 1)[0]

    assert np.all(np.diff(errors) < 0.0), f"{family} errors are not monotone: {errors}"
    assert observed_order >= minimum_order, (
        f"{family} observed order {observed_order:.3f} < {minimum_order}; errors={errors}"
    )
