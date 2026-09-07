"""Planar extraction must preserve finite-volume closure and patch meaning."""

import numpy as np
import pytest

from source.solvers.fvm.mesh.cartesian import BoxDomain, BoxPatches, structured_box
from source.solvers.fvm.mesh.cartesian.extrusion import extrude_mesh_section
from source.solvers.fvm.mesh.geometry import compute_mesh_geometry
from source.solvers.fvm.mesh.validation import validate_geometry, validate_topology


def test_section_extrusion_conserves_volume_and_shared_faces():
    source = structured_box(3, 2, 4, lx=3, ly=2, lz=4)
    domain = BoxDomain(
        bounds=(0, 3, 0, 2, -0.5, 0.5),
        patches=BoxPatches("xmin", "xmax", "ymin", "ymax", "zmin", "zmax"),
    )
    mesh = extrude_mesh_section(source, coordinate=1.3, levels=(-0.5, 0, 0.5), domain=domain)
    assert mesh["n_cells"] == 12
    assert mesh["n_interior_faces"] == 20
    validate_topology(mesh)
    geometry = compute_mesh_geometry(mesh, gradient_scheme="lsq")
    quality = validate_geometry(mesh, geometry)
    assert geometry["cell_volume"].sum() == pytest.approx(6)
    assert quality["max_skewness"] < 1e-12
    assert quality["max_lsq_condition"] < 4
    assert {p["name"] for p in mesh["boundary"]} == set(domain.patches.as_tuple())
    np.testing.assert_allclose(mesh["vertex_position"][:, 2].min(), -0.5)
    np.testing.assert_allclose(mesh["vertex_position"][:, 2].max(), 0.5)


@pytest.mark.parametrize("levels", [(-0.5, -0.5, 0.5), (-0.5, -0.6, 0.5), (-0.5, 0.4)])
def test_invalid_extrusion_levels_are_rejected(levels):
    source = structured_box(1, 1, 1)
    domain = BoxDomain(
        bounds=(0, 1, 0, 1, -0.5, 0.5),
        patches=BoxPatches("xmin", "xmax", "ymin", "ymax", "zmin", "zmax"),
    )
    with pytest.raises(ValueError, match="levels"):
        extrude_mesh_section(source, coordinate=0.4, levels=levels, domain=domain)
