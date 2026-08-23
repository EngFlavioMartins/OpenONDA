"""``coupling_box_mesh`` must keep a real outlet out of the merge.

Merging all six sides clamps a reference's outlet to the freestream.
"""

import numpy as np
import pytest

from source.solvers.fvm.mesh.geometry import compute_mesh_geometry
from source.solvers.fvm.mesh.rectilinear import OUTER_PATCH_NAMES, coupling_box_mesh
from source.solvers.fvm.utils.cavity_utils import needs_pressure_reference

BOX = (-1.0, 2.0, -1.0, 1.0, -0.25, 0.25)
SPACING = 0.25


def _mesh(**kwargs):
    return coupling_box_mesh(BOX, SPACING, **kwargs)


def _assert_patches_are_contiguous_and_complete(mesh):
    face = mesh["n_interior_faces"]
    for patch in mesh["boundary"]:
        assert patch["start_face"] == face, f"gap before patch {patch['name']!r}"
        face += patch["n_faces"]
    assert face == mesh["n_faces"], "patches do not cover every boundary face"


def test_default_merges_every_outer_face():
    mesh = _mesh()
    names = [patch["name"] for patch in mesh["boundary"]]
    assert names == ["numericalBoundary"]
    _assert_patches_are_contiguous_and_complete(mesh)


@pytest.mark.parametrize("separate", [("outlet",), ("inlet", "outlet"), ("ymin", "ymax")])
def test_separated_faces_get_their_own_contiguous_patch(separate):
    mesh = _mesh(separate_outer=separate)
    names = [patch["name"] for patch in mesh["boundary"]]
    assert names[0] == "numericalBoundary"
    assert sorted(names[1:]) == sorted(separate)
    _assert_patches_are_contiguous_and_complete(mesh)

    # Face count is conserved: nothing was dropped or duplicated by the reorder.
    assert mesh["n_faces"] == _mesh()["n_faces"]
    assert mesh["n_cells"] == _mesh()["n_cells"]


def test_separated_outlet_keeps_its_outward_normal():
    mesh = _mesh(separate_outer=("outlet",))
    geometry = compute_mesh_geometry(mesh)
    patch = next(p for p in mesh["boundary"] if p["name"] == "outlet")
    span = slice(patch["start_face"], patch["start_face"] + patch["n_faces"])
    sf = geometry["face_area_vector"][span]
    normals = sf / np.linalg.norm(sf, axis=1, keepdims=True)
    np.testing.assert_allclose(normals, np.tile([1.0, 0.0, 0.0], (len(normals), 1)), atol=1e-12)


def test_a_separated_outlet_can_anchor_the_pressure_datum():
    """The whole point: a Dirichlet pressure patch becomes expressible."""
    coupled = [
        {"name": "numericalBoundary", "n_faces": 100, "pressure_type": "fixedFluxPressure"},
    ]
    assert needs_pressure_reference(coupled), "coupled layout has a free pressure datum"

    reference = [
        {"name": "numericalBoundary", "n_faces": 84, "pressure_type": "fixedFluxPressure"},
        {
            "name": "outlet",
            "n_faces": 16,
            "pressure_type": "fixedValue",
            "kinematic_pressure_value": 0.0,
        },
    ]
    assert not needs_pressure_reference(reference)


def test_empty_spanwise_and_separate_outer_compose():
    mesh = _mesh(empty_spanwise=True, separate_outer=("outlet",))
    kinds = {patch["name"]: patch["type"] for patch in mesh["boundary"]}
    assert kinds["zmin"] == kinds["zmax"] == "empty"
    assert kinds["outlet"] == "patch"
    assert kinds["numericalBoundary"] == "patch"
    _assert_patches_are_contiguous_and_complete(mesh)


def test_unknown_face_name_is_rejected():
    with pytest.raises(ValueError, match="not outer faces"):
        _mesh(separate_outer=("xmax",))
    assert "outlet" in OUTER_PATCH_NAMES
