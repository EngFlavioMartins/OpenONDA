"""Geometry regression for the backward-facing-step tutorial."""

import importlib.util
from pathlib import Path

import numpy as np

from source.solvers.FVM.mesh import geometry
from source.solvers.FVM.mesh.validation import validate_mesh

MESH_MODULE_PATH = (
    Path(__file__).parents[2] / "tutorials" / "FVM" / "stepProfile" / "assets" / "mesh_step.py"
)
SPEC = importlib.util.spec_from_file_location("step_profile_mesh", MESH_MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MESH_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MESH_MODULE)
backward_facing_step_mesh = MESH_MODULE.backward_facing_step_mesh


def test_step_tutorial_has_a_geometric_step():
    mesh, _ = backward_facing_step_mesh(
        n_upstream=4,
        n_downstream=8,
        n_height=4,
    )
    geo = geometry.compute_mesh_geometry(mesh)
    report = validate_mesh(mesh, geo)
    centres = geo["element_centroids"][: mesh["n_elements"]]

    # The lower-left block is solid, while the same y range is fluid after
    # the expansion. This distinguishes a geometric step from a scalar front.
    assert not np.any((centres[:, 0] < 0.0) & (centres[:, 1] < 1.0))
    assert np.any((centres[:, 0] > 0.0) & (centres[:, 1] < 1.0))

    patches = {patch["name"]: patch for patch in mesh["boundary"]}
    assert patches["outlet"]["nFaces"] == 2 * patches["inlet"]["nFaces"]

    wall = patches["walls"]
    wall_centres = geo["face_centroids"][wall["startFace"] : wall["startFace"] + wall["nFaces"]]
    vertical_step = np.isclose(wall_centres[:, 0], 0.0) & (wall_centres[:, 1] < 1.0)
    assert np.count_nonzero(vertical_step) == 2
    assert report["max_non_orthogonality_deg"] < 1e-10
