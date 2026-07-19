"""Body-conformal carved-cube mesh audit (AGENT_PLAN M1).

Certifies that ``coupling_box_mesh(..., hole_box=...)`` — the generator used by
the cubeFlow tutorial — produces a valid body-fitted representation of the cube:

* no fluid cells inside the body;
* the wall patch is a closed 2-manifold (ΣSf = 0, every wall edge shared by
  exactly two wall faces);
* wall-face normals point out of the fluid (into the body);
* positive volumes, valid owner/neighbour connectivity after cell renumbering;
* no duplicated or zero-area faces;
* correct patch classification (coupling patch ``patch``, body ``wall``);
* the IBM path stays unreachable for this topology (``Solver.ibm is None``).

Checked for both the uniform grid and the graded ``wall_refined_axis`` grid the
tutorial actually runs with.
"""

from __future__ import annotations

import contextlib
import io

import numpy as np
import pytest

from source.coupler.core.helpers.fvm_backend import (
    coupling_box_mesh,
    wall_refined_axis,
)
from source.solvers.FVM.mesh.geometry import compute_mesh_geometry
from source.solvers.FVM.mesh.validation import validate_mesh

BOX = (-1.5, 1.5, -1.5, 1.5, -1.5, 1.5)
HOLE = (-0.5, 0.5, -0.5, 0.5, -0.5, 0.5)
BODY_CENTER = np.array([0.0, 0.0, 0.0])


def _build(grading: str):
    if grading == "uniform":
        return coupling_box_mesh(BOX, 0.25, hole_box=HOLE, wall_patch_name="cube")
    nodes = tuple(
        wall_refined_axis(BOX[2 * a], BOX[2 * a + 1], HOLE[2 * a], HOLE[2 * a + 1], 0.125, 0.25)
        for a in range(3)
    )
    return coupling_box_mesh(BOX, 0.25, hole_box=HOLE, wall_patch_name="cube", nodes=nodes)


@pytest.fixture(scope="module", params=["uniform", "graded"])
def carved(request):
    mesh = _build(request.param)
    geo = compute_mesh_geometry(mesh)
    return mesh, geo


def _wall_patch(mesh):
    (patch,) = [b for b in mesh["boundary"] if b["name"] == "cube"]
    return patch


def _wall_faces(mesh):
    patch = _wall_patch(mesh)
    return np.arange(patch["startFace"], patch["startFace"] + patch["nFaces"])


def test_production_validator_accepts(carved):
    mesh, geo = carved
    validate_mesh(mesh)  # raises MeshValidationError on any defect
    validate_mesh(mesh, geo)


def test_no_fluid_cells_inside_body(carved):
    mesh, geo = carved
    c = geo["element_centroids"][: mesh["n_elements"]]
    inside = (
        (c[:, 0] > HOLE[0])
        & (c[:, 0] < HOLE[1])
        & (c[:, 1] > HOLE[2])
        & (c[:, 1] < HOLE[3])
        & (c[:, 2] > HOLE[4])
        & (c[:, 2] < HOLE[5])
    )
    assert not inside.any(), f"{inside.sum()} fluid cells inside the body"


def test_positive_volumes_and_valid_connectivity(carved):
    mesh, geo = carved
    vols = geo["element_volumes"][: mesh["n_elements"]]
    assert np.all(vols > 0.0)
    owners = np.asarray(mesh["owners"])
    neighbours = np.asarray(mesh["neighbours"])
    n_int = mesh["n_interior_faces"]
    assert owners.min() >= 0 and owners.max() < mesh["n_elements"]
    assert neighbours.min() >= 0 and neighbours.max() < mesh["n_elements"]
    assert np.all(owners[:n_int] != neighbours)


def test_no_duplicate_or_zero_area_faces(carved):
    mesh, geo = carved
    areas = np.linalg.norm(geo["face_sf"], axis=1)
    assert areas.min() > 0.0, "zero-area face present"
    keys = {tuple(sorted(int(i) for i in quad)) for quad in mesh["faces"]}
    assert len(keys) == mesh["n_faces"], "duplicated face (same node set) present"


def test_wall_patch_is_closed_manifold(carved):
    mesh, geo = carved
    faces = _wall_faces(mesh)
    # Closed surface: outward area vectors sum to zero, total area = 6 L².
    net = geo["face_sf"][faces].sum(axis=0)
    assert np.allclose(net, 0.0, atol=1e-12), f"wall not closed: ΣSf={net}"
    area = np.linalg.norm(geo["face_sf"][faces], axis=1).sum()
    assert area == pytest.approx(6.0, abs=1e-9)
    # 2-manifold: every wall-face edge is shared by exactly two wall faces.
    edge_count: dict[tuple[int, int], int] = {}
    for f in faces:
        quad = mesh["faces"][f]
        for k in range(4):
            a, b = int(quad[k]), int(quad[(k + 1) % 4])
            edge = (a, b) if a < b else (b, a)
            edge_count[edge] = edge_count.get(edge, 0) + 1
    bad = {e: c for e, c in edge_count.items() if c != 2}
    assert not bad, f"{len(bad)} non-manifold wall edges (shared != 2 times)"


def test_wall_normals_point_out_of_fluid(carved):
    """For the convex axis-aligned body, 'out of the fluid' means every wall
    face normal points from its centroid toward the body interior."""
    mesh, geo = carved
    faces = _wall_faces(mesh)
    sf = geo["face_sf"][faces]
    fc = geo["face_centroids"][faces]
    inward = np.einsum("fi,fi->f", sf, BODY_CENTER - fc)
    assert np.all(inward > 0.0), (
        f"{np.sum(inward <= 0.0)} wall faces have normals pointing into the fluid"
    )


def test_patch_classification(carved):
    mesh, _ = carved
    types = {b["name"]: b["type"] for b in mesh["boundary"]}
    assert types == {"numericalBoundary": "patch", "cube": "wall"}
    # Patches tile the boundary face range exactly, without gaps or overlap.
    n_int = mesh["n_interior_faces"]
    spans = sorted((b["startFace"], b["startFace"] + b["nFaces"]) for b in mesh["boundary"])
    assert spans[0][0] == n_int
    assert spans[-1][1] == mesh["n_faces"]
    for (_, end), (start, _) in zip(spans, spans[1:], strict=False):
        assert end == start


def test_face_count_conservation():
    """Carving removes hole cells and their interior faces; every removed
    interior face between a kept and a removed cell must reappear exactly once
    as a wall face."""
    full = coupling_box_mesh(BOX, 0.25)
    carved_mesh = coupling_box_mesh(BOX, 0.25, hole_box=HOLE, wall_patch_name="cube")
    n_hole = 4**3  # (1.0 / 0.25)³ removed cells
    assert carved_mesh["n_elements"] == full["n_elements"] - n_hole
    wall = _wall_patch(carved_mesh)
    assert wall["nFaces"] == 6 * 4**2  # exposed surface of the 4³ block


def test_ibm_path_unreachable_in_tutorial_solver(tmp_path):
    """The tutorial builds its FVM through ``build_fvm_backend``; certify that
    the resulting solver has no IBM attached (spec: no IBM on this benchmark)."""
    from source.coupler.config.types import CouplerSetup
    from source.coupler.core.helpers.fvm_backend import build_fvm_backend

    setup = CouplerSetup(
        backend="fvm",
        u_inf=[1.0, 0.0, 0.0],
        nu=1e-3,
        dt=0.05,
        t_end=0.1,
        fvm_box=BOX,
        grid_spacing=0.25,
        wall_patch_name="cube",
        surface={"cube": {"side_length": 1.0, "center": [0.0, 0.0, 0.0]}},
        case_dir=str(tmp_path),
    )
    with contextlib.redirect_stdout(io.StringIO()):
        solver = build_fvm_backend(setup)
    assert solver.ibm is None
    assert getattr(solver.algorithm, "ibm", None) is None
