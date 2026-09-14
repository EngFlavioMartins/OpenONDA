"""Physical box requests follow the cfMesh lattice, including exact ratios."""

from pathlib import Path

import numpy as np
import pytest

import openonda.fvm.mesher as msh


def _reference_mesher(factor, dx=0.25):
    root = Path(__file__).resolve().parents[2]
    return msh.CartesianMesher(
        domain=msh.BoxDomain(
            (-7.5, 15.0, -7.5, 7.5, -7.5, 7.5),
            msh.BoxPatches("inlet", "outlet", "ymin", "ymax", "zmin", "zmax"),
        ),
        surfaces=(
            msh.STLSurface(
                root / "tutorials/coupled_fvm_vpm/02_cube_flow/reference_flow/assets/cube.stl",
                patch="cube",
            ),
        ),
        max_cell_size=factor * dx,
        refinements=(
            msh.BoxRefinement("nearBody", (-1.5, 2.5, -1.5, 1.5, -1.5, 1.5), 3 * dx),
            msh.BoxRefinement("wake", (0.0, 8.0, -2.0, 2.0, -2.0, 2.0), 6 * dx),
        ),
        patch_refinements=(msh.PatchRefinement("cube", dx),),
    )


@pytest.mark.parametrize(
    "factor,near,wake,wall",
    [
        (8.0, 2.0, 4.0, 1.0),
        (11.99, 2.9975, 5.995, 0.749375),
        (12.0, 1.5, 3.0, 0.75),
        (12.01, 1.50125, 3.0025, 0.750625),
        (16.0, 2.0, 4.0, 1.0),
    ],
)
def test_reference_template_matches_reported_sizes(factor, near, wake, wall):
    dx = 0.25
    mesher = _reference_mesher(factor, dx)
    mesh = mesher.build(stop_after="templateGeneration")
    centres = mesh["vertex_position"][mesh["cell_vertex_indices"]].mean(axis=1)
    reports = {entry.name: entry for entry in mesher._cfmesh_size_reports()}
    for request, expected in zip(mesher.refinements, (near * dx, wake * dx), strict=True):
        selected = mesh["cell_sizes"][request.contains(centres)]
        assert len(selected) > 0
        assert selected.max() == pytest.approx(expected)
        assert reports[request.name].effective == pytest.approx(expected)
        assert mesher.effective_cell_size(request.cell_size, strict=True) == pytest.approx(expected)
    generation = mesh["mesh_generation"]
    root_size = generation["root_box"][1] - generation["root_box"][0]
    patch_size = root_size / 2 ** generation["surface_patch_refinement_levels"]["cube"]
    assert patch_size == pytest.approx(wall * dx)
    assert reports["patch:cube"].effective == pytest.approx(patch_size)


@pytest.mark.slow
@pytest.mark.parametrize("dx", [0.25, 0.17])
@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_completed_reference_mesh_publishes_strict_box_sizes(dx):
    mesher = _reference_mesher(12.0, dx=dx)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        mesh = mesher.build()
    reports = {entry.name: entry for entry in mesher.report.sizes}
    assert reports["nearBody"].requested == pytest.approx(3.0 * dx)
    assert reports["nearBody"].effective == pytest.approx(1.5 * dx)
    assert reports["nearBody"].level == 3
    assert reports["wake"].effective == pytest.approx(3.0 * dx)
    assert reports["wake"].level == 2
    generation = mesh["mesh_generation"]
    assert generation["resolved_box_sizes"] == pytest.approx(
        {"nearBody": 1.5 * dx, "wake": 3.0 * dx}
    )
    assert generation["requested_sizes"] == [entry.as_dict() for entry in mesher.report.sizes]
    assert generation["cartesian_report"]["sizes"] == generation["requested_sizes"]


@pytest.mark.parametrize("requested", [0.0, -1.0, np.inf, np.nan])
@pytest.mark.parametrize("strict", [False, True])
def test_invalid_size_queries_fail_without_entering_refinement_loop(requested, strict):
    with pytest.raises(ValueError, match="finite and positive"):
        _reference_mesher(12.0).effective_cell_size(requested, strict=strict)
