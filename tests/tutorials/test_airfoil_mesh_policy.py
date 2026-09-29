"""The airfoil's fine wall request must not refine the remote box faces."""

from pathlib import Path

import pytest

import openonda.fvm.mesher as msh
from openonda.tutorial_runner import load_case_module
from openonda.tutorials import materialize_tutorial


def test_airfoil_wall_sizing_is_patch_local(tmp_path):
    case = materialize_tutorial("fvm/airfoil_flow", tmp_path)
    mesh = load_case_module(case).create_fvm_mesh()
    assert mesh.max_cell_size == pytest.approx(1.0)
    assert mesh.boundary_cell_size == pytest.approx(1.0)
    assert mesh.min_cell_size == pytest.approx(0.03125)
    assert mesh.patch_refinements == (msh.PatchRefinement("airfoil", 0.03125),)


def test_patch_local_request_keeps_outer_template_coarse():
    root = Path(__file__).resolve().parents[2]
    mesher = msh.CartesianMesher(
        domain=msh.BoxDomain(
            (-2.0, 2.0, -2.0, 2.0, -2.0, 2.0),
            msh.BoxPatches("xmin", "xmax", "ymin", "ymax", "zmin", "zmax"),
        ),
        surfaces=(
            msh.STLSurface(
                root / "tutorials/coupled_fvm_vpm/02_cube_flow/assets/cube.stl",
                patch="cube",
            ),
        ),
        max_cell_size=1.0,
        min_cell_size=0.25,
        patch_refinements=(msh.PatchRefinement("cube", 0.25),),
    )
    mesh = mesher.build(stop_after="templateGeneration")
    generation = mesh["mesh_generation"]
    assert generation["boundary_refinement_level"] == generation["global_refinement_level"]
    assert (
        generation["surface_patch_refinement_levels"]["cube"]
        == generation["global_refinement_level"] + 2
    )
    assert generation["resolved_background_cell_size"] == pytest.approx(1.0)
    assert generation["resolved_surface_patch_sizes"]["cube"] == pytest.approx(0.25)
