"""Declarative interface checks for the cylinder reference case."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np

import openonda.fvm.mesher as msh
from tutorials.coupled_fvm_vpm.cylinder_shedding_flow.reference_flow import setup


def capture_case(monkeypatch, name="coarse", dx=0.125):
    captured = {}

    def create(solver_setup, **kwargs):
        captured.update(setup=solver_setup, **kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(setup.fvm, "create_fvm_solver", create)
    setup.create_solver(name, dx)
    return captured


def test_single_creator_declares_cfmesh_inputs_and_output_directories(monkeypatch):
    case = capture_case(monkeypatch)
    mesh = case["mesh"]

    assert isinstance(mesh, msh.ExtrudedCartesianMesher)
    assert isinstance(mesh.source, msh.CartesianMesher)
    case_dir = Path(setup.__file__).resolve().parent
    assert mesh.source.surfaces[0].path == (case_dir.parent / "assets/cylinder_long.stl").resolve()
    assert mesh.source.surface_may_cross_domain_boundary
    assert mesh.domain.bounds == (-8.0, 24.0, -10.0, 10.0, -0.5, 0.5)
    assert mesh.source.domain.patches.as_tuple() == (
        "inlet",
        "outlet",
        "ymin",
        "ymax",
        "zmin",
        "zmax",
    )
    assert [item.name for item in mesh.source.refinements] == [
        "nearBody",
        "nearWake",
        "wake",
    ]
    np.testing.assert_allclose(
        [item.cell_size for item in mesh.source.refinements],
        [0.375, 0.375, 0.75],
    )
    np.testing.assert_allclose(
        [mesh.effective_cell_size(item.cell_size) for item in mesh.source.refinements],
        [0.25, 0.25, 0.5],
    )
    assert mesh.source.max_cell_size == 1.0
    assert mesh.source.patch_refinements == (msh.PatchRefinement("cylinder", 0.125),)
    assert mesh.levels == (-0.5, 0.0, 0.5)
    assert case["solution_dir"] == case_dir / "solution/coarse"
    assert case["samples_dir"] == case_dir / "samples/coarse"


def test_creator_sets_the_solver_and_boundary_contract(monkeypatch):
    case = capture_case(monkeypatch, "medium", 0.0625)
    config = case["setup"]

    assert config.case_name == "medium"
    assert config.mesh.max_non_orthogonality_deg == 70.0
    assert config.mesh.max_skewness == 1.0
    assert config.mesh.max_lsq_condition == 9.0
    assert config.pimple.n_orthogonal_correctors == 0
    assert [boundary.name for boundary in config.boundaries] == [
        "inlet",
        "outlet",
        "ymin",
        "ymax",
        "zmin",
        "zmax",
        "cylinder",
    ]
