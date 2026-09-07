"""The cylinder case uses the public, Pythonic cfMesh-style API."""

import inspect

import openonda.fvm.mesher as msh
from tutorials.coupled_fvm_vpm.cylinder_shedding_flow.reference_flow import setup


def test_extruded_cartesian_mesher_is_public():
    assert msh.ExtrudedCartesianMesher.__module__.endswith("cartesian.extrusion")


def test_setup_has_no_auxiliary_meshing_imports():
    source = inspect.getsource(setup)
    for forbidden in (
        "canonical_surface",
        "case_definition",
        "mesh_family",
        "study.py",
        "mesh.py",
        "meshDict",
    ):
        assert forbidden not in source
    assert "msh.CartesianMesher(" in source
    assert "msh.STLSurface(" in source
    assert "msh.BoxDomain(" in source
    assert "msh.BoxPatches(" in source
    assert "msh.BoxRefinement(" in source
