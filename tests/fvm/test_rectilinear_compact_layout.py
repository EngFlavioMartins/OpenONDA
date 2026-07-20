"""Compact structured-mesh and raw-VTU regression checks."""

from __future__ import annotations

import importlib
from pathlib import Path
import sys

import numpy as np
import pytest

from source.solvers.FVM.mesh.rectilinear import box_mesh_3d
from source.solvers.FVM.mesh.topology import MeshTopology


def _mesh():
    axis = np.linspace(-1.0, 1.0, 4)
    return box_mesh_3d(axis, axis, axis, hole_box=(-1 / 3, 1 / 3, -1 / 3, 1 / 3, -1 / 3, 1 / 3))


def test_rectilinear_mesh_uses_compact_quads_and_explicit_hexes():
    mesh = _mesh()

    assert isinstance(mesh["faces"], np.ndarray)
    assert mesh["faces"].shape == (mesh["n_faces"], 4)
    assert mesh["faces"].dtype == np.int32
    assert mesh["cell_vertices"].shape == (mesh["n_elements"], 8)
    assert np.all(mesh["cell_type_codes"] == 5)

    topology = MeshTopology.from_mesh_data(mesh)
    assert topology.face_nodes.dtype == np.int32
    assert topology.cell_faces.dtype == np.int32
    assert topology.cell_face_offsets[-1] == len(topology.cell_faces)


def test_rectilinear_vtu_defaults_to_raw_cell_data(tmp_path):
    pytest.importorskip("pyvista")
    pytest.importorskip("vtk")
    import pyvista as pv

    from source.solvers.FVM.io.vtk_exporter import VTKExporter

    mesh = _mesh()
    target = tmp_path / "structured.vtu"
    VTKExporter(mesh).export(
        str(target),
        {"U": np.zeros((mesh["n_elements"], 3)), "p": np.ones(mesh["n_elements"])},
    )

    result = pv.read(target)
    assert "U" in result.cell_data
    assert "U" not in result.point_data
    assert np.all(result.celltypes == pv.CellType.HEXAHEDRON)


def test_cube_comparison_sampler_accepts_raw_cell_vtu(tmp_path):
    pytest.importorskip("pyvista")
    pytest.importorskip("vtk")
    from source.solvers.FVM.io.vtk_exporter import VTKExporter
    from source.solvers.FVM.mesh.geometry import compute_mesh_geometry

    mesh = _mesh()
    target = tmp_path / "raw.vtu"
    velocity = np.tile([1.0, -0.25, 0.5], (mesh["n_elements"], 1))
    VTKExporter(mesh).export(str(target), {"U": velocity, "p": np.ones(mesh["n_elements"])})

    assets = Path("tutorials/coupled_FVM_VPM/cubeFlow/assets").resolve()
    sys.path.insert(0, str(assets))
    try:
        reference_util = importlib.import_module("_reference_util")
        point = compute_mesh_geometry(mesh)["element_centroids"][0:1]
        sampled = reference_util.sample_vtu(target, point)
    finally:
        sys.path.remove(str(assets))
        sys.modules.pop("_reference_util", None)

    assert sampled["valid"][0]
    np.testing.assert_allclose(sampled["U"], [[1.0, -0.25, 0.5]])
    np.testing.assert_allclose(sampled["p"], [1.0])
