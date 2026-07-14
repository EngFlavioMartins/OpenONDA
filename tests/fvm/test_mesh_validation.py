"""Mesh validation and quality-report tests."""

import copy

import numpy as np
import pytest

from source.solvers.FVM.mesh.geometry import compute_mesh_geometry
from source.solvers.FVM.mesh.validation import MeshValidationError, validate_mesh


def test_hand_built_mesh_passes_and_reports_quality(hand_built_3d_mesh):
    geo = compute_mesh_geometry(hand_built_3d_mesh)
    report = validate_mesh(hand_built_3d_mesh, geo)
    assert report["n_cells"] == 8
    assert report["min_volume"] > 0.0
    assert report["max_non_orthogonality_deg"] < 1e-10


def test_rejects_boundary_gap(hand_built_3d_mesh):
    mesh = copy.deepcopy(hand_built_3d_mesh)
    mesh["boundary"][1]["startFace"] += 1
    with pytest.raises(MeshValidationError, match="contiguous start"):
        validate_mesh(mesh)


def test_rejects_repeated_face_node(hand_built_3d_mesh):
    mesh = copy.deepcopy(hand_built_3d_mesh)
    mesh["faces"][0] = np.array([4, 13, 13, 1], dtype=np.int32)
    with pytest.raises(MeshValidationError, match="repeats a node"):
        validate_mesh(mesh)


def test_rejects_reversed_face_orientation(hand_built_3d_mesh):
    mesh = copy.deepcopy(hand_built_3d_mesh)
    mesh["faces"][0] = mesh["faces"][0][::-1].copy()
    geo = compute_mesh_geometry(mesh)
    with pytest.raises(MeshValidationError, match="orientation"):
        validate_mesh(mesh, geo)
