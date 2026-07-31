"""Mesh validation and quality-report tests."""

import copy

import numpy as np
import pytest

from source.solvers.FVM.config.types import MeshConfig
from source.solvers.FVM.mesh.geometry import MeshGeometry, compute_mesh_geometry
from source.solvers.FVM.mesh.topology import MeshTopology
from source.solvers.FVM.mesh.validation import (
    MeshValidationError,
    enforce_quality_thresholds,
    validate_mesh,
)


def test_hand_built_mesh_passes_and_reports_quality(hand_built_3d_mesh):
    geo = compute_mesh_geometry(hand_built_3d_mesh)
    report = validate_mesh(hand_built_3d_mesh, geo)
    assert report["n_cells"] == 8
    assert report["min_volume"] > 0.0
    assert report["max_non_orthogonality_deg"] < 1e-10
    assert report["max_skewness"] < 1e-12
    assert report["max_aspect_ratio"] == pytest.approx(1.0)


def test_typed_mesh_schema_is_contiguous_and_read_only(hand_built_3d_mesh):
    geo = compute_mesh_geometry(hand_built_3d_mesh, gradient_scheme="lsq")
    source_owners = hand_built_3d_mesh["owners"]
    source_volumes = geo["element_volumes"]
    assert source_owners.flags.writeable
    assert source_volumes.flags.writeable

    topology = MeshTopology.from_mesh_data(hand_built_3d_mesh)
    geometry = MeshGeometry.from_data(hand_built_3d_mesh, geo)

    assert topology.face_nodes.flags.c_contiguous
    assert topology.cell_face_offsets[-1] == len(topology.cell_faces)
    assert not topology.owners.flags.writeable
    assert not geometry.cell_volumes.flags.writeable
    assert source_owners.flags.writeable
    assert source_volumes.flags.writeable
    assert np.shares_memory(topology.owners, source_owners)
    assert np.shares_memory(geometry.cell_volumes, source_volumes)
    assert geometry.lsq_condition.shape == (hand_built_3d_mesh["n_elements"],)


def test_explicit_mesh_quality_threshold_rejects(hand_built_3d_mesh):
    geo = compute_mesh_geometry(hand_built_3d_mesh)
    report = validate_mesh(hand_built_3d_mesh, geo)
    with pytest.raises(MeshValidationError, match="max_aspect_ratio"):
        enforce_quality_thresholds(report, MeshConfig(max_aspect_ratio=0.5))


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
