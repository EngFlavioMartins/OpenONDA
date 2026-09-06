"""Regression tests for solver-facing topology and quality gates."""

from __future__ import annotations

import numpy as np
import pytest

from source.solvers.fvm.mesh.cartesian import structured_box
from source.solvers.fvm.mesh.validation import MeshValidationError, validate_topology


def test_topology_rejects_a_coarse_face_missing_a_transition_midpoint():
    mesh = structured_box(1, 1, 1)
    points = np.asarray(mesh["vertex_position"], dtype=np.float64)
    points = np.vstack((points, np.asarray(((0.0, 0.5, 1.0),))))
    faces = [np.asarray(face, dtype=np.int32).copy() for face in mesh["faces"]]
    # Split one perimeter edge on the x-min face without splitting the
    # adjacent z-max face.  This is the minimal form of the adaptive
    # coarse/fine transition defect found by the audit.
    faces[0] = np.asarray((4, 8, 6, 2, 0), dtype=np.int32)
    mesh["vertex_position"] = points
    mesh["faces"] = faces
    mesh["n_points"] = len(points)

    with pytest.raises(MeshValidationError, match="non-closed polygon edges"):
        validate_topology(mesh)
