"""Panel-mesh STL I/O tests for the installed numpy-stl API."""

from __future__ import annotations

import numpy as np

from source.solvers.vpm.boundary_elements.panels.geometry.stl_io import load_stl, save_stl


def test_panel_stl_roundtrip_preserves_triangles_and_unit_normals(tmp_path):
    vertex_position = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0]],
        ]
    )
    normal = np.array([[0.0, 0.0, 2.0], [0.0, 0.0, -3.0]])
    path = tmp_path / "panels.stl"

    save_stl(str(path), vertex_position, normal)
    loaded_vertex_position, loaded_normal = load_stl(str(path))

    np.testing.assert_allclose(loaded_vertex_position, vertex_position, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(loaded_normal, normal / np.linalg.norm(normal, axis=1)[:, None])
