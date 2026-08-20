"""Tests for STL-driven native Cartesian surface input."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from source.solvers.FVM.mesh.triangulated_surface import TriangulatedSurface

CUBE_STL = (
    Path(__file__).parents[2] / "tutorials/coupled_FVM_VPM/cubeFlow/referenceFlow/assets/cube.stl"
)


def test_loads_watertight_ascii_cube():
    surface = TriangulatedSurface.from_stl(CUBE_STL)

    assert surface.path == CUBE_STL.resolve()
    assert surface.bounds == (-0.5, 0.5, -0.5, 0.5, -0.5, 0.5)
    assert surface.triangles.shape == (12, 3, 3)
    assert len(surface.sha256) == 64
    assert not surface.triangles.flags.writeable


def test_rejects_open_surface(tmp_path):
    surface_file = tmp_path / "open.stl"
    surface_file.write_text(
        """solid open
facet normal 0 0 1
outer loop
vertex 0 0 0
vertex 1 0 0
vertex 0 1 0
endloop
endfacet
facet normal 0 0 1
outer loop
vertex 1 0 0
vertex 1 1 0
vertex 0 1 0
endloop
endfacet
facet normal 0 0 1
outer loop
vertex 0 0 1
vertex 1 0 1
vertex 0 1 1
endloop
endfacet
facet normal 0 0 1
outer loop
vertex 1 0 1
vertex 1 1 1
vertex 0 1 1
endloop
endfacet
endsolid open
"""
    )

    with pytest.raises(ValueError, match="not watertight"):
        TriangulatedSurface.from_stl(surface_file)


def test_accepts_non_axis_aligned_closed_surface(tmp_path):
    """A closed watertight non-axis-aligned surface loads on the general path."""
    surface_file = tmp_path / "tetrahedron.stl"
    triangles = np.asarray(
        [
            ((0, 0, 0), (0, 1, 0), (1, 0, 0)),
            ((0, 0, 0), (1, 0, 0), (0, 0, 1)),
            ((0, 0, 0), (0, 0, 1), (0, 1, 0)),
            ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        ],
        dtype=float,
    )
    lines = ["solid tetrahedron\n"]
    for triangle in triangles:
        lines.extend(("facet normal 0 0 0\n", "outer loop\n"))
        lines.extend(f"vertex {x:g} {y:g} {z:g}\n" for x, y, z in triangle)
        lines.extend(("endloop\n", "endfacet\n"))
    lines.append("endsolid tetrahedron\n")
    surface_file.write_text("".join(lines))

    surface = TriangulatedSurface.from_stl(surface_file)
    assert surface.kind == "general"
    assert surface.bounds == (0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
    assert surface.triangles.shape == (4, 3, 3)
