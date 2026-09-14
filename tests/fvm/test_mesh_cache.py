"""Cached meshes must not survive a changed meshing specification."""

from pathlib import Path

import numpy as np

from openonda.fvm import mesher as msh


def test_cache_hit_and_changed_resolution(tmp_path, monkeypatch):
    cube = (
        Path(__file__).resolve().parents[2]
        / "tutorials/coupled_fvm_vpm/02_cube_flow/assets/cube.stl"
    )
    builds = []

    def request(size):
        mesher = msh.CartesianMesher(
            domain=msh.BoxDomain(
                bounds=(-1.5, 1.5, -1.5, 1.5, -1.5, 1.5),
                patches=msh.BoxPatches("xmin", "xmax", "ymin", "ymax", "zmin", "zmax"),
            ),
            surfaces=(msh.STLSurface(cube, patch="cube"),),
            max_cell_size=size,
        )

        def build():
            builds.append(size)
            return msh.structured_box(2, 2, 2)

        monkeypatch.setattr(mesher, "build", build)
        return msh.CachedMesh(mesher, tmp_path / "mesh.npz")

    first = request(0.5).build()
    same = request(0.5).build()
    assert builds == [0.5]
    np.testing.assert_array_equal(first["vertex_position"], same["vertex_position"])
    assert first["cartesian_cache_identity"] == same["cartesian_cache_identity"]
    changed = request(0.25).build()
    assert builds == [0.5, 0.25]
    assert changed["cartesian_cache_identity"] != first["cartesian_cache_identity"]
