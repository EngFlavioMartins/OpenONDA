"""Native FVM construction owns portable mesh reuse and invalidation."""

from pathlib import Path
import shutil

import pytest

from openonda.fvm import mesher as msh
from source.solvers.fvm import FVMCase, FVMSolver, InitialFields
from source.solvers.fvm.factory import _materialize_mesh, create_fvm_solver
from source.solvers.fvm.io.mesh_storage import load_native_mesh, save_native_mesh
from source.solvers.fvm.mesh.cartesian.config import Refinement
from tests.fvm.cartesian_acceptance_fixtures import _write_ascii_stl, box_triangles
from tests.fvm.test_restart_and_diagnostics import _setup
from tests.support.fvm_mesh import structured_box

CUBE = (
    Path(__file__).resolve().parents[2] / "tutorials/coupled_fvm_vpm/02_cube_flow/assets/cube.stl"
)
PATCHES = msh.BoxPatches("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")


def mesher(
    *,
    extruded=False,
    size=0.5,
    surface=CUBE,
    levels=(-0.5, 0.0, 0.5),
    patches=PATCHES,
    refinements=(),
):
    source = msh.CartesianMesher(
        domain=msh.BoxDomain(bounds=(-1.5, 1.5, -1.5, 1.5, -1.5, 1.5), patches=PATCHES),
        surfaces=(msh.STLSurface(surface, patch="body"),),
        max_cell_size=size,
        refinements=refinements,
    )
    if not extruded:
        return source
    return msh.ExtrudedCartesianMesher(
        source=source,
        domain=msh.BoxDomain(bounds=(-1.5, 1.5, -1.5, 1.5, levels[0], levels[-1]), patches=patches),
        levels=levels,
    )


@pytest.mark.parametrize("extruded", [False, True], ids=["cartesian", "extruded"])
@pytest.mark.parametrize("public_case", [False, True], ids=["factory", "public_case"])
def test_native_construction_reuses_its_saved_mesh(tmp_path, monkeypatch, extruded, public_case):
    request = mesher(extruded=extruded)
    builds = []
    monkeypatch.setattr(request, "build", lambda: builds.append(1) or structured_box(2, 2, 2))
    for _ in range(2):
        if public_case:
            solver = FVMSolver(
                FVMCase(
                    name="cache_test",
                    mesh=request,
                    directory=tmp_path,
                    boundaries=tuple(_setup().boundaries),
                    initial_conditions=InitialFields(velocity=(0.2, 0.0, 0.0)),
                )
            )
        else:
            solver = create_fvm_solver(_setup(), case_dir=tmp_path, mesh=request)
        solver.close()
    assert builds == [1]
    saved = load_native_mesh(tmp_path / "solution/fvm/mesh.npz")
    assert "mesh_cache_identity" in saved
    assert not (tmp_path / "constant").exists()


@pytest.mark.parametrize(
    "change",
    [
        {"size": 0.25},
        {"levels": (-0.5, -0.25, 0.5)},
        {"levels": (-0.75, 0.0, 0.75)},
        {"patches": msh.BoxPatches("xmin", "xmax", "ymin", "ymax", "bottom", "top")},
    ],
    ids=["resolution", "layer_distribution", "span", "patches"],
)
def test_changed_meshing_inputs_rebuild(tmp_path, monkeypatch, change):
    path = tmp_path / "mesh.npz"
    builds = []
    first = mesher(extruded=True)
    monkeypatch.setattr(first, "build", lambda: builds.append(1) or structured_box(2, 2, 2))
    save_native_mesh(_materialize_mesh(first, is_root=True, cache_path=path), path)
    changed = mesher(extruded=True, **change)
    monkeypatch.setattr(changed, "build", lambda: builds.append(2) or structured_box(3, 2, 2))
    result = _materialize_mesh(changed, is_root=True, cache_path=path)
    assert builds == [1, 2]
    assert result["n_cells"] == 12


def test_copied_geometry_and_archive_reuse_without_absolute_paths(tmp_path, monkeypatch):
    original = mesher()
    monkeypatch.setattr(original, "build", lambda: structured_box(2, 2, 2))
    path = tmp_path / "first/mesh.npz"
    save_native_mesh(_materialize_mesh(original, is_root=True, cache_path=path), path)
    relocated = tmp_path / "different case"
    relocated.mkdir()
    shutil.copy2(CUBE, relocated / "geometry.stl")
    shutil.copy2(path, relocated / "mesh.npz")
    copied = mesher(surface=relocated / "geometry.stl")
    monkeypatch.setattr(copied, "build", lambda: pytest.fail("matching portable mesh was rebuilt"))
    result = _materialize_mesh(copied, is_root=True, cache_path=relocated / "mesh.npz")
    assert result["n_cells"] == 8


def test_changed_geometry_rebuilds_at_the_same_input_path(tmp_path, monkeypatch):
    geometry = tmp_path / "body.stl"
    triangles = box_triangles((-0.4, 0.4, -0.4, 0.4, -0.4, 0.4))
    _write_ascii_stl(geometry, triangles, "body")
    original = mesher(surface=geometry)
    monkeypatch.setattr(original, "build", lambda: structured_box(2, 2, 2))
    path = tmp_path / "mesh.npz"
    save_native_mesh(_materialize_mesh(original, is_root=True, cache_path=path), path)
    _write_ascii_stl(geometry, triangles + 0.1, "body")
    changed = mesher(surface=geometry)
    monkeypatch.setattr(changed, "build", lambda: structured_box(3, 2, 2))
    result = _materialize_mesh(changed, is_root=True, cache_path=path)
    assert result["n_cells"] == 12


def test_custom_mesher_retains_its_own_construction_semantics(tmp_path):
    class CustomMesher(msh.CartesianMesher):
        def __call__(self):
            return structured_box(3, 2, 2)

        def build(self):
            pytest.fail("custom callable was bypassed")

    request = CustomMesher(
        domain=msh.BoxDomain(bounds=(-1.5, 1.5, -1.5, 1.5, -1.5, 1.5), patches=PATCHES),
        surfaces=(msh.STLSurface(CUBE, patch="body"),),
        max_cell_size=0.5,
    )
    path = tmp_path / "mesh.npz"
    for _ in range(2):
        result = _materialize_mesh(request, is_root=True, cache_path=path)
        assert result["n_cells"] == 12
        assert "mesh_cache_identity" not in result
        save_native_mesh(result, path)


def test_custom_refinement_builds_without_automatic_input_serialization(tmp_path, monkeypatch):
    class HalfBox(Refinement):
        name = "half_box"
        bounds = (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
        cell_size = 0.25

        def contains(self, points):
            return points[:, 0] > 0.0

    request = mesher(refinements=(HalfBox(),))
    builds = []
    monkeypatch.setattr(request, "build", lambda: builds.append(1) or structured_box(2, 2, 2))
    path = tmp_path / "mesh.npz"
    for _ in range(2):
        result = _materialize_mesh(request, is_root=True, cache_path=path)
        assert "mesh_cache_identity" not in result
        save_native_mesh(result, path)
    assert builds == [1, 1]


@pytest.mark.parametrize("corrupt", [False, True], ids=["untagged", "corrupt"])
def test_unusable_automatic_cache_rebuilds_without_changing_explicit_file_loading(
    tmp_path, monkeypatch, corrupt
):
    path = tmp_path / "mesh.npz"
    if corrupt:
        path.write_bytes(b"incomplete archive")
    else:
        save_native_mesh(structured_box(3, 2, 2), path)
    request = mesher()
    monkeypatch.setattr(request, "build", lambda: structured_box(2, 2, 2))
    result = _materialize_mesh(request, is_root=True, cache_path=path)
    assert result["n_cells"] == 8
    if corrupt:
        with pytest.raises(ValueError):
            _materialize_mesh(path, is_root=True, cache_path=path)
    else:
        loaded = _materialize_mesh(path, is_root=True, cache_path=path)
        assert loaded["n_cells"] == 12


def test_worker_does_not_read_or_generate_mesh(tmp_path):
    assert _materialize_mesh(object(), is_root=False, cache_path=tmp_path / "mesh.npz") is None
