"""Physical volumes and nominal meshing sizes survive all FVM output paths."""

import xml.etree.ElementTree as ET

import numpy as np
import pytest

from source.solvers.fvm import BoundaryConfig, FVMSetup, FVMSolver, OutputConfig
from source.solvers.fvm.factory import _save_generated_mesh
from source.solvers.fvm.io.partitioned import write_partition_vtu
from source.solvers.fvm.io.vtk_exporter import VTKExporter, mesh_cell_fields
from source.solvers.fvm.mesh.geometry import compute_mesh_geometry
from source.solvers.fvm.mesh.partition import localize_mesh_and_geometry
from source.solvers.fvm.mesh.rectilinear import box_mesh_3d


@pytest.fixture
def sized_mesh():
    # Distinct volumes reveal partition-order mistakes. Nominal sizes are
    # deliberately different from cube-root volume, as on projected cells.
    mesh = box_mesh_3d(
        np.array([0.0, 0.1, 0.4, 1.0]), np.array([0.0, 0.5, 1.0]), np.array([0.0, 1.0])
    )
    mesh["cell_sizes"] = np.linspace(0.25, 0.5, mesh["n_cells"], dtype=np.float32)
    mesh["cell_levels"] = np.arange(mesh["n_cells"], dtype=np.int8)
    mesh["boundary_layer_index"] = np.arange(mesh["n_cells"], dtype=np.int32) % 2
    return mesh


def _assert_geometry_fields(grid, mesh, geometry, ids=None):
    ids = np.arange(mesh["n_cells"]) if ids is None else ids
    expected = mesh_cell_fields(mesh, geometry["cell_volume"])
    for name, values in expected.items():
        np.testing.assert_allclose(grid.cell_data[name], values[ids], rtol=1e-6)
        assert name not in grid.point_data
    # Independent VTK integration verifies the exported physical volumes.
    vtk_volumes = grid.compute_cell_sizes(length=False, area=False).cell_data["Volume"]
    np.testing.assert_allclose(grid.cell_data["cell_volume"], vtk_volumes, rtol=1e-6)


@pytest.mark.parametrize("asynchronous", [False, True])
def test_solver_time_steps_include_mesh_geometry(tmp_path, sized_mesh, asynchronous):
    import pyvista as pv

    setup = FVMSetup(
        case_name="sizes",
        output=OutputConfig(asynchronous=asynchronous),
        boundaries=[BoundaryConfig.wall(patch["name"]) for patch in sized_mesh["boundary"]],
    )
    with FVMSolver(setup, str(tmp_path), mesh_data=sized_mesh) as solver:
        for index in range(2):
            output = tmp_path / f"fields-{index}.vtu"
            solver.write_vtk(str(output))
            solver.flush_output()
            _assert_geometry_fields(pv.read(output), sized_mesh, solver.geo_data)


def test_mesh_backup_includes_physical_and_equivalent_sizes(tmp_path, sized_mesh):
    import pyvista as pv

    _save_generated_mesh(sized_mesh, tmp_path, OutputConfig())
    geometry = compute_mesh_geometry(sized_mesh, compute_lsq=False)
    _assert_geometry_fields(pv.read(tmp_path / "fvm" / "mesh.vtu"), sized_mesh, geometry)


@pytest.mark.parametrize("interpolate_to_points", [False, True])
def test_smoothing_keeps_geometry_as_cell_data(tmp_path, sized_mesh, interpolate_to_points):
    import pyvista as pv

    geometry = compute_mesh_geometry(sized_mesh, compute_lsq=False)
    fields = mesh_cell_fields(sized_mesh, geometry["cell_volume"])
    fields["velocity"] = np.ones((sized_mesh["n_cells"], 3))
    output = tmp_path / "smooth.vtu"
    VTKExporter(sized_mesh, OutputConfig(point_interpolation="boundary_weighted")).export(
        str(output), fields, interpolate_to_points=interpolate_to_points
    )
    grid = pv.read(output)
    _assert_geometry_fields(grid, sized_mesh, geometry)
    assert "velocity" in grid.point_data


class _SerialPublication:
    """Exercise rank-piece publication after fields already contain halo values."""

    def allgather(self, value):
        return [value]

    def bcast(self, value, root=0):
        return value


@pytest.mark.parametrize("ghost_layers", [0, 1])
def test_partition_collections_expose_geometry_in_local_cell_order(
    tmp_path, sized_mesh, monkeypatch, ghost_layers
):
    import pyvista as pv

    geometry = compute_mesh_geometry(sized_mesh, compute_lsq=False)
    for rank in (1, 0):  # Root publishes the collection after both pieces exist.
        local, geo, partition = localize_mesh_and_geometry(
            sized_mesh,
            geometry,
            rank=rank,
            size=2,
            include_visualization_ghosts=bool(ghost_layers),
        )
        # Localization already selected authoritative global geometry for
        # every local/halo row; no MPI exchange is required for these fields.
        monkeypatch.setattr(type(partition), "exchange_halo", lambda *args: None)
        fields = mesh_cell_fields(local, geo["cell_volume"])
        collection = write_partition_vtu(
            tmp_path,
            "sizes",
            local,
            partition,
            fields,
            _SerialPublication(),
            output=OutputConfig(ghost_layers=ghost_layers),
        )
        grid = pv.read(tmp_path / f"sizes-rank-{rank:05d}.vtu")
        ids = partition.local_global_ids if ghost_layers else partition.owned_global_ids
        _assert_geometry_fields(grid, sized_mesh, geometry, ids)
    declared = {
        entry.attrib["Name"] for entry in ET.parse(collection).findall(".//PCellData/PDataArray")
    }
    assert set(mesh_cell_fields(sized_mesh, geometry["cell_volume"])) <= declared
    grid = pv.read(collection)
    _assert_geometry_fields(grid, sized_mesh, geometry, grid.cell_data["global_cell_id"])
