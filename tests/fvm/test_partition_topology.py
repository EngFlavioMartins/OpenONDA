"""Deterministic domain-partition topology gates."""

from __future__ import annotations

import numpy as np
import pytest

from source.solvers.FVM.config.types import OutputSetup
from source.solvers.FVM.io.partitioned import (
    reconstruct_partition_checkpoint,
    write_partition_checkpoint,
    write_partition_vtu,
)
from source.solvers.FVM.mesh import geometry
from source.solvers.FVM.mesh.partition import (
    CellPartition,
    localize_mesh_and_geometry,
    ownership_ranges,
)
from source.solvers.FVM.mesh.rectilinear import box_mesh_3d

from ._structured_mesh import structured_box


class _SerialComm:
    def Barrier(self):
        return None


@pytest.mark.parametrize("size", (1, 2, 4))
def test_partitions_cover_owned_cells_and_physical_faces_once(size):
    mesh = structured_box(7, 3, 2)
    partitions = [CellPartition.from_mesh_data(mesh, rank, size) for rank in range(size)]

    owned = np.concatenate([partition.owned_global_ids for partition in partitions])
    physical_faces = np.concatenate(
        [partition.physical_boundary_face_global_ids for partition in partitions]
    )
    np.testing.assert_array_equal(owned, np.arange(mesh["n_elements"]))
    np.testing.assert_array_equal(
        np.sort(physical_faces), np.arange(mesh["n_interior_faces"], mesh["n_faces"])
    )


@pytest.mark.parametrize("size", (2, 4))
def test_processor_faces_and_halo_schedules_are_reciprocal(size):
    mesh = structured_box(8, 3, 2)
    partitions = [CellPartition.from_mesh_data(mesh, rank, size) for rank in range(size)]
    processor_occurrences = np.concatenate(
        [partition.processor_face_global_ids for partition in partitions]
    )
    _, counts = np.unique(processor_occurrences, return_counts=True)
    assert np.all(counts == 2)

    for source in partitions:
        for destination in range(size):
            np.testing.assert_array_equal(
                source.halo.send_global_ids[destination],
                partitions[destination].halo.receive_global_ids[source.rank],
            )
        local_ids = source.global_to_local(source.local_global_ids)
        np.testing.assert_array_equal(local_ids, np.arange(len(source.local_global_ids)))


def test_balanced_ranges_handle_more_ranks_than_cells():
    offsets = ownership_ranges(3, 5)
    np.testing.assert_array_equal(offsets, np.array([0, 1, 2, 3, 3, 3]))


def test_polyhedral_visualization_mesh_contains_complete_halo_cells():
    mesh = structured_box(4, 2, 2)
    geo = geometry.compute_mesh_geometry(mesh)

    local_mesh, _local_geo, partition = localize_mesh_and_geometry(mesh, geo, 0, 2)
    visualization = local_mesh["_visualization_mesh"]

    assert len(partition.ghost_global_ids) > 0
    assert visualization["n_elements"] == len(partition.local_global_ids)
    assert np.all(np.diff(visualization["cell_face_offsets"]) == 6)
    assert np.all(visualization["cell_faces"] >= 0)


def test_rectilinear_visualization_mesh_uses_compact_native_hexes():
    axis = np.linspace(0.0, 1.0, 5)
    mesh = box_mesh_3d(axis, axis, axis)
    geo = geometry.compute_mesh_geometry(mesh)

    local_mesh, _local_geo, partition = localize_mesh_and_geometry(mesh, geo, 0, 2)
    visualization = local_mesh["_visualization_mesh"]

    assert visualization["cell_vertices"].shape == (
        len(partition.local_global_ids),
        8,
    )
    assert visualization["faces"].size == 0
    assert np.all(visualization["cell_type_codes"] == 5)


@pytest.mark.parametrize("size", (2, 4))
def test_localized_mesh_has_complete_owned_stencils(size):
    mesh = structured_box(8, 3, 2)
    geo = geometry.compute_mesh_geometry(mesh)
    localizations = [localize_mesh_and_geometry(mesh, geo, rank, size) for rank in range(size)]
    for local_mesh, local_geo, partition in localizations:
        n_owned = len(partition.owned_global_ids)
        incident = np.zeros(n_owned, dtype=int)
        owners = local_mesh["owners"]
        neighbours = local_mesh["neighbours"]
        np.add.at(incident, owners[owners < n_owned], 1)
        owned_neighbours = neighbours < n_owned
        np.add.at(incident, neighbours[owned_neighbours], 1)
        assert np.all(incident == 6)
        np.testing.assert_array_equal(
            local_geo["element_volumes"][:n_owned],
            geo["element_volumes"][partition.owned_global_ids],
        )
        assert local_mesh["global_cell_ids"].shape[0] == local_mesh["n_elements"]
        visualization = local_mesh["_visualization_mesh"]
        assert visualization["n_elements"] == len(partition.local_global_ids)
        np.testing.assert_array_equal(
            visualization["global_cell_ids"],
            partition.local_global_ids,
        )
        if len(visualization["cell_faces"]):
            assert np.all(np.diff(visualization["cell_face_offsets"]) == 6)
        else:
            assert visualization["cell_vertices"].shape == (
                len(partition.local_global_ids),
                8,
            )
        assert np.all(visualization["global_point_ids"] >= 0)


def test_partition_checkpoint_round_trip(tmp_path):
    mesh = structured_box(3, 2, 2)
    partition = CellPartition.from_mesh_data(mesh, 0, 1)
    velocity = np.column_stack(
        (partition.local_global_ids, 2 * partition.local_global_ids, -partition.local_global_ids)
    ).astype(float)
    pressure = partition.local_global_ids.astype(float) ** 2

    write_partition_checkpoint(
        tmp_path, partition, {"velocity": velocity, "pressure": pressure}, _SerialComm()
    )
    restored = reconstruct_partition_checkpoint(tmp_path)

    np.testing.assert_array_equal(restored["velocity"], velocity)
    np.testing.assert_array_equal(restored["pressure"], pressure)


def test_partition_vtu_writes_cell_data_and_parallel_metadata(tmp_path):
    pytest.importorskip("pyvista", reason="partitioned VTK output requires PyVista")
    mesh = structured_box(2, 2, 2)
    partition = CellPartition.from_mesh_data(mesh, 0, 1)
    velocity = np.column_stack((partition.local_global_ids,) * 3).astype(float)
    pressure = partition.local_global_ids.astype(float)

    collection = write_partition_vtu(
        tmp_path,
        "state-0001",
        mesh,
        partition,
        {"velocity": velocity, "pressure": pressure},
        _SerialComm(),
    )

    assert collection.exists()
    assert (tmp_path / "state-0001-rank-00000.vtu").exists()
    text = collection.read_text()
    assert 'GhostLevel="1"' in text
    assert 'Name="velocity" NumberOfComponents="3"' in text
    assert 'Name="vtkGhostType" NumberOfComponents="1"' in text
    assert 'Name="GlobalCellIds" NumberOfComponents="1"' in text
    assert 'Name="GlobalPointIds" NumberOfComponents="1"' in text
    assert 'Piece Source="state-0001-rank-00000.vtu"' in text

    import pyvista as pv

    data = pv.read(tmp_path / "state-0001-rank-00000.vtu")
    assert data.n_cells == len(partition.local_global_ids)
    assert "velocity" in data.cell_data
    assert "velocity" not in data.point_data
    np.testing.assert_array_equal(data.cell_data["GlobalCellIds"], partition.local_global_ids)
    assert np.count_nonzero(data.cell_data["vtkGhostType"]) == 0
    assert "GlobalPointIds" in data.point_data
    smooth = data.cell_data_to_point_data()
    assert "velocity" in smooth.point_data

    parallel = pv.read(collection)
    assert parallel.n_cells == mesh["n_elements"]
    assert "pressure" in parallel.cell_data


def test_partition_vtu_can_omit_visualization_ghosts(tmp_path):
    pytest.importorskip("pyvista", reason="partitioned VTK output requires PyVista")
    mesh = structured_box(2, 2, 2)
    partition = CellPartition.from_mesh_data(mesh, 0, 1)

    collection = write_partition_vtu(
        tmp_path,
        "owned-only",
        mesh,
        partition,
        {"pressure": partition.local_global_ids.astype(float)},
        _SerialComm(),
        output=OutputSetup(ghost_layers=0),
    )

    text = collection.read_text(encoding="utf-8")
    assert 'GhostLevel="0"' in text
    assert "vtkGhostType" not in text
    assert 'Name="GlobalCellIds"' in text


def test_partition_vtu_float32_fields(tmp_path):
    pv = pytest.importorskip("pyvista", reason="partitioned VTK output requires PyVista")
    mesh = structured_box(2, 2, 2)
    partition = CellPartition.from_mesh_data(mesh, 0, 1)

    collection = write_partition_vtu(
        tmp_path,
        "float32",
        mesh,
        partition,
        {"pressure": partition.local_global_ids.astype(float)},
        _SerialComm(),
        output=OutputSetup(precision="float32"),
    )

    assert 'type="Float32" Name="pressure"' in collection.read_text(encoding="utf-8")
    data = pv.read(tmp_path / "float32-rank-00000.vtu")
    assert data.cell_data["pressure"].dtype == np.float32
