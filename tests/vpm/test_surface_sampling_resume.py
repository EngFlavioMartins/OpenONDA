"""Exact legacy geometry continuation; no solver, Taichi initialization or GPU."""

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import pyvista as pv

from source.solvers.vpm.config.artifacts import Samplers
from source.solvers.vpm.io.manifest import _sampler_identity
from source.solvers.vpm.io.sampler import OutputEvent, OutputManager, SamplingContext
from source.solvers.vpm.io.sampling.field_samplers import SurfaceSampler


def make_sampler(axis=2, **changes):
    args = {
        "point": [0.0, 0.0, 0.0], "normal": np.eye(3)[axis],
        "bounds": [1.6, 8.0, -2.0, 2.0], "spacing": 0.1,
        "file_name": "plane", "include_derivatives": False,
    }
    args.update(changes)
    return SurfaceSampler(**args)


def legacy_frame(sampler, path, *, dtype=np.float32):
    """Independent reproduction of the historical writer, not the new helper."""
    a = np.arange(sampler.bounds[0], sampler.bounds[1] + sampler.spacing / 2, sampler.spacing)
    b = np.arange(sampler.bounds[2], sampler.bounds[3] + sampler.spacing / 2, sampler.spacing)
    a, b = np.meshgrid(a, b, indexing="ij")
    axis = int(np.argmax(abs(sampler.normal)))
    coordinates = [None, None, None]
    coordinates[axis] = np.full(a.shape, sampler.point[axis], dtype=np.float32)
    free = [i for i in range(3) if i != axis]
    coordinates[free[0]] = a.astype(np.float32)
    coordinates[free[1]] = b.astype(np.float32)
    grid = pv.StructuredGrid(*(values.astype(dtype) for values in coordinates))
    grid.save(path)
    return grid


def metadata(frame):
    return json.loads(str(frame.field_data["openonda_sampling_grid"][0]))


def fake_sample(sampler):
    result = {
        f"position_{axis}": sampler.grid_points[:, i].copy()
        for i, axis in enumerate("xyz")
    }
    for name in ("velocity", "vorticity"):
        result.update({f"{name}_{axis}": np.zeros(sampler._n_points) for axis in "xyz"})
    return result


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_exact_historical_geometry_and_lossless_storage_type(tmp_path, axis, dtype):
    sampler = make_sampler(axis)
    path = tmp_path / "legacy.vts"
    saved = legacy_frame(sampler, path, dtype=dtype)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="sampling grid differs"):
        sampler.validate_existing_vtk(path)
    sampler.prepare_existing_vtk(path)
    assert sampler.grid_layout == "legacy_arange_v0"
    assert sampler.grid_resume_admission["frame_sha256"] == hashlib.sha256(before).hexdigest()
    assert "unrecorded parameter identity" in sampler.grid_resume_admission["provenance"]
    actual = np.column_stack([
        sampler.grid_points[:, i].reshape(sampler._grid_shape).ravel(order="F")
        for i in range(3)
    ])
    np.testing.assert_array_equal(actual, saved.points)
    sampler.validate_existing_vtk(path)
    assert path.read_bytes() == before
    assert _sampler_identity(sampler)["grid_layout"] == "legacy_arange_v0"


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_bounded_round_trip_is_unchanged(tmp_path, dtype):
    sampler = make_sampler(bounds=[-1, 1, -1, 1], spacing=0.31)
    shape = sampler._grid_shape
    previous = pv.StructuredGrid(*[
        sampler.grid_points[:, i].reshape(shape).astype(dtype) for i in range(3)
    ])
    path = tmp_path / "bounded.vts"
    previous.save(path)
    before = sampler.grid_points.copy()
    sampler.prepare_existing_vtk(path)
    assert sampler.grid_layout == "bounded_uniform_v1"
    np.testing.assert_array_equal(sampler.grid_points, before)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_writer_records_effective_policy_and_second_resume(tmp_path, axis):
    sampler = make_sampler(axis)
    legacy_frame(sampler, tmp_path / "old.vts")
    sampler.prepare_existing_vtk(tmp_path / "old.vts")
    sampler.sample = lambda solver: fake_sample(sampler)
    sampler.save_vtp(SimpleNamespace(write_precision="f32"), tmp_path / "new.vts")
    written = pv.read(tmp_path / "new.vts")
    contract = metadata(written)
    assert contract["grid_layout"] == "legacy_arange_v0"
    assert contract["configuration"]["bounds"] == sampler.bounds.tolist()
    assert contract["coordinates_sha256"] == hashlib.sha256(
        np.ascontiguousarray(written.points, dtype="<f8").tobytes()
    ).hexdigest()
    restarted = make_sampler(axis)
    restarted.prepare_existing_vtk(tmp_path / "new.vts")
    assert restarted.grid_layout == "legacy_arange_v0"
    assert restarted.grid_resume_admission["provenance"] == "versioned_frame_metadata"
    np.testing.assert_array_equal(restarted.grid_points, sampler.grid_points)
    fresh = make_sampler(axis)
    assert fresh.grid_layout == "bounded_uniform_v1"
    assert not np.array_equal(fresh.grid_points, sampler.grid_points)


@pytest.mark.parametrize("changes", [
    {"bounds": [1.7, 8.0, -2, 2]}, {"spacing": 0.099},
    {"point": [0, 0, 0.1]}, {"normal": [1, 0, 0]},
])
def test_real_geometry_changes_reject_without_mutation(tmp_path, changes):
    path = tmp_path / "old.vts"
    legacy_frame(make_sampler(), path)
    sampler = make_sampler(**changes)
    before = sampler.grid_points.copy()
    with pytest.raises(ValueError, match="sampling grid differs"):
        sampler.prepare_existing_vtk(path)
    np.testing.assert_array_equal(sampler.grid_points, before)
    assert sampler.grid_layout == "bounded_uniform_v1"
    assert not hasattr(sampler, "grid_resume_admission")


@pytest.mark.parametrize("change", ["order", "dimensions", "sub_f32_ulp", "nonfinite"])
def test_coordinate_changes_cannot_be_hidden_by_casting(tmp_path, change):
    sampler = make_sampler()
    path = tmp_path / "old.vts"
    saved = legacy_frame(sampler, path, dtype=np.float64)
    if change == "order":
        saved.points[[0, 1]] = saved.points[[1, 0]]
    elif change == "dimensions":
        saved.dimensions = (41, 65, 1)
    elif change == "sub_f32_ulp":
        saved.points[0, 0] = np.nextafter(saved.points[0, 0], np.inf)
    else:
        saved.points[0, 0] = np.nan
    saved.save(path)
    with pytest.raises(ValueError, match="sampling grid|coordinates"):
        sampler.prepare_existing_vtk(path)
    assert sampler.grid_layout == "bounded_uniform_v1"


@pytest.mark.parametrize("change", ["bounded_policy", "unknown_policy", "digest", "normal", "json"])
def test_explicit_metadata_never_falls_back(tmp_path, change):
    sampler = make_sampler()
    path = tmp_path / "old.vts"
    saved = legacy_frame(sampler, path)
    contract = sampler._grid_metadata("legacy_arange_v0", sampler._make_grid("legacy_arange_v0"))
    if change == "bounded_policy":
        contract["grid_layout"] = "bounded_uniform_v1"
    elif change == "unknown_policy":
        contract["grid_layout"] = "future_layout_v99"
    elif change == "digest":
        contract["coordinates_sha256"] = "0" * 64
    elif change == "normal":
        # This leaves observed coordinates identical; the explicit declaration
        # is nevertheless part of the new, checksum-bound frame contract.
        contract["configuration"]["normal"] = [0, 0, -1]
    encoded = "{invalid" if change == "json" else json.dumps(contract)
    saved.field_data["openonda_sampling_grid"] = np.array([encoded])
    saved.save(path)
    with pytest.raises(ValueError, match="sampling grid|metadata|layout"):
        sampler.prepare_existing_vtk(path)
    assert sampler.grid_layout == "bounded_uniform_v1"


def test_metadata_free_frame_cannot_prove_redundant_normal_declaration(tmp_path):
    path = tmp_path / "old.vts"
    legacy_frame(make_sampler(), path)
    sampler = make_sampler(normal=[0, 0, -1])
    sampler.prepare_existing_vtk(path)
    assert "unrecorded parameter identity" in sampler.grid_resume_admission["provenance"]


def test_restart_preflights_retained_geometry_before_evolution_or_index_changes(tmp_path):
    directory = tmp_path / "samples"
    directory.mkdir()
    sampler = make_sampler()
    legacy_frame(sampler, directory / "plane_000270.vts")
    future = make_sampler(spacing=0.2)
    future.sample = lambda solver: fake_sample(future)
    future.save_vtp(SimpleNamespace(write_precision="f32"), directory / "plane_000290.vts")
    OutputManager._write_pvd(directory, "plane", [
        (10.8, "plane_000270.vts"), (11.6, "plane_000290.vts"),
    ])
    config = Samplers(samples=(sampler,))
    solver = SimpleNamespace(case=SimpleNamespace(samplers=config), case_dir=tmp_path)
    manager = OutputManager(solver)
    sampler.sample = lambda solver: pytest.fail("Restart admission must not evaluate a field")
    manager.rewind_histories(11.12)
    assert sampler.grid_layout == "legacy_arange_v0"
    assert manager._read_pvd(directory, "plane") == [(10.8, "plane_000270.vts")]
    assert (directory / "plane_000270.vts").is_file()


def test_last_retained_explicit_policy_cannot_fall_back_to_an_older_frame(tmp_path):
    directory = tmp_path / "samples"
    directory.mkdir()
    sampler = make_sampler()
    legacy_frame(sampler, directory / "plane_000270.vts")
    declared = make_sampler(spacing=0.2)
    declared.sample = lambda solver: fake_sample(declared)
    declared.save_vtp(SimpleNamespace(write_precision="f32"), directory / "plane_000280.vts")
    OutputManager._write_pvd(directory, "plane", [
        (10.8, "plane_000270.vts"), (11.2, "plane_000280.vts"),
    ])
    before = (directory / "plane.pvd").read_bytes()
    config = Samplers(samples=(sampler,))
    solver = SimpleNamespace(case=SimpleNamespace(samplers=config), case_dir=tmp_path)
    with pytest.raises(ValueError, match="sampling grid differs"):
        OutputManager(solver).rewind_histories(11.2)
    assert (directory / "plane.pvd").read_bytes() == before
    assert sampler.grid_layout == "bounded_uniform_v1"


def test_incompatible_restart_geometry_preserves_indexes_and_histories(tmp_path):
    directory = tmp_path / "samples"
    directory.mkdir()
    legacy_frame(make_sampler(spacing=0.2), directory / "plane_000270.vts")
    OutputManager._write_pvd(directory, "plane", [(10.8, "plane_000270.vts")])
    index_before = (directory / "plane.pvd").read_bytes()
    csv = directory / "plane.csv"
    csv.write_text("time,value\n10.8,1\n11.2,2\n")
    sampler = make_sampler()
    config = Samplers(samples=(sampler,))
    solver = SimpleNamespace(case=SimpleNamespace(samplers=config), case_dir=tmp_path)
    manager = OutputManager(solver)
    with pytest.raises(ValueError, match="sampling grid differs"):
        manager.rewind_histories(11.12)
    assert (directory / "plane.pvd").read_bytes() == index_before
    assert csv.read_text() == "time,value\n10.8,1\n11.2,2\n"


def test_first_write_admits_legacy_without_a_separate_restart_hook(tmp_path):
    sampler = make_sampler()
    legacy_frame(sampler, tmp_path / "plane_000270.vts")
    OutputManager._write_pvd(tmp_path, "plane", [(10.8, "plane_000270.vts")])
    config = Samplers(samples=(sampler,))
    solver = SimpleNamespace(case=SimpleNamespace(samplers=config), write_precision="f32")
    manager = OutputManager(solver)
    sampler.sample = lambda solver: fake_sample(sampler)
    manager._write(sampler, SamplingContext(
        solver=solver, output_directory=tmp_path, step=280, time=11.2,
        event=OutputEvent.ACCEPTED_STEP, continuing_output=True,
    ))
    saved = pv.read(tmp_path / "plane_000280.vts")
    previous = pv.read(tmp_path / "plane_000270.vts")
    np.testing.assert_array_equal(saved.points, previous.points)
    assert metadata(saved)["grid_layout"] == "legacy_arange_v0"
