"""Current fixed-grid continuation validates metadata before mutating output."""

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import pyvista as pv

from source.solvers.vpm.config.artifacts import Samplers
from source.solvers.vpm.io.manifest import _sampler_identity
from source.solvers.vpm.io.sampler import OutputEvent, OutputManager, SamplingContext
from source.solvers.vpm.io.sampling.field_samplers import SAMPLER_CSV_COLUMNS, SurfaceSampler


def make_sampler(axis=2, **changes):
    args = {
        "point": [0.0, 0.0, 0.0],
        "normal": np.eye(3)[axis],
        "bounds": [1.6, 8.0, -2.0, 2.0],
        "spacing": 0.1,
        "file_name": "plane",
        "include_derivatives": False,
    }
    args.update(changes)
    return SurfaceSampler(**args)


def fake_sample(sampler):
    result = {f"position_{axis}": sampler.grid_points[:, i].copy() for i, axis in enumerate("xyz")}
    result.update(
        {name: np.zeros(sampler._n_points) for name in SAMPLER_CSV_COLUMNS if name not in result}
    )
    return result


def write_frame(sampler, path, *, time=None):
    sampler.sample = lambda solver: fake_sample(sampler)
    sampler.save_vtp(SimpleNamespace(write_precision="f32"), path, time=time)
    return pv.read(path)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_current_writer_metadata_and_continuation_are_exact(tmp_path, axis):
    sampler = make_sampler(axis)
    path = tmp_path / "plane.vts"
    written = write_frame(sampler, path, time=0.75)
    np.testing.assert_array_equal(written.field_data["time"], [0.75])
    np.testing.assert_array_equal(written.field_data["TimeValue"], [0.75])
    contract = json.loads(str(written.field_data["openonda_sampling_grid"][0]))
    assert contract["grid_layout"] == "bounded_uniform_v1"
    assert contract["configuration"]["bounds"] == sampler.bounds.tolist()
    assert (
        contract["coordinates_sha256"]
        == hashlib.sha256(np.ascontiguousarray(written.points, dtype="<f8").tobytes()).hexdigest()
    )
    restarted = make_sampler(axis)
    restarted.validate_existing_vtk(path)
    np.testing.assert_array_equal(restarted.grid_points, sampler.grid_points)
    assert _sampler_identity(restarted)["grid_layout"] == "bounded_uniform_v1"


@pytest.mark.parametrize(
    "changes",
    [
        {"bounds": [1.7, 8.0, -2, 2]},
        {"spacing": 0.099},
        {"point": [0, 0, 0.1]},
        {"normal": [0, 0, -1]},
    ],
)
def test_geometry_changes_reject_without_mutation(tmp_path, changes):
    path = tmp_path / "plane.vts"
    write_frame(make_sampler(), path)
    sampler = make_sampler(**changes)
    before = sampler.grid_points.copy()
    with pytest.raises(ValueError, match="sampling grid differs"):
        sampler.validate_existing_vtk(path)
    np.testing.assert_array_equal(sampler.grid_points, before)


@pytest.mark.parametrize("change", ["order", "dimensions", "sub_f32_ulp", "nonfinite"])
def test_coordinate_changes_cannot_be_hidden_by_casting(tmp_path, change):
    sampler = make_sampler()
    path = tmp_path / "plane.vts"
    saved = write_frame(sampler, path)
    saved.points = saved.points.astype(np.float64)
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
        sampler.validate_existing_vtk(path)


@pytest.mark.parametrize("change", ["missing", "layout", "digest", "normal", "json"])
def test_current_metadata_is_required_and_authenticated(tmp_path, change):
    sampler = make_sampler()
    path = tmp_path / "plane.vts"
    saved = write_frame(sampler, path)
    contract = json.loads(str(saved.field_data["openonda_sampling_grid"][0]))
    if change == "missing":
        del saved.field_data["openonda_sampling_grid"]
    else:
        if change == "layout":
            contract["grid_layout"] = "unknown"
        elif change == "digest":
            contract["coordinates_sha256"] = "0" * 64
        elif change == "normal":
            contract["configuration"]["normal"] = [0, 0, -1]
        encoded = "{invalid" if change == "json" else json.dumps(contract)
        saved.field_data["openonda_sampling_grid"] = np.array([encoded])
    saved.save(path)
    with pytest.raises(ValueError, match="sampling grid|metadata"):
        sampler.validate_existing_vtk(path)


def test_restart_preflights_retained_geometry_before_evolution_or_index_changes(tmp_path):
    directory = tmp_path / "samples"
    directory.mkdir()
    sampler = make_sampler()
    write_frame(sampler, directory / "plane_000270.vts")
    write_frame(make_sampler(spacing=0.2), directory / "plane_000290.vts")
    OutputManager._write_pvd(
        directory,
        "plane",
        [
            (10.8, "plane_000270.vts"),
            (11.6, "plane_000290.vts"),
        ],
    )
    config = Samplers(samples=(sampler,))
    solver = SimpleNamespace(case=SimpleNamespace(samplers=config), case_dir=tmp_path)
    manager = OutputManager(solver)
    sampler.sample = lambda solver: pytest.fail("Restart validation must not evaluate a field")
    manager.rewind_histories(11.12)
    assert manager._read_pvd(directory, "plane") == [(10.8, "plane_000270.vts")]
    assert (directory / "plane_000270.vts").is_file()


def test_last_retained_geometry_conflict_preserves_indexes_and_histories(tmp_path):
    directory = tmp_path / "samples"
    directory.mkdir()
    write_frame(make_sampler(), directory / "plane_000270.vts")
    write_frame(make_sampler(spacing=0.2), directory / "plane_000280.vts")
    OutputManager._write_pvd(
        directory,
        "plane",
        [
            (10.8, "plane_000270.vts"),
            (11.2, "plane_000280.vts"),
        ],
    )
    before = (directory / "plane.pvd").read_bytes()
    csv = directory / "plane.csv"
    csv.write_text("time,value\n10.8,1\n11.2,2\n")
    config = Samplers(samples=(make_sampler(),))
    solver = SimpleNamespace(case=SimpleNamespace(samplers=config), case_dir=tmp_path)
    with pytest.raises(ValueError, match="sampling grid differs"):
        OutputManager(solver).rewind_histories(11.2)
    assert (directory / "plane.pvd").read_bytes() == before
    assert csv.read_text() == "time,value\n10.8,1\n11.2,2\n"


def test_first_write_validates_current_grid_before_appending(tmp_path):
    sampler = make_sampler()
    write_frame(sampler, tmp_path / "plane_000270.vts")
    OutputManager._write_pvd(tmp_path, "plane", [(10.8, "plane_000270.vts")])
    config = Samplers(samples=(sampler,))
    solver = SimpleNamespace(case=SimpleNamespace(samplers=config), write_precision="f32")
    manager = OutputManager(solver)
    manager._write(
        sampler,
        SamplingContext(
            solver=solver,
            output_directory=tmp_path,
            step=280,
            time=11.2,
            event=OutputEvent.ACCEPTED_STEP,
            continuing_output=True,
        ),
    )
    saved = pv.read(tmp_path / "plane_000280.vts")
    previous = pv.read(tmp_path / "plane_000270.vts")
    np.testing.assert_array_equal(saved.points, previous.points)
