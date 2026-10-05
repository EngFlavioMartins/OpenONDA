"""Native result readers preserve accepted clocks and distributed ownership."""

import json

import numpy as np
import pytest
import pyvista as pv

from openonda.results import (
    NativeVelocity,
    history_window,
    read_csv_frame,
    read_csv_table,
    read_history_table,
    read_json_lines,
    read_numeric_table,
    read_surface_frame,
    write_csv_table,
)


def test_live_json_lines_ignore_only_an_unfinished_final_record(tmp_path):
    path = tmp_path / "diagnostics.jsonl"
    path.write_text('{"time": 1}\n{"time":')
    assert read_json_lines(path) == [{"time": 1}]
    path.write_text('{"time":\n{"time": 1}\n')
    with pytest.raises(json.JSONDecodeError):
        read_json_lines(path)


def test_profile_frame_uses_recorded_clock_and_unique_coordinates(tmp_path):
    path = tmp_path / "line.csv"
    path.write_text("time,position_x,velocity_x\n1,2,3\n1,1,4\n2,1,5\n")
    frame = read_csv_frame(path, 1, coordinates=("position_x",))
    np.testing.assert_array_equal(frame["position_x"], [1, 2])
    np.testing.assert_array_equal(frame["velocity_x"], [4, 3])
    with pytest.raises(ValueError, match="No recorded CSV frame"):
        read_csv_frame(path, 1.01, coordinates=("position_x",))
    path.write_text("time,position_x,velocity_x\n1,1,4\n1,1,5\n")
    with pytest.raises(ValueError, match="duplicate sample coordinates"):
        read_csv_frame(path, 1, coordinates=("position_x",))


def test_history_window_interpolates_endpoints_without_extrapolation():
    table = {"time": np.array([0, 1, 2, 3]), "value": np.array([0, 2, 4, 6])}
    window = history_window(table, 0.5, 2.5, columns=("value",))
    np.testing.assert_array_equal(window["time"], [0.5, 1, 2, 2.5])
    np.testing.assert_array_equal(window["value"], [1, 2, 4, 5])
    with pytest.raises(ValueError, match="requested window"):
        history_window(table, 0, 4, columns=("value",))
    table["value"] = np.array([0, 2, np.nan, 6])
    with pytest.raises(ValueError, match="finite recorded times"):
        history_window(table, 0, 3, columns=("value",))


def test_named_table_publishes_complete_rows_and_preserves_previous_output_on_failure(tmp_path):
    path = tmp_path / "tables" / "measurements.csv"
    write_csv_table(path, [(1, 2), (3, 4)], columns=("x", "velocity"))
    np.testing.assert_array_equal(read_csv_table(path)["velocity"], [2, 4])
    original = path.read_bytes()

    def interrupted_rows():
        yield [5, 6]
        raise RuntimeError("interrupted scientific analysis")

    with pytest.raises(RuntimeError, match="interrupted scientific analysis"):
        write_csv_table(path, interrupted_rows(), columns=("x", "velocity"))
    assert path.read_bytes() == original
    assert list(path.parent.iterdir()) == [path]


def test_headerless_numeric_table_preserves_physical_values_and_rejects_nonfinite(tmp_path):
    path = tmp_path / "reference.csv"
    path.write_text("0,1\n2,3\n")
    np.testing.assert_array_equal(read_numeric_table(path), [[0, 1], [2, 3]])
    path.write_text("0,nan\n")
    with pytest.raises(ValueError, match="finite measured values"):
        read_numeric_table(path)


@pytest.mark.parametrize("rows", ["1,2\n1,2\n", "1,2\n1,3\n", "2,3\n1,2\n"])
def test_history_requires_one_record_per_increasing_accepted_clock(tmp_path, rows):
    path = tmp_path / "history.csv"
    path.write_text("time,value\n" + rows)
    with pytest.raises(ValueError, match="duplicate or nonmonotonic"):
        read_history_table(path)


@pytest.mark.parametrize("steps", ["-1\n0", "0\n0.5", "0\nnan", "0\ninf", "1\n0"])
def test_native_table_rejects_malformed_accepted_steps(tmp_path, steps):
    path = tmp_path / "history.csv"
    path.write_text("step\n" + steps + "\n")
    with pytest.raises(ValueError, match="steps must be finite nonnegative integers"):
        read_csv_table(path)


def test_native_table_preserves_repeated_steps_for_distinct_scientific_entities(tmp_path):
    path = tmp_path / "forces.csv"
    path.write_text(
        "step,time,surface,force\n0,0,first,1\n0,0,second,2\n2,0.2,first,3\n2,0.2,second,4\n"
    )
    table = read_csv_table(path)
    np.testing.assert_array_equal(table["step"], [0, 0, 2, 2])
    np.testing.assert_array_equal(table["surface"], ["first", "second", "first", "second"])
    np.testing.assert_array_equal(table["force"], [1, 2, 3, 4])


def test_single_entity_history_requires_distinct_steps_at_later_times(tmp_path):
    path = tmp_path / "history.csv"
    path.write_text("step,time,value\n1,0.1,2\n1,0.2,3\n")
    with pytest.raises(ValueError, match="History steps are duplicate or nonmonotonic"):
        read_history_table(path)
    path.write_text("step,time,value\n1,0.1,2\n3,0.3,4\n")
    np.testing.assert_array_equal(read_history_table(path)["step"], [1, 3])


def test_physical_tables_without_steps_preserve_commented_frame_time(tmp_path):
    path = tmp_path / "frame.csv"
    path.write_text("# time = 0.25\nposition_x,velocity_x\n0,1\n1,2\n")
    frame = read_csv_table(path)
    np.testing.assert_array_equal(frame["time"], [0.25, 0.25])
    np.testing.assert_array_equal(frame["velocity_x"], [1, 2])


def test_native_velocity_admits_owned_ids_and_excludes_ghost_geometry(tmp_path):
    grid = pv.ImageData(dimensions=(5, 5, 5), spacing=(0.25,) * 3).cast_to_unstructured_grid()
    centres = grid.cell_centers().points
    ids = np.random.default_rng(42).permutation(grid.n_cells - 1)
    native_centres = np.empty_like(centres[:-1])
    native_centres[ids] = centres[:-1]
    grid.cell_data["global_cell_id"] = np.r_[ids, 0]
    grid.cell_data["vtkGhostType"] = np.r_[np.zeros(len(ids), np.uint8), np.uint8(1)]
    matrix = np.array([[1, 2, 3], [-2, 0.5, 4], [3, 1, -1]])
    grid.cell_data["velocity"] = centres @ matrix.T + [1, 2, 3]
    grid.cell_data["velocity"][-1] = np.nan
    path = tmp_path / "frame.vtu"
    grid.save(path)
    sampler = NativeVelocity.__new__(NativeVelocity)
    sampler.centres, sampler.k = native_centres, 12
    points = np.array([[0.36, 0.48, 0.55], [0.61, 0.72, 0.69], [0.875, 0.875, 0.875]])
    values = sampler.sample(path, {"line": points})["line"]
    np.testing.assert_allclose(values[:2], points[:2] @ matrix.T + [1, 2, 3], atol=1e-12)
    assert np.isnan(values[2]).all()

    grid.cell_data["global_cell_id"][1] = grid.cell_data["global_cell_id"][0]
    grid.save(path)
    with pytest.raises(ValueError, match="exactly once"):
        sampler.sample(path, {"line": points})
    grid.cell_data["global_cell_id"] = np.r_[ids, 0]
    grid.cell_data["vtkGhostType"][0] = 1
    grid.save(path)
    with pytest.raises(ValueError, match="exactly once"):
        sampler.sample(path, {"line": points})
    grid.cell_data["vtkGhostType"][0] = 0
    grid.cell_data["velocity"][0, 0] = np.nan
    grid.save(path)
    with pytest.raises(ValueError, match="Non-finite archived velocity"):
        sampler.sample(path, {"line": points})


def test_ensemble_surface_preserves_all_native_uncertainty_fields(tmp_path):
    grid = pv.ImageData(dimensions=(3, 2, 1)).cast_to_structured_grid()
    vectors = np.arange(grid.n_points * 3, dtype=float).reshape(-1, 3)
    for name in ("velocity", "vorticity", "velocity_standard_error", "vorticity_standard_error"):
        grid.point_data[name] = vectors
    grid.point_data["velocity_gradient_yx"] = np.arange(grid.n_points, dtype=float)
    grid.point_data["velocity_gradient_yx_standard_error"] = np.ones(grid.n_points)
    grid.field_data["ensemble_size"] = np.array([8])
    grid.field_data["confidence_multiplier"] = np.array([1.96])
    path = tmp_path / "ensemble.vts"
    grid.save(path)
    frame = read_surface_frame(path)
    np.testing.assert_array_equal(frame["velocity_standard_error_x"], vectors[:, 0].reshape(2, 3))
    np.testing.assert_array_equal(frame["vorticity_standard_error_z"], vectors[:, 2].reshape(2, 3))
    np.testing.assert_array_equal(frame["velocity_gradient_yx_standard_error"], np.ones((2, 3)))
    assert frame["ensemble_size"] == 8
    assert frame["confidence_multiplier"] == 1.96


def _surface_grid():
    grid = pv.ImageData(dimensions=(3, 2, 1)).cast_to_structured_grid()
    grid.point_data["velocity"] = np.arange(grid.n_points * 3, dtype=float).reshape(-1, 3)
    return grid


def test_deterministic_surface_preserves_actual_plane_and_optional_fields(tmp_path):
    grid = _surface_grid()
    grid.points = grid.points[:, [2, 1, 0]]
    grid.point_data["velocity_gradient_yx"] = np.arange(grid.n_points, dtype=float)
    path = tmp_path / "plane.vts"
    grid.save(path)
    frame = read_surface_frame(path)
    np.testing.assert_array_equal(frame["x"], grid.points[:, 0].reshape(2, 3))
    np.testing.assert_array_equal(frame["z"], grid.points[:, 2].reshape(2, 3))
    np.testing.assert_array_equal(frame["velocity_gradient_yx"], np.arange(6).reshape(2, 3))
    assert frame["valid"].all()
    assert "vorticity" not in frame
    assert "ensemble_size" not in frame
    assert "confidence_multiplier" not in frame
    assert "velocity_standard_error" not in frame


def test_surface_preserves_masked_points_without_admitting_nonfinite_observations(tmp_path):
    grid = _surface_grid()
    grid.point_data["vtkValidPointMask"] = np.array([0, 1, 1, 1, 1, 1], dtype=np.uint8)
    grid.point_data["velocity"][0] = np.nan
    grid.point_data["velocity_standard_error"] = np.ones((grid.n_points, 3))
    grid.point_data["velocity_standard_error"][0] = np.nan
    grid.field_data["ensemble_size"] = np.array([4])
    grid.field_data["confidence_multiplier"] = np.array([2.5])
    path = tmp_path / "masked.vts"
    grid.save(path)
    frame = read_surface_frame(path)
    assert not frame["valid"][0, 0]
    assert np.isnan(frame["velocity"][0, 0]).all()
    assert np.isnan(frame["velocity_standard_error"][0, 0]).all()
    assert frame["confidence_multiplier"] == 2.5
    grid.point_data["velocity"][1, 0] = np.nan
    grid.save(path)
    with pytest.raises(ValueError, match="finite values on valid points"):
        read_surface_frame(path)


@pytest.mark.parametrize(
    "fault", ["volume", "duplicate_point", "nonfinite_point", "cell_velocity", "mask", "cell_mask"]
)
def test_surface_requires_the_recorded_grid_and_point_association(tmp_path, fault):
    grid = _surface_grid()
    if fault == "volume":
        grid = pv.ImageData(dimensions=(3, 2, 2)).cast_to_structured_grid()
        grid.point_data["velocity"] = np.ones((grid.n_points, 3))
    elif fault == "duplicate_point":
        grid.points[0] = grid.points[1]
    elif fault == "nonfinite_point":
        grid.points[0, 0] = np.nan
    elif fault == "cell_velocity":
        del grid.point_data["velocity"]
        grid.cell_data["velocity"] = np.ones((grid.n_cells, 3))
    elif fault == "mask":
        grid.point_data["vtkValidPointMask"] = np.array([2, 1, 1, 1, 1, 1])
    else:
        grid.cell_data["vtkValidPointMask"] = np.ones(grid.n_cells)
    path = tmp_path / "invalid.vts"
    grid.save(path)
    with pytest.raises(ValueError):
        read_surface_frame(path)


@pytest.mark.parametrize(
    "fault", ["components", "negative_error", "nonfinite_error", "missing_field", "cell_error"]
)
def test_surface_uncertainty_requires_real_measured_point_fields(tmp_path, fault):
    grid = _surface_grid()
    grid.point_data["velocity_standard_error"] = np.ones((grid.n_points, 3))
    grid.field_data["ensemble_size"] = np.array([4])
    grid.field_data["confidence_multiplier"] = np.array([2.0])
    if fault == "components":
        grid.point_data["velocity_standard_error"] = np.ones((grid.n_points, 2))
    elif fault == "negative_error":
        grid.point_data["velocity_standard_error"][0, 0] = -1
    elif fault == "nonfinite_error":
        grid.point_data["velocity_standard_error"][0, 0] = np.inf
    elif fault == "missing_field":
        grid.point_data["vorticity_standard_error"] = np.ones((grid.n_points, 3))
    else:
        del grid.point_data["velocity_standard_error"]
        grid.cell_data["velocity_standard_error"] = np.ones((grid.n_cells, 3))
    path = tmp_path / "invalid_uncertainty.vts"
    grid.save(path)
    with pytest.raises(ValueError):
        read_surface_frame(path)


@pytest.mark.parametrize(
    ("size", "confidence"),
    [
        (None, [2.0]),
        ([4], None),
        ([1], [2.0]),
        ([2.5], [2.0]),
        ([4], [0.0]),
        ([4], [np.nan]),
        ([4, 4], [2.0]),
    ],
)
def test_surface_uncertainty_requires_complete_recorded_ensemble_metadata(
    tmp_path, size, confidence
):
    grid = _surface_grid()
    grid.point_data["velocity_standard_error"] = np.ones((grid.n_points, 3))
    if size is not None:
        grid.field_data["ensemble_size"] = np.asarray(size)
    if confidence is not None:
        grid.field_data["confidence_multiplier"] = np.asarray(confidence)
    path = tmp_path / "invalid_metadata.vts"
    grid.save(path)
    with pytest.raises(ValueError, match="ensemble"):
        read_surface_frame(path)
