"""Read-only observation auditing uses saved metadata, not solver construction."""

import csv
import importlib
import json

import numpy as np
import pytest

AUDIT = importlib.import_module(
    "tests.support.cylinder.audit_saved_samples"
)


def line(path, times, *, duplicate=False, moving=False, time_step_size=.01, steps=None, accepted_time_step_size=None):
    columns = ["time", "step"] + [f"{field}_{axis}" for field in ("position", "velocity", "vorticity") for axis in "xyz"]
    columns += ["kinematic_pressure"]
    if accepted_time_step_size is not None:
        columns.append("accepted_time_step_size")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for index, time in enumerate(times):
            row = dict.fromkeys(columns, 0.)
            row.update(time=time, step=round(time / time_step_size) if steps is None else steps[index],
                       position_x=1 + int(moving and index > 0), velocity_x=1)
            if accepted_time_step_size is not None:
                row["accepted_time_step_size"] = accepted_time_step_size
            writer.writerow(row)
            if duplicate:
                writer.writerow(row)


SAMPLER = {"type": "LineSampler", "n_points": 1, "file_name": "probe"}


def test_slower_sampler_need_not_end_at_observed_force_clock(tmp_path):
    path = tmp_path / "probe.csv"
    line(path, [0, .2, .4])
    result = AUDIT.audit_csv(path, SAMPLER, .2, .44, time_step_size=.01)
    assert result["last_required_event"] == .4
    assert result["checked_end"] == .44
    assert result["events"] == 3


@pytest.mark.parametrize("times, kwargs", [([0, .4], {}), ([0, .2], {"duplicate": True}),
                                         ([0, .2], {"moving": True}), ([.2, 0], {})])
def test_missing_duplicate_moving_or_reversed_samples_fail(tmp_path, times, kwargs):
    path = tmp_path / "probe.csv"
    line(path, times, **kwargs)
    with pytest.raises(ValueError):
        AUDIT.audit_csv(path, SAMPLER, .2, .4, time_step_size=.01)


@pytest.mark.parametrize("dt, particle", [(.008, False), (.04, True)])
def test_solver_specific_step_clock_accepts_accumulated_roundoff(tmp_path, dt, particle):
    path = tmp_path / "probe.csv"
    line(path, [.04, .08000000000000002], time_step_size=dt, accepted_time_step_size=dt)
    result = AUDIT.audit_csv(path, SAMPLER, .04, .08, particle, time_step_size=dt)
    assert result["step_time_consistency"] == "validated"
    assert result["time_step_size"] == dt


@pytest.mark.parametrize("step", [-1, 0.5, "5.0000000000000000001", 4, 1])
def test_invalid_or_wrong_solver_step_fails(tmp_path, step):
    path = tmp_path / "probe.csv"
    line(path, [.04], steps=[step])
    with pytest.raises(ValueError, match="step"):
        AUDIT.audit_csv(path, SAMPLER, .04, .04, time_step_size=.008)


@pytest.mark.parametrize("dt", [0, -1, float("inf"), float("nan"), True, ".008"])
def test_invalid_recorded_step_size_fails(tmp_path, dt):
    path = tmp_path / "probe.csv"
    line(path, [.04], time_step_size=.008)
    with pytest.raises(ValueError, match="fixed time step"):
        AUDIT.audit_csv(path, SAMPLER, .04, .04, time_step_size=dt)


def test_accepted_step_size_cannot_silently_differ(tmp_path):
    path = tmp_path / "probe.csv"
    line(path, [.04], time_step_size=.008, accepted_time_step_size=.01)
    with pytest.raises(ValueError, match="accepted time step"):
        AUDIT.audit_csv(path, SAMPLER, .04, .04, time_step_size=.008)


def test_normal_reference_path_and_metadata_defined_contract(tmp_path):
    directory = tmp_path / "reference_flow"
    (directory / "samples").mkdir(parents=True)
    (directory / "solution").mkdir()
    metadata = {"configuration": {"time": {"time_step_size": .01},
                "samplers": [{**SAMPLER, "schedule": {"every_n_steps": 20, "every_time": None}}]}}
    (directory / "solution/fvm_metadata.json").write_text(json.dumps(metadata))
    path = directory / "samples/probe.csv"
    line(path, [0, .2, .4])
    before = path.read_bytes()
    result = AUDIT.audit_normal(tmp_path, "reference", .44)
    assert result["directory"] == str(directory)
    assert result["csv_count"] == 1
    assert result["force_signal"]["status"] == "not assessed"
    assert path.read_bytes() == before
    assert not (tmp_path / "reference").exists()


def test_surface_audit_checks_every_valid_field_and_fixed_grid(tmp_path):
    pv = pytest.importorskip("pyvista")
    x, y = np.meshgrid([1., 2.], [0., 1.])
    frame = pv.StructuredGrid(x, y, np.zeros_like(x))
    frame.point_data["velocity"] = np.ones((4, 3))
    path = tmp_path / "frame.vts"
    frame.save(path)
    collection = tmp_path / "slice.pvd"
    collection.write_text('<VTKFile><Collection><DataSet file="frame.vts" timestep=".4"/>'
                          '</Collection></VTKFile>')
    result = AUDIT.audit_collection(collection, .4, .44, {})
    assert len(result["frames"]) == 1
    frame.point_data["velocity"][0] = np.nan
    frame.save(path)
    with pytest.raises(ValueError, match="non-finite valid"):
        AUDIT.audit_collection(collection, .4, .44, {})


def test_unknown_sampling_policy_is_not_guessed():
    with pytest.raises(ValueError, match="Unsupported"):
        AUDIT._cadence({"type": "CustomSchedule", "interval": 1}, .04, True)
