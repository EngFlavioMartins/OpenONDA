"""A stationary turbine wake needs complete native vector-field evidence."""

from types import SimpleNamespace
from xml.etree import ElementTree

import h5py
import numpy as np
import pytest
import pyvista as pv

from tutorials.vpm.rotor_flow.assets import _common as rotor_common
from tutorials.vpm.rotor_flow.assets.plot_rotor_wake_planes import (
    assess_wake_signal_onset,
    checkpoint_particle_front_brackets,
    finite_distance_profiles,
    induced_field_drift,
    native_plane_windows,
    plane_profiles,
)


def published_planes(directory, defect=None):
    times = np.arange(1, 49) / 8
    if defect == "stale":
        times = times[:-3]
    elif defect == "gap":
        times = np.delete(times, 20)
    elif defect == "duplicate":
        times[20] = times[19]
    elif defect == "off_clock":
        times[20] += 0.01
    points = np.array([[0, -1, -1], [0, 1, -1], [0, -1, 1], [0, 1, 1]], dtype=float)
    samplers = []
    for index, downstream in enumerate((12.0, 24.0)):
        name = f"slice_x{int(downstream)}m"
        samplers.append(
            {
                "type": "SurfaceSampler",
                "file_name": name,
                "schedule": {"type": "EverySteps", "interval": 1, "start_time": 0.0},
            }
        )
        if defect == "missing_plane" and index == 1:
            continue
        collection = ElementTree.Element("VTKFile", type="Collection")
        frames = ElementTree.SubElement(collection, "Collection")
        for step, time in enumerate(times):
            grid = pv.PolyData(points + [downstream, 0, 0])
            if defect == "wrong_station" and index == 1:
                grid.points += [3.0, 0.0, 0.0]
            velocity = np.tile([99.0 + 0.1 * np.sin(2 * np.pi * time), 0.0, 0.0], (4, 1))
            if defect == "cancelling_drift" and time > 3:
                velocity[:, 1] = [0.2, -0.2, 0.2, -0.2]
            if defect == "no_wake":
                velocity[:] = [100.0, 0.0, 0.0]
            if defect == "first_revolution_transient" and 1.0 < time <= 2.0:
                velocity[:, 0] = 95.0
            if defect == "nonfinite" and step == len(times) - 1:
                velocity[0, 2] = np.nan
            if defect == "moving_grid" and step == len(times) - 1:
                grid.points += [0.0, 0.0, 0.1]
            grid["velocity"] = velocity
            filename = f"{name}_{step:06d}.vtp"
            grid.save(directory / filename)
            ElementTree.SubElement(frames, "DataSet", timestep=str(time), file=filename)
        ElementTree.ElementTree(collection).write(directory / f"{name}.pvd")
    return SimpleNamespace(
        samples_dir=directory,
        rotation_period=1.0,
        rotor_radius=6.0,
        metadata={
            "state": {"initial_step": 0, "initial_time": 0.0, "step": 48, "time": 6.0},
            "configuration": {
                "numerics": {"time_step_size": 0.125, "freestream_velocity": [100.0, 0.0, 0.0]},
                "samplers": {"items": samplers},
            },
        },
    )


@pytest.mark.parametrize("schedule_type", ["EverySteps", "EveryTime"])
def test_complete_periodic_native_planes_pass(tmp_path, schedule_type):
    p = published_planes(tmp_path)
    if schedule_type == "EveryTime":
        for sampler in p.metadata["configuration"]["samplers"]["items"]:
            sampler["schedule"] = {"type": "EveryTime", "interval": 0.125, "start_time": 0.0}
    rows = native_plane_windows(p, require_complete=True)
    assert len(rows) == 2
    assert all(row["complete"] and row["induced_field_drift"] < 1e-12 for row in rows)


@pytest.mark.parametrize(
    "defect",
    [
        "stale",
        "gap",
        "duplicate",
        "off_clock",
        "missing_plane",
        "nonfinite",
        "moving_grid",
        "wrong_station",
    ],
)
def test_files_alone_do_not_qualify_a_native_wake(tmp_path, defect):
    p = published_planes(tmp_path, defect)
    with pytest.raises((ValueError, OSError)):
        native_plane_windows(p, require_complete=True)


def test_spatial_cancellation_and_large_freestream_do_not_hide_changes(tmp_path):
    p = published_planes(tmp_path, "cancelling_drift")
    rows = native_plane_windows(p, require_complete=True)
    for row in rows:
        # Axial disc means are unchanged; the resolved vector field is not.
        assert row["induced_field_drift"] > 0.05


def test_pure_freestream_is_not_a_resolved_stationary_wake(tmp_path):
    rows = native_plane_windows(published_planes(tmp_path, "no_wake"), require_complete=True)
    assert all(not np.isfinite(row["induced_field_drift"]) for row in rows)


def test_signal_onset_requires_a_persistent_induced_signal(tmp_path):
    arrived_directory = tmp_path / "arrived"
    missing_directory = tmp_path / "missing"
    arrived_directory.mkdir()
    missing_directory.mkdir()
    arrived = published_planes(arrived_directory)
    missing = published_planes(missing_directory, "no_wake")

    arrived_assessment = assess_wake_signal_onset(arrived, "slice_x12m")
    missing_assessment = assess_wake_signal_onset(missing, "slice_x12m")

    assert arrived_assessment["signal_onset_time"] is not None
    assert arrived_assessment["persistence_frames"] == 3
    assert missing_assessment["signal_onset_time"] is None


def test_operating_point_uses_shared_five_revolution_window(monkeypatch):
    import pandas as pd

    data = pd.DataFrame(
        {
            "time": np.arange(0.0, 10.1, 0.1),
            "CT": np.arange(0.0, 10.1, 0.1),
            "CP": np.arange(0.0, 10.1, 0.1) * 2.0,
        }
    )
    monkeypatch.setattr(rotor_common, "performance", lambda: data)
    monkeypatch.setattr(
        rotor_common,
        "rotor_inputs",
        lambda: SimpleNamespace(rotation_period=1.0),
    )

    ct, cp = rotor_common.read_operating_point()

    assert ct == pytest.approx(data.loc[data.time > 5.0, "CT"].mean())
    assert cp == pytest.approx(data.loc[data.time > 5.0, "CP"].mean())


def test_complete_window_must_not_be_redefined_by_a_late_sampler_start(tmp_path):
    p = published_planes(tmp_path)
    for sampler in p.metadata["configuration"]["samplers"]["items"]:
        sampler["schedule"]["start_time"] = 4.0
    with pytest.raises(ValueError, match="incomplete"):
        native_plane_windows(p, require_complete=True)


def test_periodic_wake_with_noncommensurate_samples_is_not_false_drift():
    period = 2 * np.pi / (49 / 6)
    times = np.arange(9.84, 14.4 + 1e-9, 0.06)
    velocity = np.zeros((len(times), 4, 3))
    velocity[:, :, 0] = 6 + 0.2 * np.sin(6 * np.pi * times / period)[:, None]
    drift, rotations = induced_field_drift(times, velocity, [7.0, 0.0, 0.0], period)
    assert rotations == 5
    assert drift < 0.01


def test_first_revolution_transient_is_included_in_phase_aware_drift():
    period = 1.0
    times = np.linspace(0.0, 5.0, 51)
    velocity = np.full((len(times), 4, 3), [7.0, 0.0, 0.0])
    velocity[times <= 1.0, :, 0] = 8.0

    drift, rotations = induced_field_drift(
        times,
        velocity,
        [7.0, 0.0, 0.0],
        period,
        window_start=0.0,
        window_end=5.0,
    )

    assert rotations == 5
    assert drift > 0.01


def test_phase_aware_drift_never_extrapolates_missing_window_boundaries():
    times = np.arange(1.0, 6.1, 0.5)
    velocity = np.zeros((len(times), 2, 3))
    velocity[:, :, 0] = 7.0

    with pytest.raises(ValueError, match="does not bracket boundary"):
        induced_field_drift(
            times,
            velocity,
            [7.0, 0.0, 0.0],
            1.0,
            window_start=0.0,
            window_end=5.0,
        )


def test_native_five_revolution_mean_cannot_hide_first_revolution_transient(tmp_path):
    directory = tmp_path / "first_transient"
    directory.mkdir()
    p = published_planes(directory, "first_revolution_transient")

    rows = native_plane_windows(p, require_complete=True)

    assert all(row["compared_rotations"] == 5 for row in rows)
    assert all(row["induced_field_drift"] > 0.01 for row in rows)
    assert all(row["window_mean_velocity"][:, 0].mean() < 99.0 for row in rows)


def test_checkpoint_particle_front_brackets_remain_unqualified_arrival_evidence(tmp_path):
    solution = tmp_path / "solution"
    solution.mkdir()
    for index, (time, maximum_x) in enumerate(((0.0, 5.0), (1.0, 13.0), (2.0, 25.0))):
        with h5py.File(solution / f"vpm_{index:06d}.h5", "w") as archive:
            solver = archive.create_group("solver")
            solver.attrs["time"] = time
            solver.attrs["step"] = index
            particles = archive.create_group("particles")
            particles.create_dataset("position", data=[[maximum_x, 0.0, 0.0]])
    p = SimpleNamespace(solution_dir=solution, station_radius=6.0, rotor_radius=6.0)

    records = checkpoint_particle_front_brackets(p)

    assert [record["status"] for record in records] == ["bracketed", "bracketed"]
    assert (records[0]["previous_time"], records[0]["crossing_time"]) == (0.0, 1.0)
    assert records[1]["previous_step"] == 1


def test_checkpoint_particle_front_brackets_report_unavailable_invalid_and_ordered_states(tmp_path):
    missing = SimpleNamespace(
        solution_dir=tmp_path / "missing", station_radius=6.0, rotor_radius=6.0
    )
    assert all(
        record["status"] == "unavailable" for record in checkpoint_particle_front_brackets(missing)
    )

    solution = tmp_path / "solution"
    solution.mkdir()
    with h5py.File(solution / "vpm_000000.h5", "w") as archive:
        solver = archive.create_group("solver")
        solver.attrs["time"] = 1.0
        solver.attrs["step"] = 1
        particles = archive.create_group("particles")
        particles.create_dataset("position", data=[[5.0, 0.0, 0.0]])
    with h5py.File(solution / "vpm_000001.h5", "w") as archive:
        solver = archive.create_group("solver")
        solver.attrs["time"] = 0.0
        solver.attrs["step"] = 0
        particles = archive.create_group("particles")
        particles.create_dataset("position", data=[[13.0, 0.0, 0.0]])
    unordered = SimpleNamespace(solution_dir=solution, station_radius=6.0, rotor_radius=6.0)
    assert all(
        record["status"] == "invalid_order"
        for record in checkpoint_particle_front_brackets(unordered)
    )

    malformed = tmp_path / "malformed"
    malformed.mkdir()
    for index in range(2):
        with h5py.File(malformed / f"vpm_{index:06d}.h5", "w") as archive:
            solver = archive.create_group("solver")
            solver.attrs["time"] = float(index)
            solver.attrs["step"] = index
    invalid = SimpleNamespace(solution_dir=malformed, station_radius=6.0, rotor_radius=6.0)
    assert all(
        record["status"] == "invalid" for record in checkpoint_particle_front_brackets(invalid)
    )


def test_profiles_time_weight_irregular_frames_and_exclude_partial_annuli(monkeypatch):
    y, z = np.meshgrid(np.linspace(-1, 1, 17), np.linspace(-1, 1, 17))
    points = np.column_stack((np.full(y.size, 2.0), y.ravel(), z.ravel()))
    times = np.array([0.0, 0.1, 1.0])
    velocity = np.zeros((3, len(points), 3))
    velocity[:, :, 0] = 7 * (1 + times[:, None])
    record = {
        "name": "linear_in_time",
        "points": points,
        "times": times,
        "velocity": velocity,
        "complete": True,
        "induced_field_drift": 0.0,
        "compared_rotations": 4,
    }
    monkeypatch.setattr(
        "tutorials.vpm.rotor_flow.assets.plot_rotor_wake_planes.native_plane_windows",
        lambda *args: [record],
    )
    row = plane_profiles(SimpleNamespace(rotor_radius=1.0, freestream_speed=7.0))[0]
    assert row["radius"].max() < 1.0
    # The exact average of 1+t over [0, 1] is 1.5, independent of sample density.
    np.testing.assert_allclose(row["mean"][np.isfinite(row["mean"])], 1.5, atol=1e-12)


def test_finite_distance_profiles_report_both_induction_components(monkeypatch):
    y, z = np.meshgrid(np.linspace(-1, 1, 17), np.linspace(-1, 1, 17))
    points = np.column_stack((np.full(y.size, 12.0), y.ravel(), z.ravel()))
    times = np.array([0.0, 0.5, 1.0])
    velocity = np.zeros((3, len(points), 3))
    velocity[:, :, 0] = 7.0
    record = {
        "name": "wake_1D",
        "points": points,
        "times": times,
        "velocity": velocity,
        "complete": True,
        "induced_field_drift": 0.0,
        "compared_rotations": 4,
    }
    bem = {
        "radial_position": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        "circulation": np.array([1.0, 1.0, 0.8, 0.5, 0.2]),
        "tangential_induction_factor": np.array([0.08, 0.06, 0.04, 0.02, 0.01]),
    }
    monkeypatch.setattr(
        "tutorials.vpm.rotor_flow.assets.plot_rotor_wake_planes.native_plane_windows",
        lambda *args, **kwargs: [record],
    )
    monkeypatch.setattr(
        "tutorials.vpm.rotor_flow.assets.plot_rotor_wake_planes.bem_reference",
        lambda: bem,
    )
    p = SimpleNamespace(
        rotor_radius=6.0,
        hub_radius=0.5,
        n_blades=3,
        freestream_speed=7.0,
        angular_velocity=8.0,
        metadata={
            "configuration": {
                "numerics": {"vlm": {"surfaces": [{"kinematics": {"angular_speed": -8.0}}]}}
            }
        },
    )
    row = finite_distance_profiles(p, rotations=6)[0]
    for key in (
        "actual_axial_induction",
        "reference_axial_induction",
        "actual_tangential_induction",
        "reference_tangential_induction",
        "actual_axial_velocity",
        "reference_axial_velocity",
        "actual_tangential_velocity",
        "reference_tangential_velocity",
    ):
        assert np.isfinite(row[key]).any()
