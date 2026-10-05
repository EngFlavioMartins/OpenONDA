"""Native saved-state readers own integrity and complete averaging windows."""

import json

import h5py
import numpy as np
import pandas as pd
import pytest
import pyvista as pv

from source.solvers.vpm.io.postprocess import (
    backup_frames,
    common_frame_time,
    coupled_frames,
    particle_state,
    profile_mean,
    saved_frame,
    surface_mean,
)


def write_backup(path, *, step=0, time=0.0, count=2):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as archive:
        solver = archive.create_group("solver")
        solver.attrs.update(step=step, time=time, n_particles_total=count)
        fields = archive.create_group("particles")
        for name, values in {
            "position": np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [np.nan] * 3]),
            "vortex_strength": np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0], [np.nan] * 3]),
            "core_radius": np.array([0.1, 0.2, np.nan]),
            "group_id": np.array([0, 1, -1]),
        }.items():
            fields.create_dataset(name, data=values)
        surface = solver.create_group("vlm")
        surface.create_dataset("panel_corner_position", data=np.zeros((1, 4, 3)))
        surface.create_dataset("circulation", data=np.ones(1))


def test_particle_state_reads_active_fields_and_owned_clock(tmp_path):
    path = tmp_path / "vpm_000003.h5"
    write_backup(path, step=3, time=0.3)
    state = particle_state(path)
    assert state["step"] == 3 and state["time"] == 0.3
    assert state["position"].shape == (2, 3)
    np.testing.assert_array_equal(state["group_id"], [0, 1])
    assert state["vlm"]["panel_corner_position"].shape == (1, 4, 3)


@pytest.mark.parametrize("defect", ["count", "clock", "core", "field"])
def test_particle_state_rejects_malformed_native_fields(tmp_path, defect):
    path = tmp_path / "vpm_000000.h5"
    write_backup(path)
    with h5py.File(path, "a") as archive:
        if defect == "count":
            archive["solver"].attrs["n_particles_total"] = 4
        elif defect == "clock":
            archive["solver"].attrs["time"] = np.nan
        elif defect == "core":
            archive["particles/core_radius"][0] = 0.0
        else:
            archive["particles/vortex_strength"][0, 0] = np.nan
    with pytest.raises(ValueError):
        particle_state(path)


def test_coupled_frames_retains_accepted_horizon_without_mutating_later_backups(tmp_path):
    for step, time in enumerate((0.0, 0.1, 0.2)):
        write_backup(tmp_path / "vpm" / f"vpm_{step:06d}.h5", step=step, time=time)
    (tmp_path / "vpm_metadata.json").write_text(json.dumps({"state": {"step": 1, "time": 0.1}}))
    assert [time for time, _ in backup_frames(tmp_path)] == [0.0, 0.1, 0.2]
    assert [time for time, _ in coupled_frames(tmp_path)] == [0.0, 0.1]
    assert (tmp_path / "vpm/vpm_000002.h5").is_file()


def profile():
    return pd.DataFrame(
        [{"time": time, "x": x, "u": 2 * time + x} for time in (0.0, 0.3, 1.2) for x in (0.0, 1.0)]
    )


def test_profile_mean_time_weights_irregular_samples_and_interpolates_only_endpoints():
    coordinates, mean = profile_mean(profile(), 0.2, 1.1, coordinates=["x"], fields=["u"])
    np.testing.assert_allclose(coordinates[:, 0], [0.0, 1.0])
    np.testing.assert_allclose(mean[:, 0], [1.3, 2.3])


@pytest.mark.parametrize("defect", ["outside", "duplicate", "grid", "field"])
def test_profile_mean_rejects_uncovered_or_inconsistent_native_records(defect):
    data, end = profile(), 1.1
    if defect == "outside":
        end = 1.3
    elif defect == "duplicate":
        data = pd.concat([data, data.iloc[:1]])
    elif defect == "grid":
        data.loc[2, "x"] = 0.5
    else:
        data.loc[2, "u"] = np.nan
    with pytest.raises(ValueError):
        profile_mean(data, 0.2, end, coordinates=["x"], fields=["u"])


def test_surface_mean_reads_real_fields_without_extrapolating(tmp_path):
    x, y = np.meshgrid([0.0, 1.0], [0.0, 1.0], indexing="ij")
    frames = []
    for index, time in enumerate((0.0, 0.3, 0.8, 1.2)):
        grid = pv.StructuredGrid(x, y, np.zeros_like(x))
        grid["velocity"] = np.full((grid.n_points, 3), 2 * time + 3)
        path = tmp_path / f"field_{index}.vts"
        grid.save(path)
        frames.append((time, path))
    points, mean = surface_mean(frames, 0.2, 1.1, field="velocity")
    assert points.shape == mean.shape == (4, 3)
    np.testing.assert_allclose(mean, 4.3)
    with pytest.raises(ValueError):
        surface_mean(frames, 0.2, 1.3, field="velocity")


def test_common_physical_frame_selection_never_uses_a_neighbouring_timestamp():
    time = common_frame_time([0.0, 0.3, 1.2], [0.0, 0.6, 1.2])
    assert time == 1.2
    assert saved_frame([(0.0, "initial"), (1.2, "final")], time) == (1.2, "final")
    with pytest.raises(ValueError):
        saved_frame([(0.0, "initial"), (1.2, "final")], 1.1)
    with pytest.raises(ValueError):
        common_frame_time([0.3], [0.6])
