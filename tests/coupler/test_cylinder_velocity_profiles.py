"""Offline cylinder profiles retain native clocks, cell ownership and domains."""

import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

MODULE = "tutorials.coupled_fvm_vpm.01_cylinder_shedding_flow.assets.velocity_profile_data"


def test_native_velocity_uses_owned_global_cells_once():
    module = importlib.import_module(MODULE)
    expected = np.arange(9, dtype=float).reshape(3, 3)
    fields = {
        "velocity": np.vstack((expected[[2, 0, 1]], [-1, -1, -1])),
        "global_cell_id": [2, 0, 1, 0],
        "vtkGhostType": [0, 0, 0, 1],
    }
    np.testing.assert_array_equal(module.ordered_velocity(fields, 3, Path("saved.pvtu")), expected)
    fields["global_cell_id"] = [2, 0, 0, 1]
    with pytest.raises(ValueError, match="exactly once"):
        module.ordered_velocity(fields, 3, Path("saved.pvtu"))
    fields["global_cell_id"] = [2, 0, 1, 0]
    fields["velocity"][0, 0] = np.nan
    with pytest.raises(ValueError, match="Non-finite"):
        module.ordered_velocity(fields, 3, Path("saved.pvtu"))


def test_native_profiles_only_use_saved_clock_and_fvm_support(tmp_path, monkeypatch):
    module = importlib.import_module(MODULE)
    box = {"xmin": -1.6, "xmax": 2.4, "ymin": -1.6, "ymax": 1.6, "zmin": -.48, "zmax": .48}
    geometry = module.ProfileGeometry(1, 1, box, box)
    profiles = []
    for x in (2, 4):
        paths = [tmp_path / f"{name}_x{x}.csv" for name in ("reference", "vpm")]
        for path in paths:
            frame = pd.DataFrame([
                [time, x, y, 0, 1 + .1 * y, .2 * y, 0]
                for time in (.2, 1, 2, 3)
                for y in (-2, -1, 0, 1, 2)
            ], columns=["time", "position_x", "position_y", "position_z",
                        "velocity_x", "velocity_y", "velocity_z"])
            frame.to_csv(path, index=False)
        profiles.append((x, *paths, None))
    monkeypatch.setattr(module.data, "CASE_DIR", tmp_path)
    monkeypatch.setattr(module, "native_fvm_frames", lambda: (
        [(1 + 1e-13, tmp_path / "one.pvtu"), (2.00000004, tmp_path / "two.pvtu")],
        tmp_path / "mesh.npz",
    ))
    sampled = []

    def sample(_self, source, positions):
        sampled.append((source, positions.copy()))
        return np.ones_like(positions), tmp_path / "cache.npz"

    monkeypatch.setattr(module.NativeProfiles, "sample", sample)
    frames = list(module.coincident_velocity_profiles(profiles, geometry))
    assert [time for time, *_ in frames] == [1]
    assert sampled[0][0].name == "one.pvtu"
    np.testing.assert_array_equal(sampled[0][1][:, 1], [-1, 0, 1])
    assert frames[0][1][0][-1] == tmp_path / "cache.npz"
    assert frames[0][1][1][-1] is None  # x=4 is outside FVM; never extrapolate.
    assert frames[0][3]["x2"]["native_time"] == 1 + 1e-13
    assert frames[0][3]["x2"]["point_count"] == 3


def test_cache_checks_every_mpi_piece(tmp_path):
    module = importlib.import_module(MODULE)
    source = tmp_path / "field.pvtu"
    source.write_text('<VTKFile><PUnstructuredGrid><Piece Source="a.vtu"/>'
                      '<Piece Source="b.vtu"/></PUnstructuredGrid></VTKFile>')
    for name in ("a.vtu", "b.vtu"):
        (tmp_path / name).write_bytes(b"native field")
    before = module._file_stamp(source)
    (tmp_path / "b.vtu").write_bytes(b"changed native field")
    assert module._file_stamp(source) != before
    assert len(before) == 3
