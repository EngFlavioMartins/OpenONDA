"""Offline cylinder profiles retain native clocks, cell rank assignment and domains."""

import importlib

import numpy as np
import pandas as pd

from openonda import results

MODULE = "tutorials.coupled_fvm_vpm.01_cylinder_shedding_flow.assets.velocity_profile_data"


def test_native_profiles_only_use_saved_clock_and_fvm_support(tmp_path, monkeypatch):
    module = importlib.import_module(MODULE)
    box = {"xmin": -1.6, "xmax": 2.4, "ymin": -1.6, "ymax": 1.6, "zmin": -0.48, "zmax": 0.48}
    geometry = {"diameter": 1, "speed": 1, "fvm_box": box, "transfer_box": box}
    profiles = []
    for x in (2, 4):
        paths = [tmp_path / f"{name}_x{x}.csv" for name in ("reference", "vpm")]
        for path in paths:
            frame = pd.DataFrame(
                [
                    [time, x, y, 0, 1 + 0.1 * y, 0.2 * y, 0]
                    for time in (0.2, 1, 2, 3)
                    for y in (-2, -1, 0, 1, 2)
                ],
                columns=[
                    "time",
                    "position_x",
                    "position_y",
                    "position_z",
                    "velocity_x",
                    "velocity_y",
                    "velocity_z",
                ],
            )
            frame.to_csv(path, index=False)
        profiles.append((x, *paths, tmp_path / f"fvm_x{x}.csv"))
    monkeypatch.setattr(module.data, "CASE_DIR", tmp_path)
    monkeypatch.setattr(
        results,
        "read_pvd_frames",
        lambda path: [(1 + 1e-13, tmp_path / "one.pvtu"), (2.00000004, tmp_path / "two.pvtu")],
    )
    sampled = []

    def sample(source, queries):
        positions = queries["profile"]
        sampled.append((source, positions.copy()))
        return {"profile": np.ones_like(positions)}

    monkeypatch.setattr(
        results,
        "NativeVelocity",
        lambda *args, **kwargs: type("Reader", (), {"sample": staticmethod(sample)})(),
    )
    frames = list(module.coincident_velocity_profiles(profiles, geometry))
    assert [time for time, *_ in frames] == [1]
    assert sampled[0][0].name == "one.pvtu"
    np.testing.assert_array_equal(sampled[0][1][:, 1], [-1, 0, 1])
    assert frames[0][1][0][-1] == tmp_path / "fvm_x2.csv"
    assert frames[0][1][1][-1] is None  # x=4 is outside FVM; never extrapolate.
    assert frames[0][3]["x2"]["native_time"] == 1 + 1e-13
    assert frames[0][3]["x2"]["point_count"] == 3
    assert not (tmp_path / "figures").exists()
