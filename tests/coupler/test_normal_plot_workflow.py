"""Normal run/plot commands use tmp-only fixtures, never retained solutions."""

import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
CASES = ROOT / "tutorials/coupled_fvm_vpm"
CYLINDER = "01_cylinder_shedding_flow"
CUBE = "02_cube_flow"
PACKAGE = "tutorials.coupled_fvm_vpm.01_cylinder_shedding_flow.assets."


@pytest.mark.parametrize("case", [CYLINDER, CUBE, CYLINDER + "/reference_flow", CUBE + "/reference_flow"])
@pytest.mark.parametrize("launcher", ["allrun.sh", "allcontinue.sh"])
def test_launchers_preserve_outputs_and_propagate_failure(tmp_path, case, launcher):
    shutil.copy2(CASES / case / launcher, tmp_path / launcher)
    cleanup = tmp_path / "allclean.sh"
    cleanup.write_text('#!/bin/sh\necho "cleanup must not run" >&2\nexit 99\n')
    cleanup.chmod(0o755)
    python = tmp_path / "python"
    python.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> calls.txt\nexit "${TEST_EXIT:-0}"\n')
    python.chmod(0o755)
    env = {**os.environ, "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"]}
    for directory in ("solution", "samples", "figures", "constant"):
        target = tmp_path / directory / "original"
        target.parent.mkdir()
        target.write_bytes(b"preserved")
    result = subprocess.run(["bash", str(tmp_path / launcher)], cwd="/tmp", env=env)
    assert result.returncode == 0
    for directory in ("solution", "samples", "figures", "constant"):
        assert (tmp_path / directory / "original").read_bytes() == b"preserved"
    calls = (tmp_path / "calls.txt").read_text().splitlines()
    assert len(calls) == (3 if case == CUBE + "/reference_flow" else 1)
    result = subprocess.run(["bash", str(tmp_path / launcher)], env={**env, "TEST_EXIT": "17"})
    assert result.returncode == 17


def _profile(path, times):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for time in times:
        for y in (-1.0, 0.0, 1.0):
            rows.append([time, 1, y, 0, 1 - .1 * (1 - abs(y)), .01 * y, 0])
    pd.DataFrame(rows, columns=["time", "position_x", "position_y", "position_z",
                               "velocity_x", "velocity_y", "velocity_z"]).to_csv(path, index=False)


def test_profile_clocks_reject_old_rounding_and_allow_accumulation(tmp_path):
    data = importlib.import_module(PACKAGE + "postprocess")
    left, right = tmp_path / "left.csv", tmp_path / "right.csv"
    _profile(left, [1, 11])
    _profile(right, [1, 10.999999999999])
    assert data.latest_common_profile_time((left, right)) == 11
    assert len(data.profile(right, 11)) == 3
    _profile(right, [1, 11.00000004])
    assert data.latest_common_profile_time((left, right)) == 1
    with pytest.raises(ValueError, match="no profile"):
        data.profile(right, 11)


def test_profile_history_rejects_rewound_or_duplicate_spatial_rows(tmp_path):
    data = importlib.import_module(PACKAGE + "postprocess")
    path = tmp_path / "profile.csv"
    _profile(path, [2, 1])
    with pytest.raises(ValueError, match="ordered"):
        data.profile(path, 1)
    _profile(path, [1])
    frame = pd.read_csv(path)
    pd.concat([frame, frame.iloc[[0]]]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="duplicate transverse"):
        data.profile(path, 1)


def test_force_coverage_is_limited_and_explicitly_interpolated():
    data = importlib.import_module(PACKAGE + "postprocess")
    candidate = pd.DataFrame({"time": [0, 1, 2], "cd": [0, 1, 2]})
    reference = pd.DataFrame({"time": [0, .5, 1, 2, 100], "cd": [0, .5, 1, 2, 100]})
    times, left, right, errors = data.common_history(candidate, reference, ("cd",))
    assert times[-1] == 2
    np.testing.assert_array_equal(left, right)
    assert errors["cd"]["maximum"] == 0
    report = data.history_coverage(candidate, reference)
    assert report["available_time_intervals"]["reference"] == [0, 100]
    assert report["comparison_time_interval"] == [0, 2]
    assert "no time shift" in report["time_alignment"]


def test_synthetic_profile_plot_reports_reference_path_not_dataframe_name(tmp_path, monkeypatch):
    plot = importlib.import_module(PACKAGE + "plot_reference_profiles")
    data = plot.data
    reference = tmp_path / "reference_flow/samples"
    for x in (1, 2, 4):
        _profile(reference / f"transverse_x{x}.csv", [0, 2, 100])
        _profile(tmp_path / "samples" / f"vpm_transverse_x{x}.csv", [0, 2])
    _profile(tmp_path / "samples/fvm_transverse_x1.csv", [0, 2])
    monkeypatch.setattr(data, "CASE_DIR", tmp_path)
    monkeypatch.setattr(data, "FIGURES", tmp_path / "figures")
    monkeypatch.setattr(data, "AUXILIARY", tmp_path / "figures/auxiliary")
    monkeypatch.setattr(data, "reference_directory", lambda: reference)
    monkeypatch.setattr(sys, "argv", ["plot_reference_profiles.py"])
    plot.main()
    report = json.loads((tmp_path / "figures/auxiliary/reference_profile_errors.json").read_text())
    assert report["reference"] == "reference_flow/samples"
    assert report["time"] == 2
    assert "no time interpolation" in report["time_alignment"]
    assert (tmp_path / "figures/reference_profiles.png").stat().st_size > 1000


def test_cube_pvd_reader_handles_attribute_order_and_rejects_ambiguous_clocks(tmp_path):
    data = importlib.import_module("tutorials.coupled_fvm_vpm.02_cube_flow.assets.postprocess")
    path = tmp_path / "frames.pvd"
    path.write_text('<VTKFile><Collection><DataSet file="a.vts" timestep="1"/>'
                    '<DataSet file="b.vts" timestep="2"/></Collection></VTKFile>')
    assert [time for time, _ in data._pvd_frames(path)] == [1, 2]
    path.write_text(path.read_text().replace('timestep="2"', 'timestep="1.00000000001"'))
    with pytest.raises(ValueError, match="unambiguous"):
        data._pvd_frames(path)


def test_normal_cylinder_allplot_on_incomplete_synthetic_data(tmp_path):
    case = tmp_path / "case"
    assets = case / "assets"
    assets.mkdir(parents=True)
    shutil.copy2(CASES / CYLINDER / "allplot.sh", case / "allplot.sh")
    for filename in ("plot_cylinder_forces.py", "plot_reference_forces.py",
                     "plot_reference_profiles.py", "postprocess.py"):
        shutil.copy2(CASES / CYLINDER / "assets" / filename, assets / filename)
    for relative, times in (("samples", [0, 1, 2]), ("reference_flow/samples", [0, 1, 2, 100])):
        directory = case / relative
        directory.mkdir(parents=True)
        pd.DataFrame({"time": times, "drag_coefficient": np.asarray(times) * .01 + 1,
                      "lift_coefficient": np.asarray(times) * .005}).to_csv(
            directory / "forces_history.csv", index=False)
    for x in (2, 4):
        _profile(case / "reference_flow/samples" / f"transverse_x{x}.csv", [0, 2, 100])
        _profile(case / "samples" / f"vpm_transverse_x{x}.csv", [0, 2])
    environment = {**os.environ, "MPLBACKEND": "Agg", "PYTHONPATH": str(ROOT),
                   "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]}
    result = subprocess.run(["bash", str(case / "allplot.sh")], cwd="/tmp", env=environment,
                            capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "common saved coverage 0–2 s" in result.stdout
    for name in ("cylinder_forces", "reference_forces", "reference_profiles"):
        assert (case / "figures" / f"{name}.png").stat().st_size > 1000
