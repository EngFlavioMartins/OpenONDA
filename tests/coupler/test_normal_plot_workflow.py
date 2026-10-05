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


@pytest.mark.parametrize(
    "case", [CYLINDER, CUBE, CYLINDER + "/reference_flow", CUBE + "/reference_flow"]
)
@pytest.mark.parametrize("launcher", ["allrun.sh", "allcontinue.sh"])
def test_launchers_preserve_outputs_and_propagate_failure(tmp_path, case, launcher):
    shutil.copy2(CASES / case / launcher, tmp_path / launcher)
    if launcher == "allcontinue.sh":
        shutil.copy2(CASES / case / "allrun.sh", tmp_path / "allrun.sh")
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


def _profile(path, times, velocity_deficit=0.1, x_position=1):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for time in times:
        for y in (-1.0, 0.0, 1.0):
            rows.append([time, x_position, y, 0, 1 - velocity_deficit * (1 - abs(y)), 0.01 * y, 0])
    pd.DataFrame(
        rows,
        columns=[
            "time",
            "position_x",
            "position_y",
            "position_z",
            "velocity_x",
            "velocity_y",
            "velocity_z",
        ],
    ).to_csv(path, index=False)


def _profile_metadata(case):
    solution = case / "solution"
    solution.mkdir(exist_ok=True)
    fvm_box = {"xmin": -1.6, "xmax": 2.4, "ymin": -1.6, "ymax": 1.6, "zmin": -0.48, "zmax": 0.48}
    transfer_box = {
        "xmin": -1.25,
        "xmax": 2.05,
        "ymin": -1.25,
        "ymax": 1.25,
        "zmin": -0.48,
        "zmax": 0.48,
    }
    (solution / "run_metadata.json").write_text(
        json.dumps(
            {
                "fvm_solver": {"fvm_domain": fvm_box},
                "coupler": {"transfer_region_bounds": transfer_box},
            }
        )
    )
    (solution / "fvm_metadata.json").write_text(
        json.dumps(
            {
                "configuration": {
                    "samplers": [
                        {
                            "type": "ForceSampler",
                            "patch_names": ["cylinder"],
                            "reference_length": 1.0,
                            "reference_velocity": 1.0,
                        }
                    ]
                },
            }
        )
    )


def test_profile_frames_match_native_clock_precision(tmp_path):
    data = importlib.import_module(PACKAGE + "postprocess")
    left, right = tmp_path / "left.csv", tmp_path / "right.csv"
    _profile(left, [1, 11])
    _profile(right, [1, 10.999999999999])
    frames = list(data.coincident_profiles((left, right)))
    assert [time for time, _ in frames] == [1, 11]
    assert len(frames[-1][1][right]) == 3
    assert frames[-1][1][right].time.iloc[0] == 10.999999999999
    _profile(right, [1, 11.00000004])
    assert [time for time, _ in data.coincident_profiles((left, right))] == [1]
    _profile(right, [2])
    assert list(data.coincident_profiles((left, right))) == []


def test_force_coverage_is_limited_and_explicitly_interpolated():
    data = importlib.import_module(PACKAGE + "postprocess")
    candidate = pd.DataFrame({"time": [0, 1, 2], "cd": [0, 1, 2]})
    reference = pd.DataFrame({"time": [0, 0.5, 1, 2, 100], "cd": [0, 0.5, 1, 2, 100]})
    times, left, right, errors = data.common_history(candidate, reference, ("cd",))
    assert times[-1] == 2
    np.testing.assert_array_equal(left, right)
    assert errors["cd"]["maximum"] == 0
    report = data.history_coverage(candidate, reference)
    assert report["available_time_intervals"]["reference"] == [0, 100]
    assert report["comparison_time_interval"] == [0, 2]
    assert "no time shift" in report["time_alignment"]


def test_profile_sequence_uses_all_coincident_states_without_time_annotations(
    tmp_path, monkeypatch
):
    plot = importlib.import_module(PACKAGE + "plot_velocity_profiles")
    data = plot.data
    reference = tmp_path / "reference_flow/samples"
    for x in (2, 4):
        _profile(
            reference / f"transverse_x{x}.csv",
            [0, 1, 2, 100],
            velocity_deficit=0.0001,
            x_position=x,
        )
        _profile(
            tmp_path / "samples" / f"vpm_transverse_x{x}.csv",
            [0, 1, 2],
            velocity_deficit=0.0001,
            x_position=x,
        )
    _profile(
        tmp_path / "samples/fvm_transverse_x2.csv", [0, 2], velocity_deficit=0.0001, x_position=2
    )
    _profile_metadata(tmp_path)
    monkeypatch.setattr(data, "CASE_DIR", tmp_path)
    monkeypatch.setattr(data, "FIGURES", tmp_path / "figures")
    monkeypatch.setattr(data, "AUXILIARY", tmp_path / "figures/auxiliary")
    monkeypatch.setattr(data, "reference_directory", lambda: reference)
    monkeypatch.setattr(sys, "argv", ["plot_velocity_profiles.py"])
    figures = data.FIGURES
    figures.mkdir()
    save_figure = data.save_figure

    def check_figure(figure, axes, name, figure_format):
        assert not figure.texts
        assert all(not axis.get_title() for axis in figure.axes)
        assert [text.get_text() for text in figure.legends[0].get_texts()] == [
            "Reference flow",
            "Coupled FVM",
            "Coupled VPM",
        ]
        assert all(len(axis.patches) == 2 for axis in (figure.axes[0], figure.axes[2]))
        assert all(not axis.patches for axis in (figure.axes[1], figure.axes[3]))
        save_figure(figure, axes, name, figure_format)

    monkeypatch.setattr(data, "save_figure", check_figure)
    plot.main()
    report = json.loads((tmp_path / "figures/auxiliary/velocity_profile_errors.json").read_text())
    assert report["reference"] == "reference_flow/samples"
    assert [frame["time"] for frame in report["frames"]] == [0, 2]
    assert "no time interpolation" in report["time_alignment"]
    for frame in report["frames"]:
        for name in frame["files"]:
            assert (figures / name).stat().st_size > 1000


def test_cube_pvd_reader_handles_attribute_order_and_rejects_ambiguous_clocks(tmp_path):
    from openonda.results import read_pvd_frames

    path = tmp_path / "frames.pvd"
    path.write_text(
        '<VTKFile><Collection><DataSet file="a.vts" timestep="1"/>'
        '<DataSet file="b.vts" timestep="2"/></Collection></VTKFile>'
    )
    assert [time for time, _ in read_pvd_frames(path)] == [1, 2]
    path.write_text(path.read_text().replace('timestep="2"', 'timestep="1.00000000001"'))
    with pytest.raises(ValueError, match="unambiguous"):
        read_pvd_frames(path)


def _coupling_records():
    records = []
    for time, substeps in ((1, 5), (2, 4)):
        timings = {
            "vpm": 10,
            "fvm": 5,
            "vpm_boundary_condition": 2,
            "transfer": 3,
            "state_checks_and_samplers": 20,
            "reporting": 1,
            "backup": 7,
            "coupling_control_and_wait": 2,
            "total": 50,
            "evolution_total": 20,
            "last_sweep_donor_gather": 1,
        }
        records.append(
            {
                "time": time,
                "n_fvm_substeps": substeps,
                "timing_seconds": timings,
                "transfer": {
                    "n_particles_after": 100000 * time,
                    "renewal_conservation_error": 1e-9 * time,
                    "renewal_vortex_strength_tolerance": 1e-5,
                    "renewal_linear_impulse_error": 2e-9 * time,
                    "renewal_linear_impulse_tolerance": 1e-4,
                },
            }
        )
    return records


def test_cylinder_costs_include_all_exclusive_phases_and_normalize_each_record():
    plot = importlib.import_module(PACKAGE + "plot_coupling_diagnostics")
    records = _coupling_records()
    costs = plot._timing_per_fvm_step(records)
    np.testing.assert_allclose(costs.sum(axis=0), [10, 12.5])
    np.testing.assert_allclose(costs, [[2, 2.5], [1, 1.25], [1, 1.25], [6, 7.5]])


def test_cylinder_diagnostics_read_only_complete_live_records(tmp_path, monkeypatch):
    plot = importlib.import_module(PACKAGE + "plot_coupling_diagnostics")
    monkeypatch.setattr(plot.data, "CASE_DIR", tmp_path)
    path = tmp_path / "solution/coupler_diagnostics.jsonl"
    path.parent.mkdir()
    record = json.dumps(_coupling_records()[0]) + "\n"
    path.write_text(record + '{"time":')
    assert plot._records() == _coupling_records()[:1]
    path.write_text(record + '{"time":\n')
    with pytest.raises(json.JSONDecodeError):
        plot._records()
    path.write_text('{"time":\n' + record)
    with pytest.raises(json.JSONDecodeError):
        plot._records()


def test_cylinder_diagnostics_plot_costs_and_population_without_conservation_panel(monkeypatch):
    plot = importlib.import_module(PACKAGE + "plot_coupling_diagnostics")
    monkeypatch.setattr(plot, "_records", _coupling_records)

    def check_figure(figure, axes, name, figure_format):
        assert name == "coupling_diagnostics"
        assert len(axes) == 2
        assert len(axes[0].collections) == 4
        assert axes[0].get_yscale() == "log"
        assert len(axes[1].lines) == 1
        np.testing.assert_allclose(axes[1].lines[0].get_ydata(), [0.1, 0.2])
        assert [text.get_text() for text in figure.legends[0].get_texts()] == [
            "VPM",
            "FVM",
            "coupling",
            "sampling and output",
        ]
        assert len(figure.legends) == 1
        plot.plt.close(figure)

    monkeypatch.setattr(plot.data, "save_figure", check_figure)
    plot.plot("both")


def test_normal_cylinder_allplot_on_incomplete_synthetic_data(tmp_path):
    case = tmp_path / "case"
    assets = case / "assets"
    assets.mkdir(parents=True)
    shutil.copy2(CASES / CYLINDER / "allplot.sh", case / "allplot.sh")
    for filename in (
        "plot_reference_forces.py",
        "plot_velocity_profiles.py",
        "velocity_profile_data.py",
        "plot_coupling_diagnostics.py",
        "postprocess.py",
    ):
        shutil.copy2(CASES / CYLINDER / "assets" / filename, assets / filename)
    (case / "solution").mkdir()
    (case / "solution/coupler_diagnostics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in _coupling_records())
    )
    for relative, times in (("samples", [0, 1, 2]), ("reference_flow/samples", [0, 1, 2, 100])):
        directory = case / relative
        directory.mkdir(parents=True)
        pd.DataFrame(
            {
                "time": times,
                "drag_coefficient": np.asarray(times) * 0.01 + 1,
                "lift_coefficient": np.asarray(times) * 0.005,
            }
        ).to_csv(directory / "forces_history.csv", index=False)
    for x in (2, 4):
        _profile(
            case / "reference_flow/samples" / f"transverse_x{x}.csv", [0, 2, 100], x_position=x
        )
        _profile(case / "samples" / f"vpm_transverse_x{x}.csv", [0, 2], x_position=x)
    _profile(case / "samples/fvm_transverse_x2.csv", [0, 2], x_position=2)
    _profile_metadata(case)
    environment = {
        **os.environ,
        "MPLBACKEND": "Agg",
        "PYTHONPATH": str(ROOT),
        "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"],
    }
    result = subprocess.run(
        ["bash", str(case / "allplot.sh"), "both"],
        cwd="/tmp",
        env=environment,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "common saved coverage 0–2 s" in result.stdout
    assert "2 common saved states, t=0–2 s" in result.stdout
    assert "full recorded cost per FVM step" in result.stdout
    for name in (
        "coupling_diagnostics",
        "reference_forces",
        "velocity_profiles_t0",
        "velocity_profiles_t2",
    ):
        assert (case / "figures" / f"{name}.png").stat().st_size > 1000
        assert (case / "figures" / f"{name}.pdf").stat().st_size > 1000
