"""The legacy startup display gap never edits measurements or hides fresh runs."""

import importlib
import json
import sys

import numpy as np
import pandas as pd
import pytest

PLOT = importlib.import_module(
    "tutorials.coupled_fvm_vpm.01_cylinder_shedding_flow.assets.plot_reference_forces"
)
POLICY = {
    "schema": "openonda-cylinder-startup/1",
    "startup_duration": 2.0,
    "exchange_time_step": .04,
    "fvm_time_step": .008,
    "switch_step": 50,
    "startup_freestream_velocity": [1.0, .1, 0.0],
    "steady_freestream_velocity": [1.0, 0.0, 0.0],
}


def _forces():
    return pd.DataFrame({
        "time": [1.96, 2.0, 2.04, 2.08, 2.12, 20.0],
        "drag_coefficient": [1.42, 1.41, 1.42, 1.40, 1.39, 1.3],
        "lift_coefficient": [.15, .15, -7.57, -.05, -.04, 9.0],
    })


def test_display_gap_removes_interpolated_impulse_without_hiding_neighbors_or_late_peaks():
    coupled = _forces()
    before = coupled.copy(deep=True)
    time = np.array([1.96, 2.0, 2.02, 2.04, 2.06, 2.08, 2.12, 20.0])
    mask, metadata = PLOT._legacy_startup_gap(time, coupled, POLICY)
    np.testing.assert_array_equal(mask, [False, False, True, True, True, False, False, False])
    assert metadata["omitted_native_time"] == 2.04
    assert metadata["omitted_native_lift_coefficient"] == -7.57
    assert metadata["raw_history_modified"] is False
    assert metadata["error_statistics_include_omitted_sample"] is True
    pd.testing.assert_frame_equal(coupled, before)


@pytest.mark.parametrize("changed", [
    {},
    {**POLICY, "schema": "openonda-cylinder-startup/2", "startup_transition_duration": 1.0},
    {**POLICY, "startup_duration": 3.0},
    {**POLICY, "startup_freestream_velocity": [1.0, .2, 0.0]},
])
def test_unknown_or_smooth_policy_keeps_all_samples(changed):
    mask, metadata = PLOT._legacy_startup_gap(np.array([2.0, 2.04, 2.08]), _forces(), changed)
    assert not mask.any()
    assert metadata["enabled"] is False


def test_missing_native_impulse_keeps_display():
    coupled = _forces().loc[lambda frame: frame.time != 2.04]
    mask, metadata = PLOT._legacy_startup_gap(np.array([2.0, 2.04, 2.08]), coupled, POLICY)
    assert not mask.any()
    assert metadata["enabled"] is False


@pytest.mark.parametrize("include_raw", [False, True])
def test_plot_keeps_raw_error_statistics_and_preserves_csvs(tmp_path, monkeypatch, include_raw):
    coupled = _forces()
    reference = _forces()
    reference.lift_coefficient = .01
    samples = tmp_path / "samples"
    samples.mkdir()
    raw_path = samples / "forces_history.csv"
    coupled.to_csv(raw_path, index=False)
    raw_bytes = raw_path.read_bytes()
    reference_samples = tmp_path / "reference_flow/samples"
    reference_samples.mkdir(parents=True)
    reference.to_csv(reference_samples / "forces_history.csv", index=False)
    solution = tmp_path / "solution"
    solution.mkdir()
    (solution / "cylinder_startup.json").write_text(json.dumps(POLICY))
    monkeypatch.setattr(PLOT.data, "CASE_DIR", tmp_path)
    monkeypatch.setattr(PLOT.data, "reference_directory", lambda: reference_samples)
    args = ["plot_reference_forces.py", "--format", "png"]
    if include_raw:
        args.append("--include-startup-impulse")
    monkeypatch.setattr(sys, "argv", args)
    results = {}
    monkeypatch.setattr(PLOT.data, "write_json", lambda name, payload: results.update(payload))

    def check_figure(figure, axes, name, figure_format):
        assert name == "reference_forces"
        plotted = axes[1].lines[1].get_ydata()
        if include_raw:
            np.testing.assert_array_equal(plotted, coupled.lift_coefficient)
        else:
            assert np.isnan(plotted[2])
            np.testing.assert_array_equal(plotted[[0, 1, 3, 4, 5]],
                                          coupled.lift_coefficient.iloc[[0, 1, 3, 4, 5]])
            assert any("Raw data and errors unchanged" in text.get_text()
                       for text in figure.texts)
        np.testing.assert_array_equal(axes[0].lines[1].get_ydata(), coupled.drag_coefficient)
        np.testing.assert_array_equal(axes[1].lines[0].get_ydata(), reference.lift_coefficient)
        PLOT.plt.close(figure)

    monkeypatch.setattr(PLOT.data, "save_figure", check_figure)
    PLOT.main()
    _, _, _, raw_errors = PLOT.data.common_history(coupled, reference, PLOT.FORCE_COLUMNS)
    assert results["errors"] == raw_errors
    assert results["display_only_startup_omission"]["enabled"] is not include_raw
    assert raw_path.read_bytes() == raw_bytes
