"""Partial delta-wing output stays inspectable without certifying completion."""

import json

import numpy as np
import pandas as pd
import pytest

from tests._tutorial_helpers import load_tutorial_module

plots = load_tutorial_module("vpm/delta_wing", "assets._delta_wing_plots")
entry = load_tutorial_module("vpm/delta_wing", "assets.plot_delta_wing")


def test_failed_run_plots_partial_native_samples_without_accepting_lineage(tmp_path, monkeypatch):
    solution = tmp_path / "solution"
    solution.mkdir()
    (solution / "vpm_metadata.json").write_text(
        json.dumps({"lifecycle": {"status": "failed"}, "state": {"step": 100, "time": 0.25}})
    )
    samples = tmp_path / "samples/delta_wing"
    monkeypatch.setattr(entry, "CASE_DIR", tmp_path)
    monkeypatch.setattr(entry, "SAMPLES_DIR", samples)
    monkeypatch.setattr(entry, "FIGURES_DIR", tmp_path / "figures")

    def forbidden(*args, **kwargs):
        pytest.fail("partial output must not finalize lineage or create a final GIF")

    monkeypatch.setattr(entry, "finalize_lineage", forbidden)
    monkeypatch.setattr(entry, "render", forbidden)
    destinations = []

    def record(source, destination, figure_format, *, partial, **kwargs):
        assert source == samples and partial and figure_format == "pdf"
        destinations.append(destination)

    for name in ("plot_forces", "plot_circulation", "plot_wake"):
        monkeypatch.setattr(entry, name, record)
    entry.plot_results("pdf")
    assert destinations == [tmp_path / "figures/partial"] * 3


def test_short_history_has_no_invented_motion_period():
    time = np.linspace(0.0025, 0.2475, 99)
    frame = pd.DataFrame(
        {"surface": "front_wing", "time": time, "translation_velocity_z": np.sin(2 * np.pi * time)}
    )
    assert plots._available_period(frame) is None


def test_mean_wake_integrates_complete_period_with_irregular_endpoints(monkeypatch):
    points = np.zeros((3, 3))

    class Grid:
        def __init__(self, time):
            self.points = points
            self.time = time

        def __getitem__(self, key):
            assert key == "velocity"
            return np.full((3, 3), 2 * self.time + 3)

    monkeypatch.setattr(plots.pv, "read", Grid)
    frames = [(time, time, "source") for time in (0.0, 0.3, 0.8, 1.2)]
    result = plots._wake_average([frames], end=1.1, period=0.9)
    # Integral mean of 2t+3 over [0.2, 1.1] is 4.3.
    np.testing.assert_allclose(result[0][1], 4.3, rtol=0, atol=1e-12)


def test_mean_wake_rejects_less_than_one_period():
    frames = [(0.025, None, "source"), (0.225, None, "source")]
    with pytest.raises(ValueError, match="full measured heave period"):
        plots._wake_average([frames], end=0.225, period=1.0)
