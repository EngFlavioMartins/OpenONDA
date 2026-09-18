"""Partial delta-wing output stays inspectable without certifying completion."""

import json

import numpy as np
import pandas as pd
import pytest

from tests._tutorial_helpers import load_tutorial_module

postprocess = load_tutorial_module("vpm/delta_wing", "assets.postprocess")
plot_forces = load_tutorial_module("vpm/delta_wing", "assets.plot_delta_wing_forces")
plot_force_cycles = load_tutorial_module("vpm/delta_wing", "assets.plot_delta_wing_force_cycles")
plot_circulation = load_tutorial_module(
    "vpm/delta_wing", "assets.plot_delta_wing_circulation_history"
)
plot_wake_streamwise = load_tutorial_module(
    "vpm/delta_wing", "assets.plot_delta_wing_wake_streamwise"
)
plot_wake_vertical = load_tutorial_module("vpm/delta_wing", "assets.plot_delta_wing_wake_vertical")

PER_FIGURE_SCRIPTS = (
    (plot_forces, "plot_forces"),
    (plot_force_cycles, "plot_force_cycles"),
    (plot_circulation, "plot_circulation"),
    (plot_wake_streamwise, "plot_wake_streamwise"),
    (plot_wake_vertical, "plot_wake_vertical"),
)


@pytest.fixture
def failed_case(tmp_path):
    solution = tmp_path / "solution"
    solution.mkdir()
    (solution / "vpm_metadata.json").write_text(
        json.dumps({"lifecycle": {"status": "failed"}, "state": {"step": 100, "time": 0.25}})
    )
    return solution


def test_failed_run_resolves_partial_destination_without_finalizing(failed_case, monkeypatch):
    monkeypatch.setattr(postprocess, "CASE_DIR", failed_case.parent)
    sources = postprocess.resolve_plot_sources(failed_case.parent)

    assert sources.complete is False
    assert sources.status == "failed"
    assert sources.state == {"step": 100, "time": 0.25}
    assert sources.samples_arg == failed_case.parent / "samples" / "delta_wing"
    assert sources.solution_dirs == [failed_case]
    assert sources.destination == failed_case.parent / "figures" / "partial"


def test_failed_run_routes_every_per_figure_script_to_partial(failed_case, monkeypatch, capsys):
    monkeypatch.setattr(postprocess, "CASE_DIR", failed_case.parent)
    sources = postprocess.resolve_plot_sources(failed_case.parent)
    destinations = []

    def record(source, destination, figure_format, *args, partial, **kwargs):
        assert source == failed_case.parent / "samples" / "delta_wing"
        assert partial is True
        assert figure_format == "pdf"
        destinations.append(destination)

    for script, plotter in PER_FIGURE_SCRIPTS:
        monkeypatch.setattr(script, "resolve_plot_sources", lambda: sources)
        monkeypatch.setattr(script, plotter, record)
        monkeypatch.setattr("sys.argv", ["plot", "--format", "pdf"])
        script.main()

    assert destinations == [failed_case.parent / "figures" / "partial"] * len(PER_FIGURE_SCRIPTS)
    assert "partial diagnostics" in capsys.readouterr().out


def test_completed_run_resolves_accepted_lineage_target(failed_case, monkeypatch):
    metadata = {"lifecycle": {"status": "completed"}, "state": {"step": 8000, "time": 20.0}}
    (failed_case / "vpm_metadata.json").write_text(json.dumps(metadata))
    sources = postprocess.resolve_plot_sources(failed_case.parent)

    assert sources.complete is True
    assert sources.samples_arg is None
    assert sources.solution_dirs is None
    assert sources.destination == failed_case.parent / "figures"


def test_short_history_has_no_invented_motion_period():
    time = np.linspace(0.0025, 0.2475, 99)
    frame = pd.DataFrame(
        {"surface": "front_wing", "time": time, "translation_velocity_z": np.sin(2 * np.pi * time)}
    )
    assert postprocess._available_period(frame) is None


def test_mean_wake_integrates_complete_period_with_irregular_endpoints(monkeypatch):
    points = np.zeros((3, 3))

    class Grid:
        def __init__(self, time):
            self.points = points
            self.time = time

        def __getitem__(self, key):
            assert key == "velocity"
            return np.full((3, 3), 2 * self.time + 3)

    monkeypatch.setattr(postprocess.pv, "read", Grid)
    frames = [(time, time, "source") for time in (0.0, 0.3, 0.8, 1.2)]
    result = postprocess._wake_average([frames], end=1.1, period=0.9)
    # Integral mean of 2t+3 over [0.2, 1.1] is 4.3.
    np.testing.assert_allclose(result[0][1], 4.3, rtol=0, atol=1e-12)


def test_mean_wake_rejects_less_than_one_period():
    frames = [(0.025, None, "source"), (0.225, None, "source")]
    with pytest.raises(ValueError, match="full measured heave period"):
        postprocess._wake_average([frames], end=0.225, period=1.0)
