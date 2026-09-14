"""Reference scoring must not hide phase errors or invent post-merger identities."""

import numpy as np
import pandas as pd
import pytest

from tests._tutorial_helpers import load_tutorial_module

_assessment = load_tutorial_module("vpm/vortex_interactions", "assets.plot_lbm_comparison")
coherent_tracks = _assessment.coherent_tracks
radius_score = _assessment.radius_score


def test_core_identity_survives_overtaking_and_peak_rank_changes():
    rows = []
    for step, points in enumerate(
        [
            [(0.5, 1.0), (-0.5, 1.0)],
            [(0.0, 0.7), (0.6, 1.3)],
            [(0.7, 1.3), (0.8, 0.7)],
        ]
    ):
        for x, radius in points:
            rows.append(
                {
                    "step": step,
                    "time": step,
                    "x": x,
                    "radius": radius,
                    "n_peaks": 2,
                    "strongest_peak_pair_bridge_ratio": 0.01,
                }
            )
    tracks, _ = coherent_tracks(pd.DataFrame(rows))
    np.testing.assert_allclose(tracks[tracks.ring == 1].x, [0.5, 0.6, 0.7])
    np.testing.assert_allclose(tracks[tracks.ring == 2].x, [-0.5, 0.0, 0.8])
    rows[-1]["strongest_peak_pair_bridge_ratio"] = 0.7
    tracks, reason = coherent_tracks(pd.DataFrame(rows))
    assert tracks.time.max() == 1
    assert "bridge" in reason


def test_core_tracking_uses_the_first_saved_field_when_step_zero_is_missing():
    peaks = pd.DataFrame(
        [
            {
                "step": step,
                "time": time,
                "x": x,
                "radius": 1.0,
                "n_peaks": 2,
                "strongest_peak_pair_bridge_ratio": 0.0,
            }
            for step, time, positions in ((800, 3.0, (2.0, 3.0)), (840, 3.15, (2.2, 3.2)))
            for x in positions
        ]
    )
    tracks, _ = coherent_tracks(peaks)
    assert sorted(tracks.step.unique()) == [800, 840]
    np.testing.assert_allclose(tracks[tracks.ring == 1].x, [3.0, 3.2])


def test_uniform_distance_score_weights_rings_equally_and_keeps_radius_error():
    tracks = pd.DataFrame(
        [
            {"ring": ring, "time": x, "x": x, "radius": 1 + offset}
            for ring, offset, coordinates in ((1, 0.1, [0, 0.1, 1]), (2, 0.2, [0, 1]))
            for x in coordinates
        ]
    )
    reference = pd.DataFrame(
        [{"ring": ring, "x_over_R0": x + 2.5, "R_over_R0": 1} for ring in (1, 2) for x in (0, 1)]
    )
    score = radius_score(tracks, reference, 0, 1)
    assert score["rms_radius_over_R0"] == pytest.approx(np.sqrt((0.1**2 + 0.2**2) / 2))
    duplicate = tracks.iloc[[1]].copy()
    duplicate["time"] += 0.01
    repeated = pd.concat([tracks, duplicate], ignore_index=True)
    assert radius_score(repeated, reference, 0, 1) == score
    repeated.loc[repeated.index[-1], "radius"] += 0.01
    with pytest.raises(ValueError, match="monotone"):
        radius_score(repeated, reference, 0, 1)
    with pytest.raises(ValueError, match="entire requested interval"):
        radius_score(tracks, reference, 0, 1.1)
    tracks.loc[2, "x"] = -1
    with pytest.raises(ValueError, match="monotone"):
        radius_score(tracks, reference, 0, 1)


def test_sampled_peaks_recover_two_cores_and_reject_a_clipped_core():
    sampled_peaks = _assessment.sampled_peaks

    x, r = np.linspace(-1, 1, 101), np.linspace(0, 2, 101)
    xx, rr = np.meshgrid(x, r, indexing="ij")
    omega = np.exp(-((xx - 0.5) ** 2 + (rr - 1) ** 2) / 0.01) + np.exp(
        -((xx + 0.5) ** 2 + (rr - 1) ** 2) / 0.01
    )
    peaks = sampled_peaks(x, r, omega)
    assert len(peaks) == 2
    np.testing.assert_allclose(sorted(p["x"] for p in peaks), [-0.5, 0.5])
    assert all(p["radius"] == pytest.approx(1) for p in peaks)
    assert all(p["n_peaks"] == 2 and p["strongest_peak_pair_bridge_ratio"] < 0.01 for p in peaks)
    clipped = sampled_peaks(x[:76], r, omega[:76])
    assert all(p["n_peaks"] == -1 for p in clipped)


def test_temporal_comparison_uses_identical_sampler_points_and_equal_times(tmp_path, monkeypatch):
    import pyvista as pv

    analysis = _assessment

    monkeypatch.setattr(
        analysis, "sample_directory", lambda run: tmp_path / run / "samples/diagnostics"
    )
    reports = []
    for run, dt, multiplier in (("coarse", 0.01, 1.1), ("fine", 0.005, 1.0)):
        folder = tmp_path / run / "samples/diagnostics"
        folder.mkdir(parents=True)
        fields = []
        for step, time in ((0, 0.0), (10, 0.1)):
            x, y, z = np.meshgrid([0.0, 1.0], [0.0, 1.0], [0.0], indexing="ij")
            grid = pv.StructuredGrid(x, y, z)
            factor = multiplier if time else 1.0
            grid.point_data["velocity"] = np.ones((4, 3)) * factor
            grid.point_data["vorticity"] = np.ones((4, 3)) * 2 * factor
            filename = f"core_section_{step:06d}.vts"
            grid.save(folder / filename)
            fields.append({"file": filename, "time": time})
        reports.append(
            {
                "run": run,
                "settings": {
                    "dt": dt,
                    "diffusion": "CS",
                    "smagorinsky": 0.2,
                },
                "sampler_fields": fields,
            }
        )
    comparisons = analysis.temporal_field_comparisons(reports)
    assert len(comparisons) == 2
    assert comparisons[0]["velocity_initial_bitwise_equal"]
    assert comparisons[0]["vorticity_relative_l2"] == 0
    assert comparisons[1]["time"] == 0.1
    assert comparisons[1]["velocity_relative_l2"] == pytest.approx(0.1)
    assert comparisons[1]["vorticity_relative_l2"] == pytest.approx(0.1)
    reports[0]["settings"]["filter_width"] = None
    reports[1]["settings"]["filter_width"] = 0.05
    assert analysis.temporal_field_comparisons(reports) == []
    reports[0]["settings"]["filter_width"] = 0.05
    reports[0]["settings"]["dt"] = reports[1]["settings"]["dt"]
    assert analysis.temporal_field_comparisons(reports) == []


def test_high_saddle_grouping_preserves_separate_cores_and_reports_lobe_extent():
    x, r = np.arange(11.0), np.arange(5.0)
    omega = np.zeros((11, 5))
    omega[2:5, 2] = [10.0, 9.8, 9.9]
    omega[8, 2] = 10.5
    raw = _assessment.sampled_peaks(x, r, omega)
    assert len(raw) == 3
    peaks = _assessment.sampled_peaks(x, r, omega, peak_merge_bridge=0.9)
    assert len(peaks) == 2
    assert all(p["raw_n_peaks"] == 3 and p["n_peaks"] == 2 for p in peaks)
    grouped = next(p for p in peaks if p["cluster_n_peaks"] == 2)
    assert grouped["cluster_x_span"] == [2.0, 4.0]
    assert all(p["strongest_peak_pair_bridge_ratio"] == 0 for p in peaks)
    omega[3, 2] = 1.0
    assert len(_assessment.sampled_peaks(x, r, omega, peak_merge_bridge=0.9)) == 3


def test_passage_timing_resolves_a_known_period_without_inventing_reference_time():
    times = np.linspace(0, 4 * np.pi, 101)
    tracks = pd.DataFrame(
        [
            {
                "time": time,
                "ring": ring,
                "x": time + sign * np.cos(time),
                "radius": 1 + sign * 0.2 * np.sin(time),
            }
            for time in times
            for ring, sign in ((1, 1), (2, -1))
        ]
    )
    measured = _assessment.leapfrog_events(tracks)
    expected = np.pi * np.array([0.5, 1.5, 2.5, 3.5])
    np.testing.assert_allclose([row["time"] for row in measured["passages"]], expected, atol=1e-3)
    np.testing.assert_allclose(measured["full_cycle_periods"], 2 * np.pi, atol=1e-3)
    for event, time in zip(measured["passages"], expected, strict=True):
        assert event["time_bracket"][0] <= time <= event["time_bracket"][1]
    assert measured["lbm_temporal_phase_error"] is None
    assert measured["lbm_passage_times"] is None


def test_axial_contact_requires_order_reversal_and_zero_plateaus_are_counted_once():
    tracks = pd.DataFrame(
        [
            {"time": time, "ring": ring, "x": time + sign * distance, "radius": 1 + sign * 0.2}
            for time, distance in enumerate([1, 0, 1, 0, 0, -1])
            for ring, sign in ((1, 1), (2, -1))
        ]
    )
    events = _assessment.leapfrog_events(tracks)["passages"]
    assert len(events) == 1
    assert events[0]["time_bracket"] == [2, 5]
