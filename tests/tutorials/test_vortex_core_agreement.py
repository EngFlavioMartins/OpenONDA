"""Field-core tracking must stop when two distinct maxima are no longer resolved."""

import numpy as np
import pandas as pd
import pytest

from tests._tutorial_helpers import load_tutorial_module

postprocess = load_tutorial_module("vpm/vortex_interactions", "assets.postprocess")


def _peak(step, x, radius, vorticity=1.0, count=2, bridge=0.01):
    return {
        "step": step,
        "time": float(step),
        "x": x,
        "radius": radius,
        "vorticity": vorticity,
        "n_peaks": count,
        "bridge_ratio": bridge,
    }


def test_core_identity_survives_overtaking_and_stops_at_a_bridge():
    rows = [
        _peak(step, x, radius)
        for step, points in enumerate(
            (
                ((0.5, 1.0), (-0.5, 1.0)),
                ((0.0, 0.7), (0.6, 1.3)),
                ((0.7, 1.3), (0.8, 0.7)),
            )
        )
        for x, radius in points
    ]
    tracks, _ = postprocess.track_core_pair(pd.DataFrame(rows))
    np.testing.assert_allclose(tracks[tracks.core == 1].x, [0.5, 0.6, 0.7])
    np.testing.assert_allclose(tracks[tracks.core == 2].x, [-0.5, 0.0, 0.8])

    rows[-1]["bridge_ratio"] = 0.7
    tracks, reason = postprocess.track_core_pair(pd.DataFrame(rows))
    assert tracks.time.max() == 1
    assert "bridge" in reason


def test_tracking_starts_at_the_first_saved_field():
    peaks = pd.DataFrame(
        [
            {**_peak(step, x, 1.0), "time": time}
            for step, time, positions in ((800, 3.0, (2.0, 3.0)), (840, 3.15, (2.2, 3.2)))
            for x in positions
        ]
    )
    tracks, _ = postprocess.track_core_pair(peaks)
    assert sorted(tracks.step.unique()) == [800, 840]
    np.testing.assert_allclose(tracks[tracks.core == 1].x, [3.0, 3.2])


def test_competing_third_peak_ends_the_dominant_pair():
    rows = []
    for step in range(3):
        rows.extend(
            (
                _peak(step, 0.5 + 0.1 * step, 1.0, 0.4),
                _peak(step, -0.5 + 0.1 * step, 1.0, 0.3),
            )
        )
        if step == 1:
            rows.append(_peak(step, 0.1, 1.2, 0.2, count=3))
            for row in rows[-3:-1]:
                row["n_peaks"] = 3
    tracks, reason = postprocess.track_core_pair(pd.DataFrame(rows))
    assert tracks.time.max() == 0
    assert "third peak" in reason


def test_sampled_peaks_find_two_cores_and_flag_a_clipped_field():
    x, radius = np.linspace(-1, 1, 101), np.linspace(0, 2, 101)
    xx, rr = np.meshgrid(x, radius, indexing="ij")
    omega = np.exp(-((xx - 0.5) ** 2 + (rr - 1) ** 2) / 0.01) + np.exp(
        -((xx + 0.5) ** 2 + (rr - 1) ** 2) / 0.01
    )
    peaks = postprocess.sampled_peaks(x, radius, omega)
    np.testing.assert_allclose(sorted(peak["x"] for peak in peaks), [-0.5, 0.5])
    assert all(peak["radius"] == pytest.approx(1) for peak in peaks)
    assert all(peak["n_peaks"] == 2 and peak["bridge_ratio"] < 0.01 for peak in peaks)
    assert all(
        peak["n_peaks"] == -1 for peak in postprocess.sampled_peaks(x[:76], radius, omega[:76])
    )


def test_high_saddle_lobe_is_grouped_without_joining_the_two_main_cores():
    x, radius = np.arange(11.0), np.arange(3.0)
    omega = np.zeros((11, 3))
    omega[2:9, 1] = [10.0, 1.6, 1.7, 2.0, 1.7, 1.6, 9.0]
    peaks = postprocess.sampled_peaks(x, radius, omega, merge_bridge=0.7)
    assert len(peaks) == 2
    assert all(peak["raw_n_peaks"] == 3 for peak in peaks)
    assert peaks[0]["bridge_ratio"] < 0.2
