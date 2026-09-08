"""Reference scoring must not hide phase errors or invent post-merger identities."""

import numpy as np
import pandas as pd
import pytest

from tutorials.vpm.vortex_interactions.assets.assess_lbm_agreement import (
    coherent_tracks,
    radius_score,
)


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
    with pytest.raises(ValueError, match="entire requested interval"):
        radius_score(tracks, reference, 0, 1.1)
    tracks.loc[2, "x"] = -1
    with pytest.raises(ValueError, match="monotone"):
        radius_score(tracks, reference, 0, 1)


def test_sampled_peaks_recover_two_cores_and_reject_a_clipped_core():
    from tutorials.vpm.vortex_interactions.assets.assess_lbm_agreement import sampled_peaks

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

    from tutorials.vpm.vortex_interactions.assets import assess_lbm_agreement as analysis

    monkeypatch.setattr(analysis, "STUDY_DIR", tmp_path)
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
                "signature": {
                    "dt": dt,
                    "steps": int(0.1 / dt),
                    "tag": run,
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
