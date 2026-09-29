"""The blade-loading plotter averages only a reconciled native VLM window."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests._tutorial_helpers import load_tutorial_module

_loading = load_tutorial_module("vpm/rotor_flow", "assets.plot_rotor_loading_validation")
_native_vlm_logging_cadence = _loading._native_vlm_logging_cadence
_native_vlm_station_keys = _loading._native_vlm_station_keys
shared_vlm_window = _loading.shared_vlm_window


def _histories(*, chord_time_offset=0.0, chord_panels=1):
    rows = []
    chord_rows = []
    for step, time in enumerate(np.arange(0.0, 10.1, 1.0)):
        for station_id in (1, 2):
            rows.append({"step": step, "time": time, "station_id": station_id})
            for chord_index in range(chord_panels):
                chord_rows.append(
                    {
                        "step": step,
                        "time": time + chord_time_offset,
                        "station_id": station_id,
                        "chord_index": chord_index,
                    }
                )
    span = pd.DataFrame(rows)
    chord = pd.DataFrame(chord_rows)
    return span, chord


def test_shared_vlm_window_uses_final_five_revolutions():
    span, chord = _histories()

    span_window, chord_window, cutoff, end = shared_vlm_window(
        span,
        chord,
        rotation_period=1.0,
        end_time=10.0,
        expected_cadence=1.0,
        accepted_step_size=0.1,
    )

    assert (cutoff, end) == (5.0, 10.0)
    np.testing.assert_array_equal(span_window.step.unique(), np.arange(5, 11))
    np.testing.assert_array_equal(chord_window.step.unique(), np.arange(5, 11))


@pytest.mark.parametrize("offset", [0.001, -0.001])
def test_shared_vlm_window_rejects_span_chord_clock_mismatch(offset):
    span, chord = _histories(chord_time_offset=offset)

    with pytest.raises(ValueError, match="clocks do not match"):
        shared_vlm_window(span, chord, rotation_period=1.0)


def test_shared_vlm_window_rejects_conflicting_duplicate_step():
    span, chord = _histories()
    duplicate = chord[(chord.step == 4) & (chord.station_id == 1)].iloc[[0]]
    chord = pd.concat([chord, duplicate], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate step/station/chord-panel"):
        shared_vlm_window(span, chord, rotation_period=1.0)


def test_shared_vlm_window_preserves_valid_multi_panel_stations():
    span, chord = _histories(chord_panels=6)

    _, chord_window, _, _ = shared_vlm_window(
        span,
        chord,
        rotation_period=1.0,
        end_time=10.0,
        expected_cadence=1.0,
        accepted_step_size=0.1,
    )

    selected = chord_window[(chord_window.step == 5) & (chord_window.station_id == 1)]
    np.testing.assert_array_equal(selected.chord_index, np.arange(6))


def test_shared_vlm_window_rejects_missing_chord_panel():
    span, chord = _histories(chord_panels=6)
    chord = chord[~((chord.step == 7) & (chord.station_id == 2) & (chord.chord_index == 5))]

    with pytest.raises(ValueError, match="incomplete chord panels"):
        shared_vlm_window(span, chord, rotation_period=1.0)


def test_shared_vlm_window_rejects_equal_but_truncated_histories():
    span, chord = _histories()
    span = span[span.time <= 3.0]
    chord = chord[chord.time <= 3.0]

    with pytest.raises(ValueError, match="native horizon"):
        shared_vlm_window(
            span,
            chord,
            rotation_period=1.0,
            end_time=10.0,
            expected_cadence=1.0,
            accepted_step_size=0.1,
        )


def test_shared_vlm_window_rejects_gap_beyond_native_cadence():
    span, chord = _histories()
    span = span[span.step != 6]
    chord = chord[chord.step != 6]

    with pytest.raises(ValueError, match="gap beyond"):
        shared_vlm_window(
            span,
            chord,
            rotation_period=1.0,
            end_time=10.0,
            expected_cadence=1.0,
            accepted_step_size=0.1,
        )


def test_shared_vlm_window_rejects_missing_step_station_key():
    span, chord = _histories()
    chord = chord[~((chord.step == 7) & (chord.station_id == 2))]

    with pytest.raises(ValueError, match="step/station keys"):
        shared_vlm_window(span, chord, rotation_period=1.0)


def _native_chordwise_first_step():
    path = Path(__file__).with_name("fixtures") / "rotor_chordwise_first_step.csv"
    return pd.read_csv(path)


def test_native_chordwise_fixture_has_six_panels_per_station():
    chord = _native_chordwise_first_step()

    keys = _native_vlm_station_keys(
        chord,
        "native chordwise",
        panel_column="chord_index",
        expected_panel_indices=range(6),
    )

    assert len(keys) == 22
    assert chord.groupby("station_id").chord_index.nunique().eq(6).all()


def test_native_chordwise_fixture_rejects_missing_panel():
    chord = _native_chordwise_first_step()
    station_id = chord.station_id.iloc[0]
    chord = chord[~((chord.station_id == station_id) & (chord.chord_index == 5))]

    with pytest.raises(ValueError, match="incomplete chord panels"):
        _native_vlm_station_keys(
            chord,
            "native chordwise",
            panel_column="chord_index",
            expected_panel_indices=range(6),
        )


def test_native_chordwise_fixture_rejects_duplicate_panel():
    chord = _native_chordwise_first_step()
    chord = pd.concat([chord, chord.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate step/station/chord-panel"):
        _native_vlm_station_keys(
            chord,
            "native chordwise",
            panel_column="chord_index",
            expected_panel_indices=range(6),
        )


def test_native_vlm_cadence_uses_force_logging_interval():
    metadata = {"configuration": {"numerics": {"vlm": {"logging_interval_steps": 1}}}}

    assert _native_vlm_logging_cadence(metadata, 0.006) == pytest.approx(0.006)


def test_native_vlm_cadence_rejects_sparse_coupled_logging():
    metadata = {"configuration": {"numerics": {"vlm": {"logging_interval_steps": 2}}}}

    with pytest.raises(ValueError, match="every accepted owner step"):
        _native_vlm_logging_cadence(metadata, 0.006)


@pytest.mark.parametrize("with_steps", [True, False])
def test_accepted_history_preserves_source_and_excludes_live_tail(with_steps):
    common = load_tutorial_module("vpm/rotor_flow", "assets._common")
    data = pd.DataFrame({"time": [6.474, 6.48, 6.486, 6.492], "step": [1079, 1080, 1081, 1082]})
    if not with_steps:
        data = data.drop(columns="step")
    original = data.copy(deep=True)
    metadata = {
        "configuration": {"numerics": {"time_step_size": 0.006}},
        "state": {"initial_time": 6.048, "initial_step": 1008, "step": 1080, "time": 6.48},
    }
    with pytest.warns(UserWarning, match="retained on disk"):
        selected = common.accepted_history(data, metadata)
    assert selected.time.tolist() == [6.474, 6.48]
    pd.testing.assert_frame_equal(data, original)


def test_accepted_history_rejects_relabelled_step_clock():
    common = load_tutorial_module("vpm/rotor_flow", "assets._common")
    metadata = {
        "configuration": {"numerics": {"time_step_size": 0.006}},
        "state": {"initial_time": 6.048, "initial_step": 1008, "step": 1080, "time": 6.48},
    }
    data = pd.DataFrame({"time": [6.48, 6.492], "step": [1080, 1081]})
    with pytest.raises(ValueError, match="clock disagrees"):
        common.accepted_history(data, metadata)


def test_accepted_history_loading_window_ends_at_checkpoint():
    common = load_tutorial_module("vpm/rotor_flow", "assets._common")
    span, chord = _histories()
    metadata = {
        "configuration": {"numerics": {"time_step_size": 1.0}},
        "state": {"initial_time": 0.0, "initial_step": 0, "step": 8, "time": 8.0},
    }
    with pytest.warns(UserWarning):
        span = common.accepted_history(span, metadata)
    with pytest.warns(UserWarning):
        chord = common.accepted_history(chord, metadata)
    _, _, cutoff, end = shared_vlm_window(span, chord, rotation_period=1.0, end_time=8.0)
    assert (cutoff, end) == (3.0, 8.0)


def test_rotor_animation_excludes_newer_native_plane(tmp_path):
    animation = load_tutorial_module("vpm/rotor_flow", "assets.render_rotor_animation")
    pvd = tmp_path / "wake_1D.pvd"
    pvd.write_text(
        '<VTKFile><Collection><DataSet timestep="6.48" file="accepted.vts"/>'
        '<DataSet timestep="6.54" file="newer.vts"/></Collection></VTKFile>'
    )
    assert animation._pvd_frames(pvd, end_time=6.48) == [(6.48, tmp_path / "accepted.vts")]


def test_rotor_animation_excludes_newer_coupled_backup(tmp_path, monkeypatch):
    import json

    animation = load_tutorial_module("vpm/rotor_flow", "assets.render_rotor_animation")
    accepted = tmp_path / "vpm_001080.h5"
    newer = tmp_path / "vpm_001090.h5"
    (tmp_path / "vpm_metadata.json").write_text(json.dumps({"state": {"step": 1080, "time": 6.48}}))
    monkeypatch.setattr(animation, "vpm_backup_files", lambda path: [accepted, newer])
    monkeypatch.setattr(
        animation,
        "_read_backup_clock",
        lambda path: (1080, 6.48) if path == accepted else (1090, 6.54),
    )
    assert animation._coupled_frames(tmp_path) == [(6.48, accepted)]


def test_accepted_history_retains_earlier_segment_with_different_dt():
    common = load_tutorial_module("vpm/rotor_flow", "assets._common")
    metadata = {
        "configuration": {"numerics": {"time_step_size": 0.25}},
        "state": {"initial_time": 2.0, "initial_step": 2, "step": 4, "time": 2.5},
    }
    data = pd.DataFrame({"step": [0, 1, 2, 3, 4], "time": [0, 1, 2, 2.25, 2.5]})
    pd.testing.assert_frame_equal(common.accepted_history(data, metadata), data)
    data.loc[1, "time"] = 2.125
    with pytest.raises(ValueError, match="clock disagrees"):
        common.accepted_history(data, metadata)


def test_loading_cadence_only_constrains_recorded_continuation_segment():
    span, chord = _histories()
    for table in (span, chord):
        table.loc[table.step >= 6, "time"] = 5 + (table.loc[table.step >= 6, "step"] - 5) * 0.25
    _, _, cutoff, end = shared_vlm_window(
        span,
        chord,
        rotation_period=1,
        revolutions=2,
        expected_cadence=0.25,
        accepted_step_size=0.25,
        cadence_start_time=5,
    )
    assert (cutoff, end) == (4.25, 6.25)
    for table in (span, chord):
        table.loc[table.step >= 8, "time"] += 1
    with pytest.raises(ValueError, match="gap beyond"):
        shared_vlm_window(
            span,
            chord,
            rotation_period=1,
            expected_cadence=0.25,
            accepted_step_size=0.25,
            cadence_start_time=5,
        )
