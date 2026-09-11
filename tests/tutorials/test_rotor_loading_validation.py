"""The blade-loading plotter averages only a reconciled native VLM window."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tutorials.vpm.rotor_flow.assets.plot_rotor_loading_validation import (
    _native_vlm_logging_cadence,
    _native_vlm_station_keys,
    shared_vlm_window,
)


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
    path = (
        Path(__file__).resolve().parents[2]
        / "tutorials/vpm/rotor_flow/samples/rotor/vlm_chordwise_blade_0.csv"
    )
    return pd.read_csv(path, nrows=132)


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
