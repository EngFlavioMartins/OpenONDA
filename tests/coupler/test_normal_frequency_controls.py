"""Frequency-only normal forcing preserves the original spatial harmonic modes."""

import numpy as np
import pytest

from tests.support.cylinder.prepare_boundary_hybrid_inputs import components
from tests.support.cylinder.prepare_normal_frequency_controls import (
    oscillator_clock,
    replace_normal_oscillation,
)


def test_frequency_change_retains_initial_phase_and_harmonic_amplitudes():
    replay = np.linspace(100.0, 112.0, 301)
    source = replay - 63.75
    original_frequency, imposed_frequency = 0.18, 0.19
    altered = oscillator_clock(replay, source, original_frequency, imposed_frequency)
    assert altered[0] == source[0]
    np.testing.assert_allclose(
        np.diff(altered), np.diff(replay) * imposed_frequency / original_frequency, atol=2e-14
    )
    fitted = np.array(
        [
            [0.2, -0.2],
            [0.002, -0.002],
            [0.03, -0.03],
            [0.04, -0.04],
            [-0.007, 0.007],
            [0.004, -0.004],
        ]
    )
    fitted_before = fitted.copy()
    _, original = components(source, fitted, original_frequency, 40.0)
    _, changed = components(altered, fitted, original_frequency, 40.0)
    np.testing.assert_array_equal(original[0], changed[0])
    np.testing.assert_array_equal(fitted, fitted_before)
    phase = 2 * np.pi * original_frequency * (source[0] - 40.0) + 2 * np.pi * imposed_frequency * (
        replay - replay[0]
    )
    expected = (
        np.cos(phase)[:, None] * fitted[2]
        + np.sin(phase)[:, None] * fitted[3]
        + np.cos(2 * phase)[:, None] * fitted[4]
        + np.sin(2 * phase)[:, None] * fitted[5]
    )
    np.testing.assert_allclose(changed, expected, atol=8e-16)
    assert np.max(abs(changed - original)) > 0.025


def test_unchanged_frequency_keeps_every_original_source_clock_bitwise():
    replay = np.linspace(100.0, 112.0, 301)
    source = replay - 63.748491620057855
    np.testing.assert_array_equal(oscillator_clock(replay, source, 0.18, 0.18), source)


@pytest.mark.parametrize("frequency", [0.0, -0.2, np.nan])
def test_invalid_frequency_is_rejected(frequency):
    with pytest.raises(ValueError, match="Positive frequencies"):
        oscillator_clock([0.0, 0.04], [1.0, 1.04], 0.18, frequency)


def test_nonunit_original_source_clock_is_rejected():
    with pytest.raises(ValueError, match="matching unit-rate"):
        oscillator_clock([0.0, 0.04], [1.0, 1.08], 0.18, 0.19)


def test_normal_frequency_change_preserves_actual_tangent_gradient_mean_and_flux():
    replay = np.linspace(100.0, 112.0, 301)
    source = replay - 63.75
    normals = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    tangent = np.broadcast_to([0.0, 0.06, 0.0], (301, 2, 3)).copy()
    fitted = np.array(
        [
            [0.2, -0.2],
            [0.002, -0.002],
            [0.03, -0.03],
            [0.04, -0.04],
            [-0.007, 0.007],
            [0.004, -0.004],
        ]
    )
    dc, original = components(source, fitted, 0.18, 40.0)
    _, altered = components(oscillator_clock(replay, source, 0.18, 0.19), fitted, 0.18, 40.0)
    actual = {
        "normal_velocity": dc + original,
        "velocity": normals[None] * (dc + original)[..., None] + tangent,
        "tangential_gradient": tangent * np.sin(2 * np.pi * 0.18 * source)[:, None, None],
    }
    output = replace_normal_oscillation(
        actual, actual["normal_velocity"], original, altered, normals
    )
    np.testing.assert_array_equal(output["normal_velocity"][0], actual["normal_velocity"][0])
    np.testing.assert_array_equal(output["tangential_gradient"], actual["tangential_gradient"])
    np.testing.assert_allclose(output["normal_velocity"] - altered, dc, atol=6e-17)
    np.testing.assert_allclose(
        output["velocity"] - normals[None] * output["normal_velocity"][..., None],
        tangent,
        atol=1e-16,
    )
    np.testing.assert_allclose(output["normal_velocity"].sum(axis=1), 0, atol=1e-16)
    np.testing.assert_allclose(
        np.einsum("tfi,fi->tf", output["velocity"], normals), output["normal_velocity"], atol=1e-16
    )
