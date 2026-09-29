"""Analytic checks for the loading-history cadence screen."""

import numpy as np
import pytest

from studies.analyze_loading_cadence import evaluate


def test_resolved_sinusoid_preserves_endpoint_and_integral():
    times = np.arange(1001) * 0.001
    values = (2 + np.sin(2 * np.pi * times * 2))[:, None, None]
    result = evaluate(times, values, ["force_x"], 2)
    assert result["screening_decision"] == "passes_available_history_only"
    assert result["retained_time_count"] == 501
    assert result["worst"]["max_integral_error_over_absolute_integral"] < 1e-10


def test_underresolved_oscillation_is_rejected_by_nyquist_energy():
    times = np.arange(1001) * 0.001
    values = np.sin(2 * np.pi * times * 200)[:, None, None]
    result = evaluate(times, values, ["force_x"], 4)
    assert result["screening_decision"] == "rejected"
    assert result["worst"]["max_detrended_energy_above_candidate_nyquist"] > 0.99


def test_symmetry_noise_does_not_define_physical_output_cadence():
    times = np.arange(1001) * 0.001
    values = np.stack([np.ones_like(times), 1e-6 * (-1.0) ** np.arange(len(times))], axis=-1)
    result = evaluate(times, values[:, None, :], ["force_x", "force_z"], 2)
    assert result["columns"]["force_z"]["ignored_near_zero_channels"] == 1
    assert result["screening_decision"] == "passes_available_history_only"


def test_irregular_history_requires_separate_sampling_analysis():
    times = np.array([0.0, 0.01, 0.04])
    with pytest.raises(ValueError, match="uniform"):
        evaluate(times, np.ones((3, 1, 1)), ["force_x"], 2)
