import math

import pytest

from source.solvers.vpm.physics.evaluation import ParticleFieldEvaluation, _direct_integral_limit


def test_direct_integral_limit_defaults_to_conservative_value(monkeypatch):
    monkeypatch.delenv("OPENONDA_VPM_DIRECT_INTEGRAL_LIMIT", raising=False)
    assert _direct_integral_limit() == 50_000


def test_direct_integral_limit_accepts_campaign_override(monkeypatch):
    monkeypatch.setenv("OPENONDA_VPM_DIRECT_INTEGRAL_LIMIT", "200000")
    assert _direct_integral_limit() == 200_000


@pytest.mark.parametrize("value", ["-1", "not-an-integer"])
def test_direct_integral_limit_rejects_invalid_override(monkeypatch, value):
    monkeypatch.setenv("OPENONDA_VPM_DIRECT_INTEGRAL_LIMIT", value)
    with pytest.raises(ValueError, match="OPENONDA_VPM_DIRECT_INTEGRAL_LIMIT"):
        _direct_integral_limit()


def test_energy_rate_is_defined_only_between_direct_energy_measurements():
    evaluator = object.__new__(ParticleFieldEvaluation)
    evaluator._energy_history = [
        (0.0, 2.0, "direct"),
        (0.5, 1.5, "direct"),
    ]
    assert evaluator._compute_energy_dissipation_rate() == pytest.approx(-1.0)

    evaluator._energy_history.append((1.0, 1.4, "fourier_dynamic_box"))
    assert math.isnan(evaluator._compute_energy_dissipation_rate())
