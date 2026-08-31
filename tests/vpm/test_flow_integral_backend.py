import math

import pytest

from source.solvers.vpm.physics.evaluation import ParticleFieldEvaluation


class _LargeParticleCloud:
    def __len__(self) -> int:
        return 50_001


def test_large_gaussian_cloud_uses_the_fourier_integral_backend():
    evaluator = object.__new__(ParticleFieldEvaluation)
    evaluator.particle_kernel = "GAUSSIAN"
    expected = {"backend": "fourier"}
    evaluator._compute_fourier_flow_integrals = lambda particles, time, record: expected

    result = evaluator.compute_flow_integrals(_LargeParticleCloud(), time=1.0)

    assert result is expected


def test_energy_rate_is_defined_only_between_direct_energy_measurements():
    evaluator = object.__new__(ParticleFieldEvaluation)
    evaluator._energy_history = [
        (0.0, 2.0, "direct"),
        (0.5, 1.5, "direct"),
    ]
    assert evaluator._compute_energy_dissipation_rate() == pytest.approx(-1.0)

    evaluator._energy_history.append((1.0, 1.4, "fourier_dynamic_box"))
    assert math.isnan(evaluator._compute_energy_dissipation_rate())
