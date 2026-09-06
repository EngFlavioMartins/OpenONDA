import math
from types import SimpleNamespace

import numpy as np
import pytest

from source.solvers.vpm.physics.evaluation import ParticleFieldEvaluation


class _LargeParticleCloud:
    def __len__(self) -> int:
        return 10_001


def test_large_gaussian_cloud_uses_the_fourier_integral_backend():
    evaluator = object.__new__(ParticleFieldEvaluation)
    evaluator.particle_kernel = "GAUSSIAN"
    expected = {"backend": "fourier"}
    evaluator._compute_fourier_flow_integrals = lambda particles, time, record: expected

    result = evaluator.compute_flow_integrals(_LargeParticleCloud(), time=1.0)

    assert result is expected


def test_energy_rate_is_defined_between_continuity_preserving_measurements():
    evaluator = object.__new__(ParticleFieldEvaluation)
    evaluator._energy_history = [
        (0.0, 2.0, "unbounded_energy"),
        (0.5, 1.5, "unbounded_energy"),
    ]
    assert evaluator._compute_energy_dissipation_rate() == pytest.approx(-1.0)

    evaluator._energy_history.append((1.0, 1.4, "fourier_dynamic_box"))
    assert math.isnan(evaluator._compute_energy_dissipation_rate())


class _FourierParticleCloud:
    def __len__(self) -> int:
        return 10_001

    def position_cpu(self):
        return np.array([[0.0, 0.0, 0.0]])

    def vortex_strength_cpu(self):
        return np.array([[0.0, 0.0, 1.0]])

    def core_radius_cpu(self):
        return np.array([0.1])

    def particle_volume_cpu(self):
        return np.array([0.001])

    def effective_viscosity_cpu(self):
        return np.array([0.01])


def test_fourier_transition_reports_finite_energy_and_rate():
    evaluator = object.__new__(ParticleFieldEvaluation)
    evaluator._fourier_grid = None
    evaluator._fourier_energy_offset = 0.0
    evaluator._energy_history = [(0.0, 10.0, "unbounded_energy")]
    evaluator._max_history_length = 7
    spectral = SimpleNamespace(
        total_kinetic_energy=3.0,
        total_helicity=0.0,
        total_enstrophy=2.0,
        test_filtered_enstrophy=1.0,
        viscous_kinetic_energy_rate=-2.0,
    )
    evaluator._fourier_integrals_on_persistent_grid = lambda *args: (spectral, False, None)

    result = evaluator._compute_fourier_flow_integrals(
        _FourierParticleCloud(), time=0.5, record_history=True
    )

    assert result["total_kinetic_energy"] == pytest.approx(3.0)
    assert result["energy_measurement"] == "unbounded_energy"
    assert result["kinetic_energy_rate"] == pytest.approx(-2.0)
    assert result["kinetic_energy_rate_source"] == "fourier_transition_viscous_rate"


def test_fourier_grid_growth_bridges_the_rate_on_the_old_grid():
    evaluator = object.__new__(ParticleFieldEvaluation)
    evaluator._fourier_grid = None
    evaluator._fourier_energy_offset = 0.0
    evaluator._energy_history = [(0.0, 10.0, "unbounded_energy")]
    evaluator._max_history_length = 7
    new_grid = SimpleNamespace(
        total_kinetic_energy=3.0,
        total_helicity=0.0,
        total_enstrophy=2.0,
        test_filtered_enstrophy=1.0,
        viscous_kinetic_energy_rate=-2.0,
    )
    old_grid = SimpleNamespace(total_kinetic_energy=9.0)
    evaluator._fourier_integrals_on_persistent_grid = lambda *args: (
        new_grid,
        False,
        old_grid,
    )

    result = evaluator._compute_fourier_flow_integrals(
        _FourierParticleCloud(), time=0.5, record_history=True
    )

    assert result["total_kinetic_energy"] == pytest.approx(3.0)
    assert result["kinetic_energy_rate"] == pytest.approx(-2.0)
    assert result["kinetic_energy_rate_source"] == ("fourier_grid_transition_backward_difference")
    assert evaluator._energy_history[-1] == (0.5, 3.0, "unbounded_energy")
