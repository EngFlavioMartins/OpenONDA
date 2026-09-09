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


@pytest.mark.parametrize(
    ("previous", "current"),
    [
        ("periodic_fourier_energy", "unbounded_energy"),
        ("unbounded_energy", "periodic_fourier_energy"),
    ],
)
def test_energy_derivatives_never_compare_periodic_and_unbounded_definitions(previous, current):
    evaluator = object.__new__(ParticleFieldEvaluation)
    evaluator._fourier_grid = None
    evaluator._energy_history = [(0.0, 3.0, previous)]
    evaluator._max_history_length = 7
    spectral = SimpleNamespace(
        total_kinetic_energy=10.0,
        total_helicity=0.0,
        total_enstrophy=2.0,
        test_filtered_enstrophy=1.0,
        viscous_kinetic_energy_rate=-2.0,
        energy_measurement=current,
    )
    evaluator._fourier_integrals_on_persistent_grid = lambda *args: (spectral, True, None)
    result = evaluator._compute_fourier_flow_integrals(_FourierParticleCloud(), 0.5, True)
    assert result["energy_measurement"] == current
    assert result["kinetic_energy_rate"] == -2.0
    assert result["kinetic_energy_rate_source"] == "fourier_transition_viscous_rate"
    assert evaluator._energy_history[-1] == (0.5, 10.0, current)
    history = list(evaluator._energy_history)
    trial = evaluator._compute_fourier_flow_integrals(_FourierParticleCloud(), 0.5, False)
    assert trial["total_kinetic_energy"] == 10.0
    assert trial["kinetic_energy_rate"] == -2.0
    assert trial["kinetic_energy_rate_source"] == "trial_viscous_rate"
    assert evaluator._energy_history == history


def test_live_linear_impulse_does_not_require_energy_history_or_quadratic_diagnostics():
    from source.solvers.vpm.core.solver import VPMSolver

    state = SimpleNamespace(
        particle_position=np.array([[2.0, 0.0, 0.0], [-2.0, 0.0, 0.0]]),
        particle_vortex_strength=np.array([[0.0, 3.0, 0.0], [0.0, -3.0, 0.0]]),
    )
    np.testing.assert_array_equal(VPMSolver.total_linear_impulse.fget(state), [0.0, 0.0, 6.0])
    state.particle_vortex_strength *= 2
    np.testing.assert_array_equal(VPMSolver.total_linear_impulse.fget(state), [0.0, 0.0, 12.0])


def test_elongating_wake_keeps_transverse_diagnostic_grid_sizes(monkeypatch):
    from source.solvers.vpm.numerics import fourier_integrals

    evaluator = object.__new__(ParticleFieldEvaluation)
    evaluator._fourier_grid = None
    grids = []

    def measure(*args, grid, **kwargs):
        grids.append(grid)
        return SimpleNamespace(total_kinetic_energy=1.0)

    monkeypatch.setattr(fourier_integrals, "gaussian_fourier_integrals", measure)
    strength = np.array([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]])
    transverse_shape = None
    for length in np.geomspace(0.2, 20.0, 30):
        position = np.array([[0.0, -0.1, -0.05], [length, 0.1, 0.05]])
        evaluator._fourier_integrals_on_persistent_grid(
            position,
            strength,
            np.array([0.1, 0.2]),
            np.full(2, 0.001),
            np.full(2, 0.01),
        )
        grid = evaluator._fourier_grid
        if transverse_shape is None:
            transverse_shape = grid.shape[1:]
        assert grid.shape[1:] == transverse_shape
        coordinates = (position - grid.origin) / grid.spacing
        assert np.all(coordinates >= 1.0)
        assert np.all(coordinates <= np.asarray(grid.shape) - 2.0)

    assert grids[-1].shape[0] > grids[0].shape[0]
