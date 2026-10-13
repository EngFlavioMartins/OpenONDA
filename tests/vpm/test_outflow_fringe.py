"""Outflow fringes remove wake vorticity smoothly before the keep-box face."""

from types import SimpleNamespace

import numpy as np
import pytest

from openonda.vpm.config.stabilization import StabilizationConfig
from openonda.vpm.stabilization.manager import (
    StabilizationManager,
    outflow_fringe_damping_rate,
)

BOUNDS = (-5.0, 25.0, -5.0, 5.0, -0.5, 0.5)
DEPTH = (0.0, 10.0, 0.0, 0.0, 0.0, 0.0)


def _points(x):
    x = np.asarray(x, dtype=np.float64)
    return np.column_stack((x, np.zeros_like(x), np.zeros_like(x)))


def test_rate_starts_smoothly_and_integrates_to_attenuation():
    x = np.linspace(BOUNDS[0], BOUNDS[1], 600_001)
    rate = outflow_fringe_damping_rate(_points(x), BOUNDS, DEPTH, speed=2.0, attenuation=8.0)
    assert np.all(rate[x <= 15.0] == 0.0)
    # Transport at U through the fringe takes L/U; the time integral is A e-folds.
    assert np.trapezoid(rate, x) / 2.0 == pytest.approx(8.0, rel=1e-8)
    # Quintic start: the rate is third order in the fringe depth.
    near = outflow_fringe_damping_rate(
        _points([15.01, 15.02]), BOUNDS, DEPTH, speed=2.0, attenuation=8.0
    )
    assert near[1] / near[0] == pytest.approx(8.0, rel=1e-2)


def test_every_face_uses_its_inward_depth():
    depth = (1.0, 2.0, 0.5, 0.25, 0.0, 0.0)
    points = np.array(
        [[-4.5, 0, 0], [24.0, 0, 0], [0, -4.75, 0], [0, 4.875, 0], [0, 0, 0.49], [10, 0, 0]]
    )
    rate = outflow_fringe_damping_rate(points, BOUNDS, depth, speed=1.0, attenuation=4.0)
    half = 0.5**3 * (10 - 15 * 0.5 + 6 * 0.25)
    expected = [2 * 4 / width * half for width in depth[:4]] + [0.0, 0.0]
    np.testing.assert_allclose(rate, expected, rtol=1e-12)


def test_configuration_requires_bounds_and_valid_depths():
    config = StabilizationConfig.bounded_domain(BOUNDS, outflow_fringe_depth=DEPTH)
    assert config.outflow_fringe_depth == DEPTH
    with pytest.raises(ValueError, match="requires remove_particles_by_bounds"):
        StabilizationConfig(outflow_fringe_depth=DEPTH)
    for depth in ((0.0, 1.0), (0.0, -1.0, 0, 0, 0, 0), (0.0, np.inf, 0, 0, 0, 0)):
        with pytest.raises(ValueError, match="six finite non-negative"):
            StabilizationConfig.bounded_domain(BOUNDS, outflow_fringe_depth=depth)
    with pytest.raises(ValueError, match="must not overlap"):
        StabilizationConfig.bounded_domain(BOUNDS, outflow_fringe_depth=(0, 0, 6.0, 5.0, 0, 0))
    with pytest.raises(ValueError, match="attenuation"):
        StabilizationConfig(remove_particles_by_bounds=BOUNDS, outflow_fringe_attenuation=0.0)


class _Particles:
    def __init__(self, position, strength, background):
        self.position = np.asarray(position, dtype=np.float64)
        self.strength = np.asarray(strength, dtype=np.float64)
        self.vorticity = self.strength / 0.04**3
        self.background = np.asarray(background, dtype=np.float32)

    @property
    def n_particles_total(self):
        return len(self.position)

    def position_cpu(self, use_cache=True):
        return self.position.copy()

    def vortex_strength_cpu(self, use_cache=True):
        return self.strength.copy()

    def vorticity_cpu(self, use_cache=True):
        return self.vorticity.copy()

    def velocity_background_cpu(self):
        return self.background.copy()


def _manager(particles, *, time_step_size=0.04):
    def set_properties(**update):
        particles.strength = update["vortex_strength"].astype(np.float64)
        particles.vorticity = update["vorticity"].astype(np.float64)

    return SimpleNamespace(
        ctx=SimpleNamespace(
            particles=particles,
            flow_model="INVISCID",
            np_dtype=np.float64,
            state=SimpleNamespace(time_step_size=time_step_size),
            mutations=SimpleNamespace(set_properties=set_properties),
        ),
        config=StabilizationConfig.bounded_domain(BOUNDS, outflow_fringe_depth=DEPTH),
        outflow_fringe_vortex_strength_removed=np.zeros(3),
    )


def test_worker_damps_only_fringe_particles_and_accounts_removed_strength():
    strength = np.array([[0.1, -0.2, 0.3], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    particles = _Particles(_points([10.0, 20.0, 24.9]), strength, (1.0, 0.0, 0.0))
    manager = _manager(particles)
    StabilizationManager.apply_outflow_fringe(manager)
    rate = outflow_fringe_damping_rate(
        particles.position, BOUNDS, DEPTH, speed=1.0, attenuation=8.0
    )
    expected = strength * np.exp(-rate * 0.04)[:, None]
    np.testing.assert_allclose(particles.strength, expected, rtol=1e-12)
    np.testing.assert_allclose(particles.vorticity, expected / 0.04**3, rtol=1e-12)
    np.testing.assert_array_equal(particles.strength[0], strength[0])
    np.testing.assert_allclose(
        manager.outflow_fringe_vortex_strength_removed, (strength - expected).sum(axis=0)
    )


def test_transport_through_fringe_attenuates_by_configured_e_folds():
    # A particle convected at the freestream crosses the whole fringe.
    particles = _Particles(_points([15.0]), [[0.0, 0.0, 1.0]], (0.5, 0.0, 0.0))
    manager = _manager(particles, time_step_size=0.01)
    while particles.position[0, 0] < 25.0:
        particles.position[0, 0] += 0.005
        StabilizationManager.apply_outflow_fringe(manager)
    assert particles.strength[0, 2] == pytest.approx(np.exp(-8.0), rel=2e-2)


def test_worker_is_inactive_without_freestream():
    strength = np.array([[0.0, 0.0, 1.0]])
    particles = _Particles(_points([24.0]), strength, (0.0, 0.0, 0.0))
    StabilizationManager.apply_outflow_fringe(_manager(particles))
    np.testing.assert_array_equal(particles.strength, strength)
