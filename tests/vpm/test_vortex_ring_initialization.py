"""Vortex-ring initial-condition tests."""

import numpy as np

from source.solvers.vpm.initial_conditions import vortex_ring_vpm


def _vorticity(points: np.ndarray) -> np.ndarray:
    _, _, vortex_strength = vortex_ring_vpm(
        kinematic_viscosity=1.0e-3,
        ring_centre=np.zeros(3),
        tube_circulation=np.pi,
        ring_radius=1.0,
        ring_core_radius=0.1,
        mean_core_radius=0.02,
        position=points,
        particle_volume=np.ones(len(points)),
        widnall_amplitude=0.03,
        seed=7,
        n_widnall_modes=4,
    )
    return vortex_strength


def _sample_divergence() -> tuple[np.ndarray, np.ndarray]:
    theta = np.linspace(-np.pi, np.pi, 32, endpoint=False)
    radial_offset = 0.025 * np.sin(3.0 * theta)
    points = np.column_stack(
        (
            0.02 * np.cos(2.0 * theta),
            (1.0 + radial_offset) * np.cos(theta),
            (1.0 + radial_offset) * np.sin(theta),
        )
    )

    delta = 1.0e-5
    divergence = np.zeros(len(points))
    for axis in range(3):
        plus = points.copy()
        minus = points.copy()
        plus[:, axis] += delta
        minus[:, axis] -= delta
        divergence += (_vorticity(plus)[:, axis] - _vorticity(minus)[:, axis]) / (2.0 * delta)
    return divergence, _vorticity(points)


def test_solenoidal_widnall_perturbation_has_zero_divergence():
    divergence, vorticity = _sample_divergence()
    reference_gradient = np.linalg.norm(vorticity, axis=1).max() / 0.1

    assert np.abs(divergence).max() / reference_gradient < 2.0e-6
