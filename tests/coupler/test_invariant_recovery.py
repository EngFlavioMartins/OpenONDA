import numpy as np

from source.coupler.conservation import recover_invariants


def test_recovery_restores_strength_and_linear_impulse():
    rng = np.random.default_rng(4)
    positions = rng.uniform(-1.0, 1.0, (80, 3))
    vortex_strength = rng.normal(size=(80, 3)) * 1.0e-3
    volumes = np.full(80, 0.05**3)
    target = {
        "total_vortex_strength": vortex_strength.sum(axis=0) + np.array([0.01, -0.02, 0.03]),
        "linear_impulse": 0.5 * np.cross(positions, vortex_strength).sum(axis=0)
        + np.array([0.02, 0.01, -0.01]),
    }

    corrected = recover_invariants(positions, vortex_strength, target, volumes=volumes)

    np.testing.assert_allclose(corrected.sum(axis=0), target["total_vortex_strength"], atol=1.0e-12)
    impulse = 0.5 * np.cross(positions, corrected).sum(axis=0)
    np.testing.assert_allclose(impulse, target["linear_impulse"], atol=1.0e-12)


def test_zero_deficit_is_unchanged():
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    vortex_strength = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]])
    target = {
        "total_vortex_strength": vortex_strength.sum(axis=0),
        "linear_impulse": 0.5 * np.cross(positions, vortex_strength).sum(axis=0),
    }
    corrected = recover_invariants(
        positions, vortex_strength, target, volumes=np.ones(len(positions))
    )
    np.testing.assert_array_equal(corrected, vortex_strength)
