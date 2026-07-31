import numpy as np

from source.coupler.diagnostics.injection_correction import recover_invariants


def test_recovery_restores_circulation_and_linear_impulse():
    rng = np.random.default_rng(4)
    positions = rng.uniform(-1.0, 1.0, (80, 3))
    circulation = rng.normal(size=(80, 3)) * 1.0e-3
    volumes = np.full(80, 0.05**3)
    target = {
        "circulation": circulation.sum(axis=0) + np.array([0.01, -0.02, 0.03]),
        "linear_impulse": 0.5 * np.cross(positions, circulation).sum(axis=0)
        + np.array([0.02, 0.01, -0.01]),
    }

    corrected = recover_invariants(positions, circulation, target, volumes=volumes)

    np.testing.assert_allclose(corrected.sum(axis=0), target["circulation"], atol=1.0e-12)
    impulse = 0.5 * np.cross(positions, corrected).sum(axis=0)
    np.testing.assert_allclose(impulse, target["linear_impulse"], atol=1.0e-12)


def test_zero_deficit_is_unchanged():
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    circulation = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]])
    target = {
        "circulation": circulation.sum(axis=0),
        "linear_impulse": 0.5 * np.cross(positions, circulation).sum(axis=0),
    }
    corrected = recover_invariants(positions, circulation, target, volumes=np.ones(len(positions)))
    np.testing.assert_array_equal(corrected, circulation)
