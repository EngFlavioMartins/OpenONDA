import numpy as np

from source.coupler.core.helpers.continuous_overlap import (
    _gaussian_mollified_circulation,
    beale_strength_correction,
)


def test_gaussian_mollification_returns_the_physical_grid_circulation():
    shape = (11, 11, 11)
    circulation = np.zeros((*shape, 3))
    circulation[5, 5, 5, 2] = 1.0

    mollified = _gaussian_mollified_circulation(
        circulation,
        shape,
        1.0,
        sigma=1.0,
    )

    assert mollified.shape == (*shape, 3)
    assert mollified[5, 5, 5, 2] < circulation[5, 5, 5, 2]
    np.testing.assert_allclose(mollified[..., 2].sum(), 1.0, rtol=1e-12, atol=1e-12)


def test_single_strength_correction_reduces_mollification_error():
    shape = (9, 9, 9)
    target = np.zeros((*shape, 3))
    target[4, 4, 4, 2] = 1.0
    circulation = target.reshape(-1, 3).copy()
    weights = np.ones(np.prod(shape))

    _, before, after = beale_strength_correction(
        circulation,
        target.reshape(-1, 3),
        weights,
        shape,
        1.0,
        sigma=1.0,
    )

    assert after < before
