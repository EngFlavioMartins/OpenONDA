import numpy as np

from source.coupler.core.helpers.continuous_overlap import beale_strength_correction


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
