"""Check all spatial derivative traces against independent 3D polynomials."""

import numpy as np

from studies.coupler_accuracy.induction_derivatives_3d import normal_derivative_estimates


def test_normal_derivative_traces_are_quadratic_exact_and_cubic_second_order():
    position = np.array([[.3, -.7, .2], [-.2, .5, .6], [.8, -.2, .1]])
    normal = np.array([[1., 2., 3.], [-2., 3., 1.], [3., -1., 2.]])/np.sqrt(14)
    matrix = np.array([[.3, -.2, .7], [.5, -.8, .1], [-.4, .9, 1.3]])
    v = np.array([.3, -.7, 1.2])
    a = np.array([.7, -.2, .3])
    constant = np.array([4.3, -2.7, 1.6])
    step = 2e-3
    factors = {"boundary": 0, "plus": 1, "minus": -1, "plus2": 2, "minus2": -2, "plus_half": .5, "minus_half": -.5}
    for power in (2, 3):
        values = {}
        for key, factor in factors.items():
            q = position+factor*step*normal
            values[key] = constant+q @ matrix+(q @ a)[:, None]**power*v
        exact = normal @ matrix+power*(position @ a)[:, None]**(power-1)*(normal @ a)[:, None]*v
        actual = normal_derivative_estimates(values, step)
        if power == 2:
            for derivative in actual.values():
                np.testing.assert_allclose(derivative, exact, rtol=0, atol=4e-12)
        else:
            for side in ("centred", "exterior", "interior"):
                error = actual[side]-exact
                half_error = actual[side+"_half"]-exact
                assert np.linalg.norm(error) > 1e-8
                np.testing.assert_allclose(error, 4*half_error, rtol=0, atol=8e-12)
