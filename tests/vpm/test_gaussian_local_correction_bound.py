"""Outward correction envelopes against independent positive-integral fields."""

import numpy as np
import pytest

from source.solvers.vpm.physics.induction.gaussian_mesh.error_bounds import local_correction_bound
from tests.vpm._gaussian_broadening_reference import correction_fields


@pytest.mark.parametrize("ratio", [0.1, 1., 3., 5., 10.])
@pytest.mark.parametrize("scale", [2.0**-12, 1., 2.0**12])
def test_bound_dominates_independent_correction(ratio, scale):
    tau, cutoff = .125*scale, .125*scale*ratio
    gamma = np.array([[.2, -.3, .4], [-.2, .3, -.4]])
    bound = local_correction_bound(gamma, tau=tau, omitted_distance_lower=cutoff, image_count=2)
    for source in gamma:
        for core_ratio in [.01, .3, .99, 1.]:
            for distance_ratio in [1., 1.001, 2.]:
                r = np.array([.6, .8, 0.])*cutoff*distance_ratio
                u, j = correction_fields(r, source, tau*core_ratio, tau)
                assert 4*np.linalg.norm(u) <= bound.velocity_upper
                assert 4*np.linalg.norm(j) <= bound.gradient_upper


def test_strength_cancellation_cannot_remove_positive_error_budget():
    gamma = np.array([[1., 2., 3.], [-1., -2., -3.]])
    bound = local_correction_bound(gamma, tau=.1, omitted_distance_lower=.5, image_count=513)
    assert bound.absolute_strength_upper >= 12.
    assert bound.velocity_upper > 0 and bound.gradient_upper > 0


def test_zero_fields_and_no_images_are_exact_noops():
    for gamma, count in [(np.zeros((3, 3)), 4), (np.ones((2, 3)), 0), (np.empty((0, 3)), 2)]:
        bound = local_correction_bound(gamma, tau=.1, omitted_distance_lower=.5, image_count=count)
        assert bound.velocity_upper == bound.gradient_upper == 0.


@pytest.mark.parametrize("change", [
    {"tau": 0}, {"omitted_distance_lower": -1}, {"image_count": True},
    {"image_count": 1.5}, {"image_count": -1}, {"image_count": 1_000_001},
    {"strength": np.array([[np.nan, 0, 0]])}, {"strength": np.ones((2, 2))},
    {"strength": np.ones((2, 3), dtype=complex)},
    {"strength": np.ones((2, 3), dtype=np.float16)},
    {"strength": np.array([[1, 2, 3]], dtype=np.int64)},
    {"strength": np.array([[np.longdouble('1e-330'), 0, 0]], dtype=np.longdouble)},
    {"tau": np.longdouble('.1')},
])
def test_invalid_inputs_fail_closed(change):
    args = {"strength": np.ones((2, 3)), "tau": .1, "omitted_distance_lower": .5, "image_count": 513}
    args.update(change)
    with pytest.raises((ValueError, FloatingPointError)):
        local_correction_bound(**args)
