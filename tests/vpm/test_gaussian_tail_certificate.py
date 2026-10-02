"""Whole-tail policy evidence stays separate from field approximation."""

import numpy as np
import pytest

from tests.vpm._gaussian_tail_certificate import normalized_bound, tail_certificate
from tests.vpm.test_gaussian_image_tail_moments import cloud, explicit_tail


@pytest.mark.parametrize("kind", ["random", "cancelled", "axial", "translated"])
def test_whole_tail_bounds_direct_many_shell_sum_without_modifying_fields(kind):
    x, g, sigma, t, zmin, zmax, _ = cloud(kind)
    result = tail_certificate(x, g, sigma, t, z_min=zmin, z_max=zmax, shells=32)
    u, j = explicit_tail(x, g, sigma, t, zmin, zmax, 33, 512)
    beyond = tail_certificate(x, g, sigma, t, z_min=zmin, z_max=zmax, shells=512)
    assert np.all(np.linalg.norm(u, axis=1) <= result.whole_velocity+beyond.whole_velocity+2e-14)
    assert np.all(np.linalg.norm(j, axis=(1, 2)) <= result.whole_gradient+beyond.whole_gradient+2e-14)
    assert result.diagnostics["finite_image_accuracy_certified"] is False
    assert result.diagnostics["production_admissible"] is False


def test_normalization_is_outward_and_uses_supplied_scales():
    u, j = np.array([.03, .001]), np.array([.002, .2])
    result = normalized_bound(u, j, velocity_scale=2., gradient_scale=4.)
    assert np.all(result >= np.maximum(u/2., j/4.))
    with pytest.raises(ValueError):
        normalized_bound(u, j, velocity_scale=0., gradient_scale=1.)


def test_empty_query_preserves_empty_result():
    result = tail_certificate(np.zeros((1, 3)), np.ones((1, 3)), np.ones(1)*.04,
                               np.zeros((0, 3)), z_min=-.5, z_max=.5, shells=32)
    assert result.whole_velocity.shape == (0,)
    assert result.whole_gradient.shape == (0,)
