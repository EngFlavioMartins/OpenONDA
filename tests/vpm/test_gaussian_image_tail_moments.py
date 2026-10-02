"""Independent small-cloud qualification of the unwired image-tail expansion."""

import numpy as np
import pytest
from scipy.special import zeta

from tests.vpm._gaussian_image_tail_moments import moment_tail_bound
from tests.vpm._slip_periodic_gaussian_oracle import gaussian_pairs


def explicit_tail(x, gamma, sigma, targets, zmin, zmax, first, last):
    """Direct Gaussian pairs, independent of all moment/remainder arithmetic."""
    u, j = np.zeros((len(targets), 3)), np.zeros((len(targets), 3, 3))
    period = 2*(zmax-zmin)
    for odd in (False, True):
        source, strength = x.copy(), gamma.copy()
        if odd:
            source[:, 2] = 2*zmin-source[:, 2]
            strength[:, :2] *= -1
        for lo in range(first, last+1, 32):
            k = np.arange(lo, min(last+1, lo+32), dtype=float)
            shifts = np.concatenate((k, -k))*period
            d = targets[:, None, None, :]-source[None, None, :, :]
            d = np.broadcast_to(d, (len(targets), len(shifts), len(x), 3)).copy()
            d[..., 2] -= shifts[None, :, None]
            du, dj = gaussian_pairs(d, strength[None, None, :, :], sigma[None, None, :])
            u += du.sum(axis=(1, 2))
            j += dj.sum(axis=(1, 2))
    return u, j


def cloud(kind):
    rng = np.random.default_rng(8675309)
    x = rng.uniform(-.4, .4, (7, 3))
    g = rng.normal(size=x.shape)
    sigma = rng.uniform(.03, .2, len(x))
    t = rng.uniform(-.7, .7, (5, 3))
    zmin, zmax, k = -.5, .5, 4
    if kind == "cancelled":
        x = np.array([[q, 0., 0.] for q in np.linspace(-.15, .15, 5)])
        g = np.array([1., -4., 6., -4., 1.])[:, None]*np.array([.3, -.2, 1.])[None, :]
        sigma = np.full(len(x), .09)
    elif kind == "axial":
        g[:, :2] = 0
        g[:, 2] = np.arange(1, len(x)+1)
    elif kind == "translated":
        offset = np.array([1024., -512., 256.])
        x, t = x+offset, t+offset
        zmin, zmax = zmin+offset[2], zmax+offset[2]
    elif kind == "near_admission":
        x = np.array([[0., 0., .01], [.02, -.01, -.02]])
        g, sigma = g[:2], np.full(2, .01)
        t = np.array([[0., 0., 2.8], [.01, -.02, 2.75]])
        k = 2
    return x, g, sigma, t, zmin, zmax, k


@pytest.mark.parametrize("kind", ["random", "cancelled", "axial", "translated", "near_admission"])
def test_many_shell_gaussian_sum_obeys_leading_plus_remainder(kind):
    x, g, sigma, t, zmin, zmax, k = cloud(kind)
    inputs = [v.copy() for v in (x, g, sigma, t)]
    result = moment_tail_bound(x, g, sigma, t, z_min=zmin, z_max=zmax, shells=k)
    last = 512
    u, j = explicit_tail(x, g, sigma, t, zmin, zmax, k+1, last)
    # Remove only the finite part of the analytically summed leading term.
    ratio = (zeta(3, k+1)-zeta(3, last+1))/zeta(3, k+1)
    uerror = np.linalg.norm(u-ratio*result.leading_velocity, axis=1)
    jerror = np.linalg.norm(j-ratio*result.leading_gradient, axis=(1, 2))
    assert np.all(uerror <= result.singular_velocity_remainder+result.gaussian_velocity_defect+2e-14)
    assert np.all(jerror <= result.singular_gradient_remainder+result.gaussian_gradient_defect+2e-14)
    for actual, original in zip((x, g, sigma, t), inputs, strict=True):
        np.testing.assert_array_equal(actual, original)


def test_leading_gradient_is_exact_derivative_and_sign_matches_direct_tail():
    x, g, sigma, t, zmin, zmax, _ = cloud("axial")
    k = 16
    result = moment_tail_bound(x, g, sigma, t, z_min=zmin, z_max=zmax, shells=k)
    for axis in range(3):
        shifted = t.copy()
        shifted[:, axis] += 1e-3
        other = moment_tail_bound(x, g, sigma, shifted, z_min=zmin, z_max=zmax, shells=k)
        np.testing.assert_allclose((other.leading_velocity-result.leading_velocity)/1e-3,
                                   result.leading_gradient[:, :, axis], rtol=2e-11, atol=1e-16)
    u, j = explicit_tail(x, g, sigma, t, zmin, zmax, k+1, 1024)
    ratio = (zeta(3, k+1)-zeta(3, 1025))/zeta(3, k+1)
    # The leading tensor has structurally zero entries; the true cubic term
    # need not. Compare whole-vector/tensor relative error, not zero entries.
    assert np.all(np.linalg.norm(u-ratio*result.leading_velocity, axis=1)
                  < .005*np.linalg.norm(u, axis=1))
    assert np.all(np.linalg.norm(j-ratio*result.leading_gradient, axis=(1, 2))
                  < .005*np.linalg.norm(j, axis=(1, 2)))


def test_non_negligible_gaussian_defect_is_separately_charged():
    x = np.array([[0., 0., 0.]])
    g, sigma, t = np.array([[.2, -.1, 1.]]), np.array([2.]), np.array([[0., 0., 0.]])
    result = moment_tail_bound(x, g, sigma, t, z_min=-.5, z_max=.5, shells=2)
    assert np.all(result.gaussian_velocity_defect > 1e-5)
    assert np.all(result.gaussian_gradient_defect > 1e-5)
    u, j = explicit_tail(x, g, sigma, t, -.5, .5, 3, 512)
    ratio = (zeta(3, 3)-zeta(3, 513))/zeta(3, 3)
    assert np.linalg.norm(u-ratio*result.leading_velocity) <= float(
        (result.singular_velocity_remainder+result.gaussian_velocity_defect)[0])
    assert np.linalg.norm(j-ratio*result.leading_gradient) <= float(
        (result.singular_gradient_remainder+result.gaussian_gradient_defect)[0])


@pytest.mark.parametrize("key,value", [("shells", True), ("shells", 0), ("shells", 1.5),
                                        ("max_sources", 1), ("max_targets", 1)])
def test_caps_reject_invalid_inputs(key, value):
    x, g, sigma, t, zmin, zmax, k = cloud("random")
    kwargs = {"z_min": zmin, "z_max": zmax, "shells": k}
    kwargs[key] = value
    with pytest.raises(ValueError):
        moment_tail_bound(x, g, sigma, t, **kwargs)


def test_rejects_nonfinite_and_nonadmissible_distance():
    x, g, sigma, t, zmin, zmax, k = cloud("random")
    g[0, 0] = np.nan
    with pytest.raises(ValueError):
        moment_tail_bound(x, g, sigma, t, z_min=zmin, z_max=zmax, shells=k)
    g[0, 0] = 1.
    t[0, 0] = 100
    with pytest.raises(ValueError, match="every source-target extent"):
        moment_tail_bound(x, g, sigma, t, z_min=zmin, z_max=zmax, shells=k)


def test_rejects_gaussian_density_outside_monotonic_domain():
    with pytest.raises(ValueError, match="density monotonicity"):
        moment_tail_bound(np.zeros((1, 3)), np.ones((1, 3)), np.ones(1)*20,
                          np.zeros((1, 3)), z_min=-.5, z_max=.5, shells=2)


@pytest.mark.parametrize("count", [0, 4])
def test_zero_cloud_and_zero_strengths_are_exact(count):
    result = moment_tail_bound(np.zeros((count, 3)), np.zeros((count, 3)), np.ones(count),
                               np.zeros((3, 3)), z_min=-.5, z_max=.5, shells=2)
    np.testing.assert_array_equal(result.velocity_bound, 0.)
    np.testing.assert_array_equal(result.gradient_bound, 0.)
    assert result.diagnostics["runtime_admissible"] is False
