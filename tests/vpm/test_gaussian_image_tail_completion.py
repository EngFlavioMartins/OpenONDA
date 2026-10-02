"""Long direct Gaussian checks of K32/64/128 analytic image completion.

These numerical tests do not turn the padded-f64 moment helper into an
interval certificate. A long finite sum is not called the infinite truth:
its own remaining tail is accounted for explicitly below, independently
of the leading-coefficient interval used for the candidate completion.
"""

import math

import numpy as np
import pytest

from tests.vpm._gaussian_image_tail_moments import moment_tail_bound
from tests.vpm._gaussian_tail_coefficient_enclosure import coefficient_enclosure
from tests.vpm._slip_periodic_gaussian_oracle import gaussian_pairs


def _direct_tail(position, strength, core, targets, zmin, zmax, first, last):
    """Only explicit Gaussian image pairs; compensated chunk accumulation."""
    velocity, gradient, conditioning = [], [], []
    period = 2*(zmax-zmin)
    for odd in (False, True):
        source, gamma = position.copy(), strength.copy()
        if odd:
            source[:, 2] = 2*zmin-source[:, 2]
            gamma[:, :2] *= -1
        for begin in range(first, last+1, 64):
            positive = np.arange(begin, min(begin+64, last+1), dtype=float)*period
            shifts = np.concatenate((positive, -positive))
            displacement = targets[:, None, None, :]-source[None, None, :, :]
            displacement = np.broadcast_to(displacement, (len(targets), len(shifts), len(source), 3)).copy()
            displacement[..., 2] -= shifts[None, :, None]
            du, dj = gaussian_pairs(displacement, gamma[None, None, :, :], core[None, None, :])
            velocity.append(du.sum(axis=(1, 2)))
            gradient.append(dj.sum(axis=(1, 2)))
            conditioning.append(np.stack((np.linalg.norm(du, axis=-1).sum(axis=(1, 2)),
                                          np.linalg.norm(dj, axis=(-1, -2)).sum(axis=(1, 2))), axis=-1))

    def compensated(chunks):
        flattened = np.asarray(chunks).reshape(len(chunks), -1)
        values = [math.fsum(flattened[:, index]) for index in range(flattened.shape[1])]
        return np.array(values).reshape(chunks[0].shape)

    return compensated(velocity), compensated(gradient), compensated(conditioning)


def _cloud(kind):
    zmin, zmax = -.073, .12
    x = np.array([[-.041, .032, zmin], [.092, -.027, zmax], [.017, .024, .003], [-.08, -.051, .082]])
    gamma = np.array([[.3, -.2, .7], [-.4, .6, .1], [.8, .2, -.5], [.1, -.9, .4]])
    core = np.array([.025, .04, .083, .16])
    targets = np.array([x[0], x[1], [.07, .08, zmin], [-.1, -.05, zmax], [.023, -.018, .011]])
    if kind == "cancelled_moments":
        x = np.column_stack((np.linspace(-1.5, 1.5, 5), np.zeros(5), np.full(5, .003)))
        gamma = np.array([1., -4., 6., -4., 1.])[:, None]*np.array([.3, -.2, .7])
        core = np.array([.025, .04, .083, .16, .09])
    elif kind == "broad_mixed_cores":
        # Forces a non-negligible Gaussian-minus-singular charge at K32.
        # The largest core remains within the helper's stated monotonicity gate.
        core = np.array([.025, 3., .083, 6.])
        gamma[-1] = -gamma[:-1].sum(axis=0)+np.array([1e-10, -2e-10, 3e-10])
    return x, gamma, core, targets, zmin, zmax


@pytest.fixture(scope="module", params=("wall_mixed", "cancelled_moments", "broad_mixed_cores"))
def completed_cloud(request):
    data = _cloud(request.param)
    x, gamma, core, targets, zmin, zmax = data
    last = 4096
    pieces = [_direct_tail(x, gamma, core, targets, zmin, zmax, first, end)
              for first, end in ((33, 64), (65, 128), (129, last))]
    tails = {}
    for offset, shells in enumerate((32, 64, 128)):
        tails[shells] = tuple(np.sum([part[field] for part in pieces[offset:]], axis=0) for field in range(3))
    return request.param, data, last, tails


@pytest.mark.parametrize("shells", [32, 64, 128])
def test_completed_field_matches_long_gaussian_sum_with_explicit_residual_budget(completed_cloud, shells):
    kind, data, last, tails = completed_cloud
    x, gamma, core, targets, zmin, zmax = data
    before = [item.copy() for item in (x, gamma, core, targets)]
    candidate = moment_tail_bound(x, gamma, core, targets, z_min=zmin, z_max=zmax, shells=shells)
    remainder = moment_tail_bound(x, gamma, core, targets, z_min=zmin, z_max=zmax, shells=last)
    coefficient = coefficient_enclosure(shells, prefix_terms=4096)
    long_coefficient = coefficient_enclosure(last, prefix_terms=4096)
    basis_u = candidate.leading_velocity/candidate.diagnostics["leading_zeta3"]
    basis_j = candidate.leading_gradient/candidate.diagnostics["leading_zeta3"]
    # The basis does not depend on K. This catches a hidden shell dependence
    # when substituting the independently enclosed positive-series coefficient.
    np.testing.assert_allclose(remainder.leading_velocity/remainder.diagnostics["leading_zeta3"],
                               basis_u, rtol=2e-15, atol=1e-28)
    np.testing.assert_allclose(remainder.leading_gradient/remainder.diagnostics["leading_zeta3"],
                               basis_j, rtol=2e-15, atol=1e-28)
    direct_u, direct_j, condition = tails[shells]
    for field, direct, singular, defect, remote_singular, remote_defect, axis, conditioning in (
        (basis_u, direct_u, candidate.singular_velocity_remainder, candidate.gaussian_velocity_defect,
         remainder.singular_velocity_remainder, remainder.gaussian_velocity_defect, -1, condition[:, 0]),
        (basis_j, direct_j, candidate.singular_gradient_remainder, candidate.gaussian_gradient_defect,
         remainder.singular_gradient_remainder, remainder.gaussian_gradient_defect, (-1, -2), condition[:, 1]),
    ):
        completed = coefficient.midpoint*field
        long_completed = direct+long_coefficient.midpoint*field
        coefficient_error = (coefficient.radius+long_coefficient.radius)*np.linalg.norm(field, axis=axis)
        analytic_error = singular+defect+remote_singular+remote_defect+coefficient_error
        # This is a separately labelled numerical oracle allowance, not part
        # of the analytic certificate. It scales with absolute pair norms,
        # including cancellation, instead of the small final tail value.
        oracle_roundoff = 128*np.finfo(float).eps*(conditioning+np.linalg.norm(completed, axis=axis))
        assert np.all(np.linalg.norm(completed-long_completed, axis=axis) <= analytic_error+oracle_roundoff)
    if kind == "broad_mixed_cores" and shells == 32:
        assert np.max(candidate.gaussian_gradient_defect) > 1e-6
    assert not candidate.diagnostics["runtime_admissible"]
    for current, original in zip((x, gamma, core, targets), before, strict=True):
        np.testing.assert_array_equal(current, original)


@pytest.mark.parametrize("scale", [1e-5, 1e3])
def test_leading_completion_and_remainder_covary_under_units_and_translation(scale):
    x, gamma, core, targets, zmin, zmax = _cloud("wall_mixed")
    baseline = moment_tail_bound(x, gamma, core, targets, z_min=zmin, z_max=zmax, shells=64)
    shift = scale*np.array([23., -17., 31.])
    changed = moment_tail_bound(scale*x+shift, scale**2*gamma, scale*core, scale*targets+shift,
                                z_min=scale*zmin+shift[2], z_max=scale*zmax+shift[2], shells=64)
    np.testing.assert_allclose(changed.leading_velocity, baseline.leading_velocity, rtol=2e-12, atol=1e-16)
    np.testing.assert_allclose(scale*changed.leading_gradient, baseline.leading_gradient, rtol=2e-12, atol=1e-16)
    # Padding is deliberately coordinate-scale dependent; covariance of the
    # bound is approximate, not an interval/arithmetic-certification assertion.
    np.testing.assert_allclose(changed.singular_velocity_remainder, baseline.singular_velocity_remainder,
                               rtol=1e-6, atol=1e-18)
    np.testing.assert_allclose(scale*changed.singular_gradient_remainder, baseline.singular_gradient_remainder,
                               rtol=1e-6, atol=1e-18)
