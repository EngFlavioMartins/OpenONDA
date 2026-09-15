"""Labels must not improve the reported resolution of an unchanged blob field."""

import numpy as np
import pytest

from source.solvers.vpm.diagnostics.resolution import discretization_health


@pytest.mark.parametrize("sample_all", [False, True])
@pytest.mark.parametrize("count", [50, 600])
def test_coincident_group_contributions_leave_field_health_unchanged(count, sample_all):
    rng = np.random.default_rng(20)
    position = rng.normal(size=(count, 3)) * 0.2
    strength = rng.normal(size=(count, 3))
    core = np.full(count, 0.15)
    reference = discretization_health(position, strength, core, sample_all=sample_all)
    # Labels can even have different directions while their sum is identical.
    component = rng.normal(size=(count, 3)) * 0.1
    actual = discretization_health(
        np.concatenate((position, position)),
        np.concatenate((0.3 * strength + component, 0.7 * strength - component)),
        np.concatenate((core, core)),
        sample_all=sample_all,
    )
    for name in reference:
        np.testing.assert_allclose(actual[name], reference[name], rtol=1e-12, atol=1e-12)


def test_full_divergence_measurement_matches_direct_gaussian_derivatives(monkeypatch):
    from source.solvers.vpm.diagnostics import resolution

    rng = np.random.default_rng(51)
    position = rng.uniform(-0.2, 0.2, (40, 3))
    strength = rng.normal(size=(40, 3))
    core = np.full(40, 0.3)
    # All pairs lie within the Gaussian cutoff, so the direct calculation is
    # independent of neighbour search and of the bounded probe selection.
    delta = position[:, None, :] - position[None, :, :]
    gaussian = np.exp(-np.sum(delta**2, axis=2) / core[None, :] ** 2)
    gaussian /= np.pi**1.5 * core[None, :] ** 3
    gradient = np.einsum("ij,ija,jb->iab", -2.0 * gaussian / core[None, :] ** 2, delta, strength)
    local_error = np.abs(np.trace(gradient, axis1=1, axis2=2))
    local_error /= np.linalg.norm(gradient, axis=(1, 2))
    weights = np.linalg.norm(gaussian @ strength, axis=1)
    expected = np.average(local_error, weights=weights)

    monkeypatch.setattr(resolution, "_MAX_PROBES", 4)
    sampled = discretization_health(position, strength, core)
    full = discretization_health(position, strength, core, sample_all=True)
    assert sampled["vorticity_divergence_error"] != pytest.approx(expected, abs=1e-3)
    assert full["vorticity_divergence_error"] == pytest.approx(expected, abs=1e-12)
