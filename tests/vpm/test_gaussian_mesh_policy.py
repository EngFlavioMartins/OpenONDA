"""Unit-independent controls; these tests do not certify field accuracy."""

from dataclasses import FrozenInstanceError, asdict

import numpy as np
import pytest

from source.solvers.vpm.physics.induction.gaussian_mesh.policy import GaussianMeshParameters


def test_defaults_cover_all_mixed_cores_without_mutation():
    core = np.array([0.015, 0.04, 0.027], dtype=np.float32)
    original = core.copy()
    value = GaussianMeshParameters().resolve(core)
    assert value.tau == 3 * float(core.max())
    assert value.spacing == value.tau / 4
    assert value.correction_cutoff == 5 * value.tau
    assert value.order == 10
    np.testing.assert_array_equal(core, original)
    core[-1] = 0.1
    assert GaussianMeshParameters().resolve(core).tau > value.tau
    assert value.maximum_source_core == float(original.max())


@pytest.mark.parametrize("scale", [2.0**-30, 2.0**-4, 2.0**12, 2.0**30])
def test_unit_scaling(scale):
    parameters = GaussianMeshParameters()
    core = np.array([0.125, 0.25, 0.375])
    first, second = parameters.resolve(core), parameters.resolve(scale * core)
    for key in ("tau", "spacing", "correction_cutoff", "maximum_source_core"):
        assert getattr(second, key) == scale * getattr(first, key)


def test_explicit_resolution_is_not_silently_clipped():
    parameters = GaussianMeshParameters(spacing_over_tau=7 / 24)
    value = parameters.resolve(np.array([0.04]))
    assert value.spacing == pytest.approx(0.035)
    assert asdict(parameters)["spacing_over_tau"] == 7 / 24
    with pytest.raises(FrozenInstanceError):
        parameters.order = 6
    with pytest.raises(FrozenInstanceError):
        value.tau = 1


@pytest.mark.parametrize("name", ["broadening_ratio", "spacing_over_tau", "correction_radius_over_tau"])
@pytest.mark.parametrize("value", [False, "3", 0, -1, np.nan, np.inf, 1j])
def test_reject_invalid_ratios(name, value):
    with pytest.raises(ValueError):
        GaussianMeshParameters(**{name: value})


@pytest.mark.parametrize("order", [True, 3, 12, 4.0, "10", None])
def test_reject_invalid_orders(order):
    with pytest.raises(ValueError):
        GaussianMeshParameters(order=order)


@pytest.mark.parametrize("core", [[], [0], [-1], [np.inf], [np.nan], [1j], [True], [[1.0]], 0.1, [".1"]])
def test_reject_invalid_core_snapshot(core):
    with pytest.raises(ValueError):
        GaussianMeshParameters().resolve(core)


def test_reject_unrepresentable_lengths_and_under_broadening():
    with pytest.raises(ValueError):
        GaussianMeshParameters(broadening_ratio=0.9)
    with pytest.raises(ValueError, match="resolved"):
        GaussianMeshParameters().resolve([np.finfo(float).max])
    with pytest.raises(ValueError, match="resolved"):
        GaussianMeshParameters(spacing_over_tau=np.finfo(float).tiny).resolve([1e-100])
