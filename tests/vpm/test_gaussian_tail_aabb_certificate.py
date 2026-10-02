"""Qualification of immutable source moments and continuous query boxes."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from source.solvers.vpm.physics.induction.gaussian_tail import prepare_tail_source, query_tail_bound
from tests.vpm._gaussian_tail_reference import cloud, explicit_tail


@pytest.mark.parametrize("kind", ["random", "cancelled", "axial", "translated", "near_admission"])
def test_box_bound_dominates_all_target_interval_and_independent_tail(kind):
    x, g, sigma, targets, zmin, zmax, _ = cloud(kind)
    k = 32
    source = prepare_tail_source(x, g, sigma, z_min=zmin, z_max=zmax)
    box = query_tail_bound(source, targets.min(axis=0), targets.max(axis=0), shells=k)
    points = [query_tail_bound(source, target, target, shells=k) for target in targets]
    assert all(box.velocity_upper >= point.velocity_upper for point in points)
    assert all(box.gradient_upper >= point.gradient_upper for point in points)
    u, j = explicit_tail(x, g, sigma, targets, zmin, zmax, k + 1, 512)
    beyond = query_tail_bound(source, targets.min(axis=0), targets.max(axis=0), shells=512)
    assert np.all(np.linalg.norm(u, axis=1) <= box.velocity_upper + beyond.velocity_upper + 2e-14)
    assert np.all(
        np.linalg.norm(j, axis=(1, 2)) <= box.gradient_upper + beyond.gradient_upper + 2e-14
    )


def test_arbitrary_points_inside_box_not_only_corners_are_covered():
    x, g, sigma, _, zmin, zmax, _ = cloud("random")
    lower, upper = np.array([-1.0, -0.8, -0.7]), np.array([2.0, 0.9, 1.3])
    rng = np.random.default_rng(827)
    points = rng.uniform(lower, upper, (103, 3))
    source = prepare_tail_source(x, g, sigma, z_min=zmin, z_max=zmax)
    box = query_tail_bound(source, lower, upper, shells=32)
    individual = [query_tail_bound(source, point, point, shells=32) for point in points]
    assert all(box.velocity_upper >= result.velocity_upper for result in individual)
    assert all(box.gradient_upper >= result.gradient_upper for result in individual)
    # Larger boxes must remain conservative; no product-of-corner assumption.
    outer = query_tail_bound(source, lower - 1.0, upper + 1.0, shells=32)
    assert outer.velocity_upper >= box.velocity_upper
    assert outer.gradient_upper >= box.gradient_upper


def test_prepared_source_is_an_immutable_snapshot_not_identity_based_live_cache():
    x, g, sigma, targets, zmin, zmax, _ = cloud("random")
    source = prepare_tail_source(x, g, sigma, z_min=zmin, z_max=zmax)
    before = query_tail_bound(source, targets.min(axis=0), targets.max(axis=0), shells=32)
    x[:, 0] += 0.01
    g *= 2
    sigma *= 1.01
    retained = query_tail_bound(source, targets.min(axis=0), targets.max(axis=0), shells=32)
    assert retained == before
    replacement = prepare_tail_source(x, g, sigma, z_min=zmin, z_max=zmax)
    assert replacement.source_sha256 != source.source_sha256
    assert (
        query_tail_bound(
            replacement, targets.min(axis=0), targets.max(axis=0), shells=32
        ).velocity_upper
        > before.velocity_upper
    )
    with pytest.raises(FrozenInstanceError):
        source.core_max = 0.5
    assert isinstance(source.families, tuple)
    assert isinstance(source.families[0].first_moment.lower, tuple)


def test_single_point_box_dominates_corresponding_existing_bound():
    x, g, sigma, targets, zmin, zmax, _ = cloud("random")
    source = prepare_tail_source(x, g, sigma, z_min=zmin, z_max=zmax)
    t = targets[0]
    result = query_tail_bound(source, t, t, shells=128)
    u, j = explicit_tail(x, g, sigma, t[None, :], zmin, zmax, 129, 512)
    beyond = query_tail_bound(source, t, t, shells=512)
    assert np.linalg.norm(u) <= result.velocity_upper + beyond.velocity_upper + 2e-14
    assert np.linalg.norm(j) <= result.gradient_upper + beyond.gradient_upper + 2e-14


@pytest.mark.parametrize("count", [0, 7])
def test_identically_zero_snapshot_needs_no_geometric_admission(count):
    source = prepare_tail_source(
        np.zeros((count, 3)), np.zeros((count, 3)), np.ones(count), z_min=-0.5, z_max=0.5
    )
    result = query_tail_bound(source, np.ones(3) * 1e6, np.ones(3) * 2e6, shells=1)
    assert result.velocity_upper == result.gradient_upper == 0.0


def test_invalid_query_and_separation_fail_closed():
    source = prepare_tail_source(
        np.zeros((1, 3)), np.ones((1, 3)), np.ones(1) * 0.04, z_min=-0.5, z_max=0.5
    )
    with pytest.raises(ValueError, match="AABB"):
        query_tail_bound(source, np.ones(3), np.zeros(3), shells=32)
    with pytest.raises(ValueError, match="L1 extent"):
        query_tail_bound(source, np.ones(3) * 10, np.ones(3) * 11, shells=1)
    with pytest.raises(TypeError):
        query_tail_bound(None, np.zeros(3), np.ones(3), shells=32)


def test_source_caps_and_fields_checked_before_ownership():
    x, g, sigma = np.zeros((3, 3)), np.ones((3, 3)), np.ones(3) * 0.04
    with pytest.raises(ValueError):
        prepare_tail_source(x, g, sigma, z_min=-0.5, z_max=0.5, max_sources=2)
    sigma[0] = 0.0
    with pytest.raises(ValueError):
        prepare_tail_source(x, g, sigma, z_min=-0.5, z_max=0.5)
