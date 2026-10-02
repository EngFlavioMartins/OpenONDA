"""Logical cardinal domains, not FFT padding or an initial query-only AABB."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.fft import next_fast_len

from source.solvers.vpm.physics.induction.gaussian_mesh.coordinates import slab_coordinates
from source.solvers.vpm.physics.induction.gaussian_mesh.fields import (
    GaussianImageFields,
    _logical_stencils_fit,
)


def _host_owner(x, query, *, zmin=-0.5, zmax=0.5, spacing=0.035, order=10):
    owner = GaussianImageFields.__new__(GaussianImageFields)
    owner.closed, owner._owner = False, SimpleNamespace(admit=lambda: None)
    owner._prepared_images, owner.max_query_points = ((0, True),), 1000
    owner._prepared_world_images = ((2 * zmin, True),)
    owner.max_images, owner.source_only_primary = 1, False
    owner.zmin, owner.zmax, owner.order = zmin, zmax, order
    normalized, owner.steps, owner.cells = slab_coordinates(x, zmin, zmax, spacing)
    targets, _, _ = slab_coordinates(query, zmin, zmax, spacing)
    reflected = normalized.copy()
    reflected[:, 2] *= -1
    points = np.concatenate((normalized, reflected, targets))
    owner.origin = np.floor(points.min(axis=0)) - order
    owner.shape = tuple(
        (np.ceil(points.max(axis=0) - owner.origin) + order + 1).astype(int).tolist()
    )
    owner.fft_shape = tuple(next_fast_len(2 * n - 1) for n in owner.shape)
    return owner


@pytest.mark.parametrize("order", [4, 6, 8, 10])
def test_positive_admission_implies_complete_literal_device_stencil(order):
    rng = np.random.default_rng(127 + order)
    shape, fft = (37, 41, 43), (75, 81, 90)
    origin = np.array([-17.0, 3.0, -9.0])
    points = rng.uniform(-3, 49, size=(500, 3))
    # Check every point separately, including exact and adjacent grid boundaries.
    endpoints = np.array([order // 2 - 1.0, shape[0] - order // 2])
    boundary = np.array(
        [
            [v, 17.0, 19.0]
            for value in endpoints
            for v in (np.nextafter(value, -np.inf), value, np.nextafter(value, np.inf))
        ]
    )
    for coordinate in np.concatenate((points, boundary)):
        lattice = (coordinate + origin)[None]
        actual = lattice - origin
        literal_first = np.floor(actual) - order // 2 + 1
        literal = bool(np.all(literal_first >= 0) and np.all(literal_first + order <= shape))
        admitted = _logical_stencils_fit(lattice, origin, order, shape, fft)
        assert not admitted or literal
    assert _logical_stencils_fit(np.array([[17.0, 19.0, 21.0]]) + origin, origin, order, shape, fft)


def test_fft_padding_never_becomes_the_query_domain():
    shape, fft = (32, 32, 32), (64, 64, 64)
    # A complete p10 stencil fits allocation but lies outside logical output.
    query = np.array([[40.25, 15.25, 15.25]])
    assert _logical_stencils_fit(query, np.zeros(3), 10, fft, (128,) * 3)
    assert not _logical_stencils_fit(query, np.zeros(3), 10, shape, fft)
    with pytest.raises(RuntimeError, match="linear-convolution"):
        _logical_stencils_fit(query, np.zeros(3), 10, shape, (32,) * 3)


@pytest.mark.parametrize("translation", [0.0, 1024.3, -1e5])
def test_field_domain_is_source_wide_and_coordinate_consistent(translation):
    x = np.array([[-1.2, -0.6, translation + 0.01], [2.1, 0.8, translation + 0.18]])
    initial = np.array([[0.1, 0.2, translation + 0.08]])
    owner = _host_owner(x, initial, zmin=translation, zmax=translation + 0.193)
    query = np.array([[1.1, -0.3, translation + 0.02], [-0.8, 0.5, translation + 0.17]])
    copies = query.copy(), owner.origin.copy()
    assert owner.can_evaluate_targets(query)
    assert not np.all(query <= initial.max(axis=0))
    assert not owner.can_evaluate_targets(query + np.array([100.0, 0.0, 0.0]))
    assert owner.can_evaluate_targets(np.empty((0, 3)))
    np.testing.assert_array_equal(query, copies[0])
    np.testing.assert_array_equal(owner.origin, copies[1])


def test_admission_rejects_malformed_metadata_lifecycle_and_queries():
    owner = _host_owner(np.array([[0.0, 0.0, 0.0]]), np.array([[0.1, 0.2, 0.3]]))
    for value in (
        np.array([[np.nan, 0, 0]]),
        np.zeros((2, 2)),
        np.zeros((1001, 3)),
        np.zeros((1, 3), complex),
    ):
        with pytest.raises(ValueError):
            owner.can_evaluate_targets(value)
    assert not owner.can_evaluate_targets(np.array([[1e20, 0.0, 0.0]]))
    owner._prepared_world_images = ((0.0, False),)
    with pytest.raises(RuntimeError, match="descriptors changed"):
        owner.can_evaluate_targets(np.zeros((1, 3)))
    owner._prepared_images = None
    with pytest.raises(RuntimeError, match="prepared"):
        owner.can_evaluate_targets(np.zeros((1, 3)))
    owner.closed = True
    with pytest.raises(RuntimeError, match="closed"):
        owner.can_evaluate_targets(np.zeros((1, 3)))


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_cuda_disjoint_queries_match_same_logical_grid_reference(dtype):
    cp = pytest.importorskip("cupy")
    from source.solvers.vpm.physics.induction.gaussian_mesh.coordinates import finite_images
    from tests.vpm._direct_gaussian_reference import direct_finite_images

    x = np.array([[-0.08, -0.06, 0.0], [0.08, 0.06, 0.193], [0.01, -0.02, 0.11]])
    g = np.array([[0.3, -0.2, 0.7], [-0.31, 0.21, -0.69], [0.1, 0.15, -0.2]])
    sigma = np.array([0.035, 0.04, 0.045])
    first = np.array([[0.0, 0.01, 0.08]])
    later = np.array(
        [[0.035, -0.02, 0.04], [-0.045, 0.04, 0.14], [0.02, 0.0, 0.0], [0.02, 0.0, 0.193]]
    )
    images = [(0, False), (0, True), (-1, False), (1, False), (-1, True), (1, True)]
    options = {
        "zmin": 0.0,
        "zmax": 0.193,
        "tau": 0.12,
        "spacing": 0.03,
        "cutoff": 0.6,
        "dtype": dtype,
        "correction_dtype": dtype,
        "source_only_primary": True,
    }
    with GaussianImageFields(x, g, sigma, first, **options) as reused:
        reused.prepare(images)
        assert reused.can_evaluate_targets(later)
        shape, origin = reused.shape, reused.origin.copy()
        u, j, diagnostic = reused.evaluate_prepared(later)
        got = cp.asnumpy(u), cp.asnumpy(j)
        assert diagnostic["inverse_transforms"] == diagnostic["source_scatters"] == 0
        # No domain mutation on a rejected query, and the prepared field survives.
        assert not reused.can_evaluate_targets(later + np.array([10.0, 0.0, 0.0]))
        assert reused.shape == shape
        np.testing.assert_array_equal(reused.origin, origin)
        _, world, _ = finite_images(images, 0.0, 0.193, reused.cells, 10, include_primary=True)
        truth = direct_finite_images(x, g, sigma, later, world)[:2]
        for current, exact in zip(got, truth, strict=True):
            assert np.linalg.norm(current - exact) / np.linalg.norm(exact) < 1e-4
        del u, j
    # The later targets lie in source/reflection-covered grid support, so adding
    # them initially keeps the mathematical mesh exactly the same.
    with GaussianImageFields(x, g, sigma, np.concatenate((first, later)), **options) as fresh:
        assert fresh.shape == shape
        np.testing.assert_array_equal(fresh.origin, origin)
        fresh.prepare(images)
        u, j, _ = fresh.evaluate_prepared(later)
        for actual, prior in zip((cp.asnumpy(u), cp.asnumpy(j)), got, strict=True):
            scale = max(float(np.max(np.abs(prior))), np.finfo(dtype).tiny)
            np.testing.assert_allclose(
                actual, prior, rtol=16 * np.finfo(dtype).eps, atol=16 * np.finfo(dtype).eps * scale
            )
        del u, j
