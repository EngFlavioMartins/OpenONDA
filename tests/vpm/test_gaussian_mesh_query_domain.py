"""Retained cardinal windows inside unchanged linear-convolution domains."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.fft import irfftn, next_fast_len, rfftn

from source.solvers.vpm.physics.induction.gaussian_mesh.coordinates import slab_coordinates
from source.solvers.vpm.physics.induction.gaussian_mesh.fields import (
    GaussianImageFields,
    _logical_stencils_fit,
    _retained_stencils_fit,
    _target_stencil_window,
)


def _host_field(x, query, *, zmin=-0.5, zmax=0.5, spacing=0.035, order=10):
    field = GaussianImageFields.__new__(GaussianImageFields)
    field.closed, field._memory_pool = False, SimpleNamespace(check_context=lambda: None)
    field._prepared_images, field.max_query_points = ((0, True),), 1000
    field._prepared_world_images = ((2 * zmin, True),)
    field.max_images, field.source_only_primary = 1, False
    field.zmin, field.zmax, field.order = zmin, zmax, order
    normalized, field.steps, field.cells = slab_coordinates(x, zmin, zmax, spacing)
    targets, _, _ = slab_coordinates(query, zmin, zmax, spacing)
    reflected = normalized.copy()
    reflected[:, 2] *= -1
    points = np.concatenate((normalized, reflected, targets))
    field.origin = np.floor(points.min(axis=0)) - order
    field.shape = tuple(
        (np.ceil(points.max(axis=0) - field.origin) + order + 1).astype(int).tolist()
    )
    field.source_start, field.source_shape = _target_stencil_window(
        np.concatenate((normalized, reflected)), field.origin, order, field.shape
    )
    field.compact_start, field.compact_shape = _target_stencil_window(
        targets, field.origin, order, field.shape
    )
    field.fft_shape = tuple(
        next_fast_len(s + t - 1)
        for s, t in zip(field.source_shape, field.compact_shape, strict=True)
    )
    return field


@pytest.mark.parametrize("order", [4, 6, 8, 10])
def test_positive_validation_implies_complete_literal_device_stencil(order):
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
        checked = _logical_stencils_fit(lattice, origin, order, shape, fft)
        assert not checked or literal
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
def test_retained_query_domain_is_coordinate_consistent(translation):
    x = np.array([[-1.2, -0.6, translation + 0.01], [2.1, 0.8, translation + 0.18]])
    initial = np.array([[-0.9, -0.4, translation + 0.01], [1.2, 0.6, translation + 0.18]])
    field = _host_field(x, initial, zmin=translation, zmax=translation + 0.193)
    query = np.array([[1.1, -0.3, translation + 0.02], [-0.8, 0.5, translation + 0.17]])
    copies = query.copy(), field.origin.copy()
    assert field.can_evaluate_targets(query)
    assert not np.array_equal(query, initial)
    assert not field.can_evaluate_targets(x)
    source_lattice, _, _ = slab_coordinates(x, field.zmin, field.zmax, float(field.steps[0]))
    assert _logical_stencils_fit(source_lattice, field.origin, field.order, field.shape, None)
    assert not field.can_evaluate_targets(query + np.array([100.0, 0.0, 0.0]))
    assert field.can_evaluate_targets(np.empty((0, 3)))
    np.testing.assert_array_equal(query, copies[0])
    np.testing.assert_array_equal(field.origin, copies[1])


@pytest.mark.parametrize("order", [4, 6, 8, 10])
def test_retained_window_encloses_literal_stencils_and_rejects_uncached_cells(order):
    origin, shape = np.array([-17.0, 3.0, -9.0]), (37, 41, 43)
    fft = tuple(next_fast_len(2 * n - 1) for n in shape)
    boundary = np.array([[13.0, 14.0, 15.0], [19.0, 20.0, 21.0]])
    initial = (
        np.concatenate((boundary, np.nextafter(boundary, -np.inf), np.nextafter(boundary, np.inf)))
        + origin
    )
    start, retained = _target_stencil_window(initial, origin, order, shape)
    assert _retained_stencils_fit(initial, origin, order, shape, fft, start, retained)
    first = (np.floor(initial - origin) - order // 2 + 1).astype(int)
    assert np.all(first >= start)
    assert np.all(first + order <= np.asarray(start) + retained)
    for point in np.linspace(boundary[0], boundary[1], 100) + origin:
        assert _retained_stencils_fit(point[None], origin, order, shape, fft, start, retained)
    outside = np.array([[27.0, 25.0, 25.0]]) + origin
    assert _logical_stencils_fit(outside, origin, order, shape, fft)
    assert not _retained_stencils_fit(outside, origin, order, shape, fft, start, retained)
    with pytest.raises(RuntimeError, match="query-window metadata"):
        _retained_stencils_fit(initial, origin, order, shape, fft, (-1, 0, 0), retained)


def test_validation_rejects_malformed_metadata_run_events_and_queries():
    field = _host_field(np.array([[0.0, 0.0, 0.0]]), np.array([[0.1, 0.2, 0.3]]))
    for value in (
        np.array([[np.nan, 0, 0]]),
        np.zeros((2, 2)),
        np.zeros((1001, 3)),
        np.zeros((1, 3), complex),
    ):
        with pytest.raises(ValueError):
            field.can_evaluate_targets(value)
    assert not field.can_evaluate_targets(np.array([[1e20, 0.0, 0.0]]))
    field._prepared_world_images = ((0.0, False),)
    with pytest.raises(RuntimeError, match="descriptors changed"):
        field.can_evaluate_targets(np.zeros((1, 3)))
    field._prepared_images = None
    with pytest.raises(RuntimeError, match="prepared"):
        field.can_evaluate_targets(np.zeros((1, 3)))
    field.closed = True
    with pytest.raises(RuntimeError, match="closed"):
        field.can_evaluate_targets(np.zeros((1, 3)))


@pytest.mark.parametrize(
    "source_shape,target_shape",
    [((5, 3, 4), (2, 6, 3)), ((2, 6, 3), (5, 3, 4)), ((4, 4, 4), (4, 4, 4))],
)
@pytest.mark.parametrize("offset", [(11, -7, 9), (-11, 7, -9), (0, 0, 0)])
def test_asymmetric_padding_matches_full_discrete_pair_convolution(
    source_shape, target_shape, offset
):
    rng = np.random.default_rng(381)
    fft = tuple(next_fast_len(s + t - 1) for s, t in zip(source_shape, target_shape, strict=True))
    source = rng.normal(size=source_shape)
    density = np.zeros(fft)
    density[tuple(slice(0, n) for n in source_shape)] = source
    lag_axes, masks = [], []
    for s, t, f in zip(source_shape, target_shape, fft, strict=True):
        index = np.arange(f)
        lag_axes.append(np.where(index < t, index, index - f))
        masks.append((index < t) | (index >= f - s + 1))
    lag = np.stack(np.meshgrid(*lag_axes, indexing="ij"), axis=-1)
    mask = masks[0][:, None, None] & masks[1][None, :, None] & masks[2][None, None, :]
    kernel = np.where(mask, np.exp(-0.013 * np.sum((lag + offset) ** 2, axis=-1)), 0.0)
    actual = irfftn(rfftn(density) * rfftn(kernel), s=fft)[tuple(slice(0, n) for n in target_shape)]
    source_nodes = np.indices(source_shape).reshape(3, -1).T
    expected = np.empty(target_shape)
    for target in np.ndindex(target_shape):
        displacement = np.asarray(target) - source_nodes + offset
        expected[target] = np.sum(source.ravel() * np.exp(-0.013 * np.sum(displacement**2, axis=1)))
    np.testing.assert_allclose(actual, expected, rtol=3e-14, atol=2e-14)


@pytest.mark.parametrize("dtype,bits", [("float32", 24), ("float64", 53)])
def test_shifted_lag_precision_guard_checks_every_physical_axis(dtype, bits):
    field = GaussianImageFields.__new__(GaussianImageFields)
    field.dtype = np.dtype(dtype)
    field.source_shape, field.compact_shape = (17, 18, 19), (11, 12, 13)
    for axis in range(3):
        for sign in (-1, 1):
            offset = [0, 0, 0]
            safe = 2**bits - (
                field.source_shape[axis] - 1 if sign < 0 else field.compact_shape[axis] - 1
            )
            offset[axis] = sign * safe
            field.lag_offset = tuple(offset)
            field._validate_image_indices([(0, False)])
            offset[axis] += sign
            field.lag_offset = tuple(offset)
            with pytest.raises(ValueError, match="image lags"):
                field._validate_image_indices([(0, False)])


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_cuda_remote_query_uses_tight_fft_and_original_shared_cardinal_origin(dtype):
    cp = pytest.importorskip("cupy")
    from source.solvers.vpm.physics.induction.gaussian_mesh.stencil import cardinal_stencil_gpu
    from tests.vpm._direct_gaussian_reference import direct_finite_images

    x, g, sigma = np.array([[0.0, 0.0, 0.05]]), np.array([[0.3, -0.2, 0.7]]), np.array([0.04])
    q = np.array([[100_000.0, 0.01, 0.04], [100_000.01, -0.02, 0.08]])
    normalized = np.concatenate((x, q)) / 0.035
    full_origin = np.floor(normalized.min(axis=0)) - 20
    full_origin[2] = -20
    with GaussianImageFields(
        x,
        g,
        sigma,
        q,
        zmin=0.0,
        zmax=0.193,
        tau=0.12,
        spacing=0.035,
        cutoff=0.6,
        dtype=dtype,
        correction_dtype=dtype,
        _lattice_origin=full_origin,
    ) as field:
        assert field.shape[0] > 2**20
        assert field.fft_shape[0] < 32
        lattice, _, _ = slab_coordinates(q, 0.0, 0.193, 0.035)
        first, weight, _ = cardinal_stencil_gpu(
            lattice,
            field.origin,
            field.order,
            field.shape,
            dtype=dtype,
            pool=field.pool,
            max_points=len(q),
        )
        np.testing.assert_array_equal(
            cp.asnumpy(first), np.floor(lattice - full_origin).astype(int) - 4
        )
        del first, weight
        u, j, report = field.evaluate([(0, True)])
        expected = direct_finite_images(x, g, sigma, q, field._prepared_world_images)[:2]
        for current, exact in zip((cp.asnumpy(u), cp.asnumpy(j)), expected, strict=True):
            np.testing.assert_allclose(
                current, exact, rtol=2e-5 if dtype == "float32" else 2e-12, atol=0.0
            )
        assert report["correction"]["accepted_pairs"] == 0
        del u, j


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_cuda_window_queries_match_full_retained_grid_and_direct_reference(dtype, monkeypatch):
    cp = pytest.importorskip("cupy")
    from source.solvers.vpm.physics.induction.gaussian_mesh import fields
    from source.solvers.vpm.physics.induction.gaussian_mesh.coordinates import finite_images
    from tests.vpm._direct_gaussian_reference import direct_finite_images

    x = np.array([[-0.08, -0.06, 0.0], [0.08, 0.06, 0.193], [0.01, -0.02, 0.11]])
    g = np.array([[0.3, -0.2, 0.7], [-0.31, 0.21, -0.69], [0.1, 0.15, -0.2]])
    sigma = np.array([0.035, 0.04, 0.045])
    first = np.array([[-0.05, -0.03, 0.0], [0.04, 0.05, 0.193]])
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
        assert np.prod(reused.compact_shape) < np.prod(reused.shape)
        u, j, diagnostic = reused.evaluate_prepared(later)
        got = cp.asnumpy(u), cp.asnumpy(j)
        assert diagnostic["inverse_transforms"] == diagnostic["source_scatters"] == 0
        # No domain mutation on a rejected query, and the prepared field survives.
        assert not reused.can_evaluate_targets(later + np.array([10.0, 0.0, 0.0]))
        # This point still fits the FFT's logical domain, but its output
        # stencil was not retained. Reject it before GPU scratch allocation.
        assert not reused.can_evaluate_targets(x)
        with pytest.raises(ValueError, match="retained Gaussian field window"):
            reused.evaluate_prepared(x)
        assert reused._prepared_images is not None
        assert reused.shape == shape
        np.testing.assert_array_equal(reused.origin, origin)
        _, world, _ = finite_images(images, 0.0, 0.193, reused.cells, 10, include_primary=True)
        truth = direct_finite_images(x, g, sigma, later, world)[:2]
        for current, exact in zip(got, truth, strict=True):
            assert np.linalg.norm(current - exact) / np.linalg.norm(exact) < 1e-4
        del u, j
    # Retain the complete output grid in an independent field to verify that
    # cropping changes storage only, including wall traces and all derivatives.
    monkeypatch.setattr(
        fields, "_target_stencil_window", lambda q, origin, order, shape: ((0, 0, 0), shape)
    )
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
