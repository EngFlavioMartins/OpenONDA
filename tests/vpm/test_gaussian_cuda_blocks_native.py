"""Small real CUDA source/target splits against unsplit and direct fields."""

import math

import numpy as np
import pytest

from source.solvers.vpm.physics.induction.gaussian_mesh.blocked_fields import (
    GaussianBlockedCUDAFields,
)
from source.solvers.vpm.physics.induction.gaussian_mesh.fields import GaussianImageFields
from tests.vpm._direct_gaussian_reference import direct_finite_images

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("split", ["source", "target"])
def test_real_private_cap_splits_preserve_fields_and_resource_storage(monkeypatch, dtype, split):
    cp = pytest.importorskip("cupy")
    from source.solvers.vpm.physics.induction.gaussian_mesh import fields

    x = np.array([[-0.075, 0.012, 0.023], [0.018, -0.023, 0.079], [0.086, 0.031, 0.15]])
    gamma = np.array([[0.3, -0.2, 0.7], [-0.25, 0.5, -0.4], [0.1, 0.2, 0.3]])
    sigma = np.array([0.035, 0.04, 0.05])
    q = np.array(
        [[0.012, 0.014, 0.0], [-0.015, 0.053, 0.061], [0.013, -0.036, 0.193], [0.018, 0.02, 0.04]]
    )
    if split == "source":
        x[:, 0] = [-0.3, 0.0, 0.3]
    else:
        x[:, 0] = [0.0, 0.015, 0.02]
        q[:, 0] = [-0.3, -0.1, 0.1, 0.3]
    images = ((0, False), (0, True), (-1, False), (1, False), (1, True))
    options = {
        "zmin": 0.0,
        "zmax": 0.193,
        "tau": 0.12,
        "spacing": 0.035,
        "cutoff": 0.6,
        "order": 10,
        "dtype": dtype,
        "correction_dtype": dtype,
        "source_only_primary": True,
        "max_scratch_bytes": 48 * 1024**2,
        "max_plan_bytes": 8 * 1024**2,
        "max_correction_bytes": 8 * 1024**2,
        "max_total_bytes": 56 * 1024**2,
    }
    default_pool = cp.get_default_memory_pool()
    with cp.cuda.using_allocator(default_pool.malloc):
        sentinel = cp.arange(17, dtype=cp.float32)
    cp.cuda.get_current_stream().synchronize()
    allocator = cp.cuda.get_allocator()
    used_before_baseline, limit_before_baseline = (
        default_pool.used_bytes(),
        default_pool.get_limit(),
    )
    with GaussianImageFields(x, gamma, sigma, q, **options) as full:
        u, j, _ = full.evaluate(images)
        expected = cp.asnumpy(u), cp.asnumpy(j)
        world = full._prepared_world_images
        volume = math.prod(full.fft_shape)
        spectrum = full.fft_shape[0] * full.fft_shape[1] * (full.fft_shape[2] // 2 + 1)
        retained = math.prod(full.compact_shape)
        size = np.dtype(dtype).itemsize
        metadata = (
            (2 * len(x) + len(q)) * (12 + 30 * size) + (len(x) + len(q)) * 24 + len(q) * 12 * size
        )
        # Smallest legal plan: one source family, one result spectrum and
        # one radial grid. One byte below this forces real pool validation to
        # fail for the complete request, while spatial children are smaller.
        minimum = (14 * spectrum + 2 * volume + 12 * retained) * size + metadata
        cap = minimum - 1
        del u, j
    assert cp.cuda.get_allocator() is allocator
    assert (default_pool.used_bytes(), default_pool.get_limit()) == (
        used_before_baseline,
        limit_before_baseline,
    )
    # Measure subdivision separately from the unsplit baseline's cold
    # teardown, which can warm a cached CuPy block without retaining bytes.
    pool_state = default_pool.used_bytes(), default_pool.total_bytes(), default_pool.get_limit()
    assert cap < options["max_scratch_bytes"]
    options["max_plan_bytes"] = min(options["max_plan_bytes"], cap // 2)
    options["max_scratch_bytes"] = cap
    options["max_total_bytes"] = cap + options["max_correction_bytes"]
    leaves = []

    class RecordedLeaf(GaussianImageFields):
        def __init__(self, sx, strengths, cores, tq, **controls):
            leaves.append((self, len(sx), len(tq)))
            super().__init__(sx, strengths, cores, tq, **controls)

    monkeypatch.setattr(fields, "GaussianImageFields", RecordedLeaf)
    field = GaussianBlockedCUDAFields(x, gamma, sigma, q, **options)
    try:
        field.prepare(images)
        u, j, report = field.evaluate_prepared(q)
        direct = direct_finite_images(x, gamma, sigma, q, world)[:2]
        assert report["fft_blocks"] > 1 and report["block_splits"] > 0
        assert report["inverse_transforms"] >= 12 * report["fft_blocks"]
        assert report["peak_pool_reserved_bytes"] <= cap
        assert report["peak_plan_work_bytes"] <= options["max_plan_bytes"]
        assert report["peak_combined_pool_reserved_bytes"] <= options["max_total_bytes"]
        assert field._leaf is field._failed_field is None
        assert leaves[0][1:] == (len(x), len(q))
        assert any(
            (ns < len(x) and nt == len(q)) if split == "source" else (nt < len(q) and ns == len(x))
            for _, ns, nt in leaves
        )
        for actual, baseline, reference in zip((u, j), expected, direct, strict=True):
            assert actual.dtype == np.dtype(dtype)
            assert np.linalg.norm(actual - reference) / np.linalg.norm(reference) < 2e-5
            epsilon = np.finfo(dtype).eps
            np.testing.assert_allclose(
                actual, baseline, rtol=32 * epsilon, atol=32 * epsilon * np.max(np.abs(reference))
            )
        for leaf, _, _ in leaves:
            assert leaf.closed
            if leaf._memory_pool is not None:
                assert leaf._memory_pool.closed
                assert (
                    leaf._memory_pool.pool.used_bytes() == leaf._memory_pool.pool.total_bytes() == 0
                )
        assert cp.cuda.get_allocator() is allocator
        assert (
            default_pool.used_bytes(),
            default_pool.total_bytes(),
            default_pool.get_limit(),
        ) == pool_state
        np.testing.assert_array_equal(cp.asnumpy(sentinel), np.arange(17, dtype=np.float32))
    finally:
        field.close()
        del sentinel
