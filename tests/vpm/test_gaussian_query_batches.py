"""Growing queries use bounded scratch without changing their complete fields."""

import math

import numpy as np
import pytest

from source.solvers.vpm.physics.induction.gaussian_mesh.planning import query_batch_size


def test_late_query_admits_complete_output_and_only_bounded_stencil_scratch():
    targets, size, order = 1_000_000, 4, 10
    used = 1300*1024**2
    cap = used+48*targets+4096+168*1024
    batch = query_batch_size(targets, order, size, size, smooth_cap=cap,
                            smooth_used=used, correction_cap=256*1024**2,
                            correction_used=64*1024**2, device_available=600*1024**2)
    assert batch == 1024
    assert used+48*targets+4096+batch*168 <= cap
    # A previously returned device output remains owned by the caller and
    # participates in the next query's admission.
    with pytest.raises(MemoryError, match="complete Gaussian query output"):
        query_batch_size(targets, order, size, size, smooth_cap=cap,
                         smooth_used=used+48*targets, correction_cap=256*1024**2,
                         correction_used=64*1024**2, device_available=600*1024**2)


def test_correction_precision_and_live_device_memory_each_bound_the_batch():
    options = {"smooth_cap": 64*1024**2, "smooth_used": 0,
               "correction_cap": 8192+84*100, "correction_used": 0,
               "device_available": 128*1024**2}
    assert query_batch_size(1000, 10, 4, 4, **options) == 100
    assert query_batch_size(1000, 10, 4, 8, **options) == 63
    options.update(correction_cap=64*1024**2, device_available=48_128+4096+8192+252*7)
    assert query_batch_size(1000, 10, 4, 4, **options) == 7
    assert query_batch_size(0, 10, 4, 4, smooth_cap=0, smooth_used=0,
                            correction_cap=0, correction_used=0, device_available=0) == 0


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_cuda_larger_later_query_matches_full_query_with_bounded_private_pool(dtype):
    cp = pytest.importorskip("cupy")
    from source.solvers.vpm.physics.induction.gaussian_mesh.fields import GaussianImageFields
    from tests.vpm.test_gaussian_mesh_package import _case

    x, gamma, sigma, q, images = _case()
    query = np.tile(q[:1], (33, 1))
    with GaussianImageFields(x, gamma, sigma, q[:1], zmin=0., zmax=.193,
                             tau=.12, spacing=.035, cutoff=.6, dtype=dtype,
                             correction_dtype=dtype) as owner:
        owner.prepare(images)
        u, j, baseline = owner.evaluate_prepared(query)
        expected = cp.asnumpy(u), cp.asnumpy(j)
        del u, j
        owner.stream.synchronize()
        owner.pool.free_all_blocks()
        owner._correction.pool.free_all_blocks()
        retained_reserved_before = owner.pool.total_bytes()
        size = np.dtype(dtype).itemsize
        output_bytes = ((12*len(query)*size+511)//512)*512
        per_target = 48+30*size
        cap = owner.pool.used_bytes()+output_bytes+4096+3*per_target
        owner.pool.set_limit(size=cap)
        compact = owner._compact_fields
        u, j, report = owner.evaluate_prepared(query)
        assert report["query_batch_size"] == 3
        assert report["query_batches"] == math.ceil(len(query)/3)
        assert report["stencil"]["max_batch_points"] == 3
        assert report["correction"]["target_count"] == len(query)
        assert report["correction"]["candidate_pairs"] == baseline["correction"]["candidate_pairs"]
        # Lowering an existing pool limit does not release split unused
        # blocks whose siblings still hold the prepared fields. It bounds
        # fresh allocations, while their original reservation can remain.
        assert report["smooth_pool_reserved_bytes"] <= max(cap, retained_reserved_before)
        assert report["smooth_pool_reserved_bytes"] <= owner.max_scratch_bytes
        assert owner.pool.used_bytes() <= cap
        assert report["stencil"]["pool_used_after"] <= cap
        assert owner._compact_fields is compact
        for got, prior in zip((cp.asnumpy(u), cp.asnumpy(j)), expected, strict=True):
            np.testing.assert_array_equal(got, prior)
        # Holding the first result prevents another complete output from
        # fitting, and the admission failure preserves both it and the field.
        with pytest.raises(MemoryError, match="complete Gaussian query output"):
            owner.evaluate_prepared(query)
        assert owner._prepared_images is not None
        np.testing.assert_array_equal(cp.asnumpy(u), expected[0])
        del u, j


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_cuda_growing_query_stays_inside_cap_fixed_at_field_construction(dtype):
    cp = pytest.importorskip("cupy")
    from source.solvers.vpm.physics.induction.gaussian_mesh.fields import GaussianImageFields
    from tests.vpm.test_gaussian_mesh_package import _case

    x, gamma, sigma, q, images = _case()
    options = {"zmin": 0., "zmax": .193, "tau": .12, "spacing": .035,
               "cutoff": .6, "dtype": dtype, "correction_dtype": dtype,
               "max_plan_bytes": 8*1024**2}
    with GaussianImageFields(x, gamma, sigma, q[:1], **options) as baseline:
        payload = baseline.execution_plan.payload_bytes
    cap = payload+options["max_plan_bytes"]
    query = np.tile(q[:1], (65537, 1))
    with GaussianImageFields(x, gamma, sigma, q[:1], **options,
                             max_scratch_bytes=cap) as owner:
        owner.prepare(images)
        # Separate CUDA owners can scatter source atoms in a different
        # rounding order. Compare query batching on this immutable field;
        # cross-owner operator parity has a separate numerical envelope.
        u, j, _ = owner.evaluate_prepared(q[:1])
        expected = cp.asnumpy(u), cp.asnumpy(j)
        del u, j
        u, j, report = owner.evaluate_prepared(query)
        assert 0 < report["query_batch_size"] < len(query)
        assert report["query_batches"] > 1
        assert report["smooth_pool_reserved_bytes"] <= owner.pool.get_limit() <= cap
        for got, prior in zip((cp.asnumpy(u), cp.asnumpy(j)), expected, strict=True):
            np.testing.assert_array_equal(got, np.repeat(prior, len(query), axis=0))
        del u, j


def test_cuda_allocation_failure_retries_unpublished_batch_without_duplicate_correction(monkeypatch):
    cp = pytest.importorskip("cupy")
    from source.solvers.vpm.physics.induction.gaussian_mesh import fields
    from tests.vpm.test_gaussian_mesh_package import _case

    x, gamma, sigma, q, images = _case()
    query = np.tile(q[:1], (9, 1))
    with fields.GaussianImageFields(x, gamma, sigma, q[:1], zmin=0., zmax=.193,
                                   tau=.12, spacing=.035, cutoff=.6) as owner:
        owner.prepare(images)
        u, j, _ = owner.evaluate_prepared(query)
        expected = cp.asnumpy(u), cp.asnumpy(j)
        del u, j
        original = owner._correction.evaluate
        failed = False

        def fail_first_correction(targets, descriptors):
            nonlocal failed
            if not failed:
                failed = True
                raise MemoryError("simulate driver allocation failure after gather")
            return original(targets, descriptors)

        monkeypatch.setattr(fields, "query_batch_size", lambda *args, **kwargs: 4)
        monkeypatch.setattr(owner._correction, "evaluate", fail_first_correction)
        u, j, report = owner.evaluate_prepared(query)
        assert report["query_allocation_retries"] == 1
        assert report["query_batch_size"] == 2 and report["query_batches"] == 5
        for got, prior in zip((cp.asnumpy(u), cp.asnumpy(j)), expected, strict=True):
            np.testing.assert_array_equal(got, prior)
        del u, j
