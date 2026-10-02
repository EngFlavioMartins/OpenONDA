"""Complex alignment of real CuFFT channels, with no hidden scratch growth."""

import math

import numpy as np
import pytest

from source.solvers.vpm.physics.induction.gaussian_mesh.planning import field_execution_plan
from source.solvers.vpm.physics.induction.gaussian_mesh.runtime import DeviceOwner, FFTPlanPair


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("shape", [(5, 7, 9), (5, 7, 10)])
@pytest.mark.parametrize("single_workspace", [False, True])
def test_real_channel_fft_uses_only_explicit_existing_aligned_scratch(dtype, shape, single_workspace):
    cp = pytest.importorskip("cupy")
    owner = DeviceOwner(32*1024**2)
    plan = None
    try:
        with owner.allocation_scope():
            channels = cp.arange(3*math.prod(shape), dtype=dtype).reshape((3, *shape))
            scratch = cp.empty(shape, dtype=dtype)
            inverse = cp.empty(shape, dtype=dtype)
        saved = cp.asnumpy(channels)
        plan = FFTPlanPair(owner, shape, dtype, 8*1024**2, single_workspace=single_workspace)
        expected_copies = 0
        for _repeat in range(2):
            for slot in range(3):
                channel = channels[slot]
                unaligned = bool(channel.data.ptr % (2*np.dtype(dtype).itemsize))
                if unaligned:
                    before = owner.pool.total_bytes()
                    with pytest.raises(ValueError, match="explicit aligned scratch"):
                        plan.rfft(channel)
                    assert owner.pool.total_bytes() == before
                    expected_copies += 1
                result = plan.rfft(channel, aligned_scratch=scratch)
                truth = np.fft.rfftn(saved[slot])
                actual = cp.asnumpy(result)
                assert np.linalg.norm(actual-truth)/np.linalg.norm(truth) < 16*np.finfo(dtype).eps
                plan.irfft(result, inverse)
                np.testing.assert_allclose(cp.asnumpy(inverse), saved[slot],
                                           rtol=16*np.finfo(dtype).eps,
                                           atol=16*np.finfo(dtype).eps*np.abs(saved[slot]).max())
                del result, channel
            assert plan.alignment_copies == expected_copies
            assert plan.alignment_copy_bytes == expected_copies*math.prod(shape)*np.dtype(dtype).itemsize
        np.testing.assert_array_equal(cp.asnumpy(channels), saved)
        assert plan.peak_work_bytes <= plan.max_plan_bytes
        assert owner.pool.total_bytes() <= owner.max_bytes
    finally:
        if plan is not None:
            plan.close()
        owner.close()


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_alignment_scratch_admission_is_exact_and_not_an_implicit_allocation(dtype, monkeypatch):
    cp = pytest.importorskip("cupy")
    shape = (5, 7, 9)
    owner = DeviceOwner(32*1024**2)
    plan = None
    try:
        with owner.allocation_scope():
            channels = cp.ones((3, *shape), dtype=dtype)
            scratch = cp.empty(shape, dtype=dtype)
            partial = cp.ones(math.prod(shape)+2, dtype=dtype)
        plan = FFTPlanPair(owner, shape, dtype, 8*1024**2)
        def forbidden_alias_enumeration(*args, **kwargs):
            raise AssertionError("dense FFT admission must not allocate address arrays")
        monkeypatch.setattr(cp, "shares_memory", forbidden_alias_enumeration)
        with pytest.raises(ValueError, match="complex-element alignment"):
            plan.rfft(channels[1], aligned_scratch=channels[1])
        with pytest.raises(ValueError, match="must not alias"):
            plan.rfft(channels[0], aligned_scratch=channels[0])
        with pytest.raises(ValueError, match="must not alias"):
            plan.rfft(partial[:-2].reshape(shape), aligned_scratch=partial[2:].reshape(shape))
        with pytest.raises(ValueError, match="shape/dtype"):
            plan.rfft(channels[1], aligned_scratch=scratch.reshape(-1))
        # Disjoint contiguous views may share an allocation; ownership alone
        # is not an alias. The even-offset third channel is complex-aligned.
        disjoint = plan.rfft(channels[1], aligned_scratch=channels[2])
        assert cp.isfinite(disjoint).all()
        del disjoint
        calls = []
        original_empty = cp.empty

        def record_empty(shape, *args, **kwargs):
            calls.append(tuple(shape))
            return original_empty(shape, *args, **kwargs)

        monkeypatch.setattr(cp, "empty", record_empty)
        value = plan.rfft(channels[1], aligned_scratch=scratch)
        assert calls == [plan.spectrum_shape]  # Only the required returned spectrum.
        np.testing.assert_array_equal(cp.asnumpy(scratch), cp.asnumpy(channels[1]))
        with pytest.raises(ValueError, match="complex-element alignment"):
            plan.irfft(value, channels[1])
        del value
    finally:
        if plan is not None:
            plan.close()
        owner.close()


def test_fft_execution_failure_revokes_plans_and_records_exact_metadata():
    """Pure host failure injection; no CuPy import, device or allocation."""
    from types import SimpleNamespace

    plan = FFTPlanPair.__new__(FFTPlanPair)
    plan.shape, plan.dtype = (5, 7, 9), np.dtype("float32")
    plan.complex_dtype = np.dtype("complex64")
    plan.work_bytes, plan.single_workspace = 1024, True
    plan.forward, plan.inverse = object(), None
    plan.closed = False
    drains = []
    plan.owner = SimpleNamespace(admit=lambda: None,
                                 stream=SimpleNamespace(synchronize=lambda: drains.append(1)))
    source, output = [SimpleNamespace(data=SimpleNamespace(ptr=pointer)) for pointer in (1028, 2048)]
    error = RuntimeError("injected CuFFT execution failure")
    plan._execution_failed(error, direction="R2C", source=source, destination=output)
    assert plan.closed and plan.forward is plan.inverse is None and drains == [1]
    assert "shape=(5, 7, 9)" in error.__notes__[0] and "input_pointer=1028" in error.__notes__[0]


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("families", [0, 1, 2])
def test_odd_volume_fields_all_paths_match_direct_and_repeat_without_new_scratch(dtype, families, monkeypatch):
    cp = pytest.importorskip("cupy")
    from source.solvers.vpm.physics.induction.gaussian_mesh import fields
    from tests.vpm._finite_image_mesh_reference import direct_finite_images
    from tests.vpm.test_gaussian_mesh_package import _case

    # Minimal legal zero padding is odd in all axes. This deliberately tests
    # a generic supported transform, not a changed physical interpolation grid.
    monkeypatch.setattr(fields, "next_fast_len", lambda n: n)
    x, gamma, sigma, interior, images = _case()
    query = np.concatenate((interior, [[.03, .02, 0.], [.03, .02, .193]]))
    images = [(0, False), *images]
    options = {"zmin": 0., "zmax": .193, "tau": .12, "spacing": .035, "cutoff": .6,
               "dtype": dtype, "correction_dtype": dtype, "source_only_primary": True}
    with fields.GaussianImageFields(x, gamma, sigma, query, **options) as probe:
        shape, fft = probe.shape, probe.fft_shape
    assert math.prod(fft) % 2 == 1
    kwargs = {}
    if families:
        size = np.dtype(dtype).itemsize
        full = field_execution_plan(shape, fft, len(x), len(query), 10, size, 2*1024**3, 8*1024**2)
        candidates = []
        for available in np.linspace(full.payload_bytes//3, full.payload_bytes-1, 160, dtype=np.int64):
            try:
                candidate = field_execution_plan(shape, fft, len(x), len(query), 10, size,
                                                 int(available)+8*1024**2, 8*1024**2)
            except MemoryError:
                continue
            if candidate.mode == "streamed" and candidate.source_families == families and candidate.kernel_channels > 1:
                candidates.append((int(available), candidate))
        assert candidates
        available, _ = candidates[-1]
        kwargs = {"max_scratch_bytes": available+8*1024**2, "max_plan_bytes": 8*1024**2}
    with fields.GaussianImageFields(x, gamma, sigma, query, **options, **kwargs) as owner:
        assert owner.execution_plan.mode == ("streamed" if families else "all_channels")
        inverse_id = None
        originals = None
        for _repeat in range(3):
            record = owner.prepare(images)
            assert record["fft_alignment_copies"] > 0
            assert record["fft_alignment_copy_bytes"] == record["fft_alignment_copies"]*owner.volume*owner.dtype.itemsize
            if inverse_id is None:
                inverse_id = owner._inverse.data.ptr
            assert owner._inverse.data.ptr == inverse_id
            u, j, _ = owner.evaluate_prepared(query)
            current = cp.asnumpy(u), cp.asnumpy(j)
            truth = direct_finite_images(x, gamma, sigma, query, record["world_images"])[:2]
            for actual, exact in zip(current, truth, strict=True):
                assert np.linalg.norm(actual-exact)/np.linalg.norm(exact) < 1e-4
            if originals is not None:
                for actual, first in zip(current, originals, strict=True):
                    np.testing.assert_allclose(actual, first, rtol=32*np.finfo(dtype).eps,
                                               atol=32*np.finfo(dtype).eps*np.max(np.abs(first)))
            originals = current
            assert record["pool_reserved_bytes"] <= owner.max_scratch_bytes
            assert record["plan_peak_work_bytes"] <= owner.max_plan_bytes
            del u, j
