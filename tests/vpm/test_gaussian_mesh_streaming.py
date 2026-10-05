"""Execution-only bounded memory plans; no physical parameter changes."""

import math

import numpy as np
import pytest

from source.solvers.vpm.physics.induction.gaussian_mesh.planning import (
    correction_query_reserve,
    device_field_execution_plan,
    field_execution_plan,
    required_channels,
)


def test_live_device_budget_selects_exact_streaming_without_raising_caps():
    shape, fft = (411, 103, 53), (825, 210, 105)
    cap, plan_cap, correction = 2 * 1024**3, 128 * 1024**2, 256 * 1024**2
    options = (shape, fft, 293340, 293340, 10, 4)
    original = field_execution_plan(*options, cap, plan_cap)
    ample, ample_cap = device_field_execution_plan(*options, cap, plan_cap, correction, 6 * 1024**3)
    assert ample == original and ample_cap == cap
    constrained, effective = device_field_execution_plan(
        *options, cap, plan_cap, correction, 1800 * 1024**2
    )
    assert effective == 1800 * 1024**2 - correction < cap
    assert constrained != original
    assert constrained.payload_bytes + plan_cap + correction <= 1800 * 1024**2
    assert constrained.inverse_transforms in (12, 24)
    with pytest.raises(MemoryError, match="insufficient free device memory.*fft_shape"):
        device_field_execution_plan(*options, cap, plan_cap, correction, correction + plan_cap)


def test_device_budget_reserves_entire_correction_and_separate_fft_cap():
    options = ((25, 25, 27), (50, 50, 54), 10, 20, 10, 4)
    cap, plan_cap, correction = 32 * 1024**2, 1024**2, 2 * 1024**2
    full = field_execution_plan(*options, cap, plan_cap)
    free = full.payload_bytes + plan_cap + correction - 1
    selected, effective = device_field_execution_plan(*options, cap, plan_cap, correction, free)
    assert selected.mode == "streamed"
    assert effective == free - correction
    assert selected.payload_bytes + plan_cap <= effective <= cap


def test_checkpoint_275_query_window_avoids_observed_device_admission_cliff():
    # Checkpoint source count and widened midspan sampler envelope. Only the
    # retained inverse-output shape changes; the logical/FFT grids stay fixed.
    shape, fft, retained = (411, 155, 53), (825, 315, 105), (147, 143, 11)
    options = (shape, fft, 293086, 2665, 10, 4)
    cap, plan_cap, correction = 2*1024**3, 128*1024**2, 256*1024**2
    for free in (1540947968, 1603862528):
        with pytest.raises(MemoryError, match="insufficient free device memory"):
            device_field_execution_plan(*options, cap, plan_cap, correction, free)
        plan, effective = device_field_execution_plan(
            *options, cap, plan_cap, correction, free, retained_shape=retained
        )
        assert plan.mode == "streamed"
        assert plan.payload_bytes + plan_cap + correction <= free
        assert effective <= cap
    full = field_execution_plan(*options, 8*1024**3, plan_cap)
    cropped = field_execution_plan(*options, 8*1024**3, plan_cap, retained_shape=retained)
    assert full.mode == cropped.mode == "all_channels"
    assert full.payload_bytes - cropped.payload_bytes == 12*(math.prod(shape)-math.prod(retained))*4
    assert full.metadata_bytes == cropped.metadata_bytes
    assert full.radial_grid_passes == cropped.radial_grid_passes
    assert full.inverse_transforms == cropped.inverse_transforms
    for invalid in ((9, 20, 20), (412, 20, 20), (20, 20), (20.0, 20, 20)):
        with pytest.raises(ValueError, match="retained query window"):
            field_execution_plan(*options, cap, plan_cap, retained_shape=invalid)


def test_runtime_low_memory_plan_matches_unconstrained_cuda_fields(monkeypatch):
    cp = pytest.importorskip("cupy")
    from source.solvers.vpm.physics.induction.gaussian_mesh.fields import GaussianImageFields
    from tests.vpm.test_gaussian_mesh_package import _case

    x, gamma, sigma, q, images = _case()
    options = {"zmin": 0.0, "zmax": 0.193, "tau": 0.12, "spacing": 0.035, "cutoff": 0.6}
    with GaussianImageFields(x, gamma, sigma, q, **options) as unconstrained:
        u, j, _ = unconstrained.evaluate(images)
        expected = cp.asnumpy(u), cp.asnumpy(j)
        original_plan = unconstrained.execution_plan
        del u, j
    # Simulate only the admission value, not CUDA allocation success or field
    # arithmetic. Actual transforms and gathering still execute on the GPU.
    plan_cap = 8 * 1024**2
    query_reserve = correction_query_reserve(1, np.dtype("float32").itemsize)
    free = original_plan.payload_bytes + plan_cap + query_reserve - 1
    monkeypatch.setattr(cp.cuda.runtime, "memGetInfo", lambda: (free, 6 * 1024**3))
    with GaussianImageFields(x, gamma, sigma, q, **options, max_plan_bytes=plan_cap) as constrained:
        assert constrained.execution_plan.mode == "streamed"
        assert constrained.pool.get_limit() == free - query_reserve
        u, j, report = constrained.evaluate(images)
        for got, baseline in zip((cp.asnumpy(u), cp.asnumpy(j)), expected, strict=True):
            np.testing.assert_allclose(
                got,
                baseline,
                rtol=32 * np.finfo("float32").eps,
                atol=32 * np.finfo("float32").eps * np.max(np.abs(baseline)),
            )
        assert report["smooth_pool_reserved_bytes"] <= constrained.effective_smooth_pool_cap
        del u, j


def test_actual_native_and_extended_shapes_have_bounded_plans():
    cap, plan_cap = 2 * 1024**3, 128 * 1024**2
    native = field_execution_plan(
        (356, 91, 49), (720, 189, 98), 293340, 293340, 10, 4, cap, plan_cap
    )
    assert native.mode == "all_channels"
    default = field_execution_plan(
        (411, 103, 53), (825, 210, 105), 293340, 293340, 10, 4, cap, plan_cap
    )
    assert default.mode == "streamed" and default.kernel_channels > 1
    assert default.radial_grid_passes < 18
    # Hypothetical x15+configured FVM query box, keeping 1M source/query
    # buffers: capacity arithmetic only, not a fabricated future flow state.
    wake = field_execution_plan(
        (575, 129, 53), (1152, 264, 105), 1000000, 1000000, 10, 4, cap, plan_cap
    )
    assert wake.mode == "streamed" and wake.source_families == 1
    assert wake.payload_bytes + plan_cap <= cap
    with pytest.raises(MemoryError, match="even with streaming"):
        field_execution_plan(
            (688, 355, 53), (1375, 720, 105), 1000000, 1000000, 10, 4, cap, plan_cap
        )


def test_caps_choose_execution_only_and_never_raise_the_limit():
    shape, fft = (25, 25, 27), (50, 50, 54)
    plans = []
    for mib in (8, 12, 16, 24, 32):
        try:
            plan = field_execution_plan(shape, fft, 10, 20, 10, 4, mib * 1024**2, 1024**2)
        except MemoryError:
            continue
        assert plan.payload_bytes <= (mib - 1) * 1024**2
        plans.append(plan)
    assert plans[0].output_batch <= plans[-1].output_batch
    assert plans[-1].mode == "all_channels"


def test_planner_channel_inventory_matches_actual_curl_routes():
    from source.solvers.vpm.physics.induction.gaussian_mesh.fields import channel_routes

    for batch in range(1, 13):
        for first in range(0, 12, batch):
            columns = tuple(range(first, min(first + batch, 12)))
            actual = tuple(
                channel
                for channel in range(9)
                if any(column in columns for column, _, _ in channel_routes(channel))
            )
            assert required_channels(columns) == actual


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("families", [1, 2])
def test_streamed_fields_match_fast_and_direct_with_coherent_walls(dtype, families):
    cp = pytest.importorskip("cupy")
    from source.solvers.vpm.physics.induction.gaussian_mesh.coordinates import finite_images
    from source.solvers.vpm.physics.induction.gaussian_mesh.fields import GaussianImageFields
    from tests.vpm._direct_gaussian_reference import direct_finite_images
    from tests.vpm.test_gaussian_mesh_package import _case

    x, gamma, sigma, interior, images = _case()
    q = np.concatenate((interior, [[0.03, 0.02, 0.0], [0.03, 0.02, 0.193]]))
    images = [(0, False), *images]
    options = {
        "zmin": 0.0,
        "zmax": 0.193,
        "tau": 0.12,
        "spacing": 0.035,
        "cutoff": 0.6,
        "dtype": dtype,
        "correction_dtype": dtype,
        "source_only_primary": True,
    }
    with GaussianImageFields(x, gamma, sigma, q, **options) as fast:
        u, j, _ = fast.evaluate(images)
        baseline = cp.asnumpy(u), cp.asnumpy(j)
        shape, fft = fast.shape, fast.fft_shape
        retained = fast.compact_shape
        _, world, _ = finite_images(images, 0.0, 0.193, fast.cells, 513, include_primary=True)
        truth = direct_finite_images(x, gamma, sigma, q, world)[:2]
        del u, j
    size, count, queries = np.dtype(dtype).itemsize, len(x), len(q)
    volume = math.prod(fft)
    spectrum = fft[0] * fft[1] * (fft[2] // 2 + 1)
    meta = (2 * count + queries) * (12 + 30 * size) + (count + queries) * 24 + queries * 12 * size
    low = (14 * spectrum + 2 * volume + 12 * math.prod(retained)) * size + meta
    high = (40 * spectrum + 10 * volume + 12 * math.prod(retained)) * size + meta - 1
    plan_cap = 8 * 1024**2
    choices = []
    for cap in np.linspace(low, high, 64, dtype=np.int64):
        plan = field_execution_plan(
            shape, fft, count, queries, 10, size, int(cap) + plan_cap, plan_cap,
            retained_shape=retained,
        )
        if (
            plan.mode == "streamed"
            and plan.source_families == families
            and plan.kernel_channels > 1
        ):
            choices.append((int(cap), plan))
    assert choices
    cap, selected = choices[0]
    with GaussianImageFields(
        x, gamma, sigma, q, **options, max_scratch_bytes=cap + plan_cap, max_plan_bytes=plan_cap
    ) as owner:
        assert owner.execution_plan.mode == "streamed"
        assert owner.execution_plan.source_families == families
        assert owner.execution_plan == selected
        for _ in range(2):
            report = owner.prepare(images)
            u, j, _ = owner.evaluate_prepared(q)
            for got, expected, direct in zip(
                (cp.asnumpy(u), cp.asnumpy(j)), baseline, truth, strict=True
            ):
                assert np.linalg.norm(got - direct) / np.linalg.norm(direct) < 1e-4
                np.testing.assert_allclose(
                    got,
                    expected,
                    rtol=32 * np.finfo(dtype).eps,
                    atol=32 * np.finfo(dtype).eps * np.max(np.abs(direct)),
                )
            assert report["inverse_transforms"] == (24 if families == 1 else 12)
            assert report["radial_kernel_launches"] == selected.radial_grid_passes
            assert report["kernel_forward_transforms"] == selected.kernel_forward_transforms
            assert report["plan_peak_work_bytes"] <= plan_cap
            assert report["pool_reserved_bytes"] <= cap + plan_cap
            assert report["plan_builds"] > 2
            del u, j
        # Caller fields remain intact after a failed plan switch, and a
        # failed direction cannot silently reuse an over-budget plan.
        owner._plans.close()
        with pytest.raises(RuntimeError, match="closed"):
            owner.prepare(images)
        assert owner._prepared_images is None


def test_late_source_family_reuses_full_radial_scratch(monkeypatch):
    cp = pytest.importorskip("cupy")
    from source.solvers.vpm.physics.induction.gaussian_mesh.fields import GaussianImageFields
    from tests.vpm._direct_gaussian_reference import direct_finite_images
    from tests.vpm.test_gaussian_mesh_package import _case

    x, gamma, sigma, q, images = _case()
    with GaussianImageFields(
        x, gamma, sigma, q, zmin=0.0, zmax=0.193, tau=0.12, spacing=0.035, cutoff=0.6
    ) as owner:
        owner.prepare([(0, True)])
        scratch = owner._kernel_scratch
        original = cp.zeros

        def no_extra_source_grid(shape, *args, **kwargs):
            if isinstance(shape, tuple) and shape == (3, *owner.fft_shape):
                raise AssertionError("late family allocated an unbudgeted three-grid scatter")
            return original(shape, *args, **kwargs)

        monkeypatch.setattr(cp, "zeros", no_extra_source_grid)
        report = owner.prepare(images)
        u, j, _ = owner.evaluate_prepared(q)
        truth = direct_finite_images(x, gamma, sigma, q, report["world_images"])[:2]
        assert owner._kernel_scratch is scratch
        for got, exact in zip((cp.asnumpy(u), cp.asnumpy(j)), truth, strict=True):
            assert np.linalg.norm(got - exact) / np.linalg.norm(exact) < 1e-4
        del u, j
