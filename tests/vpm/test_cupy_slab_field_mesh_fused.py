"""Isolated qualification: fused field kernels, not production admission."""

import math

import numpy as np
import pytest

from tests.vpm._cupy_slab_field_mesh_fused import (
    SlabFieldMeshFusedGPU,
    channel_routes,
    fused_payload_bytes,
)


def test_symmetric_channel_routes_match_independent_curl_and_full_jacobian():
    gradient = np.array([.17, -.23, .91])
    hessian = np.array([[.71, -.31, .43], [-.31, -.83, .59], [.43, .59, .11]])
    gamma = np.array([.37, -.61, .29])
    channels = np.r_[gradient, hessian[0], hessian[1, 1:], hessian[2, 2]]
    output = np.zeros(12)
    for channel, value in enumerate(channels):
        for column, source, sign in channel_routes(channel):
            output[column] += sign*value*gamma[source]
    expected_u = np.cross(gradient, gamma)
    expected_j = np.column_stack([np.cross(hessian[:, d], gamma) for d in range(3)])
    np.testing.assert_allclose(output[:3], expected_u, rtol=1e-15, atol=1e-15)
    np.testing.assert_allclose(output[3:].reshape(3, 3), expected_j, rtol=1e-15, atol=1e-15)
    assert sum(len(channel_routes(c)) for c in range(9)) == 24


def test_native_float32_payload_admission_includes_fused_channels_and_fft_scratch():
    volume = math.prod((720, 189, 98))
    spectra = math.prod((720, 189, 50))
    amount = fused_payload_bytes(1000320480, spectra, volume, 4)
    assert amount == 1755836640
    assert amount < 2*1024**3-128*1024**2
    assert 2*amount > 2*1024**3-128*1024**2


def _case(cancelled=False):
    x = np.array([[-.082, .012, .023], [-.042, .012, .023], [-.002, .012, .023],
                  [.038, .012, .023], [.078, .012, .023]])
    gamma = np.tile([.3, -.2, .7], (5, 1))
    if cancelled:
        gamma *= np.array([1, -4, 6, -4, 1])[:, None]
    targets = np.array([[.011, .019, .018], [-.073, .058, .061], [.112, -.036, .04]])
    return x, gamma, targets


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("cancelled", [False, True])
def test_fused_finite_blocks_match_baseline_mesh_and_independent_direct_truth(dtype, cancelled):
    cp = pytest.importorskip("cupy")
    from tests.vpm._cupy_slab_field_mesh import SlabFieldMeshGPU
    from tests.vpm._finite_image_mesh_reference import direct_finite_images
    from tests.vpm._finite_slab_field_mesh_reference import (
        finite_slab_field_mesh,
        slab_world_images,
    )

    x, gamma, targets = _case(cancelled)
    sigma = np.full(len(x), .04)
    images = [(0, True), (-1, False), (1, False), (1, True)]
    world, _ = slab_world_images(images, 0., .193, 6)
    exact_u, exact_j, _ = direct_finite_images(x, gamma, sigma, targets, world)
    reference = finite_slab_field_mesh(x, gamma, sigma, targets, images, zmin=0., zmax=.193,
                                       tau=.12, spacing=.035, order=10)
    snapshots = []
    for engine_type in (SlabFieldMeshGPU, SlabFieldMeshFusedGPU):
        with engine_type(x, gamma, targets, zmin=0., zmax=.193, dtype=dtype) as engine:
            su, sj, diagnostics = engine.evaluate(images)
            snapshots.append((cp.asnumpy(su), cp.asnumpy(sj)))
            if engine_type is SlabFieldMeshFusedGPU:
                assert diagnostics["radial_kernel_launches"] == 2
                assert diagnostics["kernel_forward_transforms"] == 18
                assert diagnostics["inverse_transforms"] == 12
                assert diagnostics["pool_reserved_bytes"] <= 2*1024**3
                assert diagnostics["plan_bytes"] <= 128*1024**2
            del su, sj
    u = snapshots[1][0]+reference.correction_velocity
    j = snapshots[1][1]+reference.correction_gradient
    budget = 1e-5 if dtype == "float32" else 1e-9
    for new, truth, double, old in ((u, exact_u, reference.velocity, snapshots[0][0]+reference.correction_velocity),
                                  (j, exact_j, reference.gradient, snapshots[0][1]+reference.correction_gradient)):
        assert np.linalg.norm(new-truth)/np.linalg.norm(truth) < 1e-4
        assert np.linalg.norm(new-double)/np.linalg.norm(truth) < budget
        assert np.linalg.norm(new-old)/np.linalg.norm(truth) < budget


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_fused_raw_nine_channels_preserve_each_finite_image_kernel(dtype):
    cp = pytest.importorskip("cupy")
    from tests.vpm._cupy_slab_field_mesh_fused import _CHANNEL_COMPONENTS

    x, gamma, targets = _case()
    with SlabFieldMeshFusedGPU(x, gamma, targets, zmin=0., zmax=.193, dtype=dtype) as engine:  # noqa: SIM117
        # Includes coincidence, rho<1 series, regular formula, signed distant
        # shifts and the zero-padding gap. No cutoff or exp-tail shortcut.
        with cp.cuda.using_allocator(engine.pool.malloc):
            shifts = cp.asarray([0, -1, 7, -1200, 1200], dtype=cp.int64)
            fused = cp.empty((9, *engine.fft_shape), dtype=engine.dtype)
            arguments = (np.int64(engine.volume), *(np.int32(n) for n in engine.shape),
                         *(np.int32(n) for n in engine.fft_shape), *(engine.real_type(h) for h in engine.steps),
                         shifts, np.int32(len(shifts)), engine.real_type(engine.tau))
            engine._launch("gaussian_kernel_fused", engine.volume, (*arguments, fused))
            allowance = 12*np.finfo(dtype).eps
            for channel, components in enumerate(_CHANNEL_COMPONENTS):
                for axis, derivative in components:
                    original = cp.empty(engine.fft_shape, dtype=engine.dtype)
                    engine._launch("gaussian_kernel", engine.volume,
                                   (*arguments, np.int32(axis), np.int32(derivative), original))
                    old, new = cp.asnumpy(original), cp.asnumpy(fused[channel])
                    # Cancellation at a grid node is protected by a uniform
                    # absolute roundoff scale, not by division by that node.
                    np.testing.assert_allclose(new, old, rtol=allowance,
                                               atol=allowance*np.max(np.abs(old)))
                    del original
            del fused, shifts


@pytest.mark.parametrize("upper", [False, True])
def test_fused_reflected_plane_zero_velocity_and_finite_jacobian(upper):
    cp = pytest.importorskip("cupy")
    from tests.vpm._finite_image_mesh_reference import direct_finite_images
    from tests.vpm._finite_slab_field_mesh_reference import (
        finite_slab_field_mesh,
        slab_world_images,
    )

    zmin, zmax = 1024.3, 1024.493
    x = np.array([[.013, -.017, zmax if upper else zmin], [.036, .027, zmin+.071]])
    gamma, sigma = np.array([[.2, .7, -.4], [0., 0., 0.]]), np.full(2, .04)
    images = [(int(upper), True)]
    reference = finite_slab_field_mesh(x, gamma, sigma, x[:1], images, zmin=zmin, zmax=zmax,
                                       tau=.12, spacing=.035, order=10)
    world, _ = slab_world_images(images, zmin, zmax, 6)
    truth_u, truth_j, _ = direct_finite_images(x, gamma, sigma, x[:1], world)
    with SlabFieldMeshFusedGPU(x, gamma, x[:1], zmin=zmin, zmax=zmax) as engine:
        su, sj, _ = engine.evaluate(images)
        u, j = cp.asnumpy(su)+reference.correction_velocity, cp.asnumpy(sj)+reference.correction_gradient
        assert np.linalg.norm(u-truth_u) < 1e-5
        assert np.linalg.norm(j-truth_j)/np.linalg.norm(truth_j) < 1e-4
        del su, sj


def test_fused_preflight_and_owner_contract_fail_closed():
    cp = pytest.importorskip("cupy")
    x, gamma, q = _case()
    before = x.copy()
    with SlabFieldMeshFusedGPU(x, gamma, q, zmin=0., zmax=.193) as engine:
        extra = (16*engine.spectrum_count+6*engine.volume)*engine.dtype.itemsize
        base_payload = engine.estimated_payload_bytes-extra
        reduced_cap = base_payload+extra//2+128*1024**2
        with pytest.raises(ValueError, match="primary"):
            engine.evaluate([(0, False)])
        with cp.cuda.Stream(), pytest.raises(RuntimeError, match="original.*stream"):
            engine.evaluate([(0, True)])
        engine.close()
        engine.close()
        with pytest.raises(RuntimeError, match="closed"):
            engine.evaluate([(0, True)])
    np.testing.assert_array_equal(x, before)
    with pytest.raises(MemoryError, match="fused.*payload"):
        SlabFieldMeshFusedGPU(x, gamma, q, zmin=0., zmax=.193, max_scratch_bytes=reduced_cap)
    assert cp.fft.config.get_plan_cache().get_curr_size() == 0


def test_repeated_coalesced_blocks_reuse_scratch_without_overwriting_published_fields():
    cp = pytest.importorskip("cupy")
    x, gamma, targets = _case(cancelled=True)
    images = [(0, True)]+[(k, odd) for shell in range(1, 129)
                          for k in (-shell, shell) for odd in (False, True)]
    with SlabFieldMeshFusedGPU(x, gamma, targets, zmin=0., zmax=.193,
                              stencil_backend="gpu") as engine:
        saved_u, saved_j, _ = engine.evaluate(images)
        saved = cp.asnumpy(saved_u), cp.asnumpy(saved_j)
        pointers = (engine._kernel_scratch.data.ptr,
                    tuple(array.data.ptr for array in engine._result_spectra))
        for descriptors in ([(0, True)], images, [(1, True)], images):
            u, j, diagnostics = engine.evaluate(descriptors)
            assert (engine._kernel_scratch.data.ptr,
                    tuple(array.data.ptr for array in engine._result_spectra)) == pointers
            assert diagnostics["pool_reserved_bytes"] <= engine.max_scratch_bytes
            np.testing.assert_array_equal(cp.asnumpy(saved_u), saved[0])
            np.testing.assert_array_equal(cp.asnumpy(saved_j), saved[1])
            if descriptors is images:
                np.testing.assert_array_equal(cp.asnumpy(u), saved[0])
                np.testing.assert_array_equal(cp.asnumpy(j), saved[1])
            del u, j
        del saved_u, saved_j
    assert engine._kernel_scratch is None and not engine._result_spectra
    assert cp.fft.config.get_plan_cache().get_curr_size() == 0
