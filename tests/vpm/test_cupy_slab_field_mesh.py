"""Optional isolated CUDA qualification of the UNWIRED smooth field operator."""

import numpy as np
import pytest

from tests.vpm._cupy_slab_field_mesh import SlabFieldMeshGPU
from tests.vpm._finite_image_mesh_reference import direct_finite_images
from tests.vpm._finite_slab_field_mesh_reference import finite_slab_field_mesh, slab_world_images

cp = pytest.importorskip("cupy")


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
def test_complete_finite_block_matches_double_mesh_and_direct_truth(dtype, cancelled):
    x, gamma, targets = _case(cancelled)
    sigma = np.full(len(x), .04)
    images = [(0, True), (-1, False), (1, False), (1, True)]
    world, _ = slab_world_images(images, 0., .193, 6)
    exact_u, exact_j, _ = direct_finite_images(x, gamma, sigma, targets, world)
    reference = finite_slab_field_mesh(x, gamma, sigma, targets, images, zmin=0., zmax=.193,
                                       tau=.12, spacing=.035, order=10)
    with SlabFieldMeshGPU(x, gamma, targets, zmin=0., zmax=.193, dtype=dtype) as engine:
        smooth_u, smooth_j, diagnostics = engine.evaluate(images)
        u = cp.asnumpy(smooth_u)+reference.correction_velocity
        j = cp.asnumpy(smooth_j)+reference.correction_gradient
        assert diagnostics["inverse_transforms"] == 12
        assert diagnostics["pool_reserved_bytes"] <= 2*1024**3
        assert diagnostics["plan_bytes"] <= 128*1024**2
        assert not engine.host_x.flags.writeable
        for candidate, truth, double in ((u, exact_u, reference.velocity), (j, exact_j, reference.gradient)):
            assert np.linalg.norm(candidate-truth)/np.linalg.norm(truth) < 1e-4
            budget = 1e-5 if dtype == "float32" else 1e-9
            assert np.linalg.norm(candidate-double)/np.linalg.norm(truth) < budget
        del smooth_u, smooth_j


@pytest.mark.parametrize("upper", [False, True])
def test_reflected_finite_self_j_and_zero_velocity_without_override(upper):
    zmin, zmax = 1024.3, 1024.493
    x = np.array([[.013, -.017, zmax if upper else zmin], [.036, .027, zmin+.071]])
    gamma, sigma = np.array([[.2, .7, -.4], [0., 0., 0.]]), np.full(2, .04)
    images = [(int(upper), True)]
    reference = finite_slab_field_mesh(x, gamma, sigma, x[:1], images, zmin=zmin, zmax=zmax,
                                       tau=.12, spacing=.035, order=10)
    world, _ = slab_world_images(images, zmin, zmax, 6)
    truth_u, truth_j, _ = direct_finite_images(x, gamma, sigma, x[:1], world)
    with SlabFieldMeshGPU(x, gamma, x[:1], zmin=zmin, zmax=zmax) as engine:
        su, sj, _ = engine.evaluate(images)
        u, j = cp.asnumpy(su)+reference.correction_velocity, cp.asnumpy(sj)+reference.correction_gradient
        assert np.linalg.norm(u-truth_u) < 1e-5
        assert np.linalg.norm(j-truth_j)/np.linalg.norm(truth_j) < 1e-4
        del su, sj


def test_memory_primary_closed_and_stream_admission_are_fail_closed():
    x, gamma, q = _case()
    before = x.copy()
    with pytest.raises(MemoryError, match="payload"):
        SlabFieldMeshGPU(x, gamma, q, zmin=0., zmax=.193, max_scratch_bytes=1024)
    engine = SlabFieldMeshGPU(x, gamma, q, zmin=0., zmax=.193)
    with pytest.raises(ValueError, match="primary"):
        engine.evaluate([(0, False)])
    with cp.cuda.Stream(), pytest.raises(RuntimeError, match="original.*stream"):
        engine.evaluate([(0, True)])
    engine.close()
    engine.close()
    with pytest.raises(RuntimeError, match="closed"):
        engine.evaluate([(0, True)])
    np.testing.assert_array_equal(x, before)
