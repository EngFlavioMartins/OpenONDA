"""Additional GPU roundoff/derivative gates; run only in reserved CUDA slot."""

import numpy as np
import pytest

from tests.vpm._cupy_slab_field_mesh import SlabFieldMeshGPU
from tests.vpm._finite_image_mesh_reference import direct_finite_images
from tests.vpm._finite_slab_field_mesh_reference import finite_slab_field_mesh, slab_world_images

cp = pytest.importorskip("cupy")


def _run(x, gamma, query, images, spacing=.035, dtype="float32"):
    with SlabFieldMeshGPU(x, gamma, query, zmin=0., zmax=.96, spacing=spacing, dtype=dtype) as engine:
        u, j, diagnostics = engine.evaluate(images)
        result = cp.asnumpy(u), cp.asnumpy(j), diagnostics
        del u, j
        return result


def test_repeated_atomic_scatter_preserves_direct_accuracy_and_conditioned_roundoff():
    rng = np.random.default_rng(62002)
    x = rng.uniform([-.1, -.08, .02], [.1, .08, .13], (64, 3))
    gamma = rng.normal(0., .02, x.shape)
    gamma[-1] = -gamma[:-1].sum(axis=0)
    sigma = np.full(len(x), .04)
    query = np.array([[.012, -.013, .068], [-.021, .027, .059], [.06, .04, .023]])
    images = [(0, True), (-1, False), (1, False)]
    world, _ = slab_world_images(images, 0., .96, 1)
    true_u, true_j, conditioning = direct_finite_images(x, gamma, sigma, query, world)
    reference = finite_slab_field_mesh(x, gamma, sigma, query, images, zmin=0., zmax=.96,
                                       tau=.12, spacing=.035, order=10)
    runs = []
    for _ in range(3):
        su, sj, _ = _run(x, gamma, query, images)
        runs.append((su+reference.correction_velocity, sj+reference.correction_gradient))
    for field, truth in enumerate((true_u, true_j)):
        for fields in runs:
            assert np.linalg.norm(fields[field]-truth)/np.linalg.norm(truth) < 1e-4
        for fields in runs[1:]:
            point_error = np.linalg.norm((fields[field]-runs[0][field]).reshape(len(query), -1), axis=1)
            assert np.all(point_error <= 16*np.finfo(np.float32).eps*conditioning[:, field]+1e-12)


@pytest.mark.parametrize("spacing", [.035, .03])
@pytest.mark.parametrize("switch", [False, True])
@pytest.mark.parametrize("dtype,epsilon", [("float64", 1e-6), ("float64", 2e-6),
                                           ("float32", 1e-4), ("float32", 2e-4)])
def test_actual_parameter_gpu_fd_and_stencil_switch_discrepancy(spacing, switch, dtype, epsilon):
    x = np.array([[-.047, .018, .073], [.056, -.036, .041], [.011, .029, .096]])
    gamma = np.array([[.3, -.2, .5], [-.7, .4, .2], [.4, -.2, .3]])
    sigma = np.array([.04, .07, .09])
    query = np.array([[.012, -.013, .068], [-.021, .027, .059]])
    if switch:
        query[0, 0] = 0.  # Auxiliary-grid node: central difference crosses stencil selection.
    images = [(0, True), (1, False)]
    world, _ = slab_world_images(images, 0., .96, 1)
    _, truth_j, _ = direct_finite_images(x, gamma, sigma, query, world)
    reference = finite_slab_field_mesh(x, gamma, sigma, query, images, zmin=0., zmax=.96,
                                       tau=.12, spacing=spacing, order=10)
    _, sj, geometry = _run(x, gamma, query, images, spacing, dtype)
    gathered_j = sj+reference.correction_gradient
    # Predeclared paired step sizes expose FD roundoff/truncation rather than
    # assuming float64's1e-6 is a meaningful derivative step for f32 output.
    fd = np.empty_like(truth_j)
    for axis in range(3):
        step = np.zeros(3)
        step[axis] = epsilon
        fields = []
        for shifted in (query+step, query-step):
            smooth, _, current = _run(x, gamma, shifted, images, spacing, dtype)
            correction = finite_slab_field_mesh(x, gamma, sigma, shifted, images, zmin=0., zmax=.96,
                                                tau=.12, spacing=spacing, order=10).correction_velocity
            assert current["shape"] == geometry["shape"]
            assert current["fft_shape"] == geometry["fft_shape"]
            fields.append(smooth+correction)
        fd[:, :, axis] = (fields[0]-fields[1])/(2*epsilon)
    true_error = np.linalg.norm(fd-truth_j)/np.linalg.norm(truth_j)
    inconsistency = np.linalg.norm(fd-gathered_j)/np.linalg.norm(truth_j)
    assert true_error < 1e-4, (spacing, switch, dtype, epsilon, true_error, inconsistency)
    assert inconsistency < 1e-4, (spacing, switch, dtype, epsilon, true_error, inconsistency)
