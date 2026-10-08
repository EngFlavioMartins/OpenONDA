"""Slip-slab images of every particle kernel against long direct image sums."""

import numpy as np
import pytest

from openonda.vpm.kernels.registry import kernel_names
from openonda.vpm.physics.induction.gaussian_mesh.session import (
    GaussianSlabFieldSession,
    GaussianSlabSettings,
)
from tests.vpm._cuda import require_gaussian_cuda
from tests.vpm._gaussian_tail_reference import completed_rows
from tests.vpm._slip_periodic_gaussian_reference import kernel_pairs

pytest.importorskip("openonda.vpm.numerics._fenv")

KERNELS = sorted(kernel_names())


def _direct_images(kernel, x, g, sigma, targets, z_min, z_max, shells, include_primary):
    """Kernel pairs over |k| <= shells of both families, then the singular rows."""
    u, j = np.zeros((len(targets), 3)), np.zeros((len(targets), 3, 3))
    period = 2 * (z_max - z_min)
    for odd in (False, True):
        source, strength = x.copy(), g.copy()
        if odd:
            source[:, 2] = 2 * z_min - source[:, 2]
            strength[:, :2] *= -1
        for k in range(-shells, shells + 1):
            if k == 0 and not odd and not include_primary:
                continue
            shifted = source.copy()
            shifted[:, 2] += k * period
            du, dj = kernel_pairs(
                targets[:, None] - shifted[None], strength[None], sigma[None], kernel
            )
            u += du.sum(axis=1)
            j += dj.sum(axis=1)
    # Hundreds of periods away every kernel is Biot-Savart to round-off.
    far_u, far_j = completed_rows(x, g, targets, z_min, z_max, shells)
    return u + far_u, j + far_j


def _session(kernel, z_min, z_max, tolerance=1e-7):
    return GaussianSlabFieldSession(
        z_min=z_min,
        z_max=z_max,
        tail_tolerance=tolerance,
        max_shells=513,
        dtype="float64",
        settings=GaussianSlabSettings(backend="cpu"),
        execution_backend="cpu",
        kernel=kernel,
    )


def _cloud(z_min, z_max, count=8):
    rng = np.random.default_rng(5)
    x = np.column_stack(
        (
            rng.uniform(-0.3, 0.3, count),
            rng.uniform(-0.3, 0.3, count),
            rng.uniform(z_min, z_max, count),
        )
    )
    return x, rng.normal(size=(count, 3)) * 1e-3, rng.uniform(0.05, 0.09, count)


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("planes", [(-0.3, 0.3), (-0.02, 0.02)], ids=["thick", "one_layer"])
@pytest.mark.parametrize("source_only", [False, True], ids=["stage", "query"])
def test_slab_images_match_long_direct_sums(kernel, planes, source_only):
    x, g, sigma = _cloud(*planes)
    targets = x.copy() if not source_only else np.vstack((x[:4] + 0.01, [[0.6, 0.2, 0.0]]))
    with _session(kernel, *planes) as session:
        u, j, _ = session.evaluate(x, g, sigma, targets, source_only=source_only)
    expected_u, expected_j = _direct_images(
        kernel, x, g, sigma, targets, *planes, shells=600, include_primary=source_only
    )
    assert np.abs(u - expected_u).max() <= 1e-5 * np.abs(expected_u).max()
    assert np.abs(j - expected_j).max() <= 1e-5 * np.abs(expected_j).max()


def test_only_an_algebraic_core_widens_the_local_correction():
    planes = (-0.02, 0.02)
    x, g, sigma = _cloud(*planes)
    cutoffs = {}
    for kernel in KERNELS:
        with _session(kernel, *planes) as session:
            _, _, report = session.evaluate(x, g, sigma, x, source_only=False)
        cutoffs[kernel] = report["correction_cutoff"] / sigma.max()
        assert report["kernel"] == kernel
    # 5 broadened cores of 3 sigma: Gaussian-type defects are negligible there.
    for kernel in ("GAUSSIAN", "HIGH_ORDER_GAUSSIAN", "SUPER_GAUSSIAN"):
        assert cutoffs[kernel] == pytest.approx(15.0)
    assert cutoffs["WINCKELMANS"] > 15.0


@pytest.mark.gpu
@pytest.mark.parametrize("kernel", KERNELS)
def test_cuda_correction_matches_the_host_for_every_kernel(kernel):
    require_gaussian_cuda()
    planes = (-0.02, 0.02)
    x, g, sigma = _cloud(*planes)
    fields = []
    for backend in ("cpu", "cupy_cuda"):
        with GaussianSlabFieldSession(
            z_min=planes[0],
            z_max=planes[1],
            tail_tolerance=1e-7,
            max_shells=513,
            dtype="float64",
            settings=GaussianSlabSettings(backend=backend),
            execution_backend=backend,
            kernel=kernel,
        ) as session:
            fields.append(session.evaluate(x, g, sigma, x, source_only=True)[:2])
    (u_host, j_host), (u_cuda, j_cuda) = fields
    np.testing.assert_allclose(u_cuda, u_host, rtol=0, atol=1e-9 * np.abs(u_host).max())
    np.testing.assert_allclose(j_cuda, j_host, rtol=0, atol=1e-9 * np.abs(j_host).max())
