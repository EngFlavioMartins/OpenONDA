"""Real CUDA confirmation of the separate cutoff-classification conditions."""

from fractions import Fraction

import numpy as np
import pytest

from source.solvers.vpm.physics.induction.gaussian_mesh.correction import GaussianCoreCorrectionGPU
from source.solvers.vpm.physics.induction.gaussian_mesh.correction_distance import (
    correction_classification_bound,
)
from source.solvers.vpm.physics.induction.gaussian_tail.error_bounds import prepare_tail_source
from tests.vpm.test_gaussian_correction_distance import _physical_squared

cp = pytest.importorskip("cupy")


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("offset", [0.0, 2.0**28, -(2.0**28)])
def test_real_gpu_retains_each_proven_interior_pair_at_cell_boundaries(dtype, offset):
    cutoff = 0.6
    source_x = np.array([[offset + 0.6, offset + 0.1, offset + 0.13]])
    gamma = np.array([[0.2, -0.3, 0.4]])
    sigma = np.array([0.04])
    zmin, zmax = offset - 0.48, offset + 0.48
    source = prepare_tail_source(source_x, gamma, sigma, z_min=zmin, z_max=zmax)
    descriptors = [(0, False), (0, True), (1, True), (-2, False)]
    proof = correction_classification_bound(
        source, source_x[0] - 5.0, source_x[0] + 5.0, cutoff=cutoff, images=descriptors
    )
    preserved = source_x.copy(), gamma.copy(), sigma.copy()
    with GaussianCoreCorrectionGPU(
        source_x, gamma, sigma, tau=0.12, cutoff=cutoff, accumulation_dtype=dtype, max_images=4
    ) as field:
        for (k, odd), image in zip(descriptors, proof.world_images, strict=True):
            shift = image[0]
            center = source_x[0].copy()
            center[2] = shift - center[2] if odd else shift + center[2]
            queries = []
            for axis in range(3):
                for sign in (-1, 1):
                    for fraction in (0.25, 0.75, 1.0):
                        q = center.copy()
                        q[axis] += sign * fraction * (2 * proof.omitted_distance_lower - cutoff)
                        if (
                            _physical_squared(source_x[0], q, k, odd, zmin, zmax)
                            < Fraction(proof.omitted_distance_lower) ** 2
                        ):
                            queries.append(q)
            q = np.asarray(queries)
            assert len(q) >= 12
            before = q.copy()
            u, j, report = field.evaluate(q, [image])
            # With exactly one source/image, every submitted pair is proven
            # interior; another accepted pair cannot mask an omitted one.
            assert report["accepted_pairs"] == len(q)
            assert np.isfinite(cp.asnumpy(u)).all() and np.isfinite(cp.asnumpy(j)).all()
            np.testing.assert_array_equal(q, before)
        for current, saved in zip((source_x, gamma, sigma), preserved, strict=True):
            np.testing.assert_array_equal(current, saved)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_real_gpu_cell_index_walk_counts_all_nonambiguous_pairs(dtype):
    cutoff, offset = 0.6, 2.0**27
    xs = []
    for cell in range(4):
        boundary = np.float64(offset + cell * cutoff)
        xs.extend((np.nextafter(boundary, -np.inf), boundary, np.nextafter(boundary, np.inf)))
    x = np.array([[value, offset + 0.1, offset + 0.13] for value in xs])
    gamma = np.tile([0.2, -0.3, 0.4], (len(x), 1))
    sigma = np.full(len(x), 0.04)
    zmin, zmax = offset - 0.48, offset + 0.48
    source = prepare_tail_source(x, gamma, sigma, z_min=zmin, z_max=zmax)
    descriptors = [(0, False), (0, True), (1, True)]
    proof = correction_classification_bound(
        source, x.min(0) - 5, x.max(0) + 5, cutoff=cutoff, images=descriptors
    )
    loss = cutoff - proof.omitted_distance_lower
    with GaussianCoreCorrectionGPU(
        x, gamma, sigma, tau=0.12, cutoff=cutoff, accumulation_dtype=dtype, max_images=3
    ) as field:
        for (k, odd), image in zip(descriptors, proof.world_images, strict=True):
            for fraction in (-1.5, -0.5, 0.25, 0.75, 1.25, 2.25, 3.5, 6.0):
                q = np.array([[offset + fraction * cutoff, offset + 0.1, offset + 0.13]])
                q[0, 2] = image[0] - q[0, 2] if odd else image[0] + q[0, 2]
                exact_squared = [_physical_squared(s, q[0], k, odd, zmin, zmax) for s in x]
                inside = [r2 < Fraction(proof.omitted_distance_lower) ** 2 for r2 in exact_squared]
                outside = [r2 > Fraction(cutoff + loss) ** 2 for r2 in exact_squared]
                assert all(i or o for i, o in zip(inside, outside, strict=True)), (
                    "avoid ambiguous cutoff annulus"
                )
                _, _, report = field.evaluate(q, [image])
                assert report["accepted_pairs"] == sum(inside)
