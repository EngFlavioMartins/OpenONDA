"""Bounded shell selection retains general 3D vector induction."""

import numpy as np
import pytest

from source.solvers.vpm.physics.induction.gaussian_mesh.session import (
    GaussianSlabFieldSession,
    GaussianSlabSettings,
)
from tests.vpm._slip_periodic_gaussian_reference import slip_periodic_gaussian


@pytest.mark.parametrize("backend", [
    "cpu", "auto", pytest.param("cupy_cuda", marks=pytest.mark.gpu),
])
def test_adaptive_full_vector_images_match_independent_long_sum(backend):
    if backend == "cupy_cuda":
        pytest.importorskip("cupy")
    position = np.array([[0.1, -0.2, 0.12], [-0.12, 0.24, 0.38]])
    strength = np.array([[0.2, -0.3, 0.7], [-0.2, 0.3, -0.699]])
    core = np.array([0.08, 0.17])
    targets = np.array([[0.3, 0.1, 0], [-0.2, 0.25, 0.5], [0.2, 0.3, 0.24]])
    reference = slip_periodic_gaussian(
        position, strength, core, targets, z_min=0, z_max=0.5, shells=2048,
    )
    with GaussianSlabFieldSession(
        z_min=0, z_max=0.5, tail_tolerance=1e-4, max_shells=129,
        dtype="float64", settings=GaussianSlabSettings(backend=backend),
    ) as model:
        velocity, gradient, record = model.evaluate(
            position, strength, core, targets, source_only=True,
        )
    assert record["shell"] < 128
    np.testing.assert_allclose(velocity, reference.velocity, rtol=0, atol=1.1e-4)
    np.testing.assert_allclose(gradient, reference.gradient, rtol=0, atol=1.1e-4)
    assert abs(velocity[2, 2]) > 1e-3
    assert np.max(np.abs(gradient[:, :, 2])) > 1e-2
