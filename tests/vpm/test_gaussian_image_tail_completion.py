"""Closed-form completion of the omitted slip-slab image rows."""

import numpy as np
import pytest

from openonda.vpm.physics.induction.gaussian_mesh.coordinates import image_tail_edges
from openonda.vpm.physics.induction.gaussian_mesh.host_fields import (
    GaussianHostImageFields,
)
from openonda.vpm.physics.induction.gaussian_mesh.session import (
    GaussianSlabFieldSession,
    GaussianSlabSettings,
)
from tests.vpm._slip_periodic_gaussian_reference import gaussian_pairs

pytestmark = pytest.mark.qualification

LAYER = 0.04  # m, one particle layer between the slip planes
Z_MIN, Z_MAX = -0.5 * LAYER, 0.5 * LAYER


def _cloud():
    rng = np.random.default_rng(3)
    position = np.column_stack(
        (rng.uniform(-0.3, 0.3, 6), rng.uniform(-0.3, 0.3, 6), rng.uniform(Z_MIN, Z_MAX, 6))
    )
    strength = rng.normal(size=(6, 3)) * 20 * LAYER**3
    targets = np.array([[0.05, 0.02, 0.0], [0.4, -0.1, 0.004], [-3.0, 2.0, 0.0]])
    return position, strength, np.full(6, LAYER), targets


def _direct_image_sum(position, strength, sigma, targets, shells, chunk=4000):
    """Gaussian images |k| <= shells of both families, summed pair by pair."""
    u, j = np.zeros((len(targets), 3)), np.zeros((len(targets), 3, 3))
    period = 2 * (Z_MAX - Z_MIN)
    for odd in (False, True):
        source, gamma = position.copy(), strength.copy()
        if odd:
            source[:, 2] = 2 * Z_MIN - source[:, 2]
            gamma[:, :2] *= -1
        for first in range(-shells, shells + 1, chunk):
            k = np.arange(first, min(shells + 1, first + chunk), dtype=float)
            d = targets[:, None, None, :] - source[None, None, :, :]
            d = np.broadcast_to(d, (len(targets), len(k), len(source), 3)).copy()
            d[..., 2] -= k[None, :, None] * period
            du, dj = gaussian_pairs(d, gamma[None, None], sigma[None, None])
            u += du.sum(axis=(1, 2))
            j += dj.sum(axis=(1, 2))
    return u, j


def _host_field(position, strength, sigma, targets, shells):
    tau = 3 * LAYER
    images = tuple((k, odd) for k in range(-shells, shells + 1) for odd in (False, True))
    with GaussianHostImageFields(
        position,
        strength,
        sigma,
        targets,
        zmin=Z_MIN,
        zmax=Z_MAX,
        tau=tau,
        spacing=tau / 4,
        cutoff=5 * tau,
        dtype="float64",
        correction_dtype="float64",
        source_only_primary=True,
        complete_image_tail=True,
    ) as field:
        field.prepare(images)
        velocity, gradient, _ = field.evaluate_prepared(targets)
    return velocity, gradient


@pytest.mark.slow
def test_completed_field_matches_a_long_direct_image_sum():
    position, strength, sigma, targets = _cloud()
    # 2*50000 rows per family: the sum's own truncation is far below the tolerance
    u_direct, j_direct = _direct_image_sum(position, strength, sigma, targets, 50_000)
    u, j = _host_field(position, strength, sigma, targets, shells=7)
    assert np.abs(u - u_direct).max() <= 1e-5 * np.abs(u_direct).max()
    assert np.abs(j - j_direct).max() <= 2e-5 * np.abs(j_direct).max()


def _planar_circulation_fraction(kernel, rho):
    """Circulation within radius rho of a spanwise-uniform line of 3-D cores.

    Integrating each kernel's density along the span gives its planar core:
    exp(-rho^2)/pi for the Gaussian, (2 - rho^2) exp(-rho^2)/pi for the
    corrected Gaussian (the super-Gaussian is it at rho/sqrt(2)), and
    2/(pi (1 + rho^2)^3) for the Winckelmans kernel.
    """
    if kernel == "GAUSSIAN":
        return 1 - np.exp(-(rho**2))
    if kernel in {"HIGH_ORDER_GAUSSIAN", "SUPER_GAUSSIAN"}:
        x2 = rho**2 / (2 if kernel == "SUPER_GAUSSIAN" else 1)
        return 1 - (1 - x2) * np.exp(-x2)
    return 1 - (1 + rho**2) ** -2


@pytest.mark.parametrize(
    "backend,kernel,core_over_layer,tolerance",
    [
        ("cpu", "GAUSSIAN", 1.0, 2e-5),
        ("cupy_cuda", "GAUSSIAN", 1.0, 2e-5),
        # The corrected Gaussian's span spectrum carries (1 + k^2 sigma^2/4).
        ("cpu", "HIGH_ORDER_GAUSSIAN", 1.5, 2e-5),
        ("cpu", "SUPER_GAUSSIAN", 1.0, 2e-5),
        # An algebraic core ripples along the span like exp(-2 pi sigma/h).
        ("cpu", "WINCKELMANS", 3.0, 1e-5),
        ("cupy_cuda", "WINCKELMANS", 3.0, 1e-5),
    ],
)
def test_one_layer_slab_query_field_is_the_two_dimensional_vortex(
    backend, kernel, core_over_layer, tolerance
):
    if backend == "cupy_cuda":
        cupy = pytest.importorskip("cupy")
        if not cupy.cuda.is_available():
            pytest.skip("CUDA device unavailable")
    pytest.importorskip("openonda.vpm.numerics._fenv")
    radius = np.array([0.02, 0.05, 0.1, 0.3, 1.0, 5.0, 30.0])
    targets = np.column_stack((radius, np.zeros((len(radius), 2))))
    with GaussianSlabFieldSession(
        z_min=Z_MIN,
        z_max=Z_MAX,
        tail_tolerance=1e-4,
        max_shells=129,
        dtype="float64",
        settings=GaussianSlabSettings(backend=backend),
        execution_backend=backend,
        kernel=kernel,
    ) as session:
        velocity, _, report = session.evaluate(
            np.zeros((1, 3)),
            np.array([[0.0, 0.0, LAYER]]),
            np.array([core_over_layer * LAYER]),
            targets,
            source_only=True,
        )
    # Unit circulation per layer: u = fraction(r/sigma)/(2 pi r)
    sigma = core_over_layer * LAYER
    exact = _planar_circulation_fraction(kernel, radius / sigma) / (2 * np.pi * radius)
    assert report["shell"] <= 31
    np.testing.assert_allclose(velocity[:, 1], exact, rtol=tolerance)
    assert np.abs(velocity[:, [0, 2]]).max() <= 1e-12


def test_completion_requires_consecutive_retained_shells():
    cells = 3
    shells = [(2 * k * cells, False) for k in range(-2, 3) if k] + [
        ((2 * k - 1) * cells, True) for k in range(-2, 3)
    ]
    edges = image_tail_edges(shells, cells, include_primary=False)
    assert edges == {False: (-15, 15), True: (-18, 12)}
    with pytest.raises(ValueError, match="consecutive"):
        image_tail_edges([row for row in shells if row != (6, False)], cells, include_primary=False)
    with pytest.raises(ValueError, match="consecutive"):
        image_tail_edges(shells, cells, include_primary=True)
    with pytest.raises(ValueError, match="both families"):
        image_tail_edges([row for row in shells if row[1]], cells, include_primary=False)
