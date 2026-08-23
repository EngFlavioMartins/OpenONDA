"""Direct algebraic certification of the WALE SGS operator (PLAN.md §13).

These tests exercise ``source.solvers.fvm.turbulence.les_models.WALE`` and its
helper ``_wale_operator`` against canonical velocity-gradient tensors with
known analytic behaviour, independent of the full FVM solver (no mesh
assembly, no PIMPLE loop) — so a failure here isolates the SGS *formula*
itself from the baseline discretisation.

Reference: Nicoud & Ducros, "Subgrid-Scale Stress Modelling Based on the
Square of the Velocity Gradient Tensor", Flow Turb. Combust. 62 (1999):

    eddy_viscosity = (wale_coefficient * Delta)^2 * (Sd_ij Sd_ij)^{3/2}
                              / ( (S_ij S_ij)^{5/2} + (Sd_ij Sd_ij)^{5/4} )

    S_ij  = symmetric part of the velocity-gradient tensor g_ij = du_i/dx_j
    Sd_ij = traceless symmetric part of g_ik g_kj (the SQUARE of g, as a
            matrix product -- not an elementwise square)
"""

from __future__ import annotations

import numpy as np
import pytest

from source.solvers.fvm.turbulence.les_models import WALE, _strain_rate, _wale_operator


def _random_gradient_tensors(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, 3, 3))


def test_wale_operator_solid_body_rotation_matches_closed_form():
    """Solid-body rotation has S_ij = 0 identically (g is pure-antisymmetric),
    which zeroes the WALE denominator's (S:S)^{5/2} term -- but this does
    *not* make eddy_viscosity zero, because Sd_ij Sd_ij (built from g^2, not S) stays
    nonzero for a genuine rotation. For g = [[0,-w,0],[w,0,0],[0,0,0]]:

        g^2 = diag(-w^2, -w^2, 0)  (already symmetric)
        Sd  = g^2 - trace(g^2)/3 * I = diag(-w^2/3, -w^2/3, 2w^2/3)
        Sd:Sd = 2*(w^2/3)^2 + (2w^2/3)^2 = 2 w^4 / 3

    and since S:S = 0, op = (Sd:Sd)^{3/2} / (Sd:Sd)^{5/4} = (Sd:Sd)^{1/4}
                        = (2/3)^{1/4} * w.

    This is a hand-derived closed form independent of the implementation,
    used to certify the code rather than assert an unverified "WALE is zero
    in rigid rotation" folklore claim (it is Smagorinsky, which depends on
    |S| alone, that is trivially zero here -- WALE is not, and should not
    be tuned/forced to be).
    """
    omega = 3.7  # rotation rate, s^-1
    g = np.zeros((5, 3, 3))
    g[:, 0, 1] = -omega
    g[:, 1, 0] = omega
    S, strain_squared = _strain_rate(g)
    assert np.allclose(S, 0.0)
    assert np.allclose(strain_squared, 0.0)

    op = _wale_operator(g)
    expected = (2.0 / 3.0) ** 0.25 * omega
    np.testing.assert_allclose(op, expected, rtol=1e-10)
    assert np.all(np.isfinite(op)) and np.all(op >= 0.0)


def test_wale_operator_zero_for_canonical_simple_shear():
    """Canonical simple shear u = (gamma*y, 0, 0) gives g^2 = 0 identically
    (g is nilpotent: g_ij has a single nonzero entry g_01), so the WALE
    traceless-square invariant Sd:Sd is exactly zero and eddy_viscosity = 0 -- one of
    WALE's literature-cited advantages over plain Smagorinsky (Nicoud &
    Ducros 1999, §4.1): it does not spuriously flag mean shear as SGS
    turbulence.
    """
    shear_rate = 2.3
    g = np.zeros((4, 3, 3))
    g[:, 0, 1] = shear_rate
    gradient_squared = np.einsum("cik,ckj->cij", g, g)
    assert np.allclose(gradient_squared, 0.0), "g must be nilpotent for this case"
    op = _wale_operator(g)
    assert np.allclose(op, 0.0, atol=1e-12)


def test_wale_eddy_viscosity_non_negative_for_arbitrary_gradients():
    """eddy_viscosity = (wale_coefficient Delta)^2 * op must be >= 0 for arbitrary velocity gradients:
    op is a ratio of a non-negative numerator (an even power of a real
    quantity) over a strictly non-negative denominator (sum of squares plus
    epsilon), so it cannot go negative regardless of the input tensor.
    """
    g = _random_gradient_tensors(200, seed=1)
    op = _wale_operator(g)
    assert np.all(op >= 0.0)
    assert np.all(np.isfinite(op))


def test_wale_operator_rotation_invariant():
    """The WALE operator is built from proper tensor contractions (S_ij S_ij,
    Sd_ij Sd_ij), so it must be invariant under a rigid rotation of the
    velocity-gradient tensor: g' = Q g Q^T for any orthogonal Q. This is the
    "tensor invariance under rotation" check PLAN.md §13 asks for -- a
    physically meaningful SGS model cannot depend on the orientation of the
    coordinate frame.
    """
    g = _random_gradient_tensors(50, seed=2)

    # A fixed, non-trivial rotation matrix (about an arbitrary axis).
    axis = np.array([1.0, 2.0, 3.0])
    axis = axis / np.linalg.norm(axis)
    theta = 0.83
    K = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    Q = np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)
    assert np.allclose(Q @ Q.T, np.eye(3), atol=1e-10), "Q must be orthogonal"

    g_rotated = np.einsum("ij,cjk,lk->cil", Q, g, Q)

    op = _wale_operator(g)
    op_rotated = _wale_operator(g_rotated)
    np.testing.assert_allclose(op_rotated, op, rtol=1e-9, atol=1e-12)


def test_wale_compute_eddy_viscosity_rejects_invalid_coefficient():
    """wale_coefficient must be validated eagerly at construction (PLAN.md's "do not tune
    wale_coefficient to force the DNS peak" only makes sense if wale_coefficient is a real, checked
    physical parameter, not a silently-accepted arbitrary float).
    """
    with pytest.raises(ValueError):
        WALE(mesh_data={}, geo_data={"cell_volume": np.array([1.0])}, wale_coefficient=-0.1)
    with pytest.raises(ValueError):
        WALE(mesh_data={}, geo_data={"cell_volume": np.array([1.0])}, wale_coefficient=float("nan"))


def test_wale_operator_matches_literature_formula_directly():
    """Cross-check the vectorised implementation against a hand-written,
    unvectorised evaluation of the Nicoud & Ducros (1999) formula for a
    handful of general (non-degenerate) velocity-gradient tensors.
    """
    rng = np.random.default_rng(7)
    for _ in range(10):
        g = rng.normal(size=(3, 3))
        S = 0.5 * (g + g.T)
        strain_squared = float(np.sum(S * S))

        g2 = g @ g
        trace = float(np.trace(g2))
        Sd = 0.5 * (g2 + g2.T) - (trace / 3.0) * np.eye(3)
        sd_squared = float(np.sum(Sd * Sd))

        expected = sd_squared**1.5 / (strain_squared**2.5 + sd_squared**1.25 + 1e-30)

        got = _wale_operator(g[None, :, :])[0]
        assert got == pytest.approx(expected, rel=1e-10)
