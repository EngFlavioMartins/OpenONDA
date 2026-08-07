"""Structure-preservation tests for the VPM discretization.

These pin the three properties the vortex-ring study depends on, each of which
is an identity rather than a tolerance:

1. The regularised field integrals (E, H, enstrophy) use the *convolved* pair
   width sqrt(s_i^2 + s_j^2), which is what makes dE/dt = -nu*int|omega|^2 hold
   exactly under core spreading.
2. TRANSPOSED stretching conserves sum(Gamma); the other supported formulations
   do not.
3. The linear-impulse leak that TRANSPOSED introduces is a property of the
   semi-discrete system, so no choice of integrator removes it.

They are pure NumPy: they mirror the Taichi kernels rather than running the
solver, so they are fast and precision-independent.
"""

import numpy as np
import pytest
from scipy.special import erf

pytestmark = pytest.mark.unit

INV_PI15 = np.pi**-1.5
TWO_OVER_SQRT_PI = 2.0 / np.sqrt(np.pi)
ONE_OVER_FOUR_PI = 1.0 / (4.0 * np.pi)


# -- NumPy mirrors of source/solvers/VPM/numerics/kernels_common.py -----------


def _q(rho):
    return (erf(rho) - TWO_OVER_SQRT_PI * rho * np.exp(-rho * rho)) * ONE_OVER_FOUR_PI


def _zeta(rho):
    return INV_PI15 * np.exp(-rho * rho)


def _g(rho):
    out = np.empty_like(rho)
    small = rho < 1e-8
    out[small] = INV_PI15 * (0.5 - rho[small] ** 2 / 6.0)
    big = rho[~small]
    out[~small] = erf(big) / big * ONE_OVER_FOUR_PI
    return out


def _pairs(x, s, convolved):
    r = x[:, None, :] - x[None, :, :]
    r_mag = np.linalg.norm(r, axis=-1)
    if convolved:
        sigma = np.sqrt(s[:, None] ** 2 + s[None, :] ** 2)
    else:
        sigma = 0.5 * (s[:, None] + s[None, :])
    return r, r_mag, sigma


def velocity(x, g, s):
    """u_i = -sum_j q(rho) (r_ij x Gamma_j) / r^3  (compute_velocities_kernel)."""
    r, r_mag, sigma = _pairs(x, s, convolved=False)
    weight = np.where(r_mag > 1e-12, _q(r_mag / sigma) / np.maximum(r_mag, 1e-300) ** 3, 0.0)
    return -np.einsum("ij,ijk->ik", weight, np.cross(r, g[None, :, :]))


def stretch_rate(x, g, s, mode):
    """dGamma_i/dt  (_stretching_contribution, mode 0..3)."""
    r, r_mag, sigma = _pairs(x, s, convolved=False)
    rho = r_mag / sigma
    finite = r_mag > 1e-12
    c1 = np.where(finite, _q(rho) / (sigma**3 * rho**3), 0.0)
    c2 = np.where(finite, (3.0 * _q(rho) - _zeta(rho) * rho**3) / (sigma**5 * rho**5), 0.0)

    gi = np.broadcast_to(g[:, None, :], r.shape)
    gj = np.broadcast_to(g[None, :, :], r.shape)
    gxg = np.cross(gi, gj)
    r_x_gj = np.cross(r, gj)
    gi_dot_r = np.einsum("ik,ijk->ij", g, r)
    gi_dot_rxgj = np.einsum("ik,ijk->ij", g, r_x_gj)

    if mode == "DIRECT":
        term = -c1[..., None] * gxg + c2[..., None] * gi_dot_r[..., None] * r_x_gj
    elif mode == "TRANSPOSED":
        term = c1[..., None] * gxg + c2[..., None] * gi_dot_rxgj[..., None] * r
    elif mode == "MIXED":
        term = 0.5 * c2[..., None] * (gi_dot_r[..., None] * r_x_gj + gi_dot_rxgj[..., None] * r)
    else:
        raise ValueError(mode)
    return term.sum(axis=1)


def energy(x, g, s, convolved=True):
    r, r_mag, sigma = _pairs(x, s, convolved)
    return 0.5 * np.einsum("ij,ij->", _g(r_mag / sigma) / sigma, g @ g.T)


def enstrophy(x, g, s, convolved=True):
    r, r_mag, sigma = _pairs(x, s, convolved)
    return np.einsum("ij,ij->", _zeta(r_mag / sigma) / sigma**3, g @ g.T)


def impulse(x, g):
    return 0.5 * np.cross(x, g).sum(axis=0)


@pytest.fixture
def cloud():
    rng = np.random.default_rng(20260727)
    n = 90
    return (
        rng.normal(size=(n, 3)) * 0.7,
        rng.normal(size=(n, 3)) * 0.05,
        0.25 * (1.0 + 0.8 * rng.random(n)),  # deliberately unequal cores
    )


# -- 1. Kernel consistency ---------------------------------------------------


def test_convolved_pair_width_closes_the_viscous_energy_budget(cloud):
    """d/dt E = -nu*int|omega|^2 exactly, under core spreading d(s^2)/dt = 4 nu_eff.

    This is what the LES energy-budget contract measures.  It holds only when E
    and the enstrophy use the convolved pair width; with the pair mean the ratio
    is off by tens of percent and no tolerance can absorb it.
    """
    x, g, s = cloud
    rng = np.random.default_rng(5)
    nu = rng.uniform(5e-4, 2e-3, len(s))  # per-particle nu_eff, as LES produces

    step = 1e-7
    advance = lambda k: np.sqrt(s**2 + 4.0 * nu * k * step)  # noqa: E731
    sink = -(0.5 * (nu[:, None] + nu[None, :]) * _pair_enstrophy(x, g, s)).sum()

    dedt = (energy(x, g, advance(1)) - energy(x, g, advance(-1))) / (2.0 * step)
    assert dedt == pytest.approx(sink, rel=1e-6)


def _pair_enstrophy(x, g, s):
    r, r_mag, sigma = _pairs(x, s, convolved=True)
    return _zeta(r_mag / sigma) / sigma**3 * (g @ g.T)


def test_pair_mean_width_does_not_close_the_budget(cloud):
    """Guard the fix: the old convention is wrong by a wide, s-dependent margin."""
    x, g, s = cloud
    nu = np.full(len(s), 1e-3)
    step = 1e-7
    advance = lambda k: np.sqrt(s**2 + 4.0 * nu * k * step)  # noqa: E731

    dedt = (energy(x, g, advance(1), convolved=False) - energy(x, g, advance(-1), False)) / (
        2.0 * step
    )
    sink = -(nu[0] * enstrophy(x, g, s))
    assert abs(dedt / sink) > 1.2


# -- 2. Semi-discrete invariants of the stretching exchange ------------------


@pytest.mark.parametrize("mode", ["DIRECT", "TRANSPOSED", "MIXED"])
def test_only_transposed_mode_conserves_total_circulation(cloud, mode):
    x, g, s = cloud
    total = np.linalg.norm(stretch_rate(x, g, s, mode).sum(axis=0))
    scale = np.linalg.norm(g, axis=1).sum()
    if mode == "TRANSPOSED":
        assert total / scale < 1e-14
    else:
        assert total / scale > 1e-4


@pytest.mark.parametrize("mode", ["DIRECT", "TRANSPOSED", "MIXED"])
def test_supported_modes_do_not_enforce_linear_impulse_conservation(cloud, mode):
    """None of the supported stretching formulations makes dI/dt vanish."""
    x, g, s = cloud
    rate = stretch_rate(x, g, s, mode)
    didt = 0.5 * (np.cross(velocity(x, g, s), g) + np.cross(x, rate)).sum(axis=0)
    relative = np.linalg.norm(didt) / np.linalg.norm(impulse(x, g))
    assert relative > 1e-4


# -- 3. Quadratic-invariant preservation by the integrator -------------------


def _rhs(x, g, s, mode):
    return velocity(x, g, s), stretch_rate(x, g, s, mode)


def _explicit_rk2(x, g, s, dt, mode):
    k1x, k1g = _rhs(x, g, s, mode)
    k2x, k2g = _rhs(x + dt * k1x, g + dt * k1g, s, mode)
    return x + 0.5 * dt * (k1x + k2x), g + 0.5 * dt * (k1g + k2g)


def test_transposed_mode_leaks_the_impulse_at_first_order_in_dt(cloud):
    """The leak TRANSPOSED introduces is a property of the semi-discrete system,
    so refining dt does not remove it -- it converges to a non-zero rate."""
    x, g, s = cloud
    scale = np.linalg.norm(impulse(x, g))

    rates = []
    for dt in (2e-2, 1e-2, 5e-3):
        xr, gr = _explicit_rk2(x, g, s, dt, "TRANSPOSED")
        rates.append(np.linalg.norm(impulse(xr, gr) - impulse(x, g)) / scale / dt)

    assert min(rates) > 1e-3
    assert max(rates) / min(rates) < 1.1
