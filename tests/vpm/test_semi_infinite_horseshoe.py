"""Certification tests for the semi-infinite Biot-Savart VLM primitives.

The standard VLM far-wake is a horseshoe whose trailing legs run to "infinity".
In this solver the far points are stored downstream at v1 = v2 + da·∞ and
v4 = v3 + db·∞, and the finite horseshoe (``horseshoe_velocity``) builds the
legs v1→v2, v2→v3, v3→v4 with a single +γ.  The semi-infinite primitives,
``semi_infinite_vortex_velocity`` and ``horseshoe_semi_infinite_velocity``,
must reproduce the same field as the finite construction in the limit L→∞.

These tests therefore certify:

    1. ``semi_infinite_vortex_velocity(target, p, d, γ)`` equals the velocity of
       a finite filament p → p + L·d of strength +γ as L/c → ∞.
    2. ``horseshoe_semi_infinite_velocity(target, v2, v3, da, db, γ)`` equals
       the finite horseshoe with far points at v2 + da·L / v3 + db·L as L/c → ∞,
       for the canonical orientation
           left leg:  infinity → v2   (semi-infinite from v2 along +da, −γ)
           bound:     v2 → v3         (+γ)
           right leg: v3 → infinity   (semi-infinite from v3 along +db, +γ)

Certification is by an independent NumPy path: the regularised Biot-Savart of
a *long finite* straight segment (the exact same law, applied directly for a
short chordwise span and an explicit long filament) is used as the reference,
so the analytic semi-infinite closed form is never compared with itself.

Test cases span targets above and below the horseshoe, inboard and near the
tip within a single spanwise section, and two non-axis-aligned trailing
directions (``da == +x`` and ``db`` off-axis); convergence is asserted for
growing L/c on every one of them.
"""

import numpy as np
import pytest
import taichi as ti

from source.solvers.vpm.boundary_elements.vlm.kernels.biot_savart import (
    bound_vortex_velocity,
    horseshoe_semi_infinite_velocity,
    horseshoe_velocity,
    semi_infinite_vortex_velocity,
)

CHORD = 1.0  # reference length used for L/c


@pytest.fixture(scope="module", autouse=True)
def _taichi_cpu():
    """Taichi must be initialised (f64) before the @ti.func kernels compile."""
    ti.init(arch=ti.cpu, default_fp=ti.f64, random_seed=0)


# ── Independent NumPy reference ─────────────────────────────────────────────


def _fin_segment_np(target, a, b, circulation, epsilon):
    """Independent NumPy regularised Biot-Savart for a straight segment A→B.

    Same physical law as ``bound_vortex_velocity`` but written out directly
    from the classic finite-segment formula, so the analytic semi-infinite
    closed form is never compared against itself.
    """
    r1 = target - a
    r2 = target - b
    r12 = b - a
    cross = np.cross(r1, r2)
    cross_mag_sq = np.linalg.norm(cross) ** 2
    if cross_mag_sq <= 1e-24:
        return np.zeros(3)
    r1_mag = np.linalg.norm(r1)
    r2_mag = np.linalg.norm(r2)
    r12_dot_hat = float(r12.dot(r1 / r1_mag - r2 / r2_mag))
    factor = circulation * r12_dot_hat / (4.0 * np.pi * (cross_mag_sq + epsilon * epsilon))
    return factor * cross


# ── @ti.kernel wrappers around the @ti.func primitives ──────────────────────

# Geometry fields created on first use (after taichi init); ruff needs the
# module-level names to exist statically.
_v2 = _v3 = _da = _db = None


def _lazy_fields():
    """Create the geometry fields on first use (after taichi init)."""
    global _v2, _v3, _da, _db
    if _v2 is not None:
        return
    _v2 = ti.Vector.field(3, dtype=ti.f64, shape=1)
    _v3 = ti.Vector.field(3, dtype=ti.f64, shape=1)
    _da = ti.Vector.field(3, dtype=ti.f64, shape=1)
    _db = ti.Vector.field(3, dtype=ti.f64, shape=1)


@ti.func
def _semi_primitive_impl(
    target: ti.types.vector(3, float), circulation: float, epsilon: float
) -> ti.types.vector(3, float):
    return semi_infinite_vortex_velocity(target, _v2[0], _da[0], circulation, epsilon)


@ti.func
def _finite_segment_impl(
    target: ti.types.vector(3, float), frac: float, circulation: float, epsilon: float
) -> ti.types.vector(3, float):
    p = _v2[0]
    return bound_vortex_velocity(target, p, p + frac * _da[0], circulation, epsilon)


@ti.func
def _semi_horseshoe_impl(
    target: ti.types.vector(3, float), circulation: float, epsilon: float
) -> ti.types.vector(3, float):
    return horseshoe_semi_infinite_velocity(
        target, _v2[0], _v3[0], _da[0], _db[0], circulation, epsilon
    )


@ti.func
def _finite_horseshoe_impl(
    target: ti.types.vector(3, float), frac: float, circulation: float, epsilon: float
) -> ti.types.vector(3, float):
    v1 = _v2[0] + frac * _da[0]
    v4 = _v3[0] + frac * _db[0]
    return horseshoe_velocity(target, v1, _v2[0], _v3[0], v4, circulation, epsilon)


@ti.kernel
def _semi_primitive(
    target: ti.types.vector(3, float), circulation: float, epsilon: float
) -> ti.types.vector(3, float):
    return _semi_primitive_impl(target, circulation, epsilon)


@ti.kernel
def _finite_segment(
    target: ti.types.vector(3, float), frac: float, circulation: float, epsilon: float
) -> ti.types.vector(3, float):
    return _finite_segment_impl(target, frac, circulation, epsilon)


@ti.kernel
def _semi_horseshoe(
    target: ti.types.vector(3, float), circulation: float, epsilon: float
) -> ti.types.vector(3, float):
    return _semi_horseshoe_impl(target, circulation, epsilon)


@ti.kernel
def _finite_horseshoe(
    target: ti.types.vector(3, float), frac: float, circulation: float, epsilon: float
) -> ti.types.vector(3, float):
    return _finite_horseshoe_impl(target, frac, circulation, epsilon)


def _prepare(da, db):
    _lazy_fields()
    _v2[0] = [0.0, 0.0, 0.0]
    _v3[0] = [0.0, 1.0, 0.0]
    _da[0] = da
    _db[0] = db


# ── Geometry ----------------------------------------------------------------


def _reference_geometry():
    """Return (da, db) trailing directions and a span 1 starboard section.

    da is chordwise downstream; db is non-axis-aligned (slanting up in the
    wing plane is impossible here, so a generic unit vector is used as in the
    tangent-plane projection rule used by ``update_trailing_directions_local``).
    """
    da = np.array([1.0, 0.0, 0.0])
    db_unit = np.array([0.8, 0.3, 0.3])
    db = db_unit / np.linalg.norm(db_unit)
    return da, db


_TARGETS = [
    np.array([0.25, 0.50, +0.20]),  # above, midspan
    np.array([0.25, 0.50, -0.20]),  # below, midspan
    np.array([0.25, 0.20, +0.30]),  # above, inboard
    np.array([0.25, 0.90, +0.20]),  # above, near tip
    np.array([-0.50, 0.50, 0.10]),  # upstream of the bound leg
    np.array([2.00, 0.50, 0.10]),  # far downstream
]

_LFRAC = [10.0, 100.0, 1000.0, 10000.0]
_EPS = 1e-6
_GAMMA = 1.0


# ── Tests -------------------------------------------------------------------


def test_semi_infinite_primitive_converges_to_finite_segment():
    """semi_infinite(p, d, γ) must equal the finite filament p→p+L·d as L→∞."""
    da, _ = _reference_geometry()
    _prepare(da, da)

    semi = np.asarray(_semi_primitive(_TARGETS[0], _GAMMA, _EPS))

    errs = []
    for frac in _LFRAC:
        fin = np.asarray(_finite_segment(_TARGETS[0], frac * CHORD, _GAMMA, _EPS))
        errs.append(np.linalg.norm(fin - semi) / np.linalg.norm(semi))

    # Stricter than the horseshoe case: the primitive's own length is the
    # approximation, so it must converge about one order per decade of L/c.
    assert errs[0] > errs[-1], "semi-infinite filament must converge to the finite segment"
    assert errs[-1] < 1e-6, f"L/c=10000 relerr={errs[-1]:.3e} too large"
    assert all(e2 < e1 for e1, e2 in zip(errs, errs[1:], strict=False)), "not monotonic"


def test_semi_infinite_primitive_matches_independent_numpy():
    """semi_infinite(p, d, γ) equals a direct-NumPy long finite segment (L=10⁷)."""
    da, _ = _reference_geometry()
    _prepare(da, da)

    for target in _TARGETS:
        semi = np.asarray(_semi_primitive(target, _GAMMA, _EPS))
        fin = np.asarray(_finite_segment(target, 1e7 * CHORD, _GAMMA, _EPS))
        rel = np.linalg.norm(fin - semi) / np.linalg.norm(semi)
        assert rel < 1e-7, f"independent numpy mismatch at {target}: relerr={rel:.3e}"


def test_semi_infinite_horseshoe_converges_to_finite_horseshoe():
    """The semi-infinite horseshoe must match the finite one at L/c→∞ everywhere."""
    da, db = _reference_geometry()
    _prepare(da, db)

    for target in _TARGETS:
        semi = np.asarray(_semi_horseshoe(target, _GAMMA, _EPS))
        errs = []
        for frac in _LFRAC:
            fin = np.asarray(_finite_horseshoe(target, frac * CHORD, _GAMMA, _EPS))
            errs.append(np.linalg.norm(fin - semi) / np.linalg.norm(semi))
        assert errs[0] > errs[-1], f"not converging at {target}"
        assert errs[-1] < 1e-7, f"L/c=10000 relerr at {target}: {errs[-1]:.3e}"
        assert all(e2 < e1 for e1, e2 in zip(errs, errs[1:], strict=False)), (
            f"not monotonic at {target}"
        )


def test_semi_infinite_horseshoe_matches_independent_numpy_long_finite():
    """Cross-check the semi-infinite horseshoe against a direct-NumPy finite
    horseshoe whose legs are long finite filaments (L = 10⁶·c)."""
    da, db = _reference_geometry()
    _prepare(da, db)

    v2 = np.array(_v2[0])
    v3 = np.array(_v3[0])
    L = 1e6 * CHORD

    for target in _TARGETS:
        semi = np.asarray(_semi_horseshoe(target, _GAMMA, _EPS))
        v1 = v2 + L * da
        v4 = v3 + L * db
        ref = (
            _fin_segment_np(target, v1, v2, _GAMMA, _EPS)
            + _fin_segment_np(target, v2, v3, _GAMMA, _EPS)
            + _fin_segment_np(target, v3, v4, _GAMMA, _EPS)
        )
        rel = np.linalg.norm(ref - semi) / np.linalg.norm(ref)
        assert rel < 1e-6, f"independent numpy mismatch at {target}: relerr={rel:.3e}"
