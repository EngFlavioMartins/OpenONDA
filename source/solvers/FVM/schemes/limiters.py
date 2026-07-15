"""TVD flux limiters ψ(r) for high-resolution convection.

Each limiter maps the upwind smoothness ratio ``r`` to its standard TVD flux
limiter ``ψ ∈ [0, 2]``. The face value used by the convection assembly is

    φ_f = φ_upwind + ψ(r) · (φ_linear − φ_upwind)

so ψ = 0 recovers upwind and ψ = 1 recovers central on a midpoint face.
Values above one are required by standard Van Leer, MUSCL, and Superbee rather
than being silently clipped to a different scheme. ``r`` is built from the cell gradients in the
gradient-based (NVD/TVD) form used by OpenFOAM:

    r = 2 (d · ∇φ_upwind) / (φ_N − φ_P) − 1

(see ``assemble.convection._tvd_face_psi``).  All functions are vectorised over
faces and assume ``r`` is finite (the caller sanitises extrema).
"""

from __future__ import annotations

import numpy as np


def _van_leer(r: np.ndarray) -> np.ndarray:
    """Van Leer limiter: ψ(r) = (r + |r|) / (1 + |r|).

    A symmetric TVD limiter with a smooth, differentiable transition
    between upwind and central-differencing regions.  Moderately
    compressive.

    Args:
        r: Upwind smoothness ratio array ``(n_faces,)``.

    Returns:
        Limiter value ψ(r) in [0, 2].
    """
    return (r + np.abs(r)) / (1.0 + np.abs(r))


def _minmod(r: np.ndarray) -> np.ndarray:
    """Minmod limiter: ψ(r) = max(0, min(1, r)).

    The most diffusive TVD limiter; it is the least compressive and
    therefore the most robust.

    Args:
        r: Upwind smoothness ratio array ``(n_faces,)``.

    Returns:
        Limiter value ψ(r) in [0, 1].
    """
    return np.maximum(0.0, np.minimum(1.0, r))


def _muscl(r: np.ndarray) -> np.ndarray:
    """MUSCL (Monotone Upstream-centred Schemes for Conservation Laws) limiter.

        ψ(r) = max(0, min(2r, ½(r+1), 2))

    A symmetric TVD limiter (van Leer's MUSCL scheme).  Moderately
    compressive, a common default in aerospace CFD.

    Args:
        r: Upwind smoothness ratio array ``(n_faces,)``.

    Returns:
        Limiter value ψ(r) in [0, 2].
    """
    return np.maximum(0.0, np.minimum(np.minimum(2.0 * r, 0.5 * r + 0.5), 2.0))


def _superbee(r: np.ndarray) -> np.ndarray:
    """Superbee limiter (Roe): ψ(r) = max(0, min(2r, 1), min(r, 2)).

    The most compressive TVD limiter — steepens gradients and may
    cause numerical "cliff" formation.

    Args:
        r: Upwind smoothness ratio array ``(n_faces,)``.

    Returns:
        Limiter value ψ(r) in [0, 2].
    """
    return np.maximum.reduce([np.zeros_like(r), np.minimum(2.0 * r, 1.0), np.minimum(r, 2.0)])


def _limited_linear(r: np.ndarray, k: float = 1.0) -> np.ndarray:
    """OpenFOAM ``limitedLinear k``: symmetric, bounded, k ∈ (0, 1].

    k → 1 is most accurate (closest to linear), k → 0 is most stable (closer to
    upwind)."""
    two_by_k = 2.0 / max(k, 1e-6)
    return np.maximum(0.0, np.minimum(two_by_k * r, 1.0))


# Registry keyed by lower-case scheme name.
LIMITERS = {
    "vanleer": _van_leer,
    "minmod": _minmod,
    "muscl": _muscl,
    "superbee": _superbee,
    "limitedlinear": _limited_linear,
}


def is_limited_scheme(name: str) -> bool:
    """True if ``name`` selects a gradient-based TVD limiter (needs ∇φ)."""
    return str(name).lower() in LIMITERS


def apply_limiter(name: str, r: np.ndarray) -> np.ndarray:
    """Evaluate the named standard limiter without altering its TVD range."""
    fn = LIMITERS.get(str(name).lower())
    if fn is None:
        raise ValueError(f"Unknown TVD limiter '{name}'. Known: {sorted(LIMITERS)}")
    values = fn(r)
    if not np.all(np.isfinite(values)):
        raise FloatingPointError(f"TVD limiter {name!r} returned non-finite values")
    return values
