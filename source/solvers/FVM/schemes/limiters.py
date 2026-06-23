"""TVD flux limiters ψ(r) for high-resolution convection.

Each limiter maps the upwind smoothness ratio ``r`` to a blend factor
``ψ ∈ [0, 1]`` between first-order upwind (ψ = 0) and second-order linear/central
(ψ = 1).  The face value used by the convection assembly is

    φ_f = φ_upwind + ψ(r) · (φ_linear − φ_upwind)

so ψ = 0 recovers upwind, ψ = 1 recovers central, and a TVD ψ(r) gives a bounded
second-order scheme.  ``r`` is built from the cell gradients in the
gradient-based (NVD/TVD) form used by OpenFOAM:

    r = 2 (d · ∇φ_upwind) / (φ_N − φ_P) − 1

(see ``assemble.convection._tvd_face_psi``).  All functions are vectorised over
faces and assume ``r`` is finite (the caller sanitises extrema).
"""

from __future__ import annotations

import numpy as np


def _van_leer(r: np.ndarray) -> np.ndarray:
    return (r + np.abs(r)) / (1.0 + np.abs(r))


def _minmod(r: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, np.minimum(1.0, r))


def _muscl(r: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, np.minimum(np.minimum(2.0 * r, 0.5 * r + 0.5), 2.0))


def _superbee(r: np.ndarray) -> np.ndarray:
    return np.maximum.reduce(
        [np.zeros_like(r), np.minimum(2.0 * r, 1.0), np.minimum(r, 2.0)]
    )


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
    """Evaluate limiter ``name`` on ratio array ``r``, clipped to [0, 1]."""
    fn = LIMITERS.get(str(name).lower())
    if fn is None:
        raise ValueError(f"Unknown TVD limiter '{name}'. Known: {sorted(LIMITERS)}")
    return np.clip(fn(r), 0.0, 1.0)
