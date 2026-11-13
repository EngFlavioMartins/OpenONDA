"""TVD flux limiter ψ(r) for high-resolution convection.

The limiter maps the upwind smoothness ratio ``r`` to its TVD flux limiter
``ψ ∈ [0, 1]``. The face value used by the convection assembly is

    φ_f = φ_upwind + ψ(r) · (φ_linear − φ_upwind)

so ψ = 0 recovers upwind and ψ = 1 recovers central on a midpoint face.
``r`` is built from the cell gradients in the
gradient-based NVD/TVD form:

    r = 2 (d · ∇φ_upwind) / (φ_N − φ_P) − 1

(see ``assemble.convection._tvd_face_psi``).  All functions are vectorised over
faces and assume ``r`` is finite (the caller sanitises extrema).
"""

from __future__ import annotations

import numpy as np


def _limited_linear(r: np.ndarray, k: float = 1.0) -> np.ndarray:
    """Limited-linear interpolation: symmetric and bounded for k ∈ (0, 1].

    k → 1 is most accurate (closest to linear), k → 0 is most stable (closer to
    upwind)."""
    two_by_k = 2.0 / max(k, 1e-6)
    return np.maximum(0.0, np.minimum(two_by_k * r, 1.0))


# Registry keyed by scheme name.
LIMITERS = {
    "limited_linear": _limited_linear,
}


def is_limited_scheme(name: str) -> bool:
    """True if ``name`` selects a gradient-based TVD limiter (needs ∇φ)."""
    return name in LIMITERS


def apply_limiter(name: str, r: np.ndarray) -> np.ndarray:
    """Evaluate the named standard limiter without altering its TVD range."""
    fn = LIMITERS.get(name)
    if fn is None:
        raise ValueError(f"Unknown TVD limiter '{name}'. Known: {sorted(LIMITERS)}")
    values = fn(r)
    if not np.all(np.isfinite(values)):
        raise FloatingPointError(f"TVD limiter {name!r} returned non-finite values")
    return values
