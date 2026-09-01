"""Low-order vector multipoles for the Biot--Savart far field."""

from __future__ import annotations

import numpy as np


def p2m(position: np.ndarray, vortex_strength: np.ndarray, centre: np.ndarray) -> dict[str, np.ndarray]:
    """Build monopole and first moment coefficients for one leaf."""
    displacement = position - centre
    return {
        "circulation": vortex_strength.sum(axis=0),
        "first_moment": np.einsum("ni,nj->ij", displacement, vortex_strength),
    }


def m2m(child_multipoles: list[dict[str, np.ndarray]], child_centres, centre) -> dict[str, np.ndarray]:
    """Translate child monopoles and first moments to a parent centre."""
    circulation = sum((item["circulation"] for item in child_multipoles), start=np.zeros(3))
    first_moment = sum((item["first_moment"] for item in child_multipoles), start=np.zeros((3, 3)))
    for item, child_centre in zip(child_multipoles, child_centres, strict=True):
        first_moment += np.outer(np.asarray(child_centre) - centre, item["circulation"])
    return {"circulation": circulation, "first_moment": first_moment}


def multipole_velocity(multipole: dict[str, np.ndarray], displacement: np.ndarray) -> np.ndarray:
    """Evaluate the singular Biot--Savart monopole plus first moment."""
    displacement = np.asarray(displacement, dtype=np.float64)
    radius = float(np.linalg.norm(displacement))
    if radius == 0.0:
        return np.zeros(3, dtype=np.float64)
    circulation = np.asarray(multipole["circulation"], dtype=np.float64)
    first_moment = np.asarray(multipole["first_moment"], dtype=np.float64)
    q_infinity = 1.0 / (4.0 * np.pi)
    inverse_r3 = q_infinity / radius**3
    velocity = inverse_r3 * np.cross(circulation, displacement)

    # First-order Taylor expansion of K(R-d) and Gamma x (R-d).
    epsilon = np.array(
        [
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
            [[0.0, 0.0, -1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ]
    )
    velocity += inverse_r3 * np.einsum("iab,ab->i", epsilon, first_moment)
    for axis in range(3):
        for component in range(3):
            velocity += (
                3.0
                * q_infinity
                * first_moment[axis, component]
                * np.cross(np.eye(3)[component], displacement)
                * displacement[axis]
                / radius**5
            )
    return velocity


__all__ = ["m2m", "multipole_velocity", "p2m"]
