"""Low-order vector multipoles for the Biot--Savart far field."""

from __future__ import annotations

import numpy as np


def p2m(
    position: np.ndarray, vortex_strength: np.ndarray, centre: np.ndarray
) -> dict[str, np.ndarray]:
    """Build monopole and first moment coefficients for one leaf."""
    displacement = position - centre
    return {
        "circulation": vortex_strength.sum(axis=0),
        "first_moment": np.einsum("ni,nj->ij", displacement, vortex_strength),
        "second_moment": np.einsum("na,nc,nb->acb", displacement, displacement, vortex_strength),
    }


def m2m(
    child_multipoles: list[dict[str, np.ndarray]], child_centres, centre
) -> dict[str, np.ndarray]:
    """Translate child monopoles and first moments to a parent centre."""
    circulation = sum((item["circulation"] for item in child_multipoles), start=np.zeros(3))
    first_moment = sum((item["first_moment"] for item in child_multipoles), start=np.zeros((3, 3)))
    second_moment = sum(
        (item["second_moment"] for item in child_multipoles),
        start=np.zeros((3, 3, 3)),
    )
    for item, child_centre in zip(child_multipoles, child_centres, strict=True):
        offset = np.asarray(child_centre) - centre
        first_moment += np.outer(offset, item["circulation"])
        second_moment += (
            np.einsum("a,cb->acb", offset, item["first_moment"])
            + np.einsum("c,ab->acb", offset, item["first_moment"])
            + np.einsum("a,c,b->acb", offset, offset, item["circulation"])
        )
    return {
        "circulation": circulation,
        "first_moment": first_moment,
        "second_moment": second_moment,
    }


def multipole_velocity(multipole: dict[str, np.ndarray], displacement: np.ndarray) -> np.ndarray:
    """Evaluate the singular Biot--Savart monopole plus first moment."""
    displacement = np.asarray(displacement, dtype=np.float64)
    radius = float(np.linalg.norm(displacement))
    if radius == 0.0:
        return np.zeros(3, dtype=np.float64)
    circulation = np.asarray(multipole["circulation"], dtype=np.float64)
    first_moment = np.asarray(multipole["first_moment"], dtype=np.float64)
    second_moment = np.asarray(
        multipole.get("second_moment", np.zeros((3, 3, 3))), dtype=np.float64
    )
    q_infinity = 1.0 / (4.0 * np.pi)
    inverse_r3 = q_infinity / radius**3
    velocity = inverse_r3 * np.cross(circulation, displacement)

    # First-order Taylor expansion of the singular Biot--Savart field.
    # For a source at ``centre + d``, the target displacement is ``R - d``:
    #
    #     K(R - d, gamma) = K(R, gamma) - d_a dK(R, gamma)/dR_a + ...
    #
    # Contracting that derivative with M[a, b] = sum(d_a gamma_b) keeps the
    # translation convention identical to P2M/M2M and avoids an epsilon-sign
    # ambiguity in the vector cross-product form.
    for axis in range(3):
        for component in range(3):
            basis = np.eye(3)[component]
            cross_term = np.cross(basis, displacement)
            derivative = q_infinity * (
                np.cross(basis, np.eye(3)[axis]) / radius**3
                - 3.0 * displacement[axis] * cross_term / radius**5
            )
            velocity -= first_moment[axis, component] * derivative

    # The second-order source translation materially reduces the error for
    # admissible clustered cells while retaining an O(1) coefficient set per
    # cell.  The derivative below is d²K_i/(dR_axis dR_other) for unit source
    # strength e_component.
    inverse_r5 = q_infinity / radius**5
    inverse_r7 = q_infinity / radius**7
    for axis in range(3):
        for other_axis in range(3):
            for component in range(3):
                basis = np.eye(3)[component]
                cross_term = np.cross(basis, displacement)
                second_derivative = (
                    -3.0 * displacement[other_axis] * np.cross(basis, np.eye(3)[axis]) * inverse_r5
                    - 3.0 * (1.0 if axis == other_axis else 0.0) * cross_term * inverse_r5
                    - 3.0 * displacement[axis] * np.cross(basis, np.eye(3)[other_axis]) * inverse_r5
                    + 15.0 * displacement[axis] * displacement[other_axis] * cross_term * inverse_r7
                )
                velocity += 0.5 * second_moment[axis, other_axis, component] * second_derivative
    return velocity


def multipole_velocity_batch(
    multipole: dict[str, np.ndarray], displacement: np.ndarray
) -> np.ndarray:
    """Vectorized counterpart of :func:`multipole_velocity` for target leaves."""
    displacement = np.asarray(displacement, dtype=np.float64)
    radius = np.linalg.norm(displacement, axis=-1)
    safe_radius = np.where(radius > 0.0, radius, 1.0)
    circulation = np.asarray(multipole["circulation"], dtype=np.float64)
    first_moment = np.asarray(multipole["first_moment"], dtype=np.float64)
    second_moment = np.asarray(
        multipole.get("second_moment", np.zeros((3, 3, 3))), dtype=np.float64
    )
    q_infinity = 1.0 / (4.0 * np.pi)
    inverse_r3 = q_infinity / safe_radius**3
    inverse_r5 = q_infinity / safe_radius**5
    inverse_r7 = q_infinity / safe_radius**7
    velocity = inverse_r3[..., None] * np.cross(circulation, displacement)
    eye = np.eye(3)
    for axis in range(3):
        for component in range(3):
            basis = eye[component]
            cross_term = np.cross(basis, displacement)
            derivative = (
                inverse_r3[..., None] * np.cross(basis, eye[axis])
                - 3.0 * (displacement[..., axis] * inverse_r5)[..., None] * cross_term
            )
            velocity -= first_moment[axis, component] * derivative
    for axis in range(3):
        for other_axis in range(3):
            for component in range(3):
                basis = eye[component]
                cross_term = np.cross(basis, displacement)
                second_derivative = (
                    -3.0
                    * displacement[..., other_axis, None]
                    * np.cross(basis, eye[axis])
                    * inverse_r5[..., None]
                    - 3.0
                    * (1.0 if axis == other_axis else 0.0)
                    * cross_term
                    * inverse_r5[..., None]
                    - 3.0
                    * displacement[..., axis, None]
                    * np.cross(basis, eye[other_axis])
                    * inverse_r5[..., None]
                    + 15.0
                    * displacement[..., axis, None]
                    * displacement[..., other_axis, None]
                    * cross_term
                    * inverse_r7[..., None]
                )
                velocity += 0.5 * second_moment[axis, other_axis, component] * second_derivative
    return np.where((radius > 0.0)[..., None], velocity, 0.0)


__all__ = ["m2m", "multipole_velocity", "multipole_velocity_batch", "p2m"]
