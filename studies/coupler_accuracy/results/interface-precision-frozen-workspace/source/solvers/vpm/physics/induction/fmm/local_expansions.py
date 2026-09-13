"""Local expansion translation helpers for the FMM hierarchy."""

from __future__ import annotations

import numpy as np

from .multipoles import multipole_velocity


def m2l(multipole: dict[str, np.ndarray], displacement: np.ndarray) -> dict[str, np.ndarray]:
    """Translate one source multipole into a first-order target local record."""
    displacement = np.asarray(displacement, dtype=np.float64)
    radius = max(float(np.linalg.norm(displacement)), np.finfo(float).eps)
    difference = max(1.0e-7, 1.0e-5 * radius)
    gradient = np.empty((3, 3), dtype=np.float64)
    for axis in range(3):
        offset = np.zeros(3, dtype=np.float64)
        offset[axis] = difference
        gradient[:, axis] = (
            multipole_velocity(multipole, displacement + offset)
            - multipole_velocity(multipole, displacement - offset)
        ) / (2.0 * difference)
    return {
        "value": multipole_velocity(multipole, displacement),
        "gradient": gradient,
        "displacement": np.zeros(3, dtype=np.float64),
    }


def l2l(parent: dict[str, np.ndarray], displacement: np.ndarray) -> dict[str, np.ndarray]:
    """Translate a local record from a parent centre to a child centre."""
    result = dict(parent)
    if "value" in parent and "gradient" in parent:
        offset = np.asarray(displacement, dtype=np.float64)
        result["value"] = (
            np.asarray(parent["value"], dtype=np.float64)
            + np.asarray(parent["gradient"], dtype=np.float64) @ offset
        )
    result["displacement"] = np.asarray(
        parent.get("displacement", np.zeros(3, dtype=np.float64))
    ) + np.asarray(displacement)
    return result


def l2p(local: dict[str, np.ndarray], displacement: np.ndarray | None = None) -> np.ndarray:
    """Evaluate a first-order local expansion at a target displacement."""
    offset = np.zeros(3, dtype=np.float64) if displacement is None else np.asarray(displacement)
    return (
        np.asarray(local["value"], dtype=np.float64)
        + np.asarray(local["gradient"], dtype=np.float64) @ offset
    )


__all__ = ["l2l", "l2p", "m2l"]
