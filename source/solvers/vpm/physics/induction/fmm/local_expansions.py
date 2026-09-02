"""Local expansion translation helpers for the FMM hierarchy."""

from __future__ import annotations

import numpy as np


def m2l(multipole: dict[str, np.ndarray], displacement: np.ndarray) -> dict[str, np.ndarray]:
    """Translate a vector monopole/first moment into a target local record."""
    return {
        "centre_velocity": np.asarray(multipole["circulation"], dtype=np.float64),
        "first_moment": np.asarray(multipole["first_moment"], dtype=np.float64),
        "displacement": np.asarray(displacement, dtype=np.float64),
    }


def l2l(parent: dict[str, np.ndarray], displacement: np.ndarray) -> dict[str, np.ndarray]:
    """Translate a local record from a parent centre to a child centre."""
    result = dict(parent)
    if "value" in parent and "gradient" in parent:
        offset = np.asarray(displacement, dtype=np.float64)
        result["value"] = np.asarray(parent["value"], dtype=np.float64) + np.asarray(
            parent["gradient"], dtype=np.float64
        ) @ offset
    result["displacement"] = np.asarray(
        parent.get("displacement", np.zeros(3, dtype=np.float64))
    ) + np.asarray(displacement)
    return result


def l2p(local: dict[str, np.ndarray]) -> np.ndarray:
    """Return the local record's leading vector coefficient."""
    return np.asarray(local["centre_velocity"], dtype=np.float64)


__all__ = ["l2l", "l2p", "m2l"]
