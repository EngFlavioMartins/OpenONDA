"""Safe near/far interaction classification for FMM cells."""

from __future__ import annotations

import numpy as np

from .tree import FMMCell


def well_separated(source: FMMCell, target: FMMCell, tolerance: float) -> bool:
    """Return whether a cell pair is safe for a far-field expansion."""
    distance = float(np.linalg.norm(source.centre - target.centre))
    radius = source.half_width + target.half_width + source.max_core_radius
    return distance > (1.0 + 1.0 / max(tolerance, np.finfo(float).eps)) ** 0.25 * radius


def interaction_lists(tree, tolerance: float):
    """Return deterministic ``(target, source, far)`` cell relationships."""
    return tuple(
        (target, source, well_separated(source, target, tolerance))
        for target in tree.cells
        for source in tree.cells
    )


__all__ = ["interaction_lists", "well_separated"]
