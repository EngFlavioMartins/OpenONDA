"""Safe near/far interaction classification for FMM cells."""

from __future__ import annotations

import numpy as np

from ....kernels.base import RadialVortexKernel
from .tree import FMMCell


def well_separated(
    source: FMMCell,
    target: FMMCell,
    tolerance: float,
    kernel: RadialVortexKernel | None = None,
) -> bool:
    """Return whether a cell pair is safe for a far-field expansion."""
    distance = float(np.linalg.norm(source.centre - target.centre))
    core_bound = source.max_core_radius + target.max_core_radius
    geometric_radius = source.half_width + target.half_width
    if kernel is not None:
        core_bound = max(
            core_bound,
            kernel.near_field_cutoff(
                max(source.max_core_radius, target.max_core_radius), tolerance
            ),
        )
    # The retained source expansion is second order.  A conservative MAC is
    # still needed for regularized vortex moments because the vector kernel
    # has stronger cancellation than a scalar Laplace monopole.
    opening_factor = 2.50 + 0.15 * np.log10(1.0 / max(tolerance, np.finfo(float).eps))
    # Keep the geometric MAC and kernel regularisation checks separate.  The
    # latter is an absolute exclusion band; multiplying it by the MAC factor
    # would make modest leaf cells classify an entire bounded cloud as near.
    return distance > opening_factor * geometric_radius and distance > geometric_radius + core_bound


def interaction_lists(tree, tolerance: float):
    """Return deterministic ``(target, source, far)`` cell relationships."""
    return tuple(
        (target, source, well_separated(source, target, tolerance))
        for target in tree.cells
        for source in tree.cells
    )


__all__ = ["interaction_lists", "well_separated"]
