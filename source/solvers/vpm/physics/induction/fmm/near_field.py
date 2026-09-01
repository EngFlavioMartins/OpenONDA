"""Exact regularized near-field interactions for the FMM."""

from __future__ import annotations

import numpy as np

from ....kernels.base import RadialVortexKernel


def p2p_velocity(kernel: RadialVortexKernel, target_position, source_position, source_strength, target_core, source_core):
    """Evaluate exact regularized P2P velocity interactions."""
    displacement = np.asarray(target_position)[:, None, :] - np.asarray(source_position)[None, :, :]
    pair_velocity = kernel.velocity_pair(
        displacement,
        np.asarray(source_strength)[None, :, :],
        np.asarray(target_core)[:, None],
        np.asarray(source_core)[None, :],
    )
    diagonal = min(len(pair_velocity), len(pair_velocity[0]) if len(pair_velocity) else 0)
    if diagonal:
        pair_velocity[np.arange(diagonal), np.arange(diagonal)] = 0.0
    return pair_velocity.sum(axis=1)


__all__ = ["p2p_velocity"]
