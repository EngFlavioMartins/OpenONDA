"""Exact regularized near-field interactions for the FMM."""

from __future__ import annotations

import numpy as np

from ....kernels.base import RadialVortexKernel


def p2p_velocity(
    kernel: RadialVortexKernel,
    target_position,
    source_position,
    source_strength,
    target_core,
    source_core,
    *,
    exclude_self: bool = False,
):
    """Evaluate exact regularized P2P velocity interactions.

    ``exclude_self`` is explicit because target and source arrays may be
    distinct sets whose matching indices are valid interactions.
    """
    displacement = np.asarray(target_position)[:, None, :] - np.asarray(source_position)[None, :, :]
    pair_velocity = kernel.velocity_pair(
        displacement,
        np.asarray(source_strength)[None, :, :],
        np.asarray(target_core)[:, None],
        np.asarray(source_core)[None, :],
    )
    if exclude_self:
        diagonal = min(len(pair_velocity), len(pair_velocity[0]) if len(pair_velocity) else 0)
        if diagonal:
            pair_velocity[np.arange(diagonal), np.arange(diagonal)] = 0.0
    return pair_velocity.sum(axis=1)


def p2p_velocity_gradient(
    kernel: RadialVortexKernel,
    target_position,
    source_position,
    source_strength,
    target_core,
    source_core,
    *,
    exclude_self: bool = False,
):
    """Evaluate exact near-field velocity and Jacobian blocks.

    The returned gradient uses row-major ``∂u_i/∂x_j`` orientation and the
    same pair-radius convention as :meth:`RadialVortexKernel.gradient_pair`.
    """
    target_position = np.asarray(target_position)
    source_position = np.asarray(source_position)
    displacement = target_position[:, None, :] - source_position[None, :, :]
    pair_velocity = kernel.velocity_pair(
        displacement,
        np.asarray(source_strength)[None, :, :],
        np.asarray(target_core)[:, None],
        np.asarray(source_core)[None, :],
    )
    pair_gradient = kernel.gradient_pair(
        displacement,
        np.asarray(source_strength)[None, :, :],
        np.asarray(target_core)[:, None],
        np.asarray(source_core)[None, :],
    )
    if exclude_self:
        diagonal = min(len(target_position), len(source_position))
        if diagonal:
            diagonal_indices = np.arange(diagonal)
            pair_velocity[diagonal_indices, diagonal_indices] = 0.0
            pair_gradient[diagonal_indices, diagonal_indices] = 0.0
    return pair_velocity.sum(axis=1), pair_gradient.sum(axis=1)


__all__ = ["p2p_velocity", "p2p_velocity_gradient"]
