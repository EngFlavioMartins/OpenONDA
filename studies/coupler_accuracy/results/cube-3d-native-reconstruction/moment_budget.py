"""Offline Euclidean projection preserving circulation, impulse and a norm bound."""

from __future__ import annotations

import numpy as np

from studies.coupler_accuracy.joint_reconstruction_3d import project_strength_budget


class MomentBudgetProjector:
    """Dykstra projection onto an affine moment set and a group-norm ball.

    The fixed donor is feasible. Centring and scaling coordinates improves the
    moment matrix conditioning without changing the physical equality set.
    Every call solves its own projection; previous fitted states are not anchors.
    """

    def __init__(self, position, prior, budget, tolerance=2e-13):
        self.prior = np.asarray(prior, dtype=float).copy()
        position = np.asarray(position, dtype=float)
        self.budget = float(budget)
        if position.shape != self.prior.shape or self.prior.ndim != 2 or self.prior.shape[1] != 3:
            raise ValueError("Positions and strengths must be matching (N, 3) arrays")
        if not np.all(np.isfinite(position)) or not np.all(np.isfinite(self.prior)):
            raise ValueError("Moment inputs must be finite")
        if self.budget < np.linalg.norm(self.prior, axis=1).sum():
            raise ValueError("The immutable donor must be feasible")
        centred = position - position.mean(axis=0)
        scale = max(float(np.sqrt(np.mean(np.sum(centred**2, axis=1)))), 1e-30)
        r = centred / scale
        cross = np.zeros((len(r), 3, 3))
        cross[:, 0, 1], cross[:, 0, 2] = -r[:, 2], r[:, 1]
        cross[:, 1, 0], cross[:, 1, 2] = r[:, 2], -r[:, 0]
        cross[:, 2, 0], cross[:, 2, 1] = -r[:, 1], r[:, 0]
        moments = np.vstack((np.tile(np.eye(3), (1, len(r))),
                             cross.transpose(1, 0, 2).reshape(3, -1)))
        _, singular, vectors = np.linalg.svd(moments, full_matrices=False)
        rank = np.count_nonzero(singular > singular[0] * 1e-12)
        self.rows = vectors[:rank]
        self.target = self.rows @ self.prior.ravel()
        self.tolerance = tolerance * max(float(np.linalg.norm(self.prior)), 1e-30)
        self.max_iterations_used = 0

    def affine_projection(self, values):
        flat = values.ravel()
        return (flat - self.rows.T @ (self.rows @ flat - self.target)).reshape(-1, 3)

    def __call__(self, values):
        x = np.asarray(values, dtype=float).copy()
        p, q = np.zeros_like(x), np.zeros_like(x)
        for iteration in range(1, 2001):
            y = self.affine_projection(x + p)
            p = x + p - y
            z = project_strength_budget(y + q, self.budget)
            q = y + q - z
            change = np.linalg.norm(z - x)
            x = z
            if change <= self.tolerance and np.linalg.norm(self.rows @ x.ravel() - self.target) <= self.tolerance:
                self.max_iterations_used = max(self.max_iterations_used, iteration)
                return x
        raise RuntimeError("Moment/budget projection did not converge; no tolerance was relaxed")


def particle_moments(position, strength):
    """Physical integrated vorticity and hydrodynamic impulse at rho=1."""
    return np.sum(strength, axis=0), 0.5 * np.sum(np.cross(position, strength), axis=0)
