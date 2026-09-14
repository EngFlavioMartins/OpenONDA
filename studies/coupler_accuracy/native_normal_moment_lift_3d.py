"""Minimum surface-L2 normal-trace change for prescribed cell first moments.

The correction has zero integral on every face. Its scalar normal velocity is
linear in the coordinates about that face's area centroid. No volume velocity
or vorticity source is constructed here, and physical wall faces can be locked.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsmr

from studies.coupler_accuracy.native_flux_moments_3d import triangle_moment_geometry


@dataclass
class NormalMomentLift:
    area: np.ndarray
    centre: np.ndarray
    covariance_integral: np.ndarray
    mode_faces: np.ndarray
    mode_gradient: np.ndarray
    mode_moment: np.ndarray
    cell_volume: np.ndarray
    operator: object
    discarded_eigenvalues: int

    @classmethod
    def from_mesh(cls, mesh, native, face_origin, cell_volume, free_faces, *, dimensions=3):
        if dimensions not in (2, 3):
            raise ValueError("Use two dominant face modes or all three resolved modes")
        n, nf = mesh["n_cells"], mesh["n_faces"]
        origin, volume = np.asarray(face_origin, dtype=float), np.asarray(cell_volume, dtype=float)
        free = np.asarray(free_faces, dtype=int)
        if (origin.shape != (nf, 3) or volume.shape != (n,) or np.any(volume <= 0)
                or not np.all(np.isfinite(origin)) or not np.all(np.isfinite(volume))
                or free.ndim != 1 or len(free) != len(np.unique(free)) or np.any((free < 0) | (free >= nf))):
            raise ValueError("Finite native geometry and distinct free faces are required")
        tc, sf, triangle_covariance = triangle_moment_geometry(native)
        ta, ids = np.linalg.norm(sf, axis=1), native.face_ids
        area = np.bincount(ids, weights=ta, minlength=nf)
        first = np.zeros((nf, 3))
        np.add.at(first, ids, ta[:, None] * (tc - origin[ids]))
        offset = first / area[:, None]
        relative = (tc - origin[ids]) - offset[ids]
        covariance = np.zeros((nf, 3, 3))
        np.add.at(covariance, ids, ta[:, None, None] * (triangle_covariance + relative[:, :, None] * relative[:, None]))
        eigenvalues, vectors = np.linalg.eigh(covariance[free])
        threshold = 16 * np.finfo(float).eps * eigenvalues[:, -1:]
        if np.any(eigenvalues[:, :1] < -threshold):
            raise ValueError("Face covariance has a materially negative eigenvalue")
        keep = eigenvalues > threshold
        if dimensions == 2:
            keep[:, 0] = False
        rows, columns = np.nonzero(keep)
        mode_faces = free[rows]
        gradients = vectors[rows, :, columns] / np.sqrt(eigenvalues[rows, columns])[:, None]
        moments = np.einsum("mij,mj->mi", covariance[mode_faces], gradients)
        matrix_rows, matrix_columns, entries = [], [], []
        all_modes = np.arange(len(mode_faces))
        internal = mode_faces < mesh["n_interior_faces"]
        for modes, cells, sign in ((all_modes, mesh["owners"][mode_faces], 1),
                                  (all_modes[internal], mesh["neighbours"][mode_faces[internal]], -1)):
            matrix_rows.append((3 * cells[:, None] + np.arange(3)).ravel())
            matrix_columns.append(np.repeat(modes, 3))
            entries.append((sign * moments[modes] / np.sqrt(volume[cells])[:, None]).ravel())
        matrix = coo_matrix((np.concatenate(entries), (np.concatenate(matrix_rows), np.concatenate(matrix_columns))),
                            shape=(3 * n, len(mode_faces))).tocsr()
        return cls(area, origin + offset, covariance, mode_faces, gradients, moments, volume, matrix, int((~keep).sum()))

    def solve(self, required_cell_integral, *, tolerance=1e-11, max_iterations=4000):
        target = np.asarray(required_cell_integral, dtype=float)
        if target.shape != (len(self.cell_volume), 3) or not np.all(np.isfinite(target)):
            raise ValueError("Finite target integral corrections on every cell are required")
        rhs = (target / np.sqrt(self.cell_volume)[:, None]).ravel()
        answer = lsmr(self.operator, rhs, atol=tolerance, btol=tolerance, maxiter=max_iterations)
        coefficients = answer[0]
        gradients = np.zeros((len(self.area), 3))
        np.add.at(gradients, self.mode_faces, self.mode_gradient * coefficients[:, None])
        moments = np.einsum("fij,fj->fi", self.covariance_integral, gradients)
        recovered = (self.operator @ coefficients).reshape(-1, 3) * np.sqrt(self.cell_volume)[:, None]
        residual = (recovered - target) / self.cell_volume[:, None]
        # A dual solution provides an independent lower bound on the minimum
        # squared coefficient norm, when the equality problem is feasible.
        dual = lsmr(self.operator.T, coefficients, atol=tolerance, btol=tolerance, maxiter=max_iterations)
        dual_image = self.operator.T @ dual[0]
        energy = float(coefficients @ coefficients)
        lower_bound = float(2 * dual[0] @ rhs - dual_image @ dual_image)
        return {"face_gradient": gradients, "face_first_moment": moments, "cell_velocity_residual": residual,
                "coefficients": coefficients, "dual_variables": dual[0],
                "diagnostics": {"stop": int(answer[1]), "iterations": int(answer[2]),
                                "residual_norm": float(answer[3]), "normal_residual_norm": float(answer[4]),
                                "condition_estimate": float(answer[6]), "dual_stop": int(dual[1]), "dual_iterations": int(dual[2]),
                                "normal_change_integrated_squared": energy, "dual_energy_lower_bound": lower_bound,
                                "relative_primal_dual_energy_gap": (energy - lower_bound) / max(energy, np.finfo(float).tiny),
                                "velocity_constraint_rms": float(np.sqrt(np.sum(self.cell_volume * np.sum(residual**2, axis=1)) / self.cell_volume.sum())),
                                "velocity_constraint_maximum": float(np.linalg.norm(residual, axis=1).max())}}
