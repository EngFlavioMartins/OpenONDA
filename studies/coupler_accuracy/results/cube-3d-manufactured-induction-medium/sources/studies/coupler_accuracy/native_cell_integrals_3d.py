"""Integrate smooth particle fields over native face-fan polyhedra in 3D.

The FVM combines each face into one area vector and one centroid. On a warped
polygonal face those summaries need not give the volume of its fan triangles. Keep
the triangulated volume and the stored FVM volume separate: an integrated
particle basis is compared with the original FVM circulation omega * V_FVM.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
from scipy.special import roots_jacobi


@lru_cache(maxsize=16)
def tetrahedron_rule(order):
    """Duffy-mapped tensor Gauss--Jacobi rule, with weights summing to one.

    The collapsed coordinates have Jacobian (1-r)^2 (1-s). Integrate those
    factors with Jacobi rules rather than approximating them as part of the
    field. The returned barycentric rule is multiplied by signed tetra volume.
    """
    if order < 1:
        raise ValueError("Quadrature order must be positive")
    rules = [roots_jacobi(order, alpha, 0) for alpha in (2, 1, 0)]
    nodes = [(x + 1) / 2 for x, _ in rules]
    weights = [w / 2**(alpha + 1) for (_, w), alpha in zip(rules, (2, 1, 0), strict=True)]
    r, s, t = np.meshgrid(*nodes, indexing="ij")
    a, b, c = np.meshgrid(*weights, indexing="ij")
    barycentric = np.column_stack((r.ravel(), ((1-r)*s).ravel(),
                                   ((1-r)*(1-s)*t).ravel(), ((1-r)*(1-s)*(1-t)).ravel()))
    return barycentric, (6*a*b*c).ravel()


@dataclass
class NativeCellIntegration:
    cell_ids: np.ndarray
    centre: np.ndarray
    fvm_volume: np.ndarray
    tetrahedra: list[np.ndarray]
    tetra_volume: list[np.ndarray]
    lower: np.ndarray
    upper: np.ndarray

    @classmethod
    def from_mesh(cls, mesh, geometry, cell_ids):
        ids = np.asarray(cell_ids, dtype=int)
        if ids.ndim != 1 or not len(ids) or len(np.unique(ids)) != len(ids):
            raise ValueError("Cell indices must be a nonempty distinct one-dimensional selection")
        if np.any(ids < 0) or np.any(ids >= mesh["n_cells"]):
            raise ValueError("Cell index outside the mesh")
        incidence = [[] for _ in range(mesh["n_cells"])]
        for face, own in enumerate(mesh["owners"]):
            incidence[own].append((face, 1))
        for face, neighbour in enumerate(mesh["neighbours"]):
            incidence[neighbour].append((face, -1))
        centres = geometry["cell_centre"][ids].copy()
        tetrahedra, volumes, low, high = [], [], [], []
        for local, cell in enumerate(ids):
            triangles = []
            for face, sign in incidence[cell]:
                vertices = mesh["vertex_position"][mesh["faces"][face]]
                fan = vertices.mean(axis=0)
                triangle = np.stack((np.broadcast_to(fan, vertices.shape), vertices,
                                     np.roll(vertices, -1, axis=0)), axis=1)
                triangles.append(triangle if sign == 1 else triangle[:, [0, 2, 1]])
            triangles = np.concatenate(triangles)
            corners = np.concatenate((np.broadcast_to(centres[local], (len(triangles), 1, 3)),
                                      triangles), axis=1)
            relative = triangles - centres[local]
            volume = np.einsum("ti,ti->t", relative[:, 0],
                               np.cross(relative[:, 1], relative[:, 2])) / 6
            if volume.sum() <= 0:
                raise ValueError(f"Nonpositive triangulated volume for cell {cell}")
            tetrahedra.append(corners)
            volumes.append(volume)
            low.append(corners.min(axis=(0, 1)))
            high.append(corners.max(axis=(0, 1)))
        return cls(ids, centres, geometry["cell_volume"][ids].copy(), tetrahedra,
                   volumes, np.asarray(low), np.asarray(high))

    @property
    def polyhedron_volume(self):
        return np.array([v.sum() for v in self.tetra_volume])

    @property
    def polyhedron_centroid(self):
        return np.array([(v[:, None] * tet.mean(axis=1)).sum(axis=0) / v.sum()
                         for tet, v in zip(self.tetrahedra, self.tetra_volume, strict=True)])

    def rule(self, cell, order, *, relative=False):
        barycentric, weights = tetrahedron_rule(order)
        # Every tetra starts at this FVM cell centre. Local coordinates avoid
        # cancellation in kernels when the whole geometry is translated.
        edges = self.tetrahedra[cell][:, 1:] - self.centre[cell]
        points = np.einsum("qk,tkd->tqd", barycentric[:, 1:], edges).reshape(-1, 3)
        if not relative:
            points += self.centre[cell]
        return points, (self.tetra_volume[cell][:, None] * weights).ravel()

    def geometry_audit(self):
        relative = (self.polyhedron_volume - self.fvm_volume) / self.fvm_volume
        return {
            "cells": len(self.cell_ids),
            "maximum_fan_vs_fvm_relative_volume_difference": float(np.max(np.abs(relative), initial=0)),
            "rms_fan_vs_fvm_relative_volume_difference": float(np.sqrt(np.mean(relative**2))),
            "maximum_centroid_displacement": float(np.max(np.linalg.norm(
                self.polyhedron_centroid - self.centre, axis=1), initial=0)),
            "negative_tetrahedra": int(sum(np.count_nonzero(v < 0) for v in self.tetra_volume)),
            "maximum_signed_volume_cancellation_ratio": float(max(
                (np.sum(np.abs(v)) / np.sum(v) for v in self.tetra_volume), default=1)),
        }


def _integrate_gaussians(integration, cell, position, radius, order):
    points, weight = integration.rule(cell, order, relative=True)
    result = np.zeros(len(position))
    p2 = np.sum(position**2, axis=1)
    peak = 1 / (np.pi**1.5 * radius**3)
    for start in range(0, len(points), 256):
        q = points[start:start+256]
        distance2 = np.maximum(np.sum(q*q, axis=1)[:, None] + p2 - 2*q @ position.T, 0)
        basis = np.exp(-distance2 / radius**2) * peak
        result += weight[start:start+256] @ basis
    return result


def gaussian_cell_integrals(integration, position, radius, *, basis_tolerance=1e-9,
                            minimum_order=4, maximum_order=16, cutoff_sigma=8,
                            progress=None):
    """Return B_cp = integral_cell zeta_sigma(x-xp) dx, as a sparse matrix.

    Increase quadrature order by two until successive rows differ by at most
    basis_tolerance after division by the stored FVM volume. This is a numerical
    refinement estimate, not a rigorous error bound. The caller can request a
    stricter independent run. Sources outside cutoff_sigma radii of the cell's
    bounding box are omitted; their pointwise basis is bounded by the Gaussian
    peak times exp(-cutoff_sigma**2). No filter normalization or row renormalizing
    is applied to the particle basis.
    """
    position = np.asarray(position, dtype=float).reshape(-1, 3)
    radius = np.broadcast_to(np.asarray(radius, dtype=float), (len(position),))
    if not np.all(np.isfinite(position)) or not np.all(np.isfinite(radius)) or np.any(radius <= 0):
        raise ValueError("Finite positions and positive Gaussian radii are required")
    if (not np.isfinite(basis_tolerance) or not np.isfinite(cutoff_sigma)
            or basis_tolerance <= 0 or minimum_order < 1
            or maximum_order < minimum_order + 2 or cutoff_sigma <= 0):
        raise ValueError("Invalid integration tolerance, orders or cutoff")
    tree = cKDTree(position)
    columns, values, offsets = [], [], [0]
    orders, changes = [], []
    for cell, centre in enumerate(integration.centre):
        cell_radius = np.linalg.norm(np.maximum(np.abs(integration.lower[cell] - centre),
                                                np.abs(integration.upper[cell] - centre)))
        candidates = np.asarray(tree.query_ball_point(
            centre, cell_radius + cutoff_sigma * radius.max(initial=0)), dtype=int)
        candidates.sort()
        lower_distance = np.maximum(np.maximum(integration.lower[cell] - position[candidates],
                                                position[candidates] - integration.upper[cell]), 0)
        candidates = candidates[np.linalg.norm(lower_distance, axis=1) <= cutoff_sigma * radius[candidates]]
        local_position = position[candidates] - centre
        local_radius = radius[candidates]
        previous = _integrate_gaussians(integration, cell, local_position, local_radius, minimum_order)
        change, accepted = np.inf, None
        for order in range(minimum_order + 2, maximum_order + 1, 2):
            current = _integrate_gaussians(integration, cell, local_position, local_radius, order)
            change = np.max(np.abs(current - previous), initial=0) / integration.fvm_volume[cell]
            if change <= basis_tolerance:
                accepted = current
                break
            previous = current
        if accepted is None:
            raise RuntimeError(f"Cell {integration.cell_ids[cell]} did not converge: basis change {change:g}")
        columns.append(candidates)
        values.append(accepted)
        offsets.append(offsets[-1] + len(candidates))
        orders.append(order)
        changes.append(float(change))
        if progress is not None and (cell % 100 == 0 or cell + 1 == len(integration.cell_ids)):
            progress(cell + 1, len(integration.cell_ids), order, float(change))
    matrix = csr_matrix((np.concatenate(values), np.concatenate(columns), np.asarray(offsets)),
                        shape=(len(integration.cell_ids), len(position)))
    diagnostics = {"basis_tolerance": basis_tolerance, "cutoff_sigma": cutoff_sigma,
                   "orders": orders, "successive_basis_change_over_fvm_volume": changes,
                   "minimum_order": minimum_order, "maximum_order": maximum_order,
                   "maximum_omitted_pointwise_basis": float(np.exp(-cutoff_sigma**2) * np.max(
                       1 / (np.pi**1.5 * radius**3), initial=0)),
                   "nnz": matrix.nnz}
    return matrix, diagnostics
