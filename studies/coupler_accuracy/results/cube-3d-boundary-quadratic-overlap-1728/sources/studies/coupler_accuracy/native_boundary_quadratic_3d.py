"""Add prescribed boundary velocity to the offline 3D quadratic cell fit.

The additional least-squares objective on nearby prescribed faces is
sum_face integral_face |u_polynomial-u_boundary|^2 dA / |C_face-c_cell|^2.
It has the same velocity-squared units as the existing normalized cell
objective. Area quadrature prevents the weight from depending on the number
of face samples. This is a soft fit; supplied boundary traces remain exact
when the separate shared-face weak-curl integrator uses them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix

from studies.coupler_accuracy.native_quadratic_moments_3d import QuadraticCellLeastSquares
from studies.coupler_accuracy.native_velocity_curl_integrals_3d import triangle_rule


def quadratic_features(displacement, second):
    return np.column_stack((displacement, .5*np.diagonal(second, axis1=1, axis2=2),
                            second[:, 0, 1], second[:, 0, 2], second[:, 1, 2]))


@dataclass
class BoundaryQuadraticCellLeastSquares(QuadraticCellLeastSquares):
    boundary_operator: csr_matrix
    boundary_position: np.ndarray
    boundary_face_ids: np.ndarray
    boundary_area_weight: np.ndarray
    boundary_observation_count: np.ndarray

    @classmethod
    def from_mesh(cls, mesh, centres, volume, *, boundary_faces, average_covariance=None,
                  max_rings=3, quadrature_order=3):
        """Reuse the qualified bulk stencil, adding traces inside its rings.

        Only explicitly selected boundary faces contribute observations. All
        nine modes must already be observable from the 3D cell stencil.
        Boundary values are subsequently supplied at boundary_position.
        """
        if quadrature_order < 3:
            raise ValueError("Quadratic boundary residuals require area quadrature order at least three")
        base = QuadraticCellLeastSquares.from_mesh(mesh, centres, volume, average_covariance=average_covariance, max_rings=max_rings)
        face_ids = np.asarray(boundary_faces, dtype=int)
        if (face_ids.ndim != 1 or len(np.unique(face_ids)) != len(face_ids)
                or np.any(face_ids < mesh["n_interior_faces"]) or np.any(face_ids >= mesh["n_faces"])):
            raise ValueError("Distinct explicitly selected boundary face indices are required")
        n = mesh["n_cells"]
        by_owner, adjacency = [[] for _ in range(n)], [set() for _ in range(n)]
        for own, nei in zip(mesh["owners"][:mesh["n_interior_faces"]], mesh["neighbours"], strict=True):
            adjacency[own].add(int(nei))
            adjacency[nei].add(int(own))
        barycentric, weight = triangle_rule(quadrature_order)
        points, area_weight, point_faces, face_centres, face_point_ids = [], [], [], [], []
        offset = 0
        for local, face in enumerate(face_ids):
            vertices = mesh["vertex_position"][mesh["faces"][face]]
            fan = vertices.mean(axis=0)
            triangles = np.stack((np.broadcast_to(fan, vertices.shape), vertices, np.roll(vertices, -1, axis=0)), axis=1)
            area = np.linalg.norm(np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0]), axis=1)/2
            if np.any(area <= 0):
                raise ValueError("Boundary face fans must have positive triangle areas")
            q = np.einsum("qi,tij->tqj", barycentric, triangles).reshape(-1, 3)
            points.append(q)
            area_weight.append((area[:, None]*weight).ravel())
            point_faces.extend([face]*len(q))
            face_centres.append(np.average(triangles.mean(axis=1), weights=area, axis=0))
            face_point_ids.append(np.arange(offset, offset+len(q)))
            by_owner[mesh["owners"][face]].append(local)
            offset += len(q)
        points = np.concatenate(points) if points else np.empty((0, 3))
        area_weight = np.concatenate(area_weight) if area_weight else np.empty(0)
        face_centres = np.asarray(face_centres).reshape(-1, 3)
        point_faces = np.asarray(point_faces, dtype=int)
        count = np.zeros(n, dtype=int)
        cell_rows, cell_columns, cell_data, wall_rows, wall_columns, wall_data = [], [], [], [], [], []
        condition = base.condition.copy()
        for cell in range(n):
            visited, frontier = {cell}, {cell}
            for _ in range(base.rings[cell]):
                frontier = set().union(*(adjacency[j] for j in frontier))-visited
                visited.update(frontier)
            faces = sorted({face for j in visited for face in by_owner[j]})
            if not faces:
                continue
            ids = np.asarray(sorted(visited-{cell}), dtype=int)
            assert len(ids) == base.neighbour_count[cell]
            h = np.cbrt(volume[cell])
            d = (base.centres[ids]-base.centres[cell])/h
            second = d[:, :, None]*d[:, None]+(base.average_covariance[ids]-base.average_covariance[cell])/h**2
            cell_design = quadratic_features(d, second)
            cell_weight = 1/np.linalg.norm(d, axis=1)
            qids = np.concatenate([face_point_ids[face] for face in faces])
            dq = (points[qids]-base.centres[cell])/h
            wall_design = quadratic_features(dq, dq[:, :, None]*dq[:, None]-base.average_covariance[cell]/h**2)
            distances = np.linalg.norm(face_centres[faces]-base.centres[cell], axis=1)
            if np.any(distances <= 0):
                raise ValueError("Cell and boundary face centroids must be distinct")
            wall_weight = np.concatenate([np.sqrt(area_weight[face_point_ids[face]])/distance
                                          for face, distance in zip(faces, distances, strict=True)])
            design, weights = np.vstack((cell_design, wall_design)), np.r_[cell_weight, wall_weight]
            u, singular, vh = np.linalg.svd(design*weights[:, None], full_matrices=False)
            if singular[-1] <= 1e-8*singular[0]:
                raise ValueError("Boundary-augmented fit does not span all nine 3D modes with acceptable conditioning")
            inverse = (vh.T/singular) @ (u.T*weights[None])
            inverse[:3] /= h
            inverse[3:] /= h**2
            cell_coefficient = np.column_stack((inverse[:, :len(ids)], -inverse.sum(axis=1)))
            columns = np.r_[ids, cell]
            cell_rows.append(np.repeat(9*cell+np.arange(9), len(columns)))
            cell_columns.append(np.tile(columns, 9))
            cell_data.append(cell_coefficient.ravel())
            wall_rows.append(np.repeat(9*cell+np.arange(9), len(qids)))
            wall_columns.append(np.tile(qids, 9))
            wall_data.append(inverse[:, len(ids):].ravel())
            count[cell], condition[cell] = len(qids), singular[0]/singular[-1]
        coo = base.operator.tocoo()
        keep = count[coo.row//9] == 0
        cell_rows.insert(0, coo.row[keep])
        cell_columns.insert(0, coo.col[keep])
        cell_data.insert(0, coo.data[keep])
        operator = csr_matrix((np.concatenate(cell_data), (np.concatenate(cell_rows), np.concatenate(cell_columns))), shape=base.operator.shape)
        boundary_operator = (csr_matrix((np.concatenate(wall_data), (np.concatenate(wall_rows), np.concatenate(wall_columns))), shape=(9*n, len(points)))
                             if wall_data else csr_matrix((9*n, 0)))
        return cls(operator, base.centres, base.average_covariance, condition, base.neighbour_count, base.rings,
                   boundary_operator, points, point_faces, area_weight, count)

    def derivatives(self, values, boundary_velocity):
        """Fit cell and prescribed boundary observations of the same field."""
        value = np.asarray(values, dtype=float)
        if value.ndim == 2:
            value = value[None]
        if value.ndim != 3 or value.shape[1] != len(self.centres) or not np.all(np.isfinite(value)):
            raise ValueError("Finite native cell values are required")
        boundary = np.broadcast_to(np.asarray(boundary_velocity, dtype=float), (len(value), len(self.boundary_position), value.shape[2]))
        if not np.all(np.isfinite(boundary)):
            raise ValueError("Finite prescribed boundary values are required")
        result = np.stack([(self.operator @ (field-field[:1])+self.boundary_operator @ (wall-field[:1])).reshape(len(self.centres), 9, value.shape[2])
                           for field, wall in zip(value, boundary, strict=True)])
        hessian = np.zeros((*result.shape[:2], 3, 3, value.shape[2]))
        for axis in range(3):
            hessian[:, :, axis, axis] = result[:, :, 3+axis]
        for index, (i, j) in enumerate(((0, 1), (0, 2), (1, 2))):
            hessian[:, :, i, j] = hessian[:, :, j, i] = result[:, :, 6+index]
        return result[:, :, :3], hessian
