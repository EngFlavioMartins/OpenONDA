"""Offline quadratic velocity reconstruction and weak vorticity moments.

Point samples and cell averages are distinct input contracts. Each stencil
spans all nine nonconstant quadratic modes in 3D; deficient stencils fail.
No exact derivatives, vorticity moments or evaluation-target values enter
the reconstruction. Boundary polynomial traces are supplied explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix


@dataclass
class QuadraticCellLeastSquares:
    operator: csr_matrix
    centres: np.ndarray
    average_covariance: np.ndarray
    condition: np.ndarray
    neighbour_count: np.ndarray
    rings: np.ndarray

    @classmethod
    def from_mesh(cls, mesh, centres, volume, *, average_covariance=None, max_rings=3):
        """Build a weighted SVD fit on two rings, expanding if rank requires it.

        u_i(y) = U_i + r @ G_i + 0.5 (r r - C_i) : H_i.
        C_i is zero for point input, or the true covariance divided by volume
        for average input at the true centroid. The fit preserves U_i exactly.
        """
        centre, volume = np.asarray(centres, dtype=float), np.asarray(volume, dtype=float)
        n = mesh["n_cells"]
        covariance = np.zeros((n, 3, 3)) if average_covariance is None else np.asarray(average_covariance, dtype=float)
        if (centre.shape != (n, 3) or volume.shape != (n,) or covariance.shape != (n, 3, 3)
                or not np.all(np.isfinite(centre)) or not np.all(np.isfinite(volume))
                or not np.all(np.isfinite(covariance)) or np.any(volume <= 0) or max_rings < 2):
            raise ValueError("Finite 3D centres, positive volumes, compatible covariance and at least two rings are required")
        if not np.allclose(covariance, covariance.swapaxes(-1, -2), rtol=0, atol=1e-14):
            raise ValueError("Average covariance must be symmetric")
        adjacency = [set() for _ in range(n)]
        for own, nei in zip(mesh["owners"][:mesh["n_interior_faces"]], mesh["neighbours"], strict=True):
            adjacency[own].add(int(nei))
            adjacency[nei].add(int(own))
        row_parts, column_parts, data_parts = [], [], []
        condition, count, rings_used = np.empty(n), np.empty(n, dtype=int), np.empty(n, dtype=int)
        for cell in range(n):
            visited, frontier = {cell}, {cell}
            h = np.cbrt(volume[cell])
            for ring in range(1, max_rings+1):
                frontier = set().union(*(adjacency[j] for j in frontier))-visited
                visited.update(frontier)
                if ring < 2:
                    continue
                ids = np.asarray(sorted(visited-{cell}), dtype=int)
                displacement = (centre[ids]-centre[cell])/h
                norm = np.linalg.norm(displacement, axis=1)
                if np.any(norm <= 0):
                    raise ValueError("Distinct centres are required within each stencil")
                tensor = displacement[:, :, None]*displacement[:, None]
                tensor += (covariance[ids]-covariance[cell])/h**2
                design = np.column_stack((displacement, .5*np.diagonal(tensor, axis1=1, axis2=2),
                                          tensor[:, 0, 1], tensor[:, 0, 2], tensor[:, 1, 2]))
                if len(ids) < 9:
                    continue
                weight = 1/norm
                u, singular, vh = np.linalg.svd(design*weight[:, None], full_matrices=False)
                if singular[-1] <= 1e-8*singular[0]:
                    continue
                inverse = (vh.T/singular) @ (u.T*weight[None])
                inverse[:3] /= h
                inverse[3:] /= h**2
                coefficients = np.column_stack((inverse, -inverse.sum(axis=1)))
                columns = np.r_[ids, cell]
                row_parts.append(np.repeat(9*cell+np.arange(9), len(columns)))
                column_parts.append(np.tile(columns, 9))
                data_parts.append(coefficients.ravel())
                condition[cell], count[cell], rings_used[cell] = singular[0]/singular[-1], len(ids), ring
                break
            else:
                raise ValueError(f"Cell {cell} stencil does not span all nine quadratic modes in three dimensions")
        operator = csr_matrix((np.concatenate(data_parts), (np.concatenate(row_parts), np.concatenate(column_parts))), shape=(9*n, n))
        return cls(operator, centre, covariance, condition, count, rings_used)

    def derivatives(self, values):
        """Return derivative-first gradient and symmetric Hessian, with states."""
        values = np.asarray(values, dtype=float)
        if values.ndim == 2:
            values = values[None]
        if values.ndim != 3 or values.shape[1] != len(self.centres) or not np.all(np.isfinite(values)):
            raise ValueError("Finite native cell values are required")
        result = np.stack([(self.operator @ (field-field[:1])).reshape(len(self.centres), 9, values.shape[2]) for field in values])
        hessian = np.zeros((*result.shape[:2], 3, 3, values.shape[2]))
        for axis in range(3):
            hessian[:, :, axis, axis] = result[:, :, 3+axis]
        for index, (i, j) in enumerate(((0, 1), (0, 2), (1, 2))):
            hessian[:, :, i, j] = hessian[:, :, j, i] = result[:, :, 6+index]
        return result[:, :, :3], hessian

    def cell_integrals(self, values, gradient, hessian, volume, centroid, covariance):
        """Integrate the fitted polynomial over actual polyhedral cells."""
        value = np.asarray(values, dtype=float)
        if value.ndim == 2:
            value = value[None]
        displacement = np.asarray(centroid)-self.centres
        second = covariance/volume[:, None, None]+displacement[:, :, None]*displacement[:, None]-self.average_covariance
        mean = value+np.einsum("ci,scij->scj", displacement, gradient)+.5*np.einsum("cik,scikj->scj", second, hessian)
        return mean*volume[None, :, None]


def reconstruct_quadratic_faces(mesh, geometry, fit, values, gradient, hessian, boundary_velocity,
                                *, boundary_gradient=None, boundary_hessian=None):
    """Average the owner/neighbour polynomials into one shared face polynomial.

    Boundary value, gradient and Hessian are explicit at each face origin.
    Omitted boundary derivatives mean a prescribed constant face trace.
    """
    value = np.asarray(values, dtype=float)
    if value.ndim == 2:
        value = value[None]
    shape = (len(value), mesh["n_faces"], 3)
    face_u = np.broadcast_to(boundary_velocity, shape).copy()
    face_g = np.broadcast_to(0 if boundary_gradient is None else boundary_gradient, (*shape, 3)).copy()
    face_h = np.broadcast_to(0 if boundary_hessian is None else boundary_hessian, (*shape, 3, 3)).copy()
    # Broadcasted zeros must retain floating-point polynomial coefficients.
    face_g, face_h = face_g.astype(float), face_h.astype(float)
    n = mesh["n_interior_faces"]
    own, nei = mesh["owners"][:n], mesh["neighbours"]
    weight = geometry["face_interpolation_weight"][:n]
    face_u[:, :n], face_g[:, :n], face_h[:, :n] = 0, 0, 0
    for ids, blend in ((own, 1-weight), (nei, weight)):
        displacement = geometry["face_centre"][:n]-fit.centres[ids]
        second = displacement[:, :, None]*displacement[:, None]-fit.average_covariance[ids]
        trace = value[:, ids]+np.einsum("fi,sfij->sfj", displacement, gradient[:, ids])
        trace += .5*np.einsum("fik,sfikj->sfj", second, hessian[:, ids])
        face_u[:, :n] += blend[None, :, None]*trace
        face_g[:, :n] += blend[None, :, None, None]*(gradient[:, ids]+np.einsum("fk,sfikj->sfij", displacement, hessian[:, ids]))
        face_h[:, :n] += blend[None, :, None, None, None]*hessian[:, ids]
    return face_u, face_g, face_h


def weak_quadratic_curl_moments(native, centroid, cell_velocity_integral, face_origin, face_velocity, face_gradient, face_hessian):
    """Exact shared-face Stokes and first-moment integration for quadratic u.

    For triangle-centred vertices r_v, E[r r]=sum r_v r_v/12 and
    E[r r r]=sum r_v r_v r_v/30. These moments integrate (y-c) cross u
    without point quadrature, including warped polygonal face fans.
    """
    velocity, gradient, hessian, integral = [np.asarray(a, dtype=float) for a in (face_velocity, face_gradient, face_hessian, cell_velocity_integral)]
    if velocity.ndim == 2:
        velocity, gradient, hessian, integral = velocity[None], gradient[None], hessian[None], integral[None]
    if (velocity.ndim != 3 or velocity.shape[-1] != 3 or gradient.shape != (*velocity.shape, 3)
            or hessian.shape != (*velocity.shape, 3, 3) or integral.shape != (len(velocity), native.n_cells, 3)
            or not all(np.all(np.isfinite(a)) for a in (velocity, gradient, hessian, integral))):
        raise ValueError("Compatible finite quadratic face polynomials and cell velocity integrals are required")
    if not np.allclose(hessian, hessian.swapaxes(-2, -3), rtol=0, atol=1e-12):
        raise ValueError("Face Hessian must be symmetric in its derivative axes")
    triangle_centre = native.triangles.mean(axis=1)
    vector_area = np.cross(native.triangles[:, 1]-native.triangles[:, 0], native.triangles[:, 2]-native.triangles[:, 0])/2
    relative = native.triangles-triangle_centre[:, None]
    second = np.einsum("tvi,tvj->tij", relative, relative)/12
    third = np.einsum("tvi,tvj,tvk->tijk", relative, relative, relative)/30
    delta = triangle_centre-np.asarray(face_origin)[native.face_ids]
    mean_tensor = second+delta[:, :, None]*delta[:, None]
    gamma, moment = np.zeros((len(velocity), native.n_cells, 3)), np.zeros((len(velocity), native.n_cells, 3, 3))
    for state in range(len(velocity)):
        g, h = gradient[state, native.face_ids], hessian[state, native.face_ids]
        mean = velocity[state, native.face_ids]+np.einsum("ti,tij->tj", delta, g)+.5*np.einsum("tik,tikj->tj", mean_tensor, h)
        central_g = g+np.einsum("tk,tikj->tij", delta, h)
        local_u = second @ central_g+.5*np.einsum("tikl,tklj->tij", third, h)
        local = np.cross(vector_area[:, None], local_u)
        flux = np.cross(vector_area, mean)
        for ids, sign, rows in ((native.owners, 1, np.ones(len(native.owners), dtype=bool)),
                                (native.neighbours, -1, native.neighbours >= 0)):
            ids = ids[rows]
            np.add.at(gamma[state], ids, sign*flux[rows])
            contribution = local[rows]+(triangle_centre[rows]-centroid[ids])[:, :, None]*flux[rows, None]
            np.add.at(moment[state], ids, sign*contribution)
    moment -= np.cross(np.eye(3)[None, None], integral[:, :, None])
    return gamma, moment
