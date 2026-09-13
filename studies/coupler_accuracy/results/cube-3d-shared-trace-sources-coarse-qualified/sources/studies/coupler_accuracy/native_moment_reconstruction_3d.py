"""Recover native-cell vorticity moments using neighbouring cells and faces.

These are offline reconstruction candidates. Neither consumes manufactured
first moments or target velocities. Boundary velocity traces are supplied
explicitly by the caller; no vorticity boundary data are invented.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CellLeastSquares:
    owners: np.ndarray
    neighbours: np.ndarray
    weighted_displacement: np.ndarray
    normal_matrix: np.ndarray
    condition: np.ndarray
    neighbour_count: np.ndarray

    @classmethod
    def from_mesh(cls, mesh, centres):
        centres = np.asarray(centres, dtype=float)
        if centres.shape != (mesh["n_cells"], 3) or not np.all(np.isfinite(centres)):
            raise ValueError("Finite 3D cell centres are required")
        own = np.asarray(mesh["owners"][:mesh["n_interior_faces"]], dtype=int)
        nei = np.asarray(mesh["neighbours"], dtype=int)
        displacement = centres[nei]-centres[own]
        distance2 = np.sum(displacement**2, axis=1)
        if np.any(distance2 <= 0):
            raise ValueError("Distinct centres are required across every interior face")
        weighted = displacement/distance2[:, None]
        normal = np.zeros((len(centres), 3, 3))
        count = np.zeros(len(centres), dtype=int)
        for ids in (own, nei):
            np.add.at(normal, ids, weighted[:, :, None]*displacement[:, None])
            np.add.at(count, ids, 1)
        eigenvalues = np.linalg.eigvalsh(normal)
        if np.any(eigenvalues[:, 0] <= 1e-12*eigenvalues[:, -1]) or np.any(count < 3):
            raise ValueError("Neighbour stencil does not span all three spatial directions")
        condition = eigenvalues[:, -1]/eigenvalues[:, 0]
        if np.max(condition) > 1e8:
            raise ValueError("Neighbour least-squares stencil requires a better-conditioned reconstruction")
        return cls(own, nei, weighted, normal, condition, count)

    def gradient(self, values):
        """Return derivative-first gradients of one or several cell fields."""
        values = np.asarray(values, dtype=float)
        if values.ndim == 2:
            values = values[None]
        if values.ndim != 3 or values.shape[1] != len(self.normal_matrix) or not np.all(np.isfinite(values)):
            raise ValueError("Finite vectors/components on the native cell stencil are required")
        result = np.empty((len(values), len(self.normal_matrix), 3, values.shape[2]))
        for index, field in enumerate(values):
            difference = field[self.neighbours]-field[self.owners]
            rhs = np.zeros_like(result[index])
            for ids in (self.owners, self.neighbours):
                np.add.at(rhs, ids, self.weighted_displacement[:, :, None]*difference[:, None])
            result[index] = np.linalg.solve(self.normal_matrix, rhs)
        return result


def weak_curl_moments(native, centroid, cell_velocity_integral, face_origin, face_velocity, face_gradient=None):
    """Integrate Stokes circulation and first moment of supplied linear traces.

    u_face(y) = face_velocity + (y-face_origin) @ face_gradient.
    The actual triangle mean and second moment retain warped-face geometry.
    M_ij = boundary integral (y-c)_i (n cross u)_j - (e_i cross integral u)_j.
    """
    face_velocity = np.asarray(face_velocity, dtype=float)
    cell_velocity_integral = np.asarray(cell_velocity_integral, dtype=float)
    if face_velocity.ndim == 2:
        face_velocity, cell_velocity_integral = face_velocity[None], cell_velocity_integral[None]
        if face_gradient is not None:
            face_gradient = np.asarray(face_gradient)[None]
    if (face_velocity.ndim != 3 or face_velocity.shape[2] != 3
            or cell_velocity_integral.shape != (len(face_velocity), native.n_cells, 3)
            or not np.all(np.isfinite(face_velocity)) or not np.all(np.isfinite(cell_velocity_integral))):
        raise ValueError("Compatible finite cell integrals and face velocities are required")
    if face_gradient is not None:
        face_gradient = np.asarray(face_gradient, dtype=float)
        if face_gradient.shape != (*face_velocity.shape, 3) or not np.all(np.isfinite(face_gradient)):
            raise ValueError("Compatible finite derivative-first face gradients are required")
    triangle_centre = native.triangles.mean(axis=1)
    vector_area = np.cross(native.triangles[:, 1]-native.triangles[:, 0], native.triangles[:, 2]-native.triangles[:, 0])/2
    relative = native.triangles-triangle_centre[:, None]
    covariance_per_area = np.einsum("tvi,tvj->tij", relative, relative)/12
    gamma = np.zeros((len(face_velocity), native.n_cells, 3))
    moment = np.zeros((len(face_velocity), native.n_cells, 3, 3))
    interior = native.neighbours >= 0
    for index in range(len(face_velocity)):
        mean = face_velocity[index, native.face_ids].copy()
        local = np.zeros((len(native.triangles), 3, 3))
        if face_gradient is not None:
            gradient = face_gradient[index, native.face_ids]
            mean += np.einsum("ti,tij->tj", triangle_centre-np.asarray(face_origin)[native.face_ids], gradient)
            local = np.cross(vector_area[:, None], covariance_per_area @ gradient)
        flux = np.cross(vector_area, mean)
        for ids, sign, rows in ((native.owners, 1, np.ones(len(native.owners), dtype=bool)),
                                (native.neighbours, -1, interior)):
            ids = ids[rows]
            np.add.at(gamma[index], ids, sign*flux[rows])
            contribution = local[rows]+(triangle_centre[rows]-centroid[ids])[:, :, None]*flux[rows, None]
            np.add.at(moment[index], ids, sign*contribution)
    moment -= np.cross(np.eye(3)[None, None], cell_velocity_integral[:, :, None])
    return gamma, moment


def reconstruct_linear_face_velocity(mesh, geometry, cell_velocity, cell_gradient, boundary_velocity, *, cell_positions=None):
    """A shared weighted linear trace, with explicit constant boundary traces.

    Boundary values are constant over each supplied face. This contract is
    appropriate for the manufactured no-slip cube and negligible outer field;
    it is not a general reconstruction of variable VPM boundary data.
    """
    value, gradient = np.asarray(cell_velocity), np.asarray(cell_gradient)
    if value.ndim == 2:
        value, gradient = value[None], gradient[None]
    boundary_velocity = np.broadcast_to(boundary_velocity, (len(value), mesh["n_faces"], 3))
    result = boundary_velocity.copy()
    result_gradient = np.zeros((len(value), mesh["n_faces"], 3, 3))
    count = mesh["n_interior_faces"]
    own, nei = mesh["owners"][:count], mesh["neighbours"]
    centre = geometry["cell_centre"] if cell_positions is None else np.asarray(cell_positions)
    target = geometry["face_centre"][:count]
    owner_trace = value[:, own]+np.einsum("fi,sfij->sfj", target-centre[own], gradient[:, own])
    neighbour_trace = value[:, nei]+np.einsum("fi,sfij->sfj", target-centre[nei], gradient[:, nei])
    w = geometry["face_interpolation_weight"][:count]
    result[:, :count] = (1-w[None, :, None])*owner_trace+w[None, :, None]*neighbour_trace
    result_gradient[:, :count] = ((1-w[None, :, None, None])*gradient[:, own]
                                  + w[None, :, None, None]*gradient[:, nei])
    return result, result_gradient
