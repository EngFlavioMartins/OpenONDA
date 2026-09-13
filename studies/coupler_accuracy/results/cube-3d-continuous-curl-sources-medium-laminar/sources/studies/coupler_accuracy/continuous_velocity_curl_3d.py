"""A continuous native-cell velocity reconstruction with a solenoidal curl.

Each face uses its existing vertex fan. Connecting those triangles to the true
cell centroid gives a conforming tetrahedral subdivision, checked explicitly.
Velocity is continuous and linear on each tetrahedron. Its piecewise constant
curl has continuous normal traces, including a zero extension when the velocity
trace is zero on the domain boundary. No claim of incompressible velocity is
made: Biot--Savart projects that velocity onto its solenoidal part.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from studies.coupler_accuracy.native_linear_volume_induction_3d import curl_affine
from studies.coupler_accuracy.native_shared_trace_update_3d import compact_face_weights
from studies.coupler_accuracy.native_volume_induction_3d import (
    NativeVolumeSources,
    triangle_geometry,
)


@dataclass
class ContinuousVelocityCurl:
    position: np.ndarray
    tetrahedra: np.ndarray
    parent: np.ndarray
    volume: np.ndarray
    centroid: np.ndarray
    inverse_edges: np.ndarray
    cell_centroid: np.ndarray
    cell_volume: np.ndarray
    n_vertices: int
    n_faces: int
    boundary_nodes: np.ndarray
    face_nodes: np.ndarray
    source: NativeVolumeSources

    @property
    def cell_offset(self):
        return self.n_vertices+self.n_faces

    @classmethod
    def from_mesh(cls, mesh, cell_centroid):
        centre = np.asarray(cell_centroid, dtype=float)
        if centre.shape != (mesh["n_cells"], 3) or not np.all(np.isfinite(centre)):
            raise ValueError("Finite native cell centroids are required")
        vertices = np.asarray(mesh["vertex_position"], dtype=float)
        nv, nf = len(vertices), mesh["n_faces"]
        face_centre = np.array([vertices[face].mean(axis=0) for face in mesh["faces"]])
        position = np.vstack((vertices, face_centre, centre))
        tetrahedra, parent, boundary = [], [], []
        for face, ids in enumerate(mesh["faces"]):
            fan = np.column_stack((np.full(len(ids), nv+face), ids, np.roll(ids, -1)))
            own = int(mesh["owners"][face])
            tetrahedra.append(np.column_stack((np.full(len(ids), nv+nf+own), fan)))
            parent.extend([own]*len(ids))
            if face < mesh["n_interior_faces"]:
                nei = int(mesh["neighbours"][face])
                tetrahedra.append(np.column_stack((np.full(len(ids), nv+nf+nei), fan[:, [0, 2, 1]])))
                parent.extend([nei]*len(ids))
            else:
                boundary.append(fan)
        tetrahedra, parent = np.vstack(tetrahedra), np.asarray(parent)
        points = position[tetrahedra]
        edges = points[:, 1:]-points[:, :1]
        volume = np.linalg.det(edges)/6
        if np.any(volume <= 0):
            raise ValueError("Every face-fan tetrahedron must have positive volume")
        local_faces = np.array([[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]])
        oriented = tetrahedra[:, local_faces].reshape(-1, 3)
        canonical, first, inverse, count = np.unique(np.sort(oriented, axis=1), axis=0,
                                                     return_index=True, return_inverse=True, return_counts=True)
        if np.any(count > 2):
            raise ValueError("Nonmanifold tetrahedral faces are unsupported")
        expected_boundary = np.sort(np.vstack(boundary), axis=1)
        expected_boundary = np.unique(expected_boundary, axis=0)
        if not np.array_equal(canonical[count == 1], expected_boundary):
            raise ValueError("The native face fans do not form a conforming tetrahedral subdivision")
        face_nodes = oriented[first]
        owner = first//4
        neighbour = np.full(len(first), -1, dtype=int)
        incident_sum = np.zeros(len(first), dtype=int)
        np.add.at(incident_sum, inverse, np.repeat(np.arange(len(tetrahedra)), 4))
        neighbour[count == 2] = incident_sum[count == 2]-owner[count == 2]
        source = NativeVolumeSources(*triangle_geometry(position[face_nodes]), owner, neighbour,
                                     np.arange(len(first)), len(tetrahedra))
        cell_volume = np.bincount(parent, weights=volume, minlength=mesh["n_cells"])
        return cls(position, tetrahedra, parent, volume, points.mean(axis=1), np.linalg.inv(edges),
                   centre, cell_volume, nv, nf, np.unique(expected_boundary), face_nodes, source)

    def complete_cell_integrals(self, trace_velocity, cell_integral):
        """Set each cell's interior node to enforce its vector integral.

        The centre basis function has integral V_cell/4; all remaining nodes
        lie on shared faces. Altering this one node leaves the trace untouched.
        """
        trace, integral = np.asarray(trace_velocity, dtype=float), np.asarray(cell_integral, dtype=float)
        if trace.ndim == 2:
            trace, integral = trace[None], integral[None]
        if (trace.ndim != 3 or trace.shape[1:] != (self.cell_offset, 3)
                or integral.shape != (len(trace), len(self.cell_centroid), 3)
                or not np.all(np.isfinite(trace)) or not np.all(np.isfinite(integral))):
            raise ValueError("Finite shared-node velocities and compatible cell integrals are required")
        velocity = np.zeros((len(trace), len(self.position), 3))
        velocity[:, :self.cell_offset] = trace
        _, first = np.unique(self.parent, return_index=True)
        for state, value in enumerate(trace):
            reference = value[self.tetrahedra[first, 1]]
            difference = value[self.tetrahedra[:, 1:]]-reference[self.parent, None]
            part = self.volume[:, None]*difference.sum(axis=1)/4
            summed = np.zeros_like(integral[state])
            np.add.at(summed, self.parent, part)
            residual = integral[state]-self.cell_volume[:, None]*reference
            velocity[state, self.cell_offset:] = reference+4*(residual-summed)/self.cell_volume[:, None]
        return velocity

    def gradient(self, velocity):
        value = np.asarray(velocity, dtype=float)
        if value.ndim == 2:
            value = value[None]
        if value.shape[1:] != self.position.shape or not np.all(np.isfinite(value)):
            raise ValueError("Finite velocities at every shared node are required")
        nodal = value[:, self.tetrahedra]
        return np.einsum("tij,stjk->stik", self.inverse_edges, nodal[:, :, 1:]-nodal[:, :, :1])

    def curl(self, velocity):
        return curl_affine(self.gradient(velocity))

    def velocity_integrals(self, velocity):
        value = np.asarray(velocity, dtype=float)
        if value.ndim == 2:
            value = value[None]
        result = np.zeros((len(value), len(self.cell_centroid), 3))
        for state in range(len(value)):
            np.add.at(result[state], self.parent, self.volume[:, None]*value[state, self.tetrahedra].mean(axis=1))
        return result

    def curl_moments(self, vorticity):
        omega = np.asarray(vorticity, dtype=float)
        if omega.ndim == 2:
            omega = omega[None]
        if omega.shape[1:] != (len(self.tetrahedra), 3) or not np.all(np.isfinite(omega)):
            raise ValueError("Finite constant vorticity in each tetrahedron is required")
        gamma = np.zeros((len(omega), len(self.cell_centroid), 3))
        moment = np.zeros((*gamma.shape, 3))
        relative = self.centroid-self.cell_centroid[self.parent]
        for state in range(len(omega)):
            circulation = self.volume[:, None]*omega[state]
            np.add.at(gamma[state], self.parent, circulation)
            np.add.at(moment[state], self.parent, relative[:, :, None]*circulation[:, None])
        return gamma, moment

    def normal_curl_jumps(self, vorticity):
        omega = np.asarray(vorticity, dtype=float)
        if omega.ndim == 2:
            omega = omega[None]
        jump = omega[:, self.source.owners].copy()
        interior = self.source.neighbours >= 0
        jump[:, interior] -= omega[:, self.source.neighbours[interior]]
        return np.einsum("sti,ti->st", jump, self.source.normals)

    def active_source(self, vorticity):
        """Remove exactly zero face coefficients, without a magnitude cutoff."""
        coefficient = self.source.coefficients(vorticity)
        active = np.any(coefficient != 0, axis=(1, 2))
        base = self.source
        compact = NativeVolumeSources(*(getattr(base, key)[active] for key in
            ("triangles", "normals", "tangents", "outward", "lengths", "owners", "neighbours", "face_ids")), base.n_cells)
        return compact, coefficient[active]


def compact_polynomial_trace(mesh, geometry, subdivision, cell_weight, face_polynomial):
    """Average one shared polynomial trace at each vertex and face-fan centre.

    Face-centre weights are minima over adjacent cells. Vertex weights are
    minima over all incident cells, keeping every zero-weight cell untouched.
    Vertex values average incident face-polynomial values by native face area.
    Physical boundary nodes are zero, making the correction compact there.
    The supplied polynomials are a velocity difference, not the total field.
    """
    velocity, gradient, hessian = [np.asarray(value, dtype=float) for value in face_polynomial]
    if velocity.ndim != 3 or velocity.shape[1:] != (mesh["n_faces"], 3):
        raise ValueError("Shared face differences must have an explicit state axis")
    if gradient.shape != (*velocity.shape, 3) or hessian.shape != (*velocity.shape, 3, 3):
        raise ValueError("Compatible three-dimensional polynomial differences are required")
    if not all(np.all(np.isfinite(value)) for value in (velocity, gradient, hessian)):
        raise ValueError("Finite face polynomial differences are required")
    weight = compact_face_weights(mesh, cell_weight)
    vertex_weight = np.ones(subdivision.n_vertices)
    vertex_area = np.zeros(subdivision.n_vertices)
    trace = np.zeros((len(velocity), subdivision.cell_offset, 3))
    for face, ids in enumerate(mesh["faces"]):
        ids = np.asarray(ids)
        np.minimum.at(vertex_weight, ids, weight[face])
        vertex_area[ids] += geometry["face_area"][face]
        for nodes, blend in ((ids, geometry["face_area"][face]),
                              (np.array([subdivision.n_vertices+face]), weight[face])):
            r = subdivision.position[nodes]-geometry["face_centre"][face]
            values = velocity[:, face, None]+np.einsum("pi,sij->spj", r, gradient[:, face])
            values += .5*np.einsum("pi,pk,sikj->spj", r, r, hessian[:, face])
            trace[:, nodes] += blend*values
    if np.any(vertex_area <= 0):
        raise ValueError("Every mesh vertex must belong to a face")
    trace[:, :subdivision.n_vertices] *= (vertex_weight/vertex_area)[None, :, None]
    trace[:, subdivision.boundary_nodes] = 0
    return trace, weight, vertex_weight
