"""Direct Biot--Savart induction from constant fields in native 3D cells.

For r = target - source, integral_cell r/|r|^3 dV equals
sum_face n integral_face 1/|r| dA. Each actual face-fan triangle is integrated
analytically. Interior faces carry the owner-minus-neighbour field jump;
there is no Gaussian core, point-cell approximation or source threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from numba import njit
import numpy as np


@njit(cache=True)
def _triangle_integrals(point, vertices, normal, tangents, outward, lengths):
    """Return positive single-layer integral P and vector K = -grad(P).

    K is the principal-value trace at a panel-interior point and undefined
    (NaN) on an edge. P is finite on the whole closed triangle. A stable
    endpoint-distance sum avoids subtracting almost equal lengths near edges.
    """
    x0, y0, z0 = vertices[0] - point
    x1, y1, z1 = vertices[1] - point
    x2, y2, z2 = vertices[2] - point
    d0 = math.sqrt(x0*x0+y0*y0+z0*z0)
    d1 = math.sqrt(x1*x1+y1*y1+z1*z1)
    d2 = math.sqrt(x2*x2+y2*y2+z2*z2)
    height = x0*normal[0]+y0*normal[1]+z0*normal[2]
    determinant = x0*(y1*z2-z1*y2)+y0*(z1*x2-x1*z2)+z0*(x1*y2-y1*x2)
    denominator = (d0*d1*d2+(x0*x1+y0*y1+z0*z1)*d2
                   + (x1*x2+y1*y2+z1*z2)*d0+(x2*x0+y2*y0+z2*z0)*d1)
    omega = 2*math.atan2(determinant, denominator) if height != 0 else 0.
    potential = -height*omega
    kx, ky, kz = -omega*normal[0], -omega*normal[1], -omega*normal[2]
    distances = (d0, d1, d2)
    for edge in range(3):
        rx, ry, rz = vertices[edge]-point
        tx, ty, tz = tangents[edge]
        mx, my, mz = outward[edge]
        length = lengths[edge]
        along = rx*tx+ry*ty+rz*tz
        transverse = rx*mx+ry*my+rz*mz
        cx, cy, cz = ry*tz-rz*ty, rz*tx-rx*tz, rx*ty-ry*tx
        perpendicular2 = cx*cx+cy*cy+cz*cz
        a, b = distances[edge], distances[(edge+1) % 3]
        first = a+along
        last = b-along-length
        if along < 0:
            first = perpendicular2/(a-along)
        if along+length > 0:
            last = perpendicular2/(b+along+length)
        distance_sum = first+last
        if distance_sum > 0:
            logarithm = math.log1p(2*length/distance_sum)
            potential += transverse*logarithm
            kx += mx*logarithm
            ky += my*logarithm
            kz += mz*logarithm
        else:
            # On the edge, transverse*log(...) has the finite limit zero.
            # The vector surface integral itself is logarithmically singular.
            kx, ky, kz = np.nan, np.nan, np.nan
    return potential, kx, ky, kz


def triangle_geometry(triangles):
    triangles = np.ascontiguousarray(triangles, dtype=float)
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3) or not np.all(np.isfinite(triangles)):
        raise ValueError("Finite three-dimensional triangles are required")
    edges = np.roll(triangles, -1, axis=1)-triangles
    lengths = np.linalg.norm(edges, axis=2)
    vector = np.cross(edges[:, 0], -edges[:, 2])
    double_area = np.linalg.norm(vector, axis=1)
    if np.any(double_area <= 0) or np.any(lengths <= 0):
        raise ValueError("Degenerate triangles are unsupported")
    normals = vector/double_area[:, None]
    tangents = edges/lengths[:, :, None]
    outward = np.cross(tangents, normals[:, None])
    return triangles, normals, tangents, outward, lengths


@njit(cache=True)
def _sample_integrals(points, triangles, normals, tangents, outward, lengths):
    potential = np.empty((len(points), len(triangles)))
    vector = np.empty((len(points), len(triangles), 3))
    for i in range(len(points)):
        for j in range(len(triangles)):
            p, x, y, z = _triangle_integrals(points[i], triangles[j], normals[j],
                                             tangents[j], outward[j], lengths[j])
            potential[i, j] = p
            vector[i, j, 0], vector[i, j, 1], vector[i, j, 2] = x, y, z
    return potential, vector


def triangle_source_integrals(points, triangles):
    """Return integral_triangle 1/r and integral_triangle r/r^3 (no 4*pi)."""
    points = np.ascontiguousarray(points, dtype=float).reshape(-1, 3)
    if not np.all(np.isfinite(points)):
        raise ValueError("Finite targets are required")
    return _sample_integrals(points, *triangle_geometry(triangles))


@njit(cache=True)
def _accumulate_potential(points, triangles, normals, tangents, outward, lengths, coefficients):
    result = np.zeros((len(points), coefficients.shape[1], 3))
    for i in range(len(points)):
        for j in range(len(triangles)):
            potential = _triangle_integrals(points[i], triangles[j], normals[j],
                                             tangents[j], outward[j], lengths[j])[0]/(4*np.pi)
            for state in range(coefficients.shape[1]):
                for axis in range(3):
                    result[i, state, axis] += potential*coefficients[j, state, axis]
    return result


@dataclass
class NativeVolumeSources:
    triangles: np.ndarray
    normals: np.ndarray
    tangents: np.ndarray
    outward: np.ndarray
    lengths: np.ndarray
    owners: np.ndarray
    neighbours: np.ndarray
    face_ids: np.ndarray
    n_cells: int

    def cell_geometry(self, cell_centres):
        """Signed face-fan volume/centroid, distinct from FVM face summaries."""
        volume = np.zeros(self.n_cells)
        moment = np.zeros((self.n_cells, 3))
        for ids, sign, rows in ((self.owners, 1, np.ones(len(self.owners), dtype=bool)),
                                (self.neighbours, -1, self.neighbours >= 0)):
            ids = ids[rows]
            reference = np.asarray(cell_centres)[ids]
            relative = self.triangles[rows]-reference[:, None]
            tetra_volume = sign*np.einsum("ti,ti->t", relative[:, 0],
                                           np.cross(relative[:, 1], relative[:, 2]))/6
            centroid = (reference+self.triangles[rows].sum(axis=1))/4
            np.add.at(volume, ids, tetra_volume)
            np.add.at(moment, ids, tetra_volume[:, None]*centroid)
        if np.any(volume <= 0):
            raise ValueError("Nonpositive native polyhedron volume")
        return volume, moment/volume[:, None]

    @classmethod
    def from_mesh(cls, mesh):
        triangles, owners, neighbours, face_ids = [], [], [], []
        for face, own in enumerate(mesh["owners"]):
            vertices = mesh["vertex_position"][mesh["faces"][face]]
            fan = vertices.mean(axis=0)
            triangles.append(np.stack((np.broadcast_to(fan, vertices.shape), vertices,
                                       np.roll(vertices, -1, axis=0)), axis=1))
            owners.extend([own]*len(vertices))
            neighbour = mesh["neighbours"][face] if face < len(mesh["neighbours"]) else -1
            neighbours.extend([neighbour]*len(vertices))
            face_ids.extend([face]*len(vertices))
        return cls(*triangle_geometry(np.concatenate(triangles)), np.asarray(owners),
                   np.asarray(neighbours), np.asarray(face_ids), mesh["n_cells"])

    def coefficients(self, vorticity, divergence=None):
        """Return per-triangle vector coefficients for one or several fields.

        Input omega has shape (cells, 3) or (states, cells, 3). Optional scalar
        divergence adds its *diagnostic* gradient-potential contribution to
        those states; it is not a proposed incompressible VPM correction.
        """
        omega = np.asarray(vorticity, dtype=float)
        if omega.ndim == 2:
            omega = omega[None]
        if omega.ndim != 3 or omega.shape[1:] != (self.n_cells, 3) or not np.all(np.isfinite(omega)):
            raise ValueError("Vorticity must have one finite vector per native cell")
        jump = omega[:, self.owners].copy()
        interior = self.neighbours >= 0
        jump[:, interior] -= omega[:, self.neighbours[interior]]
        coefficients = np.cross(jump, self.normals[None])
        if divergence is not None:
            divergence = np.broadcast_to(np.asarray(divergence, dtype=float), omega.shape[:2])
            if not np.all(np.isfinite(divergence)):
                raise ValueError("Finite cell divergence is required")
            jump = divergence[:, self.owners].copy()
            jump[:, interior] -= divergence[:, self.neighbours[interior]]
            coefficients += jump[:, :, None]*self.normals[None]
        return np.ascontiguousarray(coefficients.transpose(1, 0, 2))

    def evaluate(self, points, coefficients, *, progress=None, chunk_size=32):
        """Stream shared-face induction; no target-by-source matrix is stored."""
        points = np.ascontiguousarray(points, dtype=float).reshape(-1, 3)
        coefficients = np.ascontiguousarray(coefficients, dtype=float)
        if (coefficients.ndim != 3 or coefficients.shape[0] != len(self.triangles)
                or coefficients.shape[2] != 3 or not np.all(np.isfinite(coefficients))
                or not np.all(np.isfinite(points)) or chunk_size < 1):
            raise ValueError("Finite compatible targets/coefficients and positive chunk size are required")
        result = np.empty((len(points), coefficients.shape[1], 3))
        geometry = (self.triangles, self.normals, self.tangents, self.outward, self.lengths)
        for start in range(0, len(points), chunk_size):
            result[start:start+chunk_size] = _accumulate_potential(
                points[start:start+chunk_size], *geometry, coefficients)
            if progress:
                progress(min(start+chunk_size, len(points)), len(points))
        return result


def piecewise_constant_boundary_completion(points, triangles, velocity):
    """Bounded Helmholtz boundary term, with normals outward from fluid.

    Return -integral[((n.u) r + (n cross u) cross r)/(4*pi*r^3)] dA.
    Velocity is constant on each triangle. Targets must be off the surface.
    """
    triangles, normals, tangents, outward, lengths = triangle_geometry(triangles)
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    velocity = np.broadcast_to(np.asarray(velocity, dtype=float), (len(triangles), 3))
    result = np.zeros_like(points)
    for start in range(0, len(points), 32):
        targets = points[start:start+32]
        height = np.einsum("pti,ti->pt", targets[:, None]-triangles[None, :, 0], normals)
        if np.any(height == 0):
            # Conservative contract: points coplanar with any source triangle
            # should be handled by an explicit trace convention elsewhere.
            raise ValueError("Boundary-completion targets must be off every source plane")
        vector = _sample_integrals(targets, triangles, normals, tangents, outward, lengths)[1]/(4*np.pi)
        normal_velocity = np.sum(normals*velocity, axis=1)
        result[start:start+32] = -np.sum(normal_velocity[None, :, None]*vector
                                        + np.cross(np.cross(normals, velocity)[None], vector), axis=1)
    return result
