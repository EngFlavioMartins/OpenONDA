"""Direct 3D Biot--Savart induction retaining cell circulation and first moment.

For Omega(y) = Gamma/V + (y-c) @ B, B = S^{-1} M, where
S = integral (y-c)(y-c)^T and M = integral (y-c) Omega^T.
Integration by parts gives a linear face term plus curl(Omega) times the
cell Newton potential. No solenoidality constraint on the raw P1 density is
implied; its induced velocity is the Biot--Savart projection of that density.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from numba import njit
import numpy as np

from studies.coupler_accuracy.native_volume_induction_3d import (
    NativeVolumeSources,
    triangle_geometry,
)


@njit(cache=True)
def _triangle_potential_moment(point, vertices, normal, tangents, outward, lengths):
    """Return P=int 1/r, T=int (y-triangle_centroid)/r and signed height.

    In-plane integration by parts gives int y/r = projection(point)*P
    + sum_edge m int_edge r dl. The edge primitive is evaluated with a
    rationalized endpoint-distance difference and the finite edge limit.
    """
    x0, y0, z0 = vertices[0]-point
    x1, y1, z1 = vertices[1]-point
    x2, y2, z2 = vertices[2]-point
    d0 = math.sqrt(x0*x0+y0*y0+z0*z0)
    d1 = math.sqrt(x1*x1+y1*y1+z1*z1)
    d2 = math.sqrt(x2*x2+y2*y2+z2*z2)
    height = x0*normal[0]+y0*normal[1]+z0*normal[2]
    determinant = x0*(y1*z2-z1*y2)+y0*(z1*x2-x1*z2)+z0*(x1*y2-y1*x2)
    denominator = (d0*d1*d2+(x0*x1+y0*y1+z0*z1)*d2
                   + (x1*x2+y1*y2+z1*z2)*d0+(x2*x0+y2*y0+z2*z0)*d1)
    solid_angle = 2*math.atan2(determinant, denominator) if height != 0 else 0.
    potential = -height*solid_angle
    moment_x, moment_y, moment_z = 0., 0., 0.
    distance = (d0, d1, d2)
    for edge in range(3):
        rx, ry, rz = vertices[edge]-point
        tx, ty, tz = tangents[edge]
        mx, my, mz = outward[edge]
        length = lengths[edge]
        along = rx*tx+ry*ty+rz*tz
        transverse = rx*mx+ry*my+rz*mz
        cx, cy, cz = ry*tz-rz*ty, rz*tx-rx*tz, rx*ty-ry*tx
        perpendicular2 = cx*cx+cy*cy+cz*cz
        a, b = distance[edge], distance[(edge+1) % 3]
        first, last = a+along, b-along-length
        if along < 0:
            first = perpendicular2/(a-along)
        if along+length > 0:
            last = perpendicular2/(b+along+length)
        distance_sum = first+last
        logarithm = math.log1p(2*length/distance_sum) if distance_sum > 0 else 0.
        potential += transverse*logarithm
        radial_integral = .5*(length*b+along*length*(2*along+length)/(a+b)
                              + perpendicular2*logarithm)
        moment_x += mx*radial_integral
        moment_y += my*radial_integral
        moment_z += mz*radial_integral
    moment_x += (-(x0+x1+x2)/3+height*normal[0])*potential
    moment_y += (-(y0+y1+y2)/3+height*normal[1])*potential
    moment_z += (-(z0+z1+z2)/3+height*normal[2])*potential
    return potential, moment_x, moment_y, moment_z, height


@njit(cache=True)
def _sample_moments(points, triangles, normals, tangents, outward, lengths):
    potential = np.empty((len(points), len(triangles)))
    moment = np.empty((len(points), len(triangles), 3))
    for i in range(len(points)):
        for j in range(len(triangles)):
            p, x, y, z, _ = _triangle_potential_moment(points[i], triangles[j], normals[j], tangents[j], outward[j], lengths[j])
            potential[i, j] = p
            moment[i, j, 0], moment[i, j, 1], moment[i, j, 2] = x, y, z
    return potential, moment


def triangle_potential_moments(points, triangles):
    return _sample_moments(np.ascontiguousarray(points, dtype=float).reshape(-1, 3), *triangle_geometry(triangles))


def cell_covariance(native, centroid):
    """Exact signed-tetrahedron second central moment of the face-fan cells."""
    covariance = np.zeros((native.n_cells, 3, 3))
    for ids, sign, rows in ((native.owners, 1, np.ones(len(native.owners), dtype=bool)),
                            (native.neighbours, -1, native.neighbours >= 0)):
        ids = ids[rows]
        relative = native.triangles[rows]-centroid[ids, None]
        tetra_volume = sign*np.einsum("ti,ti->t", relative[:, 0], np.cross(relative[:, 1], relative[:, 2]))/6
        total = relative.sum(axis=1)
        tensor = (total[:, :, None]*total[:, None]+np.einsum("tvi,tvj->tij", relative, relative))/20
        np.add.at(covariance, ids, tetra_volume[:, None, None]*tensor)
    if np.any(np.linalg.eigvalsh(covariance) <= 0):
        raise ValueError("Native cell covariance must be positive definite")
    return covariance


def curl_affine(gradient):
    """Curl with derivative axis first and vector component last."""
    return np.stack((gradient[..., 1, 2]-gradient[..., 2, 1],
                     gradient[..., 2, 0]-gradient[..., 0, 2],
                     gradient[..., 0, 1]-gradient[..., 1, 0]), axis=-1)


@njit(cache=True)
def _accumulate_linear(points, triangles, normals, tangents, outward, lengths, coefficients):
    result = np.zeros((len(points), coefficients.shape[1], 3))
    for i in range(len(points)):
        for j in range(len(triangles)):
            potential, tx, ty, tz, height = _triangle_potential_moment(
                points[i], triangles[j], normals[j], tangents[j], outward[j], lengths[j])
            for state in range(coefficients.shape[1]):
                for component in range(3):
                    value = potential*(coefficients[j, state, 0, component]+height*coefficients[j, state, 4, component])
                    value += (tx*coefficients[j, state, 1, component]+ty*coefficients[j, state, 2, component]
                              + tz*coefficients[j, state, 3, component])
                    result[i, state, component] += value/(4*np.pi)
    return result


@dataclass
class LinearNativeVolumeSources:
    native: NativeVolumeSources
    volume: np.ndarray
    centroid: np.ndarray
    covariance: np.ndarray

    @classmethod
    def from_native(cls, native, cell_centres):
        volume, centroid = native.cell_geometry(cell_centres)
        return cls(native, volume, centroid, cell_covariance(native, centroid))

    def coefficients(self, circulation, first_moment):
        gamma, moment = np.asarray(circulation, dtype=float), np.asarray(first_moment, dtype=float)
        if gamma.ndim == 2:
            gamma, moment = gamma[None], moment[None]
        if (gamma.ndim != 3 or gamma.shape[1:] != (self.native.n_cells, 3)
                or moment.shape != (*gamma.shape, 3) or not np.all(np.isfinite(gamma)) or not np.all(np.isfinite(moment))):
            raise ValueError("Finite cell circulation and first central moment are required")
        gradient = np.linalg.solve(self.covariance[None], moment)
        omega = gamma/self.volume[None, :, None]
        own, nei = self.native.owners, self.native.neighbours
        interior = nei >= 0
        triangle_centre = self.native.triangles.mean(axis=1)
        jump = omega[:, own]+np.einsum("ti,stij->stj", triangle_centre-self.centroid[own], gradient[:, own])
        jump_gradient = gradient[:, own].copy()
        jump_curl = curl_affine(gradient)[:, own].copy()
        jump[:, interior] -= (omega[:, nei[interior]]
                             + np.einsum("ti,stij->stj", triangle_centre[interior]-self.centroid[nei[interior]], gradient[:, nei[interior]]))
        jump_gradient[:, interior] -= gradient[:, nei[interior]]
        jump_curl[:, interior] -= curl_affine(gradient)[:, nei[interior]]
        coefficients = np.empty((len(self.native.triangles), len(gamma), 5, 3))
        coefficients[:, :, 0] = np.cross(jump, self.native.normals[None]).transpose(1, 0, 2)
        coefficients[:, :, 1:4] = np.cross(jump_gradient, self.native.normals[None, :, None]).transpose(1, 0, 2, 3)
        coefficients[:, :, 4] = .5*jump_curl.transpose(1, 0, 2)
        return np.ascontiguousarray(coefficients), gradient

    def evaluate(self, points, coefficients, *, progress=None, chunk_size=16):
        points, coefficients = np.ascontiguousarray(points, dtype=float).reshape(-1, 3), np.ascontiguousarray(coefficients, dtype=float)
        if (coefficients.ndim != 4 or coefficients.shape[0] != len(self.native.triangles)
                or coefficients.shape[2:] != (5, 3) or not np.all(np.isfinite(points))
                or not np.all(np.isfinite(coefficients)) or chunk_size < 1):
            raise ValueError("Finite compatible targets/coefficients and a positive chunk size are required")
        result = np.empty((len(points), coefficients.shape[1], 3))
        native = self.native
        for start in range(0, len(points), chunk_size):
            result[start:start+chunk_size] = _accumulate_linear(
                points[start:start+chunk_size], native.triangles, native.normals, native.tangents,
                native.outward, native.lengths, coefficients)
            if progress:
                progress(min(start+chunk_size, len(points)), len(points))
        return result
