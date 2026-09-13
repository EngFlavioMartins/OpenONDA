"""Smooth, solenoidal 3D velocity with exact no slip on all unit-cube faces.

A = c psi, psi = amplitude prod_i [(x_i^2-a^2)^2 exp(-x_i^2/(2 width^2))].
Then u = grad(psi) cross c and omega = Hessian(psi)c - c Laplacian(psi).
Every factor and its first derivative vanish at x_i = +/-a. All components
and all spatial directions are active; this is a kinematic manufactured field.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.polynomial import Polynomial

from studies.coupler_accuracy.native_velocity_curl_integrals_3d import triangle_rule


@dataclass
class NoSlipCubeField:
    width: float
    half_width: float = 0.5
    direction: np.ndarray = field(default_factory=lambda: np.array([1., 2., 3.])/np.sqrt(14))
    amplitude: float = field(init=False, default=1.)

    def __post_init__(self):
        if not np.isfinite(self.width) or self.width <= 0 or not np.isfinite(self.half_width) or self.half_width <= 0:
            raise ValueError("Positive finite length scales are required")
        self.direction = np.asarray(self.direction, dtype=float)
        if self.direction.shape != (3,) or not np.all(np.isfinite(self.direction)) or np.linalg.norm(self.direction) == 0:
            raise ValueError("A finite nonzero three-component potential direction is required")
        self.direction = self.direction/np.linalg.norm(self.direction)
        self.amplitude = 1/np.linalg.norm(self.velocity(self.reference_position[None])[0])

    @property
    def reference_position(self):
        return np.array([self.half_width+self.width**2/self.half_width, 0.2*self.width, 0.35*self.width])

    def factors(self, points, *, second=False):
        points = np.asarray(points)
        q = points**2-self.half_width**2
        p, dp = q*q, 4*points*q
        beta = 1/self.width**2
        exponential = np.exp(-0.5*beta*points**2)
        f = p*exponential
        df = (dp-beta*points*p)*exponential
        if not second:
            return f, df
        d2p = 12*points**2-4*self.half_width**2
        d2f = (d2p-2*beta*points*dp+(beta**2*points**2-beta)*p)*exponential
        return f, df, d2f

    def potential_gradient(self, points):
        f, df = self.factors(points)
        psi = self.amplitude*np.prod(f, axis=-1)
        gradient = np.stack((df[..., 0]*f[..., 1]*f[..., 2],
                             f[..., 0]*df[..., 1]*f[..., 2],
                             f[..., 0]*f[..., 1]*df[..., 2]), axis=-1)*self.amplitude
        return psi, gradient

    def velocity(self, points):
        return np.cross(self.potential_gradient(points)[1], self.direction)

    def vorticity(self, points):
        f, df, d2f = self.factors(points, second=True)
        hessian = np.empty((*f.shape[:-1], 3, 3), dtype=np.result_type(f, float))
        for axis in range(3):
            other = [j for j in range(3) if j != axis]
            hessian[..., axis, axis] = d2f[..., axis]*f[..., other[0]]*f[..., other[1]]
            for j in range(axis+1, 3):
                third = 3-axis-j
                hessian[..., axis, j] = df[..., axis]*df[..., j]*f[..., third]
                hessian[..., j, axis] = hessian[..., axis, j]
        return self.amplitude*(hessian @ self.direction-np.trace(hessian, axis1=-2, axis2=-1)[..., None]*self.direction)

    def outer_box_velocity_bound(self, bounds):
        """Conservative component-product bound using all 1D stationary points.

        The Gaussian-polynomial factors vanish at infinity. Roots of f' and
        f'' therefore locate their finite global maxima. This is a floating-
        point evaluation of the bound, not interval arithmetic.
        """
        p = Polynomial([self.half_width**4, 0, -2*self.half_width**2, 0, 1])
        multiplier = Polynomial([0, -1/self.width**2])
        derivative = p.deriv()+multiplier*p
        second = derivative.deriv()+multiplier*derivative
        stationary = []
        for polynomial in (derivative, second):
            roots = polynomial.roots()
            stationary.append(np.asarray([x.real for x in roots if abs(x.imag) < 1e-10]))
        fmax = np.max(np.abs(self.factors(stationary[0])[0]))
        dfmax = np.max(np.abs(self.factors(stationary[1])[1]))
        fixed, dfixed = self.factors(np.asarray(bounds))
        bound = self.amplitude*np.hypot(dfixed*fmax*fmax, np.sqrt(2)*fixed*dfmax*fmax)
        return float(np.max(bound))


def accumulate_faces(mesh, values):
    """Accumulate oriented per-face vectors into their native cell integrals."""
    values = np.asarray(values)
    result = np.zeros((mesh["n_cells"], *values.shape[1:]))
    np.add.at(result, mesh["owners"], values)
    np.add.at(result, mesh["neighbours"], -values[:mesh["n_interior_faces"]])
    return result


def native_manufactured_integrals(native, mesh, fields, *, order, chunk_size=128, progress=None):
    """Integrate u and omega in all cells via A and u on actual face fans.

    integral u = integral_boundary psi (n cross c),
    integral omega = integral_boundary n cross u.
    Also retain the face integral of velocity for an aggregate-face control.
    No normalization, support threshold or source cutoff is applied.
    """
    barycentric, weight = triangle_rule(order)
    face_u = np.zeros((mesh["n_faces"], len(fields), 3))
    face_curl = np.zeros_like(face_u)
    face_potential_cross = np.zeros_like(face_u)
    for start in range(0, len(native.triangles), chunk_size):
        rows = slice(start, start+chunk_size)
        triangle = native.triangles[rows]
        # Anchored coordinates preserve an exactly constant face coordinate,
        # including x_i=+/-a on the no-slip cube, despite barycentric roundoff.
        points = triangle[:, None, 0]+np.einsum("qa,tad->tqd", barycentric[:, 1:],
                                               triangle[:, 1:]-triangle[:, :1])
        vector_area = np.cross(triangle[:, 1]-triangle[:, 0], triangle[:, 2]-triangle[:, 0])/2
        area = np.linalg.norm(vector_area, axis=1)
        for index, field_ in enumerate(fields):
            psi, gradient = field_.potential_gradient(points)
            average_psi = psi @ weight
            average_u = np.einsum("q,tqi->ti", weight, np.cross(gradient, field_.direction))
            np.add.at(face_u[:, index], native.face_ids[rows], area[:, None]*average_u)
            np.add.at(face_curl[:, index], native.face_ids[rows], np.cross(vector_area, average_u))
            np.add.at(face_potential_cross[:, index], native.face_ids[rows],
                      average_psi[:, None]*np.cross(vector_area, field_.direction))
        if progress and (start % 16384 == 0 or start+chunk_size >= len(native.triangles)):
            progress(min(start+chunk_size, len(native.triangles)), len(native.triangles), order)
    return {"velocity_integral": accumulate_faces(mesh, face_potential_cross).transpose(1, 0, 2),
            "vorticity_integral": accumulate_faces(mesh, face_curl).transpose(1, 0, 2),
            "face_velocity_integral": face_u.transpose(1, 0, 2)}
