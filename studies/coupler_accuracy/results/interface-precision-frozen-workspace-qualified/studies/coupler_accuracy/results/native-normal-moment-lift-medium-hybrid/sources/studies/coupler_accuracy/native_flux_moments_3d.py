"""Normal-flux integrals and first moments on actual native face triangles.

These observations do not construct a volume velocity. For a divergence-free
reconstruction, its cell integral must equal the first boundary-flux moment
about any cell origin. Total face flux alone does not specify that moment.
"""

from __future__ import annotations

import numpy as np


def triangle_moment_geometry(native):
    triangle = native.triangles
    centre = triangle.mean(axis=1)
    vector_area = .5 * np.cross(triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0])
    relative = triangle - centre[:, None]
    covariance = np.einsum("tvi,tvj->tij", relative, relative) / 12
    return centre, vector_area, covariance


def linear_face_flux_moments(native, face_origin, face_velocity, face_gradient=None):
    """Return integral u.n and integral (x-origin) u.n on every native face.

    The shared trace is u(x) = face_velocity + (x-face_origin) @ face_gradient;
    gradients have derivative index first. Integration is exact for this
    affine trace on the face's arithmetic-centre fan, including warped faces.
    """
    origin, value = np.asarray(face_origin, dtype=float), np.asarray(face_velocity, dtype=float)
    if (origin.ndim != 2 or origin.shape[1] != 3 or value.shape != origin.shape
            or not np.all(np.isfinite(origin)) or not np.all(np.isfinite(value))):
        raise ValueError("Finite origins and vector velocities on every face are required")
    if len(origin) != int(np.max(native.face_ids)) + 1:
        raise ValueError("Native triangle face indices must cover the supplied faces")
    if face_gradient is not None:
        gradient = np.asarray(face_gradient, dtype=float)
        if gradient.shape != (len(origin), 3, 3) or not np.all(np.isfinite(gradient)):
            raise ValueError("Finite derivative-first face gradients are required")
    centre, vector_area, covariance = triangle_moment_geometry(native)
    ids = native.face_ids
    relative = centre - origin[ids]
    mean = value[ids].copy()
    local_moment = np.zeros_like(mean)
    if face_gradient is not None:
        gradient = gradient[ids]
        mean += np.einsum("ti,tij->tj", relative, gradient)
        local_moment = np.einsum("tij,tjk,tk->ti", covariance, gradient, vector_area)
    flux = np.einsum("ti,ti->t", mean, vector_area)
    result = np.bincount(ids, weights=flux, minlength=len(origin))
    first = np.zeros_like(origin)
    np.add.at(first, ids, relative * flux[:, None] + local_moment)
    return result, first


def enforce_face_flux(native, face_origin, flux, first_moment, target_flux):
    """Add a constant scalar normal trace on each polygon to match its flux.

    On each fan triangle the velocity adjustment is delta_un * n_triangle.
    Thus its normal trace is constant even on a warped face. This changes no
    first moment about the true area centroid, apart from summation roundoff.
    A general supplied origin is supported explicitly.
    """
    origin = np.asarray(face_origin, dtype=float)
    flux, target = np.asarray(flux, dtype=float), np.asarray(target_flux, dtype=float)
    first = np.asarray(first_moment, dtype=float)
    if (origin.ndim != 2 or origin.shape[1] != 3 or first.shape != origin.shape
            or flux.shape != (len(origin),) or target.shape != flux.shape
            or not all(np.all(np.isfinite(a)) for a in (origin, flux, target, first))):
        raise ValueError("Finite compatible face origins, fluxes and moments are required")
    centre, vector_area, _ = triangle_moment_geometry(native)
    area = np.linalg.norm(vector_area, axis=1)
    ids = native.face_ids
    face_area = np.bincount(ids, weights=area, minlength=len(origin))
    if np.any(face_area <= 0):
        raise ValueError("Every face needs positive triangle area")
    area_moment = np.zeros_like(origin)
    np.add.at(area_moment, ids, area[:, None] * (centre - origin[ids]))
    delta_un = (target - flux) / face_area
    return target.copy(), first + delta_un[:, None] * area_moment, delta_un


def cell_flux_moments(mesh, face_origin, face_flux, face_first_moment, cell_origin):
    """Accumulate net outward flux and its first moment about each cell origin.

    The first moment equals integral u + integral (x-cell_origin) div(u).
    It is a cell velocity integral only if that divergence moment vanishes.
    """
    n, ni = mesh["n_cells"], mesh["n_interior_faces"]
    origin, centre = np.asarray(face_origin), np.asarray(cell_origin)
    flux, moment = np.asarray(face_flux), np.asarray(face_first_moment)
    if (origin.shape != (mesh["n_faces"], 3) or centre.shape != (n, 3)
            or moment.shape != origin.shape or flux.shape != (len(origin),)
            or not all(np.all(np.isfinite(a)) for a in (origin, centre, flux, moment))):
        raise ValueError("Finite compatible cell and face moment data are required")
    net, first = np.zeros(n), np.zeros((n, 3))
    for ids, sign, rows in ((mesh["owners"], 1, slice(None)), (mesh["neighbours"], -1, slice(ni))):
        np.add.at(net, ids, sign * flux[rows])
        contribution = (origin[rows] - centre[ids]) * flux[rows, None] + moment[rows]
        np.add.at(first, ids, sign * contribution)
    return net, first
