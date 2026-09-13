"""Compact vorticity-moment corrections from shared weighted velocity traces.

These are source-reconstruction diagnostics. A zero boundary trace makes the
circulation increment sum to zero. The first spatial moment then has the
explicit budget -e_i cross sum(cell velocity-integral increment), without a
subsequent circulation or impulse projection.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from studies.coupler_accuracy.native_quadratic_moments_3d import weak_quadratic_curl_moments


def compact_face_weights(mesh, cell_weight):
    """Largest shared weight that does not exceed either adjacent cell weight.

    The minimum keeps a zero-weight cell outside the update's support. Weights
    are constant on each face polynomial; they are a discrete overlap rule,
    not a claim to integrate a continuous taper exactly within every face.
    """
    weight = np.asarray(cell_weight, dtype=float)
    if (weight.shape != (mesh["n_cells"],) or not np.all(np.isfinite(weight))
            or np.any(weight < 0) or np.any(weight > 1)):
        raise ValueError("Finite cell weights in [0, 1] are required")
    face_weight = weight[np.asarray(mesh["owners"])].copy()
    n = mesh["n_interior_faces"]
    face_weight[:n] = np.minimum(face_weight[:n], weight[np.asarray(mesh["neighbours"])])
    return face_weight


def global_first_moment(centroid, circulation, first_moment):
    """Return integral x_i omega_j from cell-centred first moments."""
    return np.sum(first_moment, axis=-3)+np.einsum("ni,...nj->...ij", centroid, circulation)


def impulse(first_spatial_moment):
    """Return half the integral of x cross omega, using the source moments."""
    m = np.asarray(first_spatial_moment)
    return .5*np.stack((m[..., 1, 2]-m[..., 2, 1], m[..., 2, 0]-m[..., 0, 2],
                        m[..., 0, 1]-m[..., 1, 0]), axis=-1)


@dataclass
class SharedTraceUpdate:
    circulation: np.ndarray
    first_moment: np.ndarray
    cell_velocity_integral_change: np.ndarray
    face_weight: np.ndarray


def shared_trace_update(mesh, native, centroid, cell_weight, face_origin,
                        base_cell_integral, new_cell_integral, base_faces, new_faces):
    """Curl a weighted change of shared quadratic face traces.

    Both face tuples contain (velocity, derivative-first gradient, Hessian),
    with explicit state axes. Cell-integral changes use the cell weight;
    face traces use the smaller weight of the two neighbours. Updated traces
    must vanish at every physical boundary. The resulting first-moment budget
    is determined by the supplied integral changes, including zero change when
    both reconstructions preserve the same per-cell momentum integrals.
    """
    weight = np.asarray(cell_weight, dtype=float)
    face_weight = compact_face_weights(mesh, weight)
    base_integral, new_integral = np.asarray(base_cell_integral), np.asarray(new_cell_integral)
    if (base_integral.ndim != 3 or base_integral.shape[1:] != (mesh["n_cells"], 3)
            or new_integral.shape != base_integral.shape
            or not np.all(np.isfinite(base_integral)) or not np.all(np.isfinite(new_integral))):
        raise ValueError("Finite cell velocity integrals with explicit state axes are required")
    if len(base_faces) != 3 or len(new_faces) != 3:
        raise ValueError("Velocity, gradient and Hessian face traces are required")
    change = []
    states = len(base_integral)
    for degree, (old, new) in enumerate(zip(base_faces, new_faces, strict=True)):
        old, new = np.asarray(old), np.asarray(new)
        expected = (states, mesh["n_faces"])+(3,)*(degree+1)
        if old.shape != expected or new.shape != expected:
            raise ValueError("Face polynomials must have compatible state and derivative axes")
        value = (new-old)*face_weight.reshape((1, -1)+(1,)*(degree+1))
        if not np.all(np.isfinite(value)):
            raise ValueError("Face polynomial changes must be finite")
        if np.any(value[:, mesh["n_interior_faces"]:] != 0):
            raise ValueError("The weighted velocity-trace update must vanish on physical boundaries")
        change.append(value)
    integral = weight[None, :, None]*(new_integral-base_integral)
    gamma, moment = weak_quadratic_curl_moments(native, centroid, integral, face_origin, *change)
    return SharedTraceUpdate(gamma, moment, integral, face_weight)
