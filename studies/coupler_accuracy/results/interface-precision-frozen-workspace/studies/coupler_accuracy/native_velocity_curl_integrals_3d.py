"""Cell integrals of the continuous Gaussian Biot--Savart velocity curl.

Observe all vector components through Stokes' theorem on native face fans.
The long-range curl of an individual vector blob must not be truncated at a
Gaussian support radius. No raw-vorticity identity is imposed on this matrix.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.special import erf, roots_jacobi

from studies.coupler_accuracy.joint_reconstruction_3d import gaussian_velocity_curl_operator


@lru_cache(maxsize=16)
def triangle_rule(order):
    rules = [roots_jacobi(order, alpha, 0) for alpha in (1, 0)]
    nodes = [(x + 1) / 2 for x, _ in rules]
    weights = [w / 2**(alpha + 1) for (_, w), alpha in zip(rules, (1, 0), strict=True)]
    r, s = np.meshgrid(*nodes, indexing="ij")
    a, b = np.meshgrid(*weights, indexing="ij")
    barycentric = np.column_stack((r.ravel(), ((1-r)*s).ravel(), ((1-r)*(1-s)).ravel()))
    return barycentric, (2*a*b).ravel()


def gaussian_radial_velocity_factor(distance2, radius):
    """Return f in u = Gamma cross r f, including its finite value at r=0."""
    q2 = distance2 / radius[None, :]**2
    factor = np.divide(1, 4*np.pi*distance2**1.5, out=np.zeros_like(distance2), where=distance2 > 0)
    # At eight Gaussian radii the enclosed fraction differs from one by much
    # less than f64 roundoff. Keep the algebraic velocity tail everywhere.
    near = (q2 < 64) & (q2 >= 1e-4)
    q = np.sqrt(q2[near])
    factor[near] *= erf(q) - 2/np.sqrt(np.pi) * q * np.exp(-q*q)
    small_rows, small_columns = np.where(q2 < 1e-4)
    z = q2[small_rows, small_columns]
    factor[small_rows, small_columns] = (
        (1/3 - z/5 + z*z/14 - z**3/54 + z**4/264)
        / (np.pi**1.5 * radius[small_columns]**3))
    return factor


def gaussian_face_curl_integral(vertices, position, radius, *, order):
    """Return one oriented face's integral of n cross u, as (source, 3, 3).

    For u = Gamma cross r f, n cross u = [(n dot r) I - r n^T] f Gamma.
    The fan triangles supply oriented vector-area weights directly; no aggregate
    face area or normal approximation enters the Stokes integral.
    """
    vertices = np.asarray(vertices, dtype=float)
    position = np.asarray(position, dtype=float).reshape(-1, 3)
    radius = np.broadcast_to(np.asarray(radius, dtype=float), (len(position),))
    if order < 1 or np.any(radius <= 0) or not all(np.all(np.isfinite(x)) for x in (vertices, position, radius)):
        raise ValueError("Finite geometry/sources and positive radii/order are required")
    reference = vertices.mean(axis=0)
    local_vertices = vertices - reference
    local_position = position - reference
    triangles = np.stack((np.zeros_like(local_vertices), local_vertices,
                          np.roll(local_vertices, -1, axis=0)), axis=1)
    vector_area = np.cross(triangles[:, 1], triangles[:, 2]) / 2
    barycentric, weights = triangle_rule(order)
    points = np.einsum("qv,tvd->tqd", barycentric, triangles).reshape(-1, 3)
    area_weights = (vector_area[:, None, :] * weights[None, :, None]).reshape(-1, 3)
    moment = np.zeros((3, 3*len(position)))
    for start in range(0, len(points), 128):
        delta = points[start:start+128, None, :] - local_position
        f = gaussian_radial_velocity_factor(np.sum(delta**2, axis=2), radius)
        moment += area_weights[start:start+128].T @ (f[:, :, None] * delta).reshape(len(delta), -1)
    moment = moment.reshape(3, len(position), 3).transpose(1, 0, 2)
    return np.trace(moment, axis1=1, axis2=2)[:, None, None] * np.eye(3) - moment.transpose(0, 2, 1)


def gaussian_velocity_curl_volume_integral(integration, cell, position, radius, *, order):
    """Independent volume observation of the existing analytical point-curl map."""
    points, weights = integration.rule(cell, order)
    result = np.zeros((3, 3*len(position)))
    for start in range(0, len(points), 64):
        selected = points[start:start+64]
        operator = gaussian_velocity_curl_operator(selected, position, radius).reshape(len(selected), 3, -1)
        result += np.einsum("q,qij->ij", weights[start:start+64], operator)
    return result


def native_velocity_curl_cell_integrals(mesh, geometry, cell_ids, position, radius, *,
                                       basis_tolerance=1e-9, minimum_order=4, maximum_order=14,
                                       progress=None, matrix=None):
    """Return the dense integrated-curl map, with shared faces evaluated once.

    Each face's successive-rule remainder is divided by the smallest adjacent
    selected-cell volume per face. Thus the sum of estimated face remainders
    per cell respects basis_tolerance after division by the stored FVM volume.
    This estimate still requires independent volume/refinement checks.
    """
    ids = np.asarray(cell_ids, dtype=int)
    if ids.ndim != 1 or not len(ids) or len(np.unique(ids)) != len(ids):
        raise ValueError("A nonempty distinct cell selection is required")
    if np.any(ids < 0) or np.any(ids >= mesh["n_cells"]):
        raise ValueError("Cell index outside mesh")
    if basis_tolerance <= 0 or minimum_order < 1 or maximum_order < minimum_order + 2:
        raise ValueError("Invalid quadrature orders or tolerance")
    position = np.asarray(position, dtype=float).reshape(-1, 3)
    radius = np.broadcast_to(np.asarray(radius, dtype=float), (len(position),))
    if np.any(radius <= 0) or not np.all(np.isfinite(radius)) or not np.all(np.isfinite(position)):
        raise ValueError("Finite sources and positive radii are required")
    mapping = np.full(mesh["n_cells"], -1, dtype=int)
    mapping[ids] = np.arange(len(ids))
    left = mapping[mesh["owners"]]
    right = np.full(mesh["n_faces"], -1, dtype=int)
    right[:mesh["n_interior_faces"]] = mapping[mesh["neighbours"]]
    faces = np.flatnonzero((left >= 0) | (right >= 0))
    count = np.bincount(np.r_[left[left >= 0], right[right >= 0]], minlength=len(ids))
    volume_per_face = geometry["cell_volume"][ids] / count
    shape = (3*len(ids), 3*len(position))
    if matrix is None:
        matrix = np.zeros(shape)
    elif matrix.shape != shape or matrix.dtype != np.float64:
        raise ValueError("Provided matrix must have the full f64 block-map shape")
    else:
        matrix[:] = 0
    blocks = matrix.reshape(len(ids), 3, len(position), 3)
    orders, changes = [], []
    cell_change = np.zeros(len(ids))
    for index, face in enumerate(faces):
        adjacent = [i for i in (left[face], right[face]) if i >= 0]
        scale = min(volume_per_face[adjacent])
        vertices = mesh["vertex_position"][mesh["faces"][face]]
        previous = gaussian_face_curl_integral(vertices, position, radius, order=minimum_order)
        for order in range(minimum_order+2, maximum_order+1, 2):
            current = gaussian_face_curl_integral(vertices, position, radius, order=order)
            change = float(np.max(np.abs(current-previous), initial=0))
            if change <= basis_tolerance * scale:
                break
            previous = current
        else:
            raise RuntimeError(f"Face {face} failed quadrature refinement: change/scale {change/scale:g}")
        for local, sign in ((left[face], 1), (right[face], -1)):
            if local >= 0:
                blocks[local] += sign * current.transpose(1, 0, 2)
                cell_change[local] += change / geometry["cell_volume"][ids[local]]
        orders.append(order)
        changes.append(change / scale)
        if progress is not None and (index % 100 == 0 or index+1 == len(faces)):
            progress(index+1, len(faces), order, change/scale)
    diagnostic = {"face_ids": faces.tolist(), "orders": orders,
                  "successive_face_basis_change_over_volume_per_face": changes,
                  "cell_summed_remainder_over_fvm_volume": cell_change.tolist(),
                  "basis_tolerance": basis_tolerance, "minimum_order": minimum_order,
                  "maximum_order": maximum_order, "source_cutoff": None}
    return matrix, diagnostic
