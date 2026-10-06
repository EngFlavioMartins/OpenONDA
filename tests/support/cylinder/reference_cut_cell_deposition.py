"""Bounded exterior deposition of complete reference circulation control cells.

Inside control-cell centres are quadrature locations, never physical particles.
This verification primitive preserves each cell's circulation and nominal XY
first moments on wall-visible exterior nodes. It does not change native renewal
or supply a production model. Failed native support/weight bounds reject the
control explicitly.
"""

from __future__ import annotations

import numpy as np


def _m4_prime(relative):
    distance = np.abs(relative)
    return np.where(distance < 1, 1 - 2.5 * distance**2 + 1.5 * distance**3,
                    np.where(distance < 2, .5 * (1 - distance) * (2 - distance)**2, 0))


def deposit_cut_cell_circulation(
    centres,
    circulation,
    lattice_positions,
    spacing,
    solid_boundary,
    output_dtype=np.float32,
    *,
    maximum_weight_l1=2.0,
    maximum_radius=4,
):
    """Return sparse complete deposits and independently inspectable donor weights.

    ``centres`` and ``lattice_positions`` have shape (N,3) in metres;
    ``circulation`` is axial strength (N,) in m³/s, including the planar span.
    Nodes may be the full lattice or only its available exterior subset.
    Returned indices address the caller's node array. Returned strength is a
    complete additional deposit, not a correction to a pre-existing deposit.

    Constraints use nominal lattice coordinates, as native wall M4 correction
    does. Solid/visibility queries use their actual output-storage coordinates.
    The finite rounding contribution to first moments is reported separately.
    """
    centres = np.asarray(centres, dtype=np.float64)
    gamma = np.asarray(circulation, dtype=np.float64)
    nodes = np.asarray(lattice_positions, dtype=np.float64)
    dtype = np.dtype(output_dtype)
    if (centres.ndim != 2 or centres.shape[1:] != (3,) or gamma.shape != (len(centres),)
            or nodes.ndim != 2 or nodes.shape[1:] != (3,) or not len(nodes)
            or not np.isfinite(centres).all() or not np.isfinite(gamma).all()
            or not np.isfinite(nodes).all() or not np.isfinite(spacing) or spacing <= 0
            or dtype not in (np.dtype(np.float32), np.dtype(np.float64))):
        raise ValueError("Finite planar centres, axial circulation, lattice and storage dtype are required")
    if (not np.isfinite(maximum_weight_l1) or not 0 < maximum_weight_l1 <= 2.0
            or isinstance(maximum_radius, bool) or not isinstance(maximum_radius, int)
            or not 1 <= maximum_radius <= 4):
        raise ValueError("Cut-cell control cannot relax native weight L1<=2 or radius<=4")
    h = float(spacing)
    coordinate_scale = max(h, float(np.max(np.abs(nodes))), float(np.max(np.abs(centres), initial=0)))
    storage_margin = max(4 * solid_boundary.tolerance,
                         32 * np.finfo(dtype).eps * coordinate_scale, 1e-7 * h)
    if 4 * np.finfo(dtype).eps * coordinate_scale >= .01 * h:
        raise ValueError("Output storage cannot resolve the cut-cell lattice spacing")
    plane = float(nodes[0, 2])
    if np.max(np.abs(nodes[:, 2] - plane)) > 1e-12 * coordinate_scale:
        raise ValueError("Cut-cell support must occupy one planar particle layer")
    if np.max(np.abs(centres[:, 2] - plane), initial=0) > 1e-12 * coordinate_scale:
        raise ValueError("Cut-cell centres must occupy the particle plane")
    origin = nodes[:, :2].min(axis=0)
    relative_nodes = (nodes[:, :2] - origin) / h
    grid_indices = np.rint(relative_nodes).astype(np.int64)
    if np.max(np.abs(relative_nodes - grid_indices)) > 1e-9:
        raise ValueError("Cut-cell support is not one uniform XY lattice")
    stride = int(grid_indices[:, 1].max()) + 1
    identifiers = grid_indices[:, 0] * stride + grid_indices[:, 1]
    order = np.argsort(identifiers)
    sorted_identifiers = identifiers[order]
    if np.any(np.diff(sorted_identifiers) == 0):
        raise ValueError("Cut-cell support contains duplicate lattice nodes")
    shape = grid_indices.max(axis=0) + 1
    chunks, values, donor_records = [], [], []
    for donor in np.flatnonzero(gamma != 0):
        centre = centres[donor]
        fractional = (centre[:2] - origin) / h
        nearest = np.rint(fractional)
        roundoff = 64 * np.finfo(float).eps * np.maximum(1., np.abs(fractional))
        fractional = np.where(np.abs(fractional - nearest) <= roundoff, nearest, fractional)
        inside = bool(solid_boundary.contains(centre[None], include_boundary=False)[0])
        visibility_origin = centre.copy()
        if inside or solid_boundary.signed_distance(centre[None])[0] <= storage_margin:
            surface, normal = solid_boundary.closest_surface(centre[None])
            if not np.isfinite(surface).all() or not np.isfinite(normal).all():
                raise ValueError(f"Cut-cell donor {donor} has invalid nearest wall geometry")
            distance = float(np.linalg.norm(surface[0] - centre))
            if distance > .5 * h + 64 * np.finfo(float).eps * coordinate_scale:
                raise ValueError(f"Cut-cell donor {donor} lies deeper than a midpoint circulation cell")
            visibility_origin = (surface[0] + storage_margin * normal[0]).astype(dtype).astype(float)
            if abs(visibility_origin[2] - plane) > storage_margin:
                raise ValueError(f"Cut-cell donor {donor} cannot reach planar fluid support")
            visibility_origin[2] = np.asarray(plane, dtype=dtype).item()
        if solid_boundary.contains(visibility_origin[None], include_boundary=False)[0]:
            raise ValueError(f"Cut-cell donor {donor} has no resolved fluid-side visibility origin")
        solved = False
        for radius in range(1, maximum_radius + 1):
            offsets = np.stack(np.meshgrid(np.arange(-radius, radius + 2),
                                           np.arange(-radius, radius + 2), indexing="ij"),
                               axis=-1).reshape(-1, 2)
            candidates = np.floor(fractional).astype(np.int64) + offsets
            valid = np.all((candidates >= 0) & (candidates < shape), axis=1)
            candidates = candidates[valid]
            keys = candidates[:, 0] * stride + candidates[:, 1]
            insertion = np.searchsorted(sorted_identifiers, keys)
            present = insertion < len(sorted_identifiers)
            present[present] &= sorted_identifiers[insertion[present]] == keys[present]
            indices = order[insertion[present]]
            candidate_nodes = nodes[indices]
            stored_nodes = candidate_nodes.astype(dtype).astype(np.float64)
            visible = ~solid_boundary.contains(stored_nodes, include_boundary=False)
            if visible.any():
                visible[visible] &= ~solid_boundary.blocks_segments(
                    np.broadcast_to(visibility_origin, stored_nodes[visible].shape),
                    stored_nodes[visible])
            indices, candidate_nodes = indices[visible], candidate_nodes[visible]
            if len(indices) < 3:
                continue
            constraints = np.vstack((np.ones(len(indices)),
                                     ((candidate_nodes[:, :2] - centre[:2]) / h).T))
            gram = constraints @ constraints.T
            condition = float(np.linalg.cond(gram))
            if not np.isfinite(condition) or condition > 1e10:
                continue
            weights = np.prod(_m4_prime((candidate_nodes[:, :2] - centre[:2]) / h), axis=1)
            desired = np.array([1., 0., 0.])
            weights += constraints.T @ np.linalg.solve(gram, desired - constraints @ weights)
            residual = float(np.max(np.abs(constraints @ weights - desired)))
            weight_l1 = float(np.sum(np.abs(weights)))
            if (not np.isfinite(weights).all() or weight_l1 > maximum_weight_l1
                    or residual > 1e-10):
                continue
            active = weights != 0
            indices, weights, candidate_nodes = indices[active], weights[active], candidate_nodes[active]
            stored_error = np.sum(weights[:, None] *
                                  (candidate_nodes.astype(dtype).astype(float) - candidate_nodes), axis=0)
            chunks.append(indices)
            values.append(gamma[donor] * weights)
            donor_records.append({
                "donor": int(donor), "centre": centre.tolist(), "axial_strength": float(gamma[donor]),
                "source_centre_inside_solid": inside, "visibility_origin": visibility_origin.tolist(),
                "radius": radius, "support_node_count": len(indices),
                "weight_l1": weight_l1, "constraint_condition": condition,
                "maximum_constraint_residual": residual,
                "nominal_first_moment_error": (gamma[donor] *
                    np.sum(weights[:, None] * (candidate_nodes - centre), axis=0)).tolist(),
                "stored_first_moment_rounding_error": (gamma[donor] * stored_error).tolist(),
                "node_indices": indices.tolist(), "weights": weights.tolist(),
            })
            solved = True
            break
        if not solved:
            raise ValueError(f"Cut-cell donor {donor} has insufficient wall-visible exterior support "
                             "for native bounded circulation/first-moment conservation")
    if chunks:
        all_indices = np.concatenate(chunks)
        unique, inverse = np.unique(all_indices, return_inverse=True)
        strength = np.bincount(inverse, weights=np.concatenate(values), minlength=len(unique))
        keep = strength != 0
        unique, strength = unique[keep], strength[keep]
    else:
        unique, strength = np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    if not np.isfinite(strength).all():
        raise ValueError("Cut-cell deposited circulation exceeds finite storage")
    return unique, strength, {
        "scope": "Frozen complete circulation-cell control, no native numerical/runtime changes.",
        "storage_dtype": dtype.name, "spacing": h, "maximum_weight_l1": maximum_weight_l1,
        "maximum_radius": maximum_radius, "donor_count": len(donor_records),
        "deposited_axial_strength": float(strength.sum()),
        "deposited_absolute_axial_strength": float(np.abs(strength).sum()),
        "donors": donor_records,
    }
