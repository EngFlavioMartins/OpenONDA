#!/usr/bin/env python3
"""Rectilinear quasi-2D mesh with a square hole for the cube_flow tutorial.

Builds a single-cell-thick hexahedral mesh (``empty`` front/back) directly in
the face-based ``mesh_data`` dict the FVM solver consumes — no Gmsh
required.  The square-section cylinder (side ``D``, centred at the origin) is
carved out of the grid and its four sides form the body-fitted ``cube`` wall
patch.  The grid is uniform (spacing ``h``) in a core box around the body and
near wake, and stretched geometrically towards the far-field boundaries.

Domain follows the low-blockage layout used by Sohankar et al. (1998) and
Sen, Mittal & Biswas (2011): lateral extent 20 D (5 % blockage), inlet 10 D
upstream, outlet 25 D downstream.

Patches: ``inlet`` (xmin), ``outlet`` (xmax), ``bottom``/``top`` (ymin/ymax,
slip), ``cube`` (hole sides, wall), ``front``/``back`` (zmin/zmax, empty).
"""

from __future__ import annotations

import numpy as np


def graded_coords(lo, core_lo, core_hi, hi, h, ratio=1.10):
    """1D node coordinates: uniform ``h`` in [core_lo, core_hi], geometric
    stretching (factor ``ratio``) out to ``lo`` and ``hi``."""
    n_core = max(int(round((core_hi - core_lo) / h)), 1)
    core = np.linspace(core_lo, core_hi, n_core + 1)

    def stretch(start, end, direction):
        sizes = []
        pos = start
        size = h
        while (end - pos) * direction > 1e-12:
            size *= ratio
            remaining = (end - pos) * direction
            size = min(size, remaining)
            # Merge a sliver last cell into the previous one.
            if remaining - size < 0.3 * size and remaining > size:
                size = remaining
            sizes.append(size)
            pos += direction * size
        return start + direction * np.cumsum(sizes)

    left = stretch(core_lo, lo, -1.0)[::-1]
    right = stretch(core_hi, hi, +1.0)
    return np.concatenate([left, core, right])


def _index_of(coords, value):
    """Index of ``value`` in a node-coordinate array (must be a grid line)."""
    i = int(np.argmin(np.abs(coords - value)))
    if abs(coords[i] - value) > 1e-9:
        raise ValueError(f"{value} is not a grid line (closest: {coords[i]})")
    return i


def rectilinear_box_with_hole(x, y, depth, hole):
    """``mesh_data`` for a rectilinear grid with a rectangular hole.

    Args:
        x, y:  Node coordinate arrays (monotonically increasing).  The hole
               edges must coincide with grid lines.
        depth: Extrusion thickness in z (single cell).
        hole:  ``(x0, x1, y0, y1)`` bounds of the solid rectangle.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.array([0.0, depth])
    nx, ny = len(x) - 1, len(y) - 1
    npx, npy = nx + 1, ny + 1

    hx0, hx1 = _index_of(x, hole[0]), _index_of(x, hole[1])
    hy0, hy1 = _index_of(y, hole[2]), _index_of(y, hole[3])

    def in_hole(i, j):
        return hx0 <= i < hx1 and hy0 <= j < hy1

    pid = lambda i, j, k: i + npx * (j + npy * k)  # noqa: E731

    points = np.zeros((npx * npy * 2, 3))
    for k in range(2):
        for j in range(npy):
            points[pid(0, j, k) : pid(0, j, k) + npx, 0] = x
            points[pid(0, j, k) : pid(0, j, k) + npx, 1] = y[j]
            points[pid(0, j, k) : pid(0, j, k) + npx, 2] = z[k]

    # Contiguous ids for fluid (non-hole) cells.
    cell_id = -np.ones((nx, ny), dtype=np.int64)
    counter = 0
    for j in range(ny):
        for i in range(nx):
            if not in_hole(i, j):
                cell_id[i, j] = counter
                counter += 1
    n_cells = counter

    interior_faces, interior_owners, interior_neighbours = [], [], []
    patch_order = ("inlet", "outlet", "bottom", "top", "cube", "front", "back")
    patches = {n: [] for n in patch_order}
    patch_owners = {n: [] for n in patches}

    # x-normal faces (base quad has +x normal)
    for i in range(nx + 1):
        for j in range(ny):
            left = cell_id[i - 1, j] if i > 0 else -1
            right = cell_id[i, j] if i < nx else -1
            if left < 0 and right < 0:
                continue
            quad = [pid(i, j, 0), pid(i, j + 1, 0), pid(i, j + 1, 1), pid(i, j, 1)]
            if left >= 0 and right >= 0:
                interior_faces.append(quad)
                interior_owners.append(left)
                interior_neighbours.append(right)
            elif right < 0:  # fluid on the left; outward normal +x
                name = "outlet" if i == nx else "cube"
                patches[name].append(quad)
                patch_owners[name].append(left)
            else:  # fluid on the right; outward normal -x
                name = "inlet" if i == 0 else "cube"
                patches[name].append(quad[::-1])
                patch_owners[name].append(right)

    # y-normal faces (base quad has +y normal)
    for j in range(ny + 1):
        for i in range(nx):
            below = cell_id[i, j - 1] if j > 0 else -1
            above = cell_id[i, j] if j < ny else -1
            if below < 0 and above < 0:
                continue
            quad = [pid(i, j, 0), pid(i, j, 1), pid(i + 1, j, 1), pid(i + 1, j, 0)]
            if below >= 0 and above >= 0:
                interior_faces.append(quad)
                interior_owners.append(below)
                interior_neighbours.append(above)
            elif above < 0:  # fluid below; outward normal +y
                name = "top" if j == ny else "cube"
                patches[name].append(quad)
                patch_owners[name].append(below)
            else:  # fluid above; outward normal -y
                name = "bottom" if j == 0 else "cube"
                patches[name].append(quad[::-1])
                patch_owners[name].append(above)

    # z-normal faces (front/back, empty), fluid cells only
    for k, name in ((0, "front"), (1, "back")):
        for j in range(ny):
            for i in range(nx):
                if cell_id[i, j] < 0:
                    continue
                quad = [pid(i, j, k), pid(i + 1, j, k), pid(i + 1, j + 1, k), pid(i, j + 1, k)]
                patches[name].append(quad[::-1] if k == 0 else quad)
                patch_owners[name].append(cell_id[i, j])

    faces = list(interior_faces)
    owners = list(interior_owners)
    n_interior = len(interior_faces)

    boundary = []
    start = n_interior
    for name in patch_order:
        pf, po = patches[name], patch_owners[name]
        if name in ("front", "back"):
            ptype = "empty"
        elif name == "cube":
            ptype = "wall"
        else:
            ptype = "patch"
        boundary.append({"name": name, "start_face": start, "n_faces": len(pf), "type": ptype})
        faces.extend(pf)
        owners.extend(po)
        start += len(pf)

    return {
        "points": points,
        "faces": [np.asarray(f, dtype=np.int32) for f in faces],
        "owners": np.asarray(owners, dtype=np.int32),
        "neighbours": np.asarray(interior_neighbours, dtype=np.int32),
        "boundary": boundary,
        "n_cells": n_cells,
        "n_faces": len(faces),
        "n_interior_faces": n_interior,
        "n_points": points.shape[0],
    }


def square_cylinder_mesh(
    h=0.0625,
    D=1.0,
    x_bounds=(-10.0, 25.0),
    y_bounds=(-10.0, 10.0),
    core_x=(-2.0, 8.0),
    core_y=(-2.0, 2.0),
    ratio=1.10,
):
    """Standard cube_flow mesh: uniform ``h`` around the square cylinder at the
    origin and along the near wake, stretched to the far field.  All bounds are
    in units of ``D``.  Returns ``(mesh_data, depth)``."""
    x = graded_coords(x_bounds[0] * D, core_x[0] * D, core_x[1] * D, x_bounds[1] * D, h, ratio)
    y = graded_coords(y_bounds[0] * D, core_y[0] * D, core_y[1] * D, y_bounds[1] * D, h, ratio)
    depth = h
    mesh = rectilinear_box_with_hole(x, y, depth, (-0.5 * D, 0.5 * D, -0.5 * D, 0.5 * D))
    return mesh, depth


if __name__ == "__main__":
    mesh, depth = square_cylinder_mesh()
    print(f"cube_flow square-cylinder mesh: {mesh['n_cells']} cells, depth {depth}")
