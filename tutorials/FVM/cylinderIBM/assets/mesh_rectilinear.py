#!/usr/bin/env python3
"""Rectilinear graded 2D mesh generator for the cylinderIBM tutorial.

Builds a single-cell-thick hexahedral mesh (quasi-2D, ``empty`` front/back)
directly in the face-based ``mesh_data`` dict the FVM solver consumes —
no gmsh required.  The grid is uniform (spacing ``h``) in a core box around
the cylinder (the IBM kernel requires locally uniform spacing) and stretched
geometrically towards the far-field boundaries, following the layout of
Constant et al. (docs/literature/Constant2016.pdf, Fig. 9).

Patches: ``inlet`` (xmin), ``outlet`` (xmax), ``bottom``/``top`` (ymin/ymax),
``front``/``back`` (zmin/zmax, empty).
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
        pts = start + direction * np.cumsum(sizes)
        return pts

    left = stretch(core_lo, lo, -1.0)[::-1]
    right = stretch(core_hi, hi, +1.0)
    return np.concatenate([left, core, right])


def rectilinear_box_2d(x, y, depth):
    """``mesh_data`` for a rectilinear ``(len(x)-1) × (len(y)-1) × 1`` grid.

    Args:
        x, y:  Node coordinate arrays (monotonically increasing).
        depth: Extrusion thickness in z (single cell).
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.array([0.0, depth])
    nx, ny, nz = len(x) - 1, len(y) - 1, 1
    npx, npy, npz = nx + 1, ny + 1, nz + 1

    pid = lambda i, j, k: i + npx * (j + npy * k)  # noqa: E731
    cid = lambda i, j, k: i + nx * (j + ny * k)  # noqa: E731

    points = np.zeros((npx * npy * npz, 3))
    for k in range(npz):
        for j in range(npy):
            points[pid(0, j, k) : pid(0, j, k) + npx, 0] = x
            points[pid(0, j, k) : pid(0, j, k) + npx, 1] = y[j]
            points[pid(0, j, k) : pid(0, j, k) + npx, 2] = z[k]

    interior_faces, interior_owners, interior_neighbours = [], [], []
    patches = {n: [] for n in ("inlet", "outlet", "bottom", "top", "front", "back")}
    patch_owners = {n: [] for n in patches}

    # x-normal faces
    for i in range(nx + 1):
        for j in range(ny):
            quad = [pid(i, j, 0), pid(i, j + 1, 0), pid(i, j + 1, 1), pid(i, j, 1)]
            if i == 0:
                patches["inlet"].append(quad[::-1])  # normal -x
                patch_owners["inlet"].append(cid(0, j, 0))
            elif i == nx:
                patches["outlet"].append(quad)  # normal +x
                patch_owners["outlet"].append(cid(nx - 1, j, 0))
            else:
                interior_faces.append(quad)
                interior_owners.append(cid(i - 1, j, 0))
                interior_neighbours.append(cid(i, j, 0))

    # y-normal faces
    for j in range(ny + 1):
        for i in range(nx):
            quad = [pid(i, j, 0), pid(i, j, 1), pid(i + 1, j, 1), pid(i + 1, j, 0)]
            if j == 0:
                patches["bottom"].append(quad[::-1])  # normal -y
                patch_owners["bottom"].append(cid(i, 0, 0))
            elif j == ny:
                patches["top"].append(quad)  # normal +y
                patch_owners["top"].append(cid(i, ny - 1, 0))
            else:
                interior_faces.append(quad)
                interior_owners.append(cid(i, j - 1, 0))
                interior_neighbours.append(cid(i, j, 0))

    # z-normal faces (front/back, empty)
    for k, name in ((0, "front"), (1, "back")):
        for j in range(ny):
            for i in range(nx):
                quad = [pid(i, j, k), pid(i + 1, j, k), pid(i + 1, j + 1, k), pid(i, j + 1, k)]
                if k == 0:
                    patches[name].append(quad[::-1])  # normal -z
                else:
                    patches[name].append(quad)  # normal +z
                patch_owners[name].append(cid(i, j, 0))

    faces = list(interior_faces)
    owners = list(interior_owners)
    n_interior = len(interior_faces)

    boundary = []
    start = n_interior
    for name in ("inlet", "outlet", "bottom", "top", "front", "back"):
        pf, po = patches[name], patch_owners[name]
        ptype = "empty" if name in ("front", "back") else "patch"
        boundary.append({"name": name, "startFace": start, "nFaces": len(pf), "type": ptype})
        faces.extend(pf)
        owners.extend(po)
        start += len(pf)

    return {
        "points": points,
        "faces": [np.asarray(f, dtype=np.int32) for f in faces],
        "owners": np.asarray(owners, dtype=np.int32),
        "neighbours": np.asarray(interior_neighbours, dtype=np.int32),
        "boundary": boundary,
        "n_elements": nx * ny,
        "n_faces": len(faces),
        "n_interior_faces": n_interior,
        "n_points": points.shape[0],
    }


def cylinder_ibm_mesh(
    h=0.0625,
    D=1.0,
    x_bounds=(-8.0, 16.0),
    y_bounds=(-8.0, 8.0),
    core_x=(-1.5, 3.0),
    core_y=(-1.5, 1.5),
    ratio=1.10,
):
    """Standard cylinderIBM mesh: uniform ``h`` around the cylinder at the
    origin, stretched to the far field.  Returns ``(mesh_data, depth)``."""
    x = graded_coords(x_bounds[0], core_x[0] * D, core_x[1] * D, x_bounds[1], h, ratio)
    y = graded_coords(y_bounds[0], core_y[0] * D, core_y[1] * D, y_bounds[1], h, ratio)
    depth = h
    mesh = rectilinear_box_2d(x, y, depth)
    return mesh, depth


if __name__ == "__main__":
    mesh, depth = cylinder_ibm_mesh()
    nx = len(np.unique(mesh["points"][:, 0])) - 1
    ny = len(np.unique(mesh["points"][:, 1])) - 1
    print(f"cylinderIBM mesh: {nx} x {ny} = {mesh['n_elements']} cells, depth {depth}")
