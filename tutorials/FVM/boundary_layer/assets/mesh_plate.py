#!/usr/bin/env python3
"""Rectilinear quasi-2D flat-plate mesh for the boundary_layer tutorial.

Single-cell-thick hexahedral mesh (``empty`` front/back) built directly in the
face-based ``mesh_data`` dict the FVM solver consumes. The bottom
boundary is split at the leading edge x = 0 into a frictionless run-in
(``floor``, slip) and the no-slip ``plate``, so the boundary layer starts
growing exactly at x = 0 — the configuration the Blasius similarity solution
assumes.  The wall-normal grid is geometrically stretched from the wall.

Patches: ``inlet`` (xmin), ``outlet`` (xmax), ``floor`` (ymin, x < 0),
``plate`` (ymin, x >= 0), ``top`` (ymax, slip), ``front``/``back`` (empty).
"""

from __future__ import annotations

import numpy as np


def wall_normal_coords(height, dy_wall=0.0015, ratio=1.12):
    """Node coordinates from y = 0 with geometric stretching to ``height``."""
    y = [0.0]
    dy = dy_wall
    while y[-1] < height - 1e-12:
        remaining = height - y[-1]
        step = min(dy, remaining)
        # Merge a sliver last cell into the previous one.
        if remaining - step < 0.3 * step and remaining > step:
            step = remaining
        y.append(y[-1] + step)
        dy *= ratio
    return np.asarray(y)


def plate_coords(x_up=-0.25, plate_length=1.0, n_up=10, n_plate=72):
    """Node coordinates: uniform run-in [x_up, 0], uniform plate [0, L]."""
    up = np.linspace(x_up, 0.0, n_up + 1)
    plate = np.linspace(0.0, plate_length, n_plate + 1)
    return np.concatenate([up[:-1], plate])


def flat_plate_mesh(
    x_up=-0.25,
    plate_length=1.0,
    height=0.35,
    n_up=10,
    n_plate=72,
    dy_wall=0.0015,
    ratio=1.12,
):
    """Flat-plate mesh with leading edge at the origin.

    Returns ``(mesh_data, depth)``; ``depth`` is the single-cell z thickness.
    """
    x = plate_coords(x_up, plate_length, n_up, n_plate)
    y = wall_normal_coords(height, dy_wall, ratio)
    depth = plate_length / n_plate
    z = np.array([0.0, depth])

    nx, ny = len(x) - 1, len(y) - 1
    npx, npy = nx + 1, ny + 1

    pid = lambda i, j, k: i + npx * (j + npy * k)  # noqa: E731
    cid = lambda i, j: i + nx * j  # noqa: E731

    points = np.zeros((npx * npy * 2, 3))
    for k in range(2):
        for j in range(npy):
            points[pid(0, j, k) : pid(0, j, k) + npx, 0] = x
            points[pid(0, j, k) : pid(0, j, k) + npx, 1] = y[j]
            points[pid(0, j, k) : pid(0, j, k) + npx, 2] = z[k]

    interior_faces, interior_owners, interior_neighbours = [], [], []
    patch_order = ("inlet", "outlet", "floor", "plate", "top", "front", "back")
    patches = {n: [] for n in patch_order}
    patch_owners = {n: [] for n in patches}

    # x-normal faces (base quad normal +x)
    for i in range(nx + 1):
        for j in range(ny):
            quad = [pid(i, j, 0), pid(i, j + 1, 0), pid(i, j + 1, 1), pid(i, j, 1)]
            if i == 0:
                patches["inlet"].append(quad[::-1])
                patch_owners["inlet"].append(cid(0, j))
            elif i == nx:
                patches["outlet"].append(quad)
                patch_owners["outlet"].append(cid(nx - 1, j))
            else:
                interior_faces.append(quad)
                interior_owners.append(cid(i - 1, j))
                interior_neighbours.append(cid(i, j))

    # y-normal faces (base quad normal +y)
    for j in range(ny + 1):
        for i in range(nx):
            quad = [pid(i, j, 0), pid(i, j, 1), pid(i + 1, j, 1), pid(i + 1, j, 0)]
            if j == 0:
                x_mid = 0.5 * (x[i] + x[i + 1])
                name = "floor" if x_mid < 0.0 else "plate"
                patches[name].append(quad[::-1])
                patch_owners[name].append(cid(i, 0))
            elif j == ny:
                patches["top"].append(quad)
                patch_owners["top"].append(cid(i, ny - 1))
            else:
                interior_faces.append(quad)
                interior_owners.append(cid(i, j - 1))
                interior_neighbours.append(cid(i, j))

    # z-normal faces (front/back, empty)
    for k, name in ((0, "front"), (1, "back")):
        for j in range(ny):
            for i in range(nx):
                quad = [pid(i, j, k), pid(i + 1, j, k), pid(i + 1, j + 1, k), pid(i, j + 1, k)]
                patches[name].append(quad[::-1] if k == 0 else quad)
                patch_owners[name].append(cid(i, j))

    faces = list(interior_faces)
    owners = list(interior_owners)
    n_interior = len(interior_faces)

    boundary = []
    start = n_interior
    for name in patch_order:
        pf, po = patches[name], patch_owners[name]
        if name in ("front", "back"):
            ptype = "empty"
        elif name == "plate":
            ptype = "wall"
        else:
            ptype = "patch"
        boundary.append({"name": name, "start_face": start, "n_faces": len(pf), "type": ptype})
        faces.extend(pf)
        owners.extend(po)
        start += len(pf)

    mesh = {
        "points": points,
        "faces": [np.asarray(f, dtype=np.int32) for f in faces],
        "owners": np.asarray(owners, dtype=np.int32),
        "neighbours": np.asarray(interior_neighbours, dtype=np.int32),
        "boundary": boundary,
        "n_cells": nx * ny,
        "n_faces": len(faces),
        "n_interior_faces": n_interior,
        "n_points": points.shape[0],
    }
    return mesh, depth


if __name__ == "__main__":
    mesh, depth = flat_plate_mesh()
    print(f"boundary_layer mesh: {mesh['n_cells']} cells, depth {depth}")
    for b in mesh["boundary"]:
        print(f"  {b['name']:<8} {b['n_faces']:>6} faces  ({b['type']})")
