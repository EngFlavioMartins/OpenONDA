#!/usr/bin/env python3
"""Rectilinear quasi-2D mesh for a backward-facing step."""

from __future__ import annotations

import numpy as np


def backward_facing_step_mesh(
    *,
    step_height: float = 1.0,
    upstream_length: float = 4.0,
    downstream_length: float = 20.0,
    n_upstream: int = 24,
    n_downstream: int = 120,
    n_height: int = 16,
):
    """Return a single-cell-thick hexahedral mesh with a 2:1 expansion.

    The inlet occupies ``h <= y <= 2h`` at ``x = -L_up``.  The solid
    upstream block occupies ``x < 0, y < h``; downstream of ``x = 0`` the
    fluid fills ``0 <= y <= 2h``.  The vertical face at ``x = 0, y < h`` is
    therefore a real geometric step.
    """
    if min(step_height, upstream_length, downstream_length) <= 0.0:
        raise ValueError("Step dimensions must be positive")
    if min(n_upstream, n_downstream) < 1 or n_height < 2 or n_height % 2:
        raise ValueError("Cell counts must be positive and n_height must be an even integer >= 2")

    h = float(step_height)
    x_up = np.linspace(-upstream_length * h, 0.0, n_upstream + 1)
    x_down = np.linspace(0.0, downstream_length * h, n_downstream + 1)
    x = np.concatenate((x_up[:-1], x_down))
    y = np.linspace(0.0, 2.0 * h, n_height + 1)
    depth = min(np.min(np.diff(x)), np.min(np.diff(y)))
    z = np.array([0.0, depth])

    nx, ny = len(x) - 1, len(y) - 1
    npx, npy = nx + 1, ny + 1
    step_j = n_height // 2

    def pid(i, j, k):
        return i + npx * (j + npy * k)

    points = np.zeros((npx * npy * 2, 3), dtype=np.float64)
    for k in range(2):
        for j in range(npy):
            row = slice(pid(0, j, k), pid(0, j, k) + npx)
            points[row, 0] = x
            points[row, 1] = y[j]
            points[row, 2] = z[k]

    fluid = np.zeros((nx, ny), dtype=bool)
    x_mid = 0.5 * (x[:-1] + x[1:])
    fluid[x_mid >= 0.0, :] = True
    fluid[x_mid < 0.0, step_j:] = True

    cell_id = -np.ones((nx, ny), dtype=np.int32)
    fluid_ij = []
    for j in range(ny):
        for i in range(nx):
            if fluid[i, j]:
                cell_id[i, j] = len(fluid_ij)
                fluid_ij.append((i, j))

    def oriented(quad, desired):
        nodes = np.asarray(quad, dtype=np.int32)
        xyz = points[nodes]
        normal = np.cross(xyz[1] - xyz[0], xyz[2] - xyz[0])
        return nodes if np.dot(normal, desired) > 0.0 else nodes[::-1].copy()

    interior_faces = []
    interior_owners = []
    interior_neighbours = []
    patch_order = ("inlet", "outlet", "walls", "front", "back")
    patches = {name: [] for name in patch_order}
    patch_owners = {name: [] for name in patch_order}
    ex = np.array([1.0, 0.0, 0.0])
    ey = np.array([0.0, 1.0, 0.0])
    ez = np.array([0.0, 0.0, 1.0])

    # Interfaces normal to x.  A fluid/solid interface below y=h at x=0 is
    # the vertical step wall.
    for i in range(nx + 1):
        for j in range(ny):
            left = i > 0 and fluid[i - 1, j]
            right = i < nx and fluid[i, j]
            if not left and not right:
                continue
            quad = [pid(i, j, 0), pid(i, j + 1, 0), pid(i, j + 1, 1), pid(i, j, 1)]
            if left and right:
                interior_faces.append(oriented(quad, ex))
                interior_owners.append(cell_id[i - 1, j])
                interior_neighbours.append(cell_id[i, j])
            elif right:
                name = "inlet" if i == 0 else "walls"
                patches[name].append(oriented(quad, -ex))
                patch_owners[name].append(cell_id[i, j])
            else:
                name = "outlet" if i == nx else "walls"
                patches[name].append(oriented(quad, ex))
                patch_owners[name].append(cell_id[i - 1, j])

    # Interfaces normal to y.  The upstream lower wall lies at y=h and the
    # downstream lower wall at y=0.
    for j in range(ny + 1):
        for i in range(nx):
            lower = j > 0 and fluid[i, j - 1]
            upper = j < ny and fluid[i, j]
            if not lower and not upper:
                continue
            quad = [pid(i, j, 0), pid(i, j, 1), pid(i + 1, j, 1), pid(i + 1, j, 0)]
            if lower and upper:
                interior_faces.append(oriented(quad, ey))
                interior_owners.append(cell_id[i, j - 1])
                interior_neighbours.append(cell_id[i, j])
            elif upper:
                patches["walls"].append(oriented(quad, -ey))
                patch_owners["walls"].append(cell_id[i, j])
            else:
                patches["walls"].append(oriented(quad, ey))
                patch_owners["walls"].append(cell_id[i, j - 1])

    # Single-cell depth; both z patches are empty in the quasi-2D solve.
    for k, name, direction in ((0, "front", -ez), (1, "back", ez)):
        for i, j in fluid_ij:
            quad = [pid(i, j, k), pid(i + 1, j, k), pid(i + 1, j + 1, k), pid(i, j + 1, k)]
            patches[name].append(oriented(quad, direction))
            patch_owners[name].append(cell_id[i, j])

    faces = list(interior_faces)
    owners = list(interior_owners)
    n_interior = len(faces)
    boundary = []
    start = n_interior
    for name in patch_order:
        patch_faces = patches[name]
        boundary.append(
            {
                "name": name,
                "start_face": start,
                "n_faces": len(patch_faces),
                "type": "empty"
                if name in ("front", "back")
                else ("wall" if name == "walls" else "patch"),
            }
        )
        faces.extend(patch_faces)
        owners.extend(patch_owners[name])
        start += len(patch_faces)

    mesh = {
        "points": points,
        "faces": [np.asarray(face, dtype=np.int32) for face in faces],
        "owners": np.asarray(owners, dtype=np.int32),
        "neighbours": np.asarray(interior_neighbours, dtype=np.int32),
        "boundary": boundary,
        "n_cells": len(fluid_ij),
        "n_faces": len(faces),
        "n_interior_faces": n_interior,
        "n_points": len(points),
    }
    return mesh, depth


if __name__ == "__main__":
    mesh, mesh_depth = backward_facing_step_mesh()
    print(f"step_profile mesh: {mesh['n_cells']} cells, depth {mesh_depth:g}")
    for patch in mesh["boundary"]:
        print(f"  {patch['name']:<8} {patch['n_faces']:>6} faces ({patch['type']})")
