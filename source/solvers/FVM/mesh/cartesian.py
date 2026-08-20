"""Orthogonal Cartesian mesh generation for tests and benchmarks."""

from __future__ import annotations

import numpy as np


def structured_box(nx: int, ny: int, nz: int, lx=1.0, ly=1.0, lz=1.0) -> dict:
    """Return a face-based ``nx`` by ``ny`` by ``nz`` hexahedral box."""
    if min(nx, ny, nz) < 1 or min(lx, ly, lz) <= 0:
        raise ValueError("Cell counts and box lengths must be positive")
    dx, dy, dz = lx / nx, ly / ny, lz / nz
    npx, npy, npz = nx + 1, ny + 1, nz + 1

    def point_id(i, j, k):
        return i + npx * (j + npy * k)

    def cell_id(i, j, k):
        return i + nx * (j + ny * k)

    points = np.zeros((npx * npy * npz, 3), dtype=np.float64)
    for k in range(npz):
        for j in range(npy):
            for i in range(npx):
                points[point_id(i, j, k)] = (i * dx, j * dy, k * dz)

    def oriented(nodes, direction):
        node_ids = np.asarray(nodes, dtype=np.int32)
        coordinates = points[node_ids]
        normal = np.cross(coordinates[1] - coordinates[0], coordinates[2] - coordinates[0])
        return node_ids if np.dot(normal, direction) >= 0 else node_ids[::-1].copy()

    names = ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")
    patches = {name: [] for name in names}
    patch_owners = {name: [] for name in names}
    interior_faces = []
    interior_owners = []
    interior_neighbours = []
    ex = np.array([1.0, 0.0, 0.0])
    ey = np.array([0.0, 1.0, 0.0])
    ez = np.array([0.0, 0.0, 1.0])

    for i in range(nx + 1):
        for k in range(nz):
            for j in range(ny):
                face = [
                    point_id(i, j, k),
                    point_id(i, j + 1, k),
                    point_id(i, j + 1, k + 1),
                    point_id(i, j, k + 1),
                ]
                if i == 0:
                    patches["xmin"].append(oriented(face, -ex))
                    patch_owners["xmin"].append(cell_id(0, j, k))
                elif i == nx:
                    patches["xmax"].append(oriented(face, ex))
                    patch_owners["xmax"].append(cell_id(nx - 1, j, k))
                else:
                    interior_faces.append(oriented(face, ex))
                    interior_owners.append(cell_id(i - 1, j, k))
                    interior_neighbours.append(cell_id(i, j, k))

    for j in range(ny + 1):
        for k in range(nz):
            for i in range(nx):
                face = [
                    point_id(i, j, k),
                    point_id(i + 1, j, k),
                    point_id(i + 1, j, k + 1),
                    point_id(i, j, k + 1),
                ]
                if j == 0:
                    patches["ymin"].append(oriented(face, -ey))
                    patch_owners["ymin"].append(cell_id(i, 0, k))
                elif j == ny:
                    patches["ymax"].append(oriented(face, ey))
                    patch_owners["ymax"].append(cell_id(i, ny - 1, k))
                else:
                    interior_faces.append(oriented(face, ey))
                    interior_owners.append(cell_id(i, j - 1, k))
                    interior_neighbours.append(cell_id(i, j, k))

    for k in range(nz + 1):
        for j in range(ny):
            for i in range(nx):
                face = [
                    point_id(i, j, k),
                    point_id(i + 1, j, k),
                    point_id(i + 1, j + 1, k),
                    point_id(i, j + 1, k),
                ]
                if k == 0:
                    patches["zmin"].append(oriented(face, -ez))
                    patch_owners["zmin"].append(cell_id(i, j, 0))
                elif k == nz:
                    patches["zmax"].append(oriented(face, ez))
                    patch_owners["zmax"].append(cell_id(i, j, nz - 1))
                else:
                    interior_faces.append(oriented(face, ez))
                    interior_owners.append(cell_id(i, j, k - 1))
                    interior_neighbours.append(cell_id(i, j, k))

    faces = list(interior_faces)
    owners = list(interior_owners)
    n_interior = len(faces)
    boundaries = []
    start = n_interior
    for name in names:
        boundaries.append(
            {"name": name, "start_face": start, "n_faces": len(patches[name]), "type": "patch"}
        )
        faces.extend(patches[name])
        owners.extend(patch_owners[name])
        start += len(patches[name])

    return {
        "points": points,
        "faces": faces,
        "owners": np.asarray(owners, dtype=np.int32),
        "neighbours": np.asarray(interior_neighbours, dtype=np.int32),
        "boundary": boundaries,
        "n_cells": nx * ny * nz,
        "n_faces": len(faces),
        "n_interior_faces": n_interior,
        "n_points": len(points),
    }
