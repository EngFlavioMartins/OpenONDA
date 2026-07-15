"""Single-cell-thick orthogonal mesh for a doubly periodic square."""

from __future__ import annotations

import numpy as np


def periodic_square_mesh(n: int, length: float = 2.0 * np.pi) -> dict:
    """Build an ``n`` by ``n`` hexahedral mesh with cyclic-ready patches."""
    if n < 2:
        raise ValueError("n must be at least 2")

    spacing = length / n
    x = np.linspace(0.0, length, n + 1)
    y = np.linspace(0.0, length, n + 1)
    z = np.array([0.0, spacing])
    npx = n + 1

    def point_id(i: int, j: int, k: int) -> int:
        return i + npx * (j + npx * k)

    def cell_id(i: int, j: int) -> int:
        return i + n * j

    points = np.zeros((npx * npx * 2, 3), dtype=np.float64)
    for k in range(2):
        for j in range(npx):
            start = point_id(0, j, k)
            points[start : start + npx, 0] = x
            points[start : start + npx, 1] = y[j]
            points[start : start + npx, 2] = z[k]

    interior_faces: list[list[int]] = []
    interior_owners: list[int] = []
    interior_neighbours: list[int] = []
    patch_order = ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")
    patches: dict[str, list[list[int]]] = {name: [] for name in patch_order}
    patch_owners: dict[str, list[int]] = {name: [] for name in patch_order}

    for i in range(n + 1):
        for j in range(n):
            face = [
                point_id(i, j, 0),
                point_id(i, j + 1, 0),
                point_id(i, j + 1, 1),
                point_id(i, j, 1),
            ]
            if i == 0:
                patches["xmin"].append(face[::-1])
                patch_owners["xmin"].append(cell_id(0, j))
            elif i == n:
                patches["xmax"].append(face)
                patch_owners["xmax"].append(cell_id(n - 1, j))
            else:
                interior_faces.append(face)
                interior_owners.append(cell_id(i - 1, j))
                interior_neighbours.append(cell_id(i, j))

    for j in range(n + 1):
        for i in range(n):
            face = [
                point_id(i, j, 0),
                point_id(i, j, 1),
                point_id(i + 1, j, 1),
                point_id(i + 1, j, 0),
            ]
            if j == 0:
                patches["ymin"].append(face[::-1])
                patch_owners["ymin"].append(cell_id(i, 0))
            elif j == n:
                patches["ymax"].append(face)
                patch_owners["ymax"].append(cell_id(i, n - 1))
            else:
                interior_faces.append(face)
                interior_owners.append(cell_id(i, j - 1))
                interior_neighbours.append(cell_id(i, j))

    for k, name in ((0, "zmin"), (1, "zmax")):
        for j in range(n):
            for i in range(n):
                face = [
                    point_id(i, j, k),
                    point_id(i + 1, j, k),
                    point_id(i + 1, j + 1, k),
                    point_id(i, j + 1, k),
                ]
                patches[name].append(face[::-1] if k == 0 else face)
                patch_owners[name].append(cell_id(i, j))

    faces = list(interior_faces)
    owners = list(interior_owners)
    n_interior = len(interior_faces)
    boundary = []
    start = n_interior
    for name in patch_order:
        patch_faces = patches[name]
        boundary.append(
            {
                "name": name,
                "startFace": start,
                "nFaces": len(patch_faces),
                "type": "empty" if name.startswith("z") else "patch",
            }
        )
        faces.extend(patch_faces)
        owners.extend(patch_owners[name])
        start += len(patch_faces)

    return {
        "points": points,
        "faces": [np.asarray(face, dtype=np.int32) for face in faces],
        "owners": np.asarray(owners, dtype=np.int32),
        "neighbours": np.asarray(interior_neighbours, dtype=np.int32),
        "boundary": boundary,
        "n_elements": n * n,
        "n_faces": len(faces),
        "n_interior_faces": n_interior,
        "n_points": len(points),
    }
