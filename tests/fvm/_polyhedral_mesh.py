"""Deterministic first-order prism and mixed hex-prism verification meshes."""

from __future__ import annotations

import numpy as np


def _face_templates(nodes, family):
    if family == "hex":
        a, b, c, d, e, f, g, h = nodes
        return ([a, d, c, b], [e, f, g, h], [a, b, f, e], [b, c, g, f], [c, d, h, g], [d, a, e, h])
    a, b, c, d, e, f = nodes
    return ([a, c, b], [d, e, f], [a, b, e, d], [b, c, f, e], [c, a, d, f])


def _orient_face(face, points, direction):
    result = np.asarray(face, dtype=np.int32)
    coords = points[result]
    normal = np.zeros(3)
    centre = np.mean(coords, axis=0)
    for index in range(len(result)):
        p0 = coords[index] - centre
        p1 = coords[(index + 1) % len(result)] - centre
        normal += np.cross(p0, p1)
    if np.dot(normal, direction) < 0.0:
        result = result[::-1].copy()
    return result


def split_prism_box(n, *, mixed=False):
    """Build an ``n³`` box split into prisms, optionally retaining left-side hexes."""
    n_points_axis = n + 1

    def point_id(i, j, k):
        return i + n_points_axis * (j + n_points_axis * k)

    points = np.array(
        [
            (i / n, j / n, k / n)
            for k in range(n_points_axis)
            for j in range(n_points_axis)
            for i in range(n_points_axis)
        ],
        dtype=np.float64,
    )
    cells = []
    families = []
    for k in range(n):
        for j in range(n):
            for i in range(n):
                p000 = point_id(i, j, k)
                p100 = point_id(i + 1, j, k)
                p110 = point_id(i + 1, j + 1, k)
                p010 = point_id(i, j + 1, k)
                p001 = point_id(i, j, k + 1)
                p101 = point_id(i + 1, j, k + 1)
                p111 = point_id(i + 1, j + 1, k + 1)
                p011 = point_id(i, j + 1, k + 1)
                if mixed and i < n // 2:
                    cells.append((p000, p100, p110, p010, p001, p101, p111, p011))
                    families.append("hex")
                else:
                    cells.extend(
                        [
                            (p000, p100, p110, p001, p101, p111),
                            (p000, p110, p010, p001, p111, p011),
                        ]
                    )
                    families.extend(("prism", "prism"))

    cell_centres = np.asarray([np.mean(points[np.asarray(cell)], axis=0) for cell in cells])
    face_cells = {}
    face_nodes = {}
    for cell_index, (cell, family) in enumerate(zip(cells, families, strict=True)):
        for face in _face_templates(cell, family):
            key = tuple(sorted(face))
            face_cells.setdefault(key, []).append(cell_index)
            face_nodes.setdefault(key, face)

    internal = []
    boundary_groups = {name: [] for name in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")}
    for key, adjacent in face_cells.items():
        if len(adjacent) == 2:
            owner, neighbour = sorted(adjacent)
            direction = cell_centres[neighbour] - cell_centres[owner]
            internal.append((_orient_face(face_nodes[key], points, direction), owner, neighbour))
        elif len(adjacent) == 1:
            owner = adjacent[0]
            face = face_nodes[key]
            centre = np.mean(points[np.asarray(face)], axis=0)
            axis = int(np.argmax(np.maximum(np.isclose(centre, 0.0), np.isclose(centre, 1.0))))
            side = "min" if np.isclose(centre[axis], 0.0) else "max"
            name = f"{'xyz'[axis]}{side}"
            direction = centre - cell_centres[owner]
            boundary_groups[name].append((_orient_face(face, points, direction), owner))
        else:
            raise ValueError(f"Non-manifold generated face {key}: {adjacent}")

    faces = [entry[0] for entry in internal]
    owners = [entry[1] for entry in internal]
    neighbours = [entry[2] for entry in internal]
    n_internal = len(faces)
    boundary = []
    for name, entries in boundary_groups.items():
        boundary.append(
            {"name": name, "startFace": len(faces), "nFaces": len(entries), "type": "patch"}
        )
        faces.extend(entry[0] for entry in entries)
        owners.extend(entry[1] for entry in entries)

    return {
        "points": points,
        "faces": faces,
        "owners": np.asarray(owners, dtype=np.int32),
        "neighbours": np.asarray(neighbours, dtype=np.int32),
        "boundary": boundary,
        "n_elements": len(cells),
        "n_faces": len(faces),
        "n_interior_faces": n_internal,
        "n_points": len(points),
        "cell_families": np.asarray(families),
        "cell_orders": np.ones(len(cells), dtype=np.int8),
    }
