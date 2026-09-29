"""Independent polyhedral fixtures with native inward wall orientation."""

import numpy as np

from source.coupler.geometry import SolidBoundary, TriangulatedWall


def voxel_wall(cells, *, origin=(0, 0, 0), spacing=1.0, rotation=None):
    cells = set(cells)
    triangles = []
    for cell in sorted(cells):
        for axis in range(3):
            transverse = [a for a in range(3) if a != axis]
            for sign in (-1, 1):
                neighbour = list(cell)
                neighbour[axis] += sign
                if tuple(neighbour) in cells:
                    continue
                face = np.tile(np.asarray(cell, dtype=float), (4, 1))
                face[:, axis] += sign > 0
                face[:, transverse] += np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
                normal = np.cross(face[1] - face[0], face[2] - face[0])
                if normal[axis] * sign > 0:  # Native normal points into the solid.
                    face = face[::-1]
                triangles.extend((face[[0, 1, 2]], face[[0, 2, 3]]))
    triangles = np.asarray(origin) + spacing * np.asarray(triangles)
    if rotation is not None:
        triangles = triangles @ np.asarray(rotation).T
    return TriangulatedWall(triangles, [-4, 4] * 3)


def rotation():
    a, b = 0.37, 0.23
    rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    return rz @ ry


def wall_case(name):
    if name == "curved":
        angle = np.linspace(0, 2 * np.pi, 65)[:-1]
        lower = np.column_stack((0.5 * np.cos(angle), 0.5 * np.sin(angle), -np.ones(64)))
        upper = lower.copy()
        upper[:, 2] = 1
        triangles = []
        for i in range(64):
            j = (i + 1) % 64
            triangles.extend(
                (
                    [lower[i], upper[j], lower[j]],
                    [lower[i], upper[i], upper[j]],
                    [[0, 0, -1], lower[i], lower[j]],
                    [[0, 0, 1], upper[j], upper[i]],
                )
            )
        wall = TriangulatedWall(np.array(triangles), [-4, 4] * 3)
        normal = np.array([np.cos(np.pi / 64), np.sin(np.pi / 64), 0])
        return SolidBoundary((wall,)), 0.6 * normal, 0.49 * normal, normal
    if name == "rotated":
        matrix = rotation()
        wall = voxel_wall([(0, 0, 0)], origin=(-0.5, -0.5, -0.5), rotation=matrix)
        start, end = np.array([[0.6, 0.12, -0.1], [0.49, 0.12, -0.1]]) @ matrix.T
        return SolidBoundary((wall,)), start, end, matrix[:, 0]
    if name == "concave":
        wall = voxel_wall([(0, 0, 0), (1, 0, 0), (0, 1, 0)], origin=(0, 0, -0.5))
        return (
            SolidBoundary((wall,)),
            np.array([1.1, 1.5, 0]),
            np.array([0.99, 1.5, 0]),
            np.array([1, 0, 0]),
        )
    if name == "thin":
        wall = TriangulatedWall.from_box([-0.01, 0.01, -1, 1, -1, 1], [-4, 4] * 3)
        return (
            SolidBoundary((wall,)),
            np.array([0.03, 0, 0]),
            np.array([-0.02, 0, 0]),
            np.array([1, 0, 0]),
        )
    if name == "multiple":
        bodies = [
            TriangulatedWall.from_box(bounds, [-4, 4] * 3)
            for bounds in (
                [-1.1, -0.1, -0.5, 0.5, -0.5, 0.5],
                [0.1, 1.1, -0.5, 0.5, -0.5, 0.5],
            )
        ]
        return (
            SolidBoundary(bodies),
            np.array([0, 0, 0]),
            np.array([0.11, 0, 0]),
            np.array([-1, 0, 0]),
        )
    raise ValueError(name)
