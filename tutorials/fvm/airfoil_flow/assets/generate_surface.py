"""NACA 0012 surface geometry for the airfoil experiment."""

from pathlib import Path

import numpy as np


def create_airfoil_surface(path: Path, chord: float, depth: float, chord_count: int = 80) -> Path:
    """Write a closed, finite-span NACA 0012 surface."""

    x_values = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, chord_count + 1)))
    thickness_values = (
        5.0
        * 0.12
        * (
            0.2969 * np.sqrt(x_values)
            - 0.1260 * x_values
            - 0.3516 * x_values**2
            + 0.2843 * x_values**3
            - 0.1036 * x_values**4
        )
    )
    outline = np.concatenate(
        (
            np.column_stack((x_values, thickness_values)),
            np.column_stack((x_values[-2:0:-1], -thickness_values[-2:0:-1])),
        )
    )
    outline[:, 0] *= chord
    outline[:, 0] -= 0.25 * chord
    outline[:, 1] *= chord

    vertices = np.vstack(
        (
            np.column_stack((outline[:, 0], outline[:, 1], np.full(len(outline), -depth / 2))),
            np.column_stack((outline[:, 0], outline[:, 1], np.full(len(outline), depth / 2))),
        )
    )
    count = len(outline)
    centre = np.asarray((outline[:, 0].mean(), outline[:, 1].mean()), dtype=float)
    vertices = np.vstack(
        (
            vertices,
            (centre[0], centre[1], -depth / 2),
            (centre[0], centre[1], depth / 2),
        )
    )
    negative_centre, positive_centre = len(vertices) - 2, len(vertices) - 1
    faces: list[tuple[int, int, int]] = []
    for index in range(count):
        next_index = (index + 1) % count
        faces.extend(
            (
                (index, next_index, count + next_index),
                (index, count + next_index, count + index),
                (negative_centre, next_index, index),
                (positive_centre, count + index, count + next_index),
            )
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["solid openonda_naca0012"]
    for face in faces:
        triangle = vertices[np.asarray(face)]
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        magnitude = float(np.linalg.norm(normal))
        if magnitude > 0.0:
            normal /= magnitude
        lines.append(f"  facet normal {normal[0]:.9e} {normal[1]:.9e} {normal[2]:.9e}")
        lines.append("    outer loop")
        for vertex in triangle:
            lines.append(f"      vertex {vertex[0]:.9e} {vertex[1]:.9e} {vertex[2]:.9e}")
        lines.extend(("    endloop", "  endfacet"))
    lines.append("endsolid openonda_naca0012")
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return path
