# SPDX-License-Identifier: GPL-3.0-or-later
"""Deterministic STL fixtures for the Cartesian-mesher acceptance matrix.

The fixture builders deliberately live under ``tests`` and contain no mesh
generation or solver imports. They only create triangulated surfaces that the
future public mesher must ingest. Coordinates are generated from integer grid
indices or fixed analytic parameters so repeated test runs produce identical
bytes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class AcceptanceFixture:
    """A named surface fixture and the file paths that represent it."""

    name: str
    paths: tuple[Path, ...]
    purpose: str


def _write_ascii_stl(path: Path, triangles: np.ndarray, name: str) -> Path:
    """Write triangles as deterministic ASCII STL and return ``path``."""
    lines = [f"solid {name}"]
    for triangle in np.asarray(triangles, dtype=np.float64):
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        length = float(np.linalg.norm(normal))
        if length:
            normal /= length
        lines.append(f"  facet normal {normal[0]:.17g} {normal[1]:.17g} {normal[2]:.17g}")
        lines.append("    outer loop")
        for point in triangle:
            lines.append(f"      vertex {point[0]:.17g} {point[1]:.17g} {point[2]:.17g}")
        lines.extend(("    endloop", "  endfacet"))
    lines.append(f"endsolid {name}")
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return path


def box_triangles(
    bounds: tuple[float, float, float, float, float, float],
) -> np.ndarray:
    """Return twelve outward-oriented triangles for an axis-aligned box."""
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    points = np.asarray(
        (
            (xmin, ymin, zmin),
            (xmax, ymin, zmin),
            (xmax, ymax, zmin),
            (xmin, ymax, zmin),
            (xmin, ymin, zmax),
            (xmax, ymin, zmax),
            (xmax, ymax, zmax),
            (xmin, ymax, zmax),
        ),
        dtype=np.float64,
    )
    faces = np.asarray(
        (
            (0, 2, 1),
            (0, 3, 2),
            (4, 5, 6),
            (4, 6, 7),
            (0, 4, 7),
            (0, 7, 3),
            (1, 2, 6),
            (1, 6, 5),
            (0, 1, 5),
            (0, 5, 4),
            (3, 7, 6),
            (3, 6, 2),
        ),
        dtype=np.int64,
    )
    return points[faces]


def _rotate_z(
    triangles: np.ndarray, angle_degrees: float, translation: tuple[float, ...]
) -> np.ndarray:
    """Rotate triangles around z and then translate them."""
    angle = np.deg2rad(angle_degrees)
    matrix = np.asarray(
        (
            (np.cos(angle), -np.sin(angle), 0.0),
            (np.sin(angle), np.cos(angle), 0.0),
            (0.0, 0.0, 1.0),
        )
    )
    return np.einsum("ij,nkj->nki", matrix, triangles) + np.asarray(translation)


def sphere_triangles(
    radii: tuple[float, float, float] = (0.55, 0.45, 0.40),
    *,
    latitude_count: int = 8,
    longitude_count: int = 16,
) -> np.ndarray:
    """Return a closed latitude/longitude ellipsoid tessellation."""
    if latitude_count < 2 or longitude_count < 3:
        raise ValueError(
            "The ellipsoid fixture needs at least two latitude bands and three longitudes"
        )
    rx, ry, rz = radii
    vertices = [np.asarray((0.0, 0.0, rz), dtype=np.float64)]
    for latitude in range(1, latitude_count):
        theta = np.pi * latitude / latitude_count
        for longitude in range(longitude_count):
            phi = 2.0 * np.pi * longitude / longitude_count
            vertices.append(
                (
                    rx * np.sin(theta) * np.cos(phi),
                    ry * np.sin(theta) * np.sin(phi),
                    rz * np.cos(theta),
                )
            )
    bottom = len(vertices)
    vertices.append(np.asarray((0.0, 0.0, -rz), dtype=np.float64))

    def ring_id(latitude: int, longitude: int) -> int:
        return 1 + (latitude - 1) * longitude_count + longitude % longitude_count

    faces: list[tuple[int, int, int]] = []
    for longitude in range(longitude_count):
        next_longitude = (longitude + 1) % longitude_count
        faces.append((0, ring_id(1, next_longitude), ring_id(1, longitude)))
    for latitude in range(1, latitude_count - 1):
        for longitude in range(longitude_count):
            next_longitude = (longitude + 1) % longitude_count
            lower_left = ring_id(latitude, longitude)
            lower_right = ring_id(latitude, next_longitude)
            upper_left = ring_id(latitude + 1, longitude)
            upper_right = ring_id(latitude + 1, next_longitude)
            faces.extend(
                (
                    (lower_left, lower_right, upper_right),
                    (lower_left, upper_right, upper_left),
                )
            )
    for longitude in range(longitude_count):
        next_longitude = (longitude + 1) % longitude_count
        faces.append(
            (
                bottom,
                ring_id(latitude_count - 1, longitude),
                ring_id(latitude_count - 1, next_longitude),
            )
        )
    return np.asarray(vertices, dtype=np.float64)[np.asarray(faces, dtype=np.int64)]


def torus_triangles(
    major_radius: float = 0.70,
    minor_radius: float = 0.22,
    *,
    major_count: int = 16,
    minor_count: int = 8,
) -> np.ndarray:
    """Return a closed concave torus tessellation."""
    vertices = np.empty((major_count * minor_count, 3), dtype=np.float64)
    for major in range(major_count):
        u = 2.0 * np.pi * major / major_count
        for minor in range(minor_count):
            v = 2.0 * np.pi * minor / minor_count
            radius = major_radius + minor_radius * np.cos(v)
            vertices[major * minor_count + minor] = (
                radius * np.cos(u),
                radius * np.sin(u),
                minor_radius * np.sin(v),
            )

    def vertex_id(major: int, minor: int) -> int:
        return (major % major_count) * minor_count + minor % minor_count

    faces = []
    for major in range(major_count):
        for minor in range(minor_count):
            faces.extend(
                (
                    (
                        vertex_id(major, minor),
                        vertex_id(major + 1, minor),
                        vertex_id(major + 1, minor + 1),
                    ),
                    (
                        vertex_id(major, minor),
                        vertex_id(major + 1, minor + 1),
                        vertex_id(major, minor + 1),
                    ),
                )
            )
    return vertices[np.asarray(faces, dtype=np.int64)]


def finite_wing_triangles(
    *, chord_count: int = 12, span: float = 0.80, thickness: float = 0.12
) -> np.ndarray:
    """Return a closed finite NACA-like wing as a thin extruded solid."""
    x_values = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, chord_count + 1)))
    thickness_values = (
        5.0
        * thickness
        * (
            0.2969 * np.sqrt(x_values)
            - 0.1260 * x_values
            - 0.3516 * x_values**2
            + 0.2843 * x_values**3
            - 0.1015 * x_values**4
        )
    )
    outline = np.concatenate(
        (
            np.column_stack((x_values, thickness_values)),
            np.column_stack((x_values[-2:0:-1], -thickness_values[-2:0:-1])),
        )
    )
    vertices = np.vstack(
        (
            np.column_stack((outline[:, 0], np.full(len(outline), -span / 2.0), outline[:, 1])),
            np.column_stack((outline[:, 0], np.full(len(outline), span / 2.0), outline[:, 1])),
        )
    )
    count = len(outline)
    cap_centres = np.asarray(
        (
            (outline[:, 0].mean(), -span / 2.0, outline[:, 1].mean()),
            (outline[:, 0].mean(), span / 2.0, outline[:, 1].mean()),
        ),
        dtype=np.float64,
    )
    vertices = np.vstack((vertices, cap_centres))
    negative_centre = len(vertices) - 2
    positive_centre = len(vertices) - 1
    faces: list[tuple[int, int, int]] = []
    for index in range(count):
        next_index = (index + 1) % count
        faces.extend(
            (
                (index, next_index, count + next_index),
                (index, count + next_index, count + index),
            )
        )
    for index in range(count):
        next_index = (index + 1) % count
        faces.append((negative_centre, next_index, index))
        faces.append((positive_centre, count + index, count + next_index))
    return vertices[np.asarray(faces, dtype=np.int64)]


def make_acceptance_fixtures(directory: Path) -> dict[str, AcceptanceFixture]:
    """Create all valid and deliberately broken acceptance fixtures."""
    directory.mkdir(parents=True, exist_ok=True)

    rotated_box = _rotate_z(
        box_triangles((-0.45, 0.45, -0.35, 0.35, -0.30, 0.30)), 41.0, (0.2, -0.1, 0.0)
    )
    two_body_a = box_triangles((-0.85, -0.35, -0.30, 0.30, -0.30, 0.30))
    two_body_b = box_triangles((0.35, 0.85, -0.25, 0.25, -0.25, 0.25))
    valid = {
        "rotated_box": (
            "rotated_box.stl",
            rotated_box,
            "planar patches, sharp edges, and orientation invariance",
        ),
        "ellipsoid": ("ellipsoid.stl", sphere_triangles(), "smooth arbitrary curvature"),
        "torus": ("torus.stl", torus_triangles(), "concavity and genus"),
        "finite_naca_wing": (
            "finite_naca_wing.stl",
            finite_wing_triangles(),
            "mixed curvature and finite tips",
        ),
    }
    result = {
        name: AcceptanceFixture(
            name, (_write_ascii_stl(directory / filename, triangles, name),), purpose
        )
        for name, (filename, triangles, purpose) in valid.items()
    }
    result["two_disjoint_bodies"] = AcceptanceFixture(
        "two_disjoint_bodies",
        (
            _write_ascii_stl(directory / "body_a.stl", two_body_a, "body_a"),
            _write_ascii_stl(directory / "body_b.stl", two_body_b, "body_b"),
        ),
        "multiple surfaces and patch identity",
    )

    box = box_triangles((-0.4, 0.4, -0.4, 0.4, -0.4, 0.4))
    result["open_edge"] = AcceptanceFixture(
        "open_edge",
        (_write_ascii_stl(directory / "open_edge.stl", box[:-1], "open_edge"),),
        "failure on an open edge",
    )
    result["non_manifold_edge"] = AcceptanceFixture(
        "non_manifold_edge",
        (
            _write_ascii_stl(
                directory / "non_manifold_edge.stl",
                np.concatenate((box, box[:1])),
                "non_manifold_edge",
            ),
        ),
        "failure on a non-manifold edge",
    )
    inverted_component = np.concatenate(
        (box, box_triangles((0.8, 1.2, -0.25, 0.25, -0.25, 0.25))[::-1, ::-1])
    )
    result["inverted_component"] = AcceptanceFixture(
        "inverted_component",
        (
            _write_ascii_stl(
                directory / "inverted_component.stl", inverted_component, "inverted_component"
            ),
        ),
        "failure on an inverted component",
    )
    degenerate = box.copy()
    degenerate[0, 1] = degenerate[0, 0]
    result["degenerate_triangle"] = AcceptanceFixture(
        "degenerate_triangle",
        (
            _write_ascii_stl(
                directory / "degenerate_triangle.stl", degenerate, "degenerate_triangle"
            ),
        ),
        "failure on a degenerate triangle",
    )
    return result


__all__ = [
    "AcceptanceFixture",
    "box_triangles",
    "finite_wing_triangles",
    "make_acceptance_fixtures",
    "sphere_triangles",
    "torus_triangles",
]
