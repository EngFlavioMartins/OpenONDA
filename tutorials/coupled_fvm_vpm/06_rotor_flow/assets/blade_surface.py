"""Closed blade surfaces lofted through the sections of the VLM rotor blade, written as STL.

The VLM blade of ``tutorials/vpm/06_rotor_flow`` is a flat lifting surface
whose segments give the leading and trailing edge at 23 radial stations. The
same chord lines carry a NACA four-digit thickness here, so the two rotors share
their planform, twist and operating point and differ only in the blade
section model.
"""

import json

import numpy as np

from openonda.core.file_output import write_text_atomic

ROTATION_AXIS = np.array([1.0, 0.0, 0.0])
# The section ends in a flat base at this chord fraction: a sharp trailing edge thinner than
# the blade cells breaks the mesher's wall projection
TRAILING_EDGE_CUT = 0.95


def naca_half_thickness(x, thickness):
    """Half thickness of a closed-trailing-edge NACA four-digit section at chord fraction x."""
    return (
        5.0
        * thickness
        * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1036 * x**4)
    )


def blade_stations(blade_file):
    """Leading and trailing edge points, shape ``(n_stations, 3)`` each, from root to tip."""
    segments = json.loads(blade_file.read_text(encoding="utf-8"))["wings"][0]["segments"]
    leading = [np.array(segment["vertex_position"]["a"]) for segment in segments]
    trailing = [np.array(segment["vertex_position"]["d"]) for segment in segments]
    leading.append(np.array(segments[-1]["vertex_position"]["b"]))
    trailing.append(np.array(segments[-1]["vertex_position"]["c"]))
    return np.array(leading), np.array(trailing)


def section_ring(leading_edge, trailing_edge, thickness, n_points):
    """Outline of one section: lower surface to the blunt trailing edge, upper surface back.

    The thickness is laid off normal to the chord line, in the plane normal
    to the radial direction of the station; the base closes the section at
    ``TRAILING_EDGE_CUT`` of the chord.
    """
    chord = trailing_edge - leading_edge
    radial = np.cross(ROTATION_AXIS, np.cross(ROTATION_AXIS, 0.5 * (leading_edge + trailing_edge)))
    radial /= np.linalg.norm(radial)
    normal = np.cross(radial, chord)
    normal /= np.linalg.norm(normal)
    x = TRAILING_EDGE_CUT * 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, n_points)))
    offset = np.outer(naca_half_thickness(x, thickness) * np.linalg.norm(chord), normal)
    lower = leading_edge + np.outer(x, chord) - offset
    upper = leading_edge + np.outer(x, chord) + offset
    # The leading edge appears once; the base edge joins the two trailing-edge points
    return np.vstack((lower, upper[:0:-1]))


def rotate_about_axis(points, angle_degrees):
    """Rotate points about the rotor axis by ``angle_degrees`` (right-hand rule)."""
    angle = np.radians(angle_degrees)
    cos, sin = np.cos(angle), np.sin(angle)
    rotation = np.array([[1.0, 0.0, 0.0], [0.0, cos, -sin], [0.0, sin, cos]])
    return points @ rotation.T


def write_blade_stl(path, blade_file, thickness, n_points, azimuth_degrees):
    """Write one blade, lofted through its stations and closed at root and tip."""
    leading, trailing = blade_stations(blade_file)
    rings = [
        rotate_about_axis(section_ring(le, te, thickness, n_points), azimuth_degrees)
        for le, te in zip(leading, trailing)
    ]
    n_outline = len(rings[0])
    n_chord = (n_outline + 1) // 2  # points of the lower surface, the base point included
    vertices = np.vstack(rings)

    triangles = []
    for k in range(len(rings) - 1):
        root, tip = k * n_outline, (k + 1) * n_outline
        for i in range(n_outline):
            j = (i + 1) % n_outline
            triangles += [(root + i, root + j, tip + j), (root + i, tip + j, tip + i)]

    # Caps: strips between lower point i and upper point i at equal chord position; the last
    # strip ends on the base edge
    def lower_point(i):
        return i

    def upper_point(i):
        return (n_outline - i) % n_outline

    cap = [(lower_point(i), lower_point(i + 1), upper_point(i + 1)) for i in range(n_chord - 1)]
    cap += [(lower_point(i), upper_point(i + 1), upper_point(i)) for i in range(1, n_chord - 1)]
    tip = (len(rings) - 1) * n_outline
    triangles += [(tip + a, tip + b, tip + c) for a, b, c in cap]
    triangles += [(a, c, b) for a, b, c in cap]

    # Orient every triangle outward: the cap normals must point away from the blade
    lines = [f"solid blade{azimuth_degrees:g}"]
    for triangle in vertices[np.asarray(triangles)]:
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        normal /= np.linalg.norm(normal)
        lines += [
            f"  facet normal {normal[0]:.9e} {normal[1]:.9e} {normal[2]:.9e}",
            "    outer loop",
        ]
        lines += [f"      vertex {x:.9e} {y:.9e} {z:.9e}" for x, y, z in triangle]
        lines += ["    endloop", "  endfacet"]
    lines.append(f"endsolid blade{azimuth_degrees:g}")
    write_text_atomic(path, "\n".join(lines) + "\n")
    return path
