"""Closed NACA four-digit wing with flat end caps, written as an STL surface."""

import numpy as np

from openonda.core.file_output import write_text_atomic


def naca4_section(code, chord, n_points):
    """Lower and upper surface points (x, y) from the leading to the trailing edge.

    The chord lies along x, centred on the origin. Cosine spacing clusters the
    points at both edges, and the thickness polynomial closes the trailing edge.
    """
    camber = int(code[0]) / 100.0
    camber_position = int(code[1]) / 10.0
    thickness = int(code[2:]) / 100.0
    x = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, n_points)))

    half_thickness = (
        5.0
        * thickness
        * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1036 * x**4)
    )
    front = x < camber_position
    camber_line = np.where(
        front,
        camber / camber_position**2 * (2.0 * camber_position * x - x**2),
        camber
        / (1.0 - camber_position) ** 2
        * (1.0 - 2.0 * camber_position + 2.0 * camber_position * x - x**2),
    )
    slope = np.where(
        front,
        2.0 * camber / camber_position**2 * (camber_position - x),
        2.0 * camber / (1.0 - camber_position) ** 2 * (camber_position - x),
    )

    # Thickness is laid off normal to the camber line
    angle = np.arctan(slope)
    lower = np.column_stack(
        (x + half_thickness * np.sin(angle), camber_line - half_thickness * np.cos(angle))
    )
    upper = np.column_stack(
        (x - half_thickness * np.sin(angle), camber_line + half_thickness * np.cos(angle))
    )
    centre = np.array([0.5, 0.0])
    return chord * (lower - centre), chord * (upper - centre)


def section_outline(code, chord, n_points):
    """Counter-clockwise outline: lower surface to the trailing edge, upper surface back.

    The leading and trailing edges appear once, so both surfaces share them.
    """
    lower, upper = naca4_section(code, chord, n_points)
    return np.vstack((lower[:-1], upper[:0:-1]))


def write_wing_stl(path, code, chord, span, n_points, n_span_segments):
    """Write the wing extruded along z over the span, closed by flat tip caps.

    Every edge is shared by exactly two triangles and every normal points out
    of the solid, so the surface is watertight and consistently oriented.
    """
    outline = section_outline(code, chord, n_points)
    n_outline = len(outline)
    n_chord = n_outline // 2  # outline index of the trailing edge
    stations = np.linspace(-0.5 * span, 0.5 * span, n_span_segments + 1)

    # One outline ring of vertices per spanwise station
    vertices = np.vstack([np.column_stack((outline, np.full(n_outline, z))) for z in stations])

    triangles = []
    for k in range(n_span_segments):
        root, tip = k * n_outline, (k + 1) * n_outline
        for i in range(n_outline):
            j = (i + 1) % n_outline
            triangles += [(root + i, root + j, tip + j), (root + i, tip + j, tip + i)]

    # Tip caps: strips between lower point i and upper point i at equal chord position
    def lower_point(i):
        return i

    def upper_point(i):
        return (n_outline - i) % n_outline

    cap = [(lower_point(i), lower_point(i + 1), upper_point(i + 1)) for i in range(n_chord - 1)]
    cap += [(lower_point(i), upper_point(i + 1), upper_point(i)) for i in range(1, n_chord)]
    tip = n_span_segments * n_outline
    triangles += [(tip + a, tip + b, tip + c) for a, b, c in cap]
    triangles += [(a, c, b) for a, b, c in cap]

    lines = ["solid naca" + code]
    for triangle in vertices[np.asarray(triangles)]:
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        normal /= np.linalg.norm(normal)
        lines += [
            f"  facet normal {normal[0]:.9e} {normal[1]:.9e} {normal[2]:.9e}",
            "    outer loop",
        ]
        lines += [f"      vertex {x:.9e} {y:.9e} {z:.9e}" for x, y, z in triangle]
        lines += ["    endloop", "  endfacet"]
    lines.append("endsolid naca" + code)

    # Each MPI rank writes the same file; an atomic replace never exposes a partial one
    write_text_atomic(path, "\n".join(lines) + "\n")
    return path
