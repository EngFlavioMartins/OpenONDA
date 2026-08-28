"""Generate the benchmark's long watertight cylinder STL.

The remote caps lie outside the solved four-diameter span. The mesh therefore
contains the canonical circular-cylinder side wall and clips it at two slip
planes without introducing finite-cylinder cap forces.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

CASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT = CASE_DIR / "assets" / "cylinder_long.stl"
RADIUS = 0.5
Z_MIN = -6.0
Z_MAX = 6.0
CIRCUMFERENTIAL_SEGMENTS = 128
AXIAL_SEGMENTS = 4


def _facet(stream, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> None:
    normal = np.cross(b - a, c - a)
    normal /= np.linalg.norm(normal)
    stream.write(f"  facet normal {normal[0]:.12g} {normal[1]:.12g} {normal[2]:.12g}\n")
    stream.write("    outer loop\n")
    for point in (a, b, c):
        stream.write(f"      vertex {point[0]:.12g} {point[1]:.12g} {point[2]:.12g}\n")
    stream.write("    endloop\n  endfacet\n")


def main() -> None:
    angles = np.linspace(0.0, 2.0 * math.pi, CIRCUMFERENTIAL_SEGMENTS + 1)
    z_values = np.linspace(Z_MIN, Z_MAX, AXIAL_SEGMENTS + 1)
    with OUTPUT.open("w", encoding="ascii", newline="\n") as stream:
        # Preserve the original ASCII solid label so regeneration remains
        # byte-for-byte identical to the versioned geometry.
        stream.write("solid openonda_cylinder_ar4\n")
        for k in range(AXIAL_SEGMENTS):
            z0, z1 = z_values[k : k + 2]
            for i in range(CIRCUMFERENTIAL_SEGMENTS):
                t0, t1 = angles[i : i + 2]
                b0 = np.array([RADIUS * math.cos(t0), RADIUS * math.sin(t0), z0])
                b1 = np.array([RADIUS * math.cos(t1), RADIUS * math.sin(t1), z0])
                u0 = np.array([RADIUS * math.cos(t0), RADIUS * math.sin(t0), z1])
                u1 = np.array([RADIUS * math.cos(t1), RADIUS * math.sin(t1), z1])
                _facet(stream, b0, b1, u1)
                _facet(stream, b0, u1, u0)

        top_centre = np.array([0.0, 0.0, Z_MAX])
        bottom_centre = np.array([0.0, 0.0, Z_MIN])
        for i in range(CIRCUMFERENTIAL_SEGMENTS):
            t0, t1 = angles[i : i + 2]
            top0 = np.array([RADIUS * math.cos(t0), RADIUS * math.sin(t0), Z_MAX])
            top1 = np.array([RADIUS * math.cos(t1), RADIUS * math.sin(t1), Z_MAX])
            bottom0 = np.array([RADIUS * math.cos(t0), RADIUS * math.sin(t0), Z_MIN])
            bottom1 = np.array([RADIUS * math.cos(t1), RADIUS * math.sin(t1), Z_MIN])
            _facet(stream, top_centre, top0, top1)
            _facet(stream, bottom_centre, bottom1, bottom0)
        stream.write("endsolid openonda_cylinder_ar4\n")
    print(
        f"wrote {OUTPUT.relative_to(CASE_DIR)}: "
        f"{2 * CIRCUMFERENTIAL_SEGMENTS * (AXIAL_SEGMENTS + 1)} triangles"
    )


if __name__ == "__main__":
    main()
