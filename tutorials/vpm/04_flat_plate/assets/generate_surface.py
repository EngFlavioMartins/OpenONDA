"""
Generate Flat Plate Surface Geometry
====================================
Utilities for creating and exporting flat plate VLM surface geometries.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: January 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

import numpy as np
from source.solvers.vpm.boundary_elements.vlm.geometry.surface_io import load_surface, save_surface

from source.solvers.vpm.boundary_elements.vlm.geometry.aircraft import Aircraft, Wing, WingSegment


def create_flat_plate_vertices(
    chord: float,
    half_span: float,
    angle_of_attack_degrees: float = 0.0,
) -> dict:
    """
    Create vertex_position for a flat rectangular plate at angle of attack.

    The plate is aligned with X axis at root (Y=0) and extends to half_span.
    Angle of attack rotates the plate about the Y axis.

    Vertex layout (looking from above, +Z up)::

        a -------- b    (Leading edge)
        |          |
        |          |
        d -------- c    (Trailing edge)

    For a rectangular plate:
        - Point a: LE root at (0, 0, 0)
        - Point b: LE tip at (0, half_span, 0)
        - Point c: TE tip at (chord, half_span, 0)
        - Point d: TE root at (chord, 0, 0)

    Args:
        chord: Chord length [m]
        half_span: Half-span (from root to tip) [m]
        angle_of_attack_degrees: Angle of attack [degrees]

    Returns:
        Dictionary with vertex_position 'a', 'b', 'c', 'd' as numpy arrays
    """
    alpha = np.radians(angle_of_attack_degrees)

    # Define base vertex_position (before alpha rotation)
    # a: LE root
    a_base = np.array([0.0, 0.0, 0.0])
    # b: LE tip
    b_base = np.array([0.0, half_span, 0.0])
    # c: TE tip
    c_base = np.array([chord, half_span, 0.0])
    # d: TE root
    d_base = np.array([chord, 0.0, 0.0])

    # Rotation matrix about Y axis (positive alpha = nose up)
    cos_a = np.cos(alpha)
    sin_a = np.sin(alpha)

    def rotate_point(point):
        """Rotate point about Y axis by alpha."""
        x_rot = point[0] * cos_a + point[2] * sin_a
        z_rot = -point[0] * sin_a + point[2] * cos_a
        return np.array([x_rot, point[1], z_rot])

    vertex_position = {
        "a": rotate_point(a_base),
        "b": rotate_point(b_base),
        "c": rotate_point(c_base),
        "d": rotate_point(d_base),
    }

    return vertex_position


def create_flat_plate(
    chord: float = 0.5,
    span: float = 1.0,
    angle_of_attack_degrees: float = 0.0,
    n_chordwise_panels: int = 8,
    n_spanwise_panels: int = 20,
) -> Aircraft:
    """
    Create a flat rectangular plate aircraft geometry.

    Args:
        chord: Chord length [m]
        span: Full span [m] (will be divided by 2 for half_span)
        angle_of_attack_degrees: Angle of attack [deg]
        n_chordwise_panels: Chordwise panels
        n_spanwise_panels: Spanwise panels (per side)

    Returns:
        Aircraft: Configured aircraft object
    """
    aircraft = Aircraft(uid="flat_plate")

    # Create wing with Y-symmetry (XZ plane mirror)
    wing = Wing(uid="main_wing", symmetry=2)

    # Create vertex_position for flat plate (use half-span)
    vertex_position = create_flat_plate_vertices(
        chord,
        span / 2.0,
        angle_of_attack_degrees,
    )

    # Wing segment: flat plate
    segment = WingSegment(
        uid="segment_0",
        vertex_position=vertex_position,
        n_chordwise_panels=n_chordwise_panels,
        n_spanwise_panels=n_spanwise_panels,
        airfoils={"inner": "flat", "outer": "flat"},
    )

    wing.add_segment(segment)
    aircraft.add_wing(wing)
    aircraft.compute_default_refs()

    return aircraft


if __name__ == "__main__":
    # Example: Generate and save a flat plate surface
    surface = create_flat_plate(
        chord=0.5,
        span=1.0,
        angle_of_attack_degrees=0.0,
        n_chordwise_panels=8,
        n_spanwise_panels=20,
    )
    save_surface(surface, "flat_plate_surface")
