"""
Generate Flat Plate Surface Geometry
====================================
Utilities for creating and exporting flat plate VLM surface geometries.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: January 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

import numpy as np
import json
from typing import Optional

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


def save_surface(aircraft: Aircraft, filepath: str):
    """
    Save aircraft surface geometry to JSON file.

    Args:
        aircraft: Aircraft object to save
        filepath: Output filepath (without extension, .json will be added)
    """
    data = {"uid": aircraft.uid, "wings": []}

    for wing in aircraft.wings.values():
        wing_data = {"uid": wing.uid, "symmetry": wing.symmetry, "segments": []}

        for segment in wing.segments.values():
            seg_data = {
                "uid": segment.uid,
                "vertex_position": {k: v.tolist() for k, v in segment.vertex_position.items()},
                "n_chordwise_panels": segment.n_chordwise_panels,
                "n_spanwise_panels": segment.n_spanwise_panels,
                "airfoils": segment.airfoils,
            }
            wing_data["segments"].append(seg_data)

        data["wings"].append(wing_data)

    # Add reference values
    refs = aircraft.refs
    data["refs"] = {
        "area": refs.get("area", 1.0),
        "chord": refs.get("chord", 1.0),
        "span": refs.get("span", 1.0),
        "geometry_centre": refs.get("geometry_centre", np.zeros(3)).tolist()
        if hasattr(refs.get("geometry_centre", np.zeros(3)), "tolist")
        else list(refs.get("geometry_centre", [0, 0, 0])),
        "reference_point": refs.get("reference_point", np.zeros(3)).tolist()
        if hasattr(refs.get("reference_point", np.zeros(3)), "tolist")
        else list(refs.get("reference_point", [0, 0, 0])),
    }

    output_path = filepath if filepath.endswith(".json") else f"{filepath}.json"
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Surface saved to: {output_path}")
    return output_path


def load_surface(filepath: str) -> Aircraft:
    """
    Load aircraft surface geometry from JSON file.

    Args:
        filepath: Path to surface JSON file

    Returns:
        Aircraft: Loaded aircraft object
    """
    input_path = filepath if filepath.endswith(".json") else f"{filepath}.json"

    with open(input_path, "r") as f:
        data = json.load(f)

    aircraft = Aircraft(uid=data["uid"])

    for wing_data in data["wings"]:
        wing = Wing(uid=wing_data["uid"], symmetry=wing_data.get("symmetry", 1))

        for seg_data in wing_data["segments"]:
            vertex_position = {k: np.array(v) for k, v in seg_data["vertex_position"].items()}
            segment = WingSegment(
                uid=seg_data["uid"],
                vertex_position=vertex_position,
                n_chordwise_panels=seg_data["n_chordwise_panels"],
                n_spanwise_panels=seg_data["n_spanwise_panels"],
                airfoils=seg_data.get("airfoils", {"inner": "flat", "outer": "flat"}),
            )
            wing.add_segment(segment)

        aircraft.add_wing(wing)

    # Restore reference values if present
    if "refs" in data:
        aircraft.refs = {
            "area": data["refs"].get("area", 1.0),
            "chord": data["refs"].get("chord", 1.0),
            "span": data["refs"].get("span", 1.0),
            "geometry_centre": np.array(data["refs"].get("geometry_centre", [0.0, 0.0, 0.0])),
            "reference_point": np.array(data["refs"].get("reference_point", [0.0, 0.0, 0.0])),
        }
    else:
        aircraft.compute_default_refs()

    print(f"Surface loaded from: {input_path}")
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
