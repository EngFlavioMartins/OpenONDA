"""
Generate Rotor Blade Surface Geometry
=====================================
Creates a flat-plate rotor blade for quadcopter hover simulations.

The blade lies in the X-Y plane (rotation about Z-axis) with:
  - Span along Y-axis: from R_hub to R_tip
  - Chord along X-axis: leading edge at +X, trailing edge at -X
  - Pitch (nose-up about Y): creates angle of attack relative to
    the tangential velocity (omega × r), producing downward thrust (-Z)

Vertex convention (looking from +Z, blade pointing in +Y):

    a -------- b    (Leading edge, +X side)
    |          |
    |          |
    d -------- c    (Trailing edge, -X side)

  a = LE root (x > 0, y = R_hub)
  b = LE tip  (x > 0, y = R_tip)
  c = TE tip  (x < 0, y = R_tip)
  d = TE root (x < 0, y = R_hub)

For CCW rotation (omega > 0 about Z), the blade moves in the +X
direction. With the LE at +X and TE at -X, the flow arrives from
+X → the blade has positive AoA when pitched nose-up.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: February 2026
"""

import numpy as np
import json
from source.solvers.vpm.boundary_elements.vlm.geometry.aircraft import Aircraft, Wing, WingSegment


def create_rotor_blade(
    R_hub: float = 0.03,
    R_tip: float = 0.15,
    chord_root: float = 0.025,
    chord_tip: float = 0.015,
    pitch_root_deg: float = 12.0,
    pitch_tip_deg: float = 6.0,
    n_chord: int = 4,
    n_span: int = 10,
    clockwise: bool = False,
) -> Aircraft:
    """
    Create a rotor blade surface as an Aircraft object.

    The blade lies in the X-Y plane (rotation about Z-axis).
    """
    aircraft = Aircraft(uid="rotor_blade")
    wing = Wing(uid="blade_0", symmetry=0)

    def pitched_vertices(chord, y, pitch_deg, cw=False):
        pitch = np.radians(pitch_deg)
        cos_p = np.cos(pitch)
        sin_p = np.sin(pitch)

        if not cw:
            # CCW: LE at +X, TE at -X (VLM convention)
            x_le = 0.75 * chord
            x_te = -0.25 * chord
            sign_z = +1.0
        else:
            # CW: LE at -X, TE at +X
            x_le = -0.75 * chord
            x_te = 0.25 * chord
            sign_z = -1.0

        le = np.array([x_le * cos_p, y, sign_z * x_le * sin_p])
        te = np.array([x_te * cos_p, y, sign_z * x_te * sin_p])
        return le, te

    # Root station
    le_root, te_root = pitched_vertices(chord_root, R_hub, pitch_root_deg, cw=clockwise)
    # Tip station
    le_tip, te_tip = pitched_vertices(chord_tip, R_tip, pitch_tip_deg, cw=clockwise)

    if clockwise:
        le_root, le_tip = le_tip, le_root
        te_root, te_tip = te_tip, te_root

    # Vertex mapping: a=LE_root, b=LE_tip, c=TE_tip, d=TE_root
    segment = WingSegment(
        uid="blade_segment",
        vertex_position={"a": le_root, "b": le_tip, "c": te_tip, "d": te_root},
        n_chordwise_panels=n_chord,
        n_spanwise_panels=n_span,
        airfoils={"inner": "flat", "outer": "flat"},
    )
    wing.add_segment(segment)
    aircraft.add_wing(wing)

    return aircraft


def save_blade(aircraft: Aircraft, filepath: str):
    """Save blade surface JSON using standard format."""
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

    path = filepath if filepath.endswith(".json") else f"{filepath}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Blade surface saved to: {path}")
    return path


if __name__ == "__main__":
    blade = create_rotor_blade()
    save_blade(blade, "blade.json")

    # Print summary
    verts = blade["wings"][0]["segments"][0]["vertex_position"]
    for k, v in verts.items():
        print(f"  {k}: [{v[0]:+.6f}, {v[1]:+.6f}, {v[2]:+.6f}]")

    span = np.linalg.norm(np.array(verts["b"]) - np.array(verts["a"]))
    chord_r = np.linalg.norm(np.array(verts["d"]) - np.array(verts["a"]))
    chord_t = np.linalg.norm(np.array(verts["c"]) - np.array(verts["b"]))
    print(f"\n  Span: {span:.4f} m")
    print(f"  Root chord: {chord_r:.4f} m")
    print(f"  Tip chord:  {chord_t:.4f} m")
