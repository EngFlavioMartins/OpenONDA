"""
Generate Single Blade Surface Geometry
==================
Utilities for creating and exporting a single wind turbine blade for VLM.
Wind Turbine Configuration:
- Wind comes from the negative X direction.
- Rotor rotates about X-axis with positive angular_velocity
- Rotor plane is in YZ
- Blade at azimuth=0 points in +Y direction
- Thrust is in +X direction (extracting momentum from wind)
Blade Orientation:
- Leading edge faces the incoming relative wind
- Trailing edge is toward +X (downstream)
- Twist (beta) rotates the chord to set proper angle of attack

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: January 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

import json
import numpy as np
from source.solvers.vpm.boundary_elements.vlm.geometry.aircraft import Aircraft, Wing, WingSegment


def create_blade(
    rotor_radius: float = 6.0,
    hub_radius: float = 0.6,
    root_chord: float = 0.8,
    tip_chord: float = 0.3,
    root_pitch_deg: float = 10.0,
    tip_pitch_deg: float = 3.0,
    beta_root_deg: float = 70.0,
    beta_tip_deg: float = 10.0,
    n_span: int = 12,
    n_chord: int = 4,
    r_schedule: "np.ndarray | None" = None,
    beta_schedule: "np.ndarray | None" = None,
    pitch_schedule: "np.ndarray | None" = None,
) -> Aircraft:
    """
    Create a single wind turbine blade as an Aircraft object.

    Configuration:
    - Wind from -X (freestream velocity [-U, 0, 0])
    - Rotor rotates about +X axis (right-hand rule: +Y rotates toward +Z)
    - Blade at azimuth=0 points in +Y direction
    - Trailing edge faces +X (downstream)

    Twist Convention:
    - beta is the angle between blade chord and rotor plane (YZ plane)
    - At root, beta is large (~70°) because relative wind is mostly axial (-X)
    - At tip, beta is small (~10°) because high rotational velocity adds
      tangential component, changing relative wind angle

    For a blade at azimuth=0 (pointing in +Y):
    - Rotation creates tangential velocity in +Z direction
    - Relative wind = [-U, 0, -angular_velocity*r] (from -X and -Z)
    - Leading edge should face this relative wind

    Args:
        radius: Blade tip radius [m]
        hub_radius: Hub cutout radius [m]
        root_chord: Chord at blade root [m]
        tip_chord: Chord at blade tip [m]
        root_pitch_deg: Pitch angle at root [degrees]
        tip_pitch_deg: Pitch angle at tip [degrees]
        beta_root_deg: Twist at root [degrees]
        beta_tip_deg: Twist at tip [degrees]
        n_span: Number of spanwise panels
        n_chord: Number of chordwise panels

    Returns:
        Aircraft object containing the blade wing
    """
    aircraft = Aircraft(uid="blade")
    wing = Wing(uid="blade_0", symmetry=0)  # No symmetry for rotor blade

    def blade_to_global(r_blade, chord_frac, pitch_deg, beta_deg):
        """
        Transform blade coordinates to global frame.

        Coordinate system:
        - Global X: axial (wind from -X to +X)
        - Global Y: radial at azimuth=0
        - Global Z: tangential at azimuth=0

        Blade section at radius r:
        - Chord lies in a plane perpendicular to radial (Y) direction
        - At beta=0, pitch=0: chord aligned with Z-axis
        - Beta rotates the chord about Y (twist) to face relative wind
        - Pitch is additional rotation for angle of attack

        Args:
            r_blade: Radial position along blade
            chord_frac: Chordwise position (0=LE, 1=TE)
            pitch_deg: Local pitch angle
            beta_deg: Local twist angle
        """
        pitch_rad = np.radians(pitch_deg)
        beta_rad = np.radians(beta_deg)

        # Local 2D airfoil coordinates (before rotation)
        # x_local: toward trailing edge (+X direction at beta=0)
        # z_local: thickness direction (normal to chord)
        x_local = chord_frac  # 0 at LE, 1 at TE
        z_local = 0.0  # Flat plate approximation

        # Apply twist (beta): rotate about Y-axis
        # This angles the chord from the rotor plane
        # Positive beta: LE rotates toward -X (into the wind)
        x_twisted = x_local * np.cos(beta_rad) - z_local * np.sin(beta_rad)
        z_twisted = x_local * np.sin(beta_rad) + z_local * np.cos(beta_rad)

        # Apply pitch: additional rotation about Y-axis for angle of attack
        x_pitched = x_twisted * np.cos(pitch_rad) - z_twisted * np.sin(pitch_rad)
        z_pitched = x_twisted * np.sin(pitch_rad) + z_twisted * np.cos(pitch_rad)

        # Map to global coordinates
        # LE is toward -X, TE is toward +X (after twist)
        # Center chord at origin, so chord goes from -0.5*c to +0.5*c
        x_global = x_pitched  # After twist/pitch, this is the axial position
        y_global = r_blade  # Radial position
        z_global = z_pitched  # Tangential offset (near zero for flat plate)

        return np.array([x_global, y_global, z_global])

    # Vertex convention for VLM:
    # a = root LE, b = tip LE, c = tip TE, d = root TE
    le_frac = -0.25  # quarter-chord ahead of mid-chord reference
    te_frac = 0.75

    if r_schedule is not None and beta_schedule is not None:
        # Multi-segment path: one trapezoidal segment per radial interval so that
        # the nonlinear twist profile is piecewise-linearly captured.
        r_s = np.asarray(r_schedule, dtype=float)
        b_s = np.asarray(beta_schedule, dtype=float)
        p_s = (
            np.asarray(pitch_schedule, dtype=float)
            if pitch_schedule is not None
            else np.linspace(root_pitch_deg, tip_pitch_deg, len(r_s))
        )
        chord = np.interp(r_s, [hub_radius, rotor_radius], [root_chord, tip_chord])

        N = len(r_s) - 1  # number of segments
        total_span = r_s[-1] - r_s[0]

        # Distribute n_span panels proportionally; every segment gets >= 1
        seg_spans = np.diff(r_s)
        seg_npan = np.maximum(1, np.round(n_span * seg_spans / total_span).astype(int))
        diff = n_span - int(seg_npan.sum())
        idx_adj = np.argsort(-seg_spans if diff > 0 else seg_spans)
        for i in range(abs(diff)):
            k = idx_adj[i % N]
            if diff > 0 or seg_npan[k] > 1:
                seg_npan[k] += int(np.sign(diff))

        for k in range(N):
            a_v = blade_to_global(r_s[k], le_frac * chord[k], p_s[k], b_s[k])
            d_v = blade_to_global(r_s[k], te_frac * chord[k], p_s[k], b_s[k])
            b_v = blade_to_global(r_s[k + 1], le_frac * chord[k + 1], p_s[k + 1], b_s[k + 1])
            c_v = blade_to_global(r_s[k + 1], te_frac * chord[k + 1], p_s[k + 1], b_s[k + 1])
            segment = WingSegment(
                uid=f"blade_segment_{k}",
                vertex_position={"a": a_v, "b": b_v, "c": c_v, "d": d_v},
                n_chordwise_panels=n_chord,
                n_spanwise_panels=int(seg_npan[k]),
                airfoils={"inner": "flat", "outer": "flat"},
            )
            wing.add_segment(segment)
    else:
        # Original single-segment path (linear-beta root → tip)
        r_inner, r_outer = hub_radius, rotor_radius
        c_inner, c_outer = root_chord, tip_chord
        a = blade_to_global(r_inner, le_frac * c_inner, root_pitch_deg, beta_root_deg)
        d = blade_to_global(r_inner, te_frac * c_inner, root_pitch_deg, beta_root_deg)
        b = blade_to_global(r_outer, le_frac * c_outer, tip_pitch_deg, beta_tip_deg)
        c = blade_to_global(r_outer, te_frac * c_outer, tip_pitch_deg, beta_tip_deg)
        segment = WingSegment(
            uid="blade_segment_0",
            vertex_position={"a": a, "b": b, "c": c, "d": d},
            n_chordwise_panels=n_chord,
            n_spanwise_panels=n_span,
            airfoils={"inner": "flat", "outer": "flat"},
        )
        wing.add_segment(segment)

    aircraft.add_wing(wing)

    # Set reference values
    aircraft.refs["area"] = np.pi * rotor_radius**2 / 3
    aircraft.refs["span"] = rotor_radius - hub_radius
    aircraft.refs["chord"] = (root_chord + tip_chord) / 2

    return aircraft


def save_surface(aircraft: Aircraft, filepath: str) -> str:
    """
    Save aircraft surface geometry to JSON file.
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

    refs = aircraft.refs
    data["refs"] = {
        "area": refs.get("area", 1.0),
        "chord": refs.get("chord", 1.0),
        "span": refs.get("span", 1.0),
    }

    output_path = filepath if filepath.endswith(".json") else f"{filepath}.json"
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Surface saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    blade = create_blade(
        rotor_radius=6.0,
        hub_radius=0.6,
        root_chord=0.8,
        tip_chord=0.3,
        root_pitch_deg=10.0,
        tip_pitch_deg=3.0,
        beta_root_deg=70.0,
        beta_tip_deg=10.0,
        n_span=12,
        n_chord=4,
    )
    save_surface(blade, "blade")

    print(f"\nBlade created")
    print(f"Total panels: {blade.total_n_panels()}")
