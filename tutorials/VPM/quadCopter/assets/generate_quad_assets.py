import json
import numpy as np
from source.solvers.VPM.boundary_elements.vlm.geometry.aircraft import Aircraft, Wing, WingSegment


def create_blade(
    radius: float = 0.15,
    hub_radius: float = 0.02,
    root_chord: float = 0.04,
    tip_chord: float = 0.02,
    root_pitch_deg: float = 15.0,
    tip_pitch_deg: float = 5.0,
    beta_root_deg: float = 60.0,
    beta_tip_deg: float = 10.0,
    n_span: int = 10,
    n_chord: int = 4,
) -> Aircraft:
    """
    Create a single quadcopter blade scale.
    """
    aircraft = Aircraft(uid="blade")
    wing = Wing(uid="blade_0", symmetry=0)

    def blade_to_global(r_blade, chord_frac, pitch_deg, beta_deg):
        pitch_rad = np.radians(pitch_deg)
        beta_rad = np.radians(beta_deg)

        x_local = chord_frac
        z_local = 0.0

        # Twist
        x_twisted = x_local * np.cos(beta_rad) - z_local * np.sin(beta_rad)
        z_twisted = x_local * np.sin(beta_rad) + z_local * np.cos(beta_rad)

        # Pitch
        x_pitched = x_twisted * np.cos(pitch_rad) - z_twisted * np.sin(pitch_rad)
        z_pitched = x_twisted * np.sin(pitch_rad) + z_twisted * np.cos(pitch_rad)

        x_global = x_pitched
        y_global = r_blade
        z_global = z_pitched

        return np.array([x_global, y_global, z_global])

    r_inner = hub_radius
    r_outer = radius
    c_inner = root_chord
    c_outer = tip_chord

    le_frac = -0.25
    te_frac = 0.75

    a = blade_to_global(r_inner, le_frac * c_inner, root_pitch_deg, beta_root_deg)
    d = blade_to_global(r_inner, te_frac * c_inner, root_pitch_deg, beta_root_deg)
    b = blade_to_global(r_outer, le_frac * c_outer, tip_pitch_deg, beta_tip_deg)
    c = blade_to_global(r_outer, te_frac * c_outer, tip_pitch_deg, beta_tip_deg)

    segment = WingSegment(
        uid="blade_segment",
        vertices={"a": a, "b": b, "c": c, "d": d},
        panels_chord=n_chord,
        panels_span=n_span,
        airfoils={"inner": "flat", "outer": "flat"},
    )
    wing.add_segment(segment)
    aircraft.add_wing(wing)

    return aircraft


def create_body(length=0.4, width=0.1) -> Aircraft:
    """
    Create a simple rectangular body (X-frame center).
    Simplified as a flat VLM surface.
    """
    aircraft = Aircraft(uid="body")
    wing = Wing(uid="fuselage", symmetry=0)

    # Simple rectangle centered at origin
    half_l = length / 2
    half_w = width / 2

    # Vertices
    # a: rear left
    # b: front left
    # c: front right
    # d: rear right
    # NOTE: VLM convention: a=rootLE, b=tipLE, c=tipTE, d=rootTE
    # Let's verify 'root' and 'tip'.
    # If root is left side (-Y), tip is right side (+Y).
    # LE is Front (+X), TE is Rear (-X).
    # Vertices structure:
    # a: root_LE: -Y, +X
    # b: tip_LE: +Y, +X
    # c: tip_TE: +Y, -X
    # d: root_TE: -Y, -X

    a = np.array([half_l, -half_w, 0.0])  # Front Left
    b = np.array([half_l, half_w, 0.0])  # Front Right
    c = np.array([-half_l, half_w, 0.0])  # Rear Right
    d = np.array([-half_l, -half_w, 0.0])  # Rear Left

    # Wait, my comments above:
    # Convention usually: X is flow direction?
    # If quad moves forward (+X), flow is (-X).
    # LE is front (+X), TE is rear (-X).
    # Span is Y.
    # Root is -Y/2, Tip is +Y/2?

    segment = WingSegment(
        uid="body_segment",
        vertices={"a": a, "b": b, "c": c, "d": d},
        panels_chord=8,
        panels_span=4,
        airfoils={"inner": "flat", "outer": "flat"},
    )
    wing.add_segment(segment)
    aircraft.add_wing(wing)
    return aircraft


def save_surface(aircraft: Aircraft, filepath: str) -> str:
    data = {"uid": aircraft.uid, "wings": []}
    for wing in aircraft.wings.values():
        wing_data = {"uid": wing.uid, "symmetry": wing.symmetry, "segments": []}
        for segment in wing.segments.values():
            seg_data = {
                "uid": segment.uid,
                "vertices": {k: v.tolist() for k, v in segment.vertices.items()},
                "panels_chord": segment.panels_chord,
                "panels_span": segment.panels_span,
                "airfoils": segment.airfoils,
            }
            wing_data["segments"].append(seg_data)
        data["wings"].append(wing_data)

    output_path = filepath if filepath.endswith(".json") else f"{filepath}.json"
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Surface saved to: {output_path}")
    return output_path


def create_quad_layout(rpm: float = 3000.0, radius: float = 0.15, arm_length: float = 0.25):
    """
    Generate a layout file for the quadcopter assembly.
    """
    omega = rpm * 2 * np.pi / 60.0
    L = arm_length

    layout = {"surfaces": []}

    # 1. Body
    layout["surfaces"].append(
        {
            "file": "assets/body.json",
            "name": "fuselage",
            "translation": [0.0, 0.0, 0.0],
            "kinematics": {"type": "StaticVLM"},
        }
    )

    # 2. Rotors
    # Rotor positions (Quad X)
    # 0: FR (+L, +L), CCW
    # 1: RR (-L, +L), CW
    # 2: RL (-L, -L), CCW
    # 3: FL (+L, -L), CW
    positions = [[L, L, 0.0], [-L, L, 0.0], [-L, -L, 0.0], [L, -L, 0.0]]

    # Directions (1=CCW, -1=CW)
    # Using all CCW (1.0) for this tutorial as discussed in previous steps
    # to avoid needing mirrored blade geometry.
    directions = [1.0, 1.0, 1.0, 1.0]

    for i in range(4):
        pos = positions[i]
        rot_speed = omega * directions[i]

        layout["surfaces"].append(
            {
                "file": "assets/blade.json",
                "name": f"rotor_{i}",
                "translation": pos,
                "kinematics": {
                    "type": "RotatingVLM",
                    "rotation_speed": rot_speed,
                    "rotation_axis": [0.0, 0.0, 1.0],
                    "rotation_center": pos,
                },
            }
        )

    output_path = "./assets/quad_layout.json"
    with open(output_path, "w") as f:
        json.dump(layout, f, indent=2)
    print(f"Layout saved to: {output_path}")


if __name__ == "__main__":
    # 1. Create Blade
    blade = create_blade()
    save_surface(blade, "./assets/blade")

    # 2. Create Body
    body = create_body()
    save_surface(body, "./assets/body")

    # 3. Create Layout
    create_quad_layout()
