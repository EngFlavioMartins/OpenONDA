"""Generate mirrored flat-plate blades with their quarter chord on the radial axis.

At a positive-y station, positive rotation about z moves the blade toward -x.
The CCW blade therefore has its leading edge at -x and its trailing edge at +x.
Positive pitch raises the leading edge and produces upward thrust in climb.
"""

import numpy as np

from source.solvers.vpm.boundary_elements.vlm.geometry.aircraft import (
    Aircraft,
    Wing,
    WingSegment,
)
from source.solvers.vpm.boundary_elements.vlm.geometry.surface_io import (
    save_surface as save_blade,
)


def create_rotor_blade(
    R_hub: float,
    R_tip: float,
    chord_root: float,
    chord_tip: float,
    pitch_root_deg: float,
    pitch_tip_deg: float,
    n_chord: int,
    n_span: int,
    clockwise: bool,
) -> Aircraft:
    """
    Create a rotor blade surface as an Aircraft object.

    The blade lies in the X-Y plane (rotation about Z-axis).
    """
    aircraft = Aircraft(uid="rotor_blade")
    wing = Wing(uid="blade_0", symmetry=0)

    def pitched_vertices(chord, y, pitch_deg, cw):
        pitch = np.radians(pitch_deg)
        cos_p = np.cos(pitch)
        sin_p = np.sin(pitch)
        direction = -1.0 if cw else 1.0
        le = np.array([-direction * 0.25 * chord * cos_p, y, 0.25 * chord * sin_p])
        te = np.array([direction * 0.75 * chord * cos_p, y, -0.75 * chord * sin_p])
        return (le, te)

    le_root, te_root = pitched_vertices(chord_root, R_hub, pitch_root_deg, cw=clockwise)
    le_tip, te_tip = pitched_vertices(chord_tip, R_tip, pitch_tip_deg, cw=clockwise)
    if not clockwise:
        le_root, le_tip = (le_tip, le_root)
        te_root, te_tip = (te_tip, te_root)
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


__all__ = ["save_blade", "create_rotor_blade"]
