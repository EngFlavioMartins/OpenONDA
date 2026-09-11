"""Test-side flat-plate geometry and attached-flow reference.

The VLM test modules historically import ``tutorials.vpm.flat_plate``, whose
official name is now ``04_flat_plate`` (a directory that cannot be imported as
Python identifier because of its numeric prefix).  Rather than depend on a
tutorial layout, these tests get a self-contained re-implementation of the two
assets they use: rectangular-plate surface construction and the lifting-line
attached-flow reference (MIT 16.100, "Force Calculations for Lifting Line",
Lecture 18).  ``save_surface`` is re-exported from the production geometry IO
module because it is a public source-level utility with no tutorial dependency.
"""

from functools import lru_cache

import numpy as np
import pandas as pd

from source.solvers.vpm.boundary_elements.vlm.geometry import surface_io
from source.solvers.vpm.boundary_elements.vlm.geometry.aircraft import (
    Aircraft,
    Wing,
    WingSegment,
)

save_surface = surface_io.save_surface


def create_flat_plate_vertices(
    chord: float,
    half_span: float,
    angle_of_attack_degrees: float = 0.0,
) -> dict:
    """Create vertex_position for a flat rectangular plate at angle of attack.

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
    a_base = np.array([0.0, 0.0, 0.0])
    b_base = np.array([0.0, half_span, 0.0])
    c_base = np.array([chord, half_span, 0.0])
    d_base = np.array([chord, 0.0, 0.0])
    cos_a = np.cos(alpha)
    sin_a = np.sin(alpha)

    def rotate_point(point):
        x_rot = point[0] * cos_a + point[2] * sin_a
        z_rot = -point[0] * sin_a + point[2] * cos_a
        return np.array([x_rot, point[1], z_rot])

    return {
        "a": rotate_point(a_base),
        "b": rotate_point(b_base),
        "c": rotate_point(c_base),
        "d": rotate_point(d_base),
    }


def create_flat_plate(
    chord: float = 0.5,
    span: float = 1.0,
    angle_of_attack_degrees: float = 0.0,
    n_chordwise_panels: int = 8,
    n_spanwise_panels: int = 20,
) -> Aircraft:
    """Create a flat rectangular plate aircraft geometry.

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
    wing = Wing(uid="main_wing", symmetry=2)
    segment = WingSegment(
        uid="segment_0",
        vertex_position=create_flat_plate_vertices(
            chord,
            span / 2.0,
            angle_of_attack_degrees,
        ),
        n_chordwise_panels=n_chordwise_panels,
        n_spanwise_panels=n_spanwise_panels,
        airfoils={"inner": "flat", "outer": "flat"},
    )
    wing.add_segment(segment)
    aircraft.add_wing(wing)
    aircraft.compute_default_refs()
    return aircraft


@lru_cache(maxsize=16)
def _unit_lifting_line_coefficients(aspect_ratio, lift_curve_slope=2.0 * np.pi, terms=80):
    """Fourier coefficients for one radian of incidence on a rectangular wing."""
    harmonics = 2 * np.arange(1, terms + 1) - 1
    theta = np.pi * np.arange(1, terms + 1) / (2 * terms)
    mu = lift_curve_slope / (4 * aspect_ratio)
    matrix = np.sin(theta[:, None] * harmonics) * (np.sin(theta[:, None]) + mu * harmonics)
    coefficients = np.linalg.solve(matrix, mu * np.sin(theta))
    coefficients.setflags(write=False)
    return harmonics, coefficients


def lifting_line_polar(angle_of_attack_degrees, aspect_ratio, n_fourier_terms=80):
    """Signed CL and induced CD from the same converged lifting-line solution."""
    n, unit = _unit_lifting_line_coefficients(aspect_ratio, terms=n_fourier_terms)
    alpha = np.radians(angle_of_attack_degrees)
    return np.pi * aspect_ratio * unit[0] * alpha, np.pi * aspect_ratio * np.sum(
        n * unit**2
    ) * alpha**2


def lifting_line_circulation(
    span_position,
    reference_span,
    reference_chord,
    angle_of_attack_radians,
    freestream_speed=1.0,
    two_dimensional_lift_curve_slope=2.0 * np.pi,
    n_fourier_terms=80,
):
    """Return circulation, sectional lift and lifting-line downwash [m/s].

    Positive incidence produces positive lift/circulation and negative vertical
    induced velocity. Dimensional lift per span uses unit density.
    """
    y = np.asarray(span_position, dtype=float)
    n, unit = _unit_lifting_line_coefficients(
        reference_span / reference_chord,
        two_dimensional_lift_curve_slope,
        n_fourier_terms,
    )
    coefficients = unit * angle_of_attack_radians
    theta = np.arccos(np.clip(-2.0 * y / reference_span, -1.0, 1.0))
    sine = np.sin(theta[..., None] * n)
    circulation = 2 * reference_span * freestream_speed * (sine @ coefficients)
    denominator = np.sin(theta)
    quotient = np.divide(
        sine,
        denominator[..., None],
        out=np.broadcast_to(n, sine.shape).astype(float).copy(),
        where=denominator[..., None] > 1e-12,
    )
    downwash = -freestream_speed * (quotient @ (n * coefficients))
    return pd.DataFrame(
        {
            "span_coordinate": y,
            "span_coordinate_normalized": 2 * y / reference_span,
            "circulation": circulation,
            "lift_per_span": freestream_speed * circulation,
            "section_lift_coefficient": 2 * circulation / (freestream_speed * reference_chord),
            "induced_velocity_z": downwash,
        }
    )
