"""Attached-flow references for the rectangular flat-plate tutorial.

The lifting-line sine series and integrated loads follow MIT 16.100,
``Force Calculations for Lifting Line`` (Lecture 18). Symmetric loading uses
odd harmonics with independent collocation points on ONE half-span.
"""

from functools import lru_cache

import numpy as np
import pandas as pd


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


def spanwise_reference(
    distribution_model,
    span_position,
    reference_span,
    reference_chord,
    angle_of_attack_radians,
    freestream_speed=1.0,
    total_lift_coefficient=None,
    aspect_ratio=None,
    **model_options,
):
    """Lifting-line loading or an elliptic shape with the same integrated lift."""
    if distribution_model == "lifting_line":
        return lifting_line_circulation(
            span_position,
            reference_span,
            reference_chord,
            angle_of_attack_radians,
            freestream_speed,
            **model_options,
        )
    y = np.asarray(span_position, dtype=float)
    cl = (
        4
        * total_lift_coefficient
        / np.pi
        * np.sqrt(np.maximum(0, 1 - (2 * y / reference_span) ** 2))
    )
    return pd.DataFrame(
        {
            "span_coordinate": y,
            "span_coordinate_normalized": 2 * y / reference_span,
            "section_lift_coefficient": cl,
        }
    )
