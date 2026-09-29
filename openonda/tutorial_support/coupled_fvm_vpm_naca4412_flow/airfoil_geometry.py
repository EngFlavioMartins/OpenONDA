"""Analytical geometry of a NACA four-digit section."""

import numpy as np


def naca4_vertices(code: str, chord: float, n_chord: int = 161) -> np.ndarray:
    """Return a closed clockwise polygon for a four-digit NACA section."""
    m = int(code[0]) / 100.0
    max_camber_position = int(code[1]) / 10.0
    thickness = int(code[2:]) / 100.0
    beta = np.linspace(0.0, np.pi, n_chord)
    x = 0.5 * (1.0 - np.cos(beta))
    yt = (
        5.0
        * thickness
        * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1036 * x**4)
    )
    yc = np.where(
        x < max_camber_position,
        m / max_camber_position**2 * (2.0 * max_camber_position * x - x**2),
        m
        / (1.0 - max_camber_position) ** 2
        * ((1.0 - 2.0 * max_camber_position) + 2.0 * max_camber_position * x - x**2),
    )
    slope = np.where(
        x < max_camber_position,
        2.0 * m / max_camber_position**2 * (max_camber_position - x),
        2.0 * m / (1.0 - max_camber_position) ** 2 * (max_camber_position - x),
    )
    theta = np.arctan(slope)
    upper = np.column_stack((x - yt * np.sin(theta), yc + yt * np.cos(theta)))
    lower = np.column_stack((x + yt * np.sin(theta), yc - yt * np.cos(theta)))
    section = np.vstack((upper[::-1], lower[1:-1]))
    section[:, 0] = chord * (section[:, 0] - 0.5)
    section[:, 1] *= chord
    return section
