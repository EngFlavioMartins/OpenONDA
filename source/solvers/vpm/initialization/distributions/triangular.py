"""Triangular-prism particle distributions for straight vortex columns."""

from __future__ import annotations

import numpy as np

from ..data import ParticleDistribution
from ._common import Bounds3D, centred_coordinates, validate_bounds, validate_spacing


def create_triangular_prism_distribution(
    *,
    bounds: Bounds3D,
    spacing: float,
    core_radius_ratio: float,
    axis: str = "z",
    axial_spacing: float | None = None,
) -> ParticleDistribution:
    """Extrude a triangular transverse lattice along a Cartesian axis."""
    spacing, core_radius_ratio = validate_spacing(spacing, core_radius_ratio)
    axial_spacing = spacing if axial_spacing is None else float(axial_spacing)
    if not np.isfinite(axial_spacing) or axial_spacing <= 0.0:
        raise ValueError("axial_spacing must be finite and positive")
    if axis not in {"x", "y", "z"}:
        raise ValueError("axis must be 'x', 'y', or 'z'")
    limits = validate_bounds(bounds)
    axial_index = {"x": 0, "y": 1, "z": 2}[axis]
    transverse = [index for index in range(3) if index != axial_index]
    row_spacing = np.sqrt(3.0) * spacing / 2.0
    first = centred_coordinates(*limits[transverse[0]], spacing)
    second = centred_coordinates(*limits[transverse[1]], row_spacing)
    plane: list[tuple[float, float]] = []
    for row, second_value in enumerate(second):
        offset = 0.5 * spacing if row % 2 else 0.0
        shifted = first + offset
        shifted = shifted[shifted <= limits[transverse[0], 1] + np.finfo(float).eps]
        for first_value in shifted:
            plane.append((first_value, second_value))
    axial = centred_coordinates(*limits[axial_index], axial_spacing)
    position = np.empty((len(plane) * len(axial), 3), dtype=float)
    for orbit, (first_value, second_value) in enumerate(plane):
        selection = slice(orbit * len(axial), (orbit + 1) * len(axial))
        position[selection, axial_index] = axial
        position[selection, transverse[0]] = first_value
        position[selection, transverse[1]] = second_value
    count = len(position)
    cell_area = np.sqrt(3.0) * spacing**2 / 2.0
    return ParticleDistribution(
        position=position,
        core_radius=np.full(count, core_radius_ratio * spacing),
        particle_volume=np.full(count, cell_area * axial_spacing),
        spacing=spacing,
    )
