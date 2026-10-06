"""Independent circulation and induction errors from tapering a no-slip velocity.

This manufactured calculation does not load a solver or modify a case. It
isolates the exterior wall-velocity taper before Gaussian renewal, blending,
moment corrections and pruning. It cannot establish cylinder force causality.
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np


def no_slip_velocity(points, *, radius=0.5, layer_width=0.08):
    r"""Exact divergence-free circular no-slip flow from a streamfunction.

    In polar coordinates, psi=(r-a^2/r)*(1-exp(-(r-a)/delta))*sin(theta).
    Both psi and its radial derivative vanish at r=a. Velocity is extended
    by zero inside the body and approaches unit x velocity at infinity.
    """
    points = np.asarray(points, dtype=float)
    result = np.zeros_like(points)
    distance = np.linalg.norm(points[:, :2], axis=1)
    fluid = distance > radius
    r = distance[fluid]
    x, y = points[fluid, 0] / r, points[fluid, 1] / r
    decay = np.exp(-(r - radius) / layer_width)
    potential = r - radius**2 / r
    potential_derivative = 1 + radius**2 / r**2
    radial_factor = potential * (1 - decay) / r
    tangential_factor = potential_derivative * (1 - decay) + potential * decay / layer_width
    result[fluid, 0] = radial_factor * x**2 + tangential_factor * y**2
    result[fluid, 1] = (radial_factor - tangential_factor) * x * y
    return result


def sampled_velocity(points, spacing, *, layer_width=0.08, taper=False):
    """Return exact zero-extended velocity, optionally with the exterior taper."""
    velocity = no_slip_velocity(points, layer_width=layer_width)
    if taper:
        phase = np.clip((np.linalg.norm(points[:, :2], axis=1) - 0.5) / spacing, 0, 1)
        velocity *= (phase**2 * (3 - 2 * phase))[:, None]
    return velocity


def circulation_strength(points, spacing, *, layer_width=0.08, taper=False):
    """Integrate n cross u at the four half-cell faces over a unit span."""
    result = np.zeros(len(points))
    for axis, component, sign in ((0, 1, 1.0), (1, 0, -1.0)):
        for side in (-1.0, 1.0):
            faces = points.copy()
            faces[:, axis] += side * spacing / 2
            velocity = sampled_velocity(faces, spacing, layer_width=layer_width, taper=taper)
            result += sign * side * spacing * velocity[:, component]
    return result


def upper_rectangle_circulation(coordinate, spacing, *, layer_width=0.08, taper=False):
    """Independent midpoint line integral around the upper-half cell rectangle."""
    upper_y = coordinate[coordinate > 0]
    total = 0.0
    for axis, component, sign, values, bounds in (
        (0, 1, 1.0, upper_y, (coordinate[0] - spacing / 2, coordinate[-1] + spacing / 2)),
        (1, 0, -1.0, coordinate, (upper_y[0] - spacing / 2, upper_y[-1] + spacing / 2)),
    ):
        for side, bound in zip((-1.0, 1.0), bounds, strict=True):
            points = np.zeros((len(values), 3))
            points[:, axis] = bound
            points[:, 1 - axis] = values
            velocity = sampled_velocity(points, spacing, layer_width=layer_width, taper=taper)
            total += sign * side * spacing * np.sum(velocity[:, component])
    return float(total)


def planar_velocity(points, sources, strengths, spacing):
    """Independent source-radius Gaussian Biot-Savart sum, bounded in memory."""
    result = np.zeros_like(points)
    for start in range(0, len(sources), 2048):
        source = sources[start : start + 2048]
        gamma = strengths[start : start + 2048]
        delta = points[:, None, :2] - source[None, :, :2]
        squared = np.einsum("fpi,fpi->fp", delta, delta)
        coefficient = np.divide(
            -np.expm1(-squared / spacing**2),
            squared,
            out=np.full_like(squared, 1 / spacing**2),
            where=squared > 0,
        )
        coefficient *= gamma[None, :] / (2 * math.pi)
        result[:, 0] -= np.sum(coefficient * delta[:, :, 1], axis=1)
        result[:, 1] += np.sum(coefficient * delta[:, :, 0], axis=1)
    return result


def audit(spacings=(0.08, 0.04, 0.02, 0.01), *, layer_width=0.08):
    angle = 2 * math.pi * np.arange(128) / 128
    outer = np.column_stack((1.6 * np.cos(angle), 1.6 * np.sin(angle), np.zeros_like(angle)))
    rows = []
    for h in spacings:
        # The difference is supported within h of the wall. One metre covers
        # every affected midpoint face for these resolutions.
        coordinate = np.arange(-round(1 / h), round(1 / h) + 1) * h
        x, y = np.meshgrid(coordinate, coordinate, indexing="ij")
        sources = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
        untapered = circulation_strength(sources, h, layer_width=layer_width)
        tapered = circulation_strength(sources, h, layer_width=layer_width, taper=True)
        difference = tapered - untapered
        support = np.abs(difference) > 1e-16
        fluid = np.linalg.norm(sources[:, :2], axis=1) >= 0.5
        upper = sources[:, 1] > 0
        upper_circulation = {}
        for name, target, taper in (("untapered", untapered, False), ("tapered", tapered, True)):
            complete = float(np.sum(target[upper]))
            exterior = float(np.sum(target[upper & fluid]))
            contour = upper_rectangle_circulation(
                coordinate, h, layer_width=layer_width, taper=taper
            )
            upper_circulation[name] = {
                "complete_cell_circulation": complete,
                "independent_contour_circulation": contour,
                "exterior_nodes_only_circulation": exterior,
                "relative_lost_circulation_from_omitted_cut_cells": (exterior - complete)
                / complete,
            }
        values = {}
        for name, selected in (
            ("complete_cell_circulation", support),
            ("exterior_nodes_only", support & fluid),
        ):
            change = planar_velocity(outer, sources[selected], difference[selected], h)
            values[name] = {
                "affected_nodes": int(np.count_nonzero(selected)),
                "circulation_change": float(np.sum(difference[selected])),
                "absolute_strength_change": float(np.sum(np.abs(difference[selected]))),
                "linear_impulse_change_x": float(
                    0.5 * np.sum(sources[selected, 1] * difference[selected])
                ),
                "linear_impulse_change_y": float(
                    -0.5 * np.sum(sources[selected, 0] * difference[selected])
                ),
                "maximum_outer_velocity_change": float(np.linalg.norm(change, axis=1).max()),
                "rms_outer_velocity_change": float(np.sqrt(np.mean(np.sum(change**2, axis=1)))),
            }
        rows.append({"spacing": h, "upper_rectangle_circulation": upper_circulation, **values})
    for previous, current in zip(rows[:-1], rows[1:], strict=True):
        for name in ("complete_cell_circulation", "exterior_nodes_only"):
            current[name]["observed_order_from_previous_spacing"] = math.log(
                previous[name]["maximum_outer_velocity_change"]
                / current[name]["maximum_outer_velocity_change"]
            ) / math.log(previous["spacing"] / current["spacing"])
    return {
        "scope": "Manufactured wall taper only; no interpolation, Gaussian representation correction, moment correction, blending, pruning or cylinder force inference",
        "radius": 0.5,
        "layer_width": layer_width,
        "freestream_speed": 1.0,
        "outer_radius": 1.6,
        "gaussian_core_radius_over_spacing": 1.0,
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer-width", type=float, default=0.08)
    options = parser.parse_args()
    print(json.dumps(audit(layer_width=options.layer_width), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
