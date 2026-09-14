"""Measure the step-flow reattachment length and export comparison tables."""

import csv
import os

import numpy as np


def reattachment_location(fields, step_height):
    """Estimate x/h where the first downstream cell row turns to positive u."""
    centres = fields.cell_centre
    downstream = centres[:, 0] > 0.0
    y0 = np.min(centres[downstream, 1])
    near_wall = downstream & np.isclose(centres[:, 1], y0, atol=1e-10)
    order = np.argsort(centres[near_wall, 0])
    x = centres[near_wall, 0][order]
    u = fields.velocity[near_wall, 0][order]

    negative = np.flatnonzero(u < 0.0)
    if not len(negative):
        return np.nan, float(np.min(u))
    last_negative = negative[-1]
    if last_negative + 1 == len(u):
        return np.nan, float(np.min(u))
    x0, x1 = x[last_negative], x[last_negative + 1]
    u0, u1 = u[last_negative], u[last_negative + 1]
    x_re = x0 - u0 * (x1 - x0) / (u1 - u0)
    return float(x_re / step_height), float(np.min(u))


def write_solution_tables(fields, solution_dir, history, step_height):
    """Write the cell fields and the reattachment/health history."""
    os.makedirs(solution_dir, exist_ok=True)
    centres = fields.cell_centre

    fields_path = os.path.join(solution_dir, "fields.csv")
    with open(fields_path, "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "position_x_over_height",
                "position_y_over_height",
                "velocity_x",
                "velocity_y",
                "kinematic_pressure",
            ]
        )
        for centre, velocity, pressure in zip(
            centres,
            fields.velocity,
            fields.kinematic_pressure,
            strict=True,
        ):
            writer.writerow(
                [
                    centre[0] / step_height,
                    centre[1] / step_height,
                    velocity[0],
                    velocity[1],
                    pressure,
                ]
            )

    history_path = os.path.join(solution_dir, "reattachment_history.csv")
    with open(history_path, "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "time",
                "reattachment_position_over_height",
                "min_near_wall_velocity",
                "max_continuity_error",
                "max_courant_number",
            ]
        )
        writer.writerows(history)
