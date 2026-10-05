"""Measure the step-flow reattachment length and export comparison tables."""

from pathlib import Path

import numpy as np

from openonda.results import write_csv_table


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


def history_row(fields, step_height):
    """Evaluate one accepted state, including a restored backup."""
    position, velocity = reattachment_location(fields, step_height)
    return [fields.time, position, velocity, fields.max_continuity_error, fields.max_courant_number]


def write_solution_tables(fields, solution_dir, step_height):
    """Write the final velocity and pressure comparison fields."""
    centres = fields.cell_centre
    write_csv_table(
        Path(solution_dir) / "fields.csv",
        np.column_stack(
            (centres[:, :2] / step_height, fields.velocity[:, :2], fields.kinematic_pressure)
        ),
        columns=(
            "position_x_over_height",
            "position_y_over_height",
            "velocity_x",
            "velocity_y",
            "kinematic_pressure",
        ),
    )
