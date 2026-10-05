"""Boundary-layer velocity and skin-friction comparisons."""

from pathlib import Path

import numpy as np

from openonda.results import write_csv_table


def write_profiles(
    fields, sol_dir: str, kinematic_viscosity: float, freestream_velocity: float, stations
) -> None:
    """Sample u(y) at the stations and Cf(x) along the plate into CSV files."""
    cell_centre = fields.cell_centre
    u = fields.velocity
    xc, yc = cell_centre[:, 0], cell_centre[:, 1]

    # The plate uses a uniform x grid, so the column width is easy to find.
    plate_x = np.unique(np.round(xc[xc > 0], 12))
    dx = np.min(np.diff(plate_x))

    rows = []
    for station in stations:
        x_col = plate_x[np.argmin(np.abs(plate_x - station))]
        sel = np.abs(xc - x_col) < 0.5 * dx
        order = np.argsort(yc[sel])
        for y_i, u_i, v_i in zip(yc[sel][order], u[sel, 0][order], u[sel, 1][order], strict=True):
            rows.append([station, x_col, y_i, u_i, v_i])
    write_csv_table(
        Path(sol_dir) / "profiles.csv",
        rows,
        columns=("station", "position_x", "position_y", "velocity_x", "velocity_y"),
    )

    # Skin friction from the wall-adjacent cell row: tau_w ~ mu * u1 / y1.
    kinematic_pressure = fields.kinematic_pressure
    y1 = yc.min()
    y_top = yc.max()
    wall = (np.abs(yc - y1) < 1e-12) & (xc > 0.0)
    top = np.abs(yc - y_top) < 1e-12
    order = np.argsort(xc[wall])
    x_w = xc[wall][order]
    u_w = u[wall, 0][order]
    kinematic_pressure_wall = kinematic_pressure[wall][order]
    u_e = np.interp(x_w, np.sort(xc[top]), u[top, 0][np.argsort(xc[top])])
    cf = 2.0 * kinematic_viscosity * u_w / (y1 * freestream_velocity**2)
    rex = freestream_velocity * x_w / kinematic_viscosity

    write_csv_table(
        Path(sol_dir) / "cf.csv",
        zip(x_w, rex, cf, 0.664 / np.sqrt(rex), kinematic_pressure_wall, u_e, strict=True),
        columns=(
            "position_x",
            "reynolds_number",
            "skin_friction_coefficient",
            "skin_friction_coefficient_blasius",
            "kinematic_pressure_wall",
            "velocity_x_top",
        ),
    )
