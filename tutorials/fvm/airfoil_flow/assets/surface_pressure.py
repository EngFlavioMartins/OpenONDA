"""Sample the airfoil pressure coefficient from the final FVM field."""

from pathlib import Path

from openonda.results import write_csv_table


def write_surface_cp(fields, sol_dir, chord: float, freestream_velocity: float) -> None:
    """Write the surface pressure coefficient to ``surface_cp.csv``."""
    q = 0.5 * freestream_velocity**2  # kinematic pressure (rho folds out)
    rows = []
    patch = fields.boundaries["airfoil"]
    for (x, y, _z), p_i in zip(patch.face_centre, patch.kinematic_pressure, strict=True):
        rows.append((x / chord, y / chord, p_i / q))
    rows.sort()

    write_csv_table(
        Path(sol_dir) / "surface_cp.csv",
        rows,
        columns=("position_x_over_chord", "position_y_over_chord", "pressure_coefficient"),
    )
