"""Sample the airfoil pressure coefficient from the final FVM field."""

import csv
import os


def write_surface_cp(fields, sol_dir, chord: float, freestream_velocity: float) -> None:
    """Write the surface pressure coefficient to ``surface_cp.csv``."""
    q = 0.5 * freestream_velocity**2  # kinematic pressure (rho folds out)
    rows = []
    patch = fields.boundaries["airfoil"]
    for (x, y, _z), p_i in zip(patch.face_centre, patch.kinematic_pressure, strict=True):
        rows.append((x / chord, y / chord, p_i / q))
    rows.sort()

    path = os.path.join(sol_dir, "surface_cp.csv")
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["position_x_over_chord", "position_y_over_chord", "pressure_coefficient"])
        writer.writerows(rows)
