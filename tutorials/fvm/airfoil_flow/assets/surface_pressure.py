"""Sample the airfoil pressure coefficient from the final FVM field."""

import csv
import os


def write_surface_cp(fvm_solver, sol_dir, chord: float, freestream_velocity: float) -> None:
    """Write the surface pressure coefficient to ``surface_cp.csv``."""
    n = fvm_solver.mesh_data["n_cells"]
    n_interior = fvm_solver.mesh_data["n_interior_faces"]
    q = 0.5 * freestream_velocity**2  # kinematic pressure (rho folds out)
    rows = []
    for patch in fvm_solver.boundaries:
        if patch["name"] != "airfoil":
            continue
        start, nf = patch["start_face"], patch["n_faces"]
        centres = fvm_solver.geo_data["face_centre"][start : start + nf]
        ghost = n + (start - n_interior)
        p_face = fvm_solver.kinematic_pressure[ghost : ghost + nf]
        for (x, y, _z), p_i in zip(centres, p_face, strict=True):
            rows.append((x / chord, y / chord, p_i / q))
    rows.sort()

    path = os.path.join(sol_dir, "surface_cp.csv")
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["position_x_over_chord", "position_y_over_chord", "pressure_coefficient"])
        writer.writerows(rows)
