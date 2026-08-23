"""
Append aerodynamic load time-series to CSV files.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: January 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

import csv
import os

# =========================================================


def append_loads_to_csv(
    checkpoint_name: str, time: float, loads: dict, directory: str = "."
) -> None:
    """Append aerodynamic loads to CSV file.

    Args:
        checkpoint_name: Base name for output file (used to derive CSV name)
        time: Current simulation time
        loads: Dictionary containing force, moment, thrust, torque, power, Cp values
        directory: Case directory; the CSV is written below its ``samples`` directory
    """
    # Keep diagnostics separate from restart/visualization state.
    out_dir = os.path.join(directory, "samples")
    os.makedirs(out_dir, exist_ok=True)

    # Extract base name to avoid path issues
    base_name = os.path.basename(checkpoint_name)
    filename = os.path.join(out_dir, f"loads_{base_name}.csv")
    file_exists = os.path.isfile(filename)

    with open(filename, "a", newline="") as csvfile:
        fieldnames = [
            "time",
            "force_x",
            "force_y",
            "force_z",
            "moment_x",
            "moment_y",
            "moment_z",
            "thrust",
            "torque",
            "power",
            "min_pressure_coefficient",
            "max_pressure_coefficient",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(
            {
                "time": time,
                "force_x": loads["force"][0],
                "force_y": loads["force"][1],
                "force_z": loads["force"][2],
                "moment_x": loads["moment"][0],
                "moment_y": loads["moment"][1],
                "moment_z": loads["moment"][2],
                "thrust": loads["thrust"],
                "torque": loads["torque"],
                "power": loads["power"],
                "min_pressure_coefficient": loads.get("min_pressure_coefficient", 0.0),
                "max_pressure_coefficient": loads.get("max_pressure_coefficient", 0.0),
            }
        )


# append_vlm_loads_to_csv removed: VLM force CSV is now written exclusively
# by solver.py's _export_vlm_forces_to_csv, which uses the correct cached
# reference velocity.
