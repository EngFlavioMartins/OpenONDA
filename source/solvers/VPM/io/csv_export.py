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
    backup_file_name: str, flow_time: float, loads: dict, directory: str = "solution"
) -> None:
    """Append aerodynamic loads to CSV file.

    Args:
        backup_file_name: Base name for output file (used to derive CSV name)
        flow_time: Current simulation time
        loads: Dictionary containing force, moment, thrust, torque, power, Cp values
        directory: Destination directory for CSV file
    """
    # Ensure solution directory exists
    out_dir = os.path.join(directory, "samples")
    os.makedirs(out_dir, exist_ok=True)

    # Extract base name to avoid path issues
    base_name = os.path.basename(backup_file_name)
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
            "Cp_min",
            "Cp_max",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(
            {
                "time": flow_time,
                "force_x": loads["force"][0],
                "force_y": loads["force"][1],
                "force_z": loads["force"][2],
                "moment_x": loads["moment"][0],
                "moment_y": loads["moment"][1],
                "moment_z": loads["moment"][2],
                "thrust": loads["thrust"],
                "torque": loads["torque"],
                "power": loads["power"],
                "Cp_min": loads.get("Cp_min", 0.0),
                "Cp_max": loads.get("Cp_max", 0.0),
            }
        )


# append_vlm_loads_to_csv removed: VLM force CSV is now written exclusively
# by solver.py's _export_vlm_forces_to_csv, which uses the correct cached
# reference velocity.
