"""
Runtime simulation monitor (SimulationMonitor): tracks and reports per-step
diagnostics.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: January 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

import csv
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.solver import Solver

# =========================================================

class SimulationMonitor:
    """
    Handles I/O operations for VPM simulation monitoring.

    Responsibilities:
    - Logging aerodynamic loads to CSV files
    - VLM-specific load logging
    - Triggering periodic backups
    - Flow diagnostics output

    Example:
        >>> monitor = SimulationMonitor(solver)
        >>> # Called each time step
        >>> monitor.log_panel_loads(loads)
        >>> monitor.log_vlm_loads(forces)
    """

    def __init__(self, solver: "Solver", output_dir: str | None = None):
        """
        Initialize simulation monitor.

        Args:
            solver: Reference to the parent VPM solver
            output_dir: Directory for output files (default: "solution/samples")
        """
        self.solver = solver

        if output_dir is None:
            self.output_dir = os.path.join("solution", "samples")
        else:
            self.output_dir = output_dir

        # Track CSV initialization state
        self._panel_csv_initialized = False
        self._vlm_csv_initialized = False

    def _ensure_output_dir(self):
        """Ensure output directory exists."""
        os.makedirs(self.output_dir, exist_ok=True)

    def log_panel_loads(self, loads: dict) -> None:
        """
        Append panel solver aerodynamic loads to CSV file.

        Args:
            loads: Dictionary containing:
                - force: [Fx, Fy, Fz] aerodynamic force components
                - moment: [Mx, My, Mz] moment components
                - thrust, torque, power: scalar values
                - Cp_min, Cp_max: pressure coefficient extrema (optional)
        """
        self._ensure_output_dir()

        base_name = os.path.basename(self.solver.backup_file_name)
        filename = os.path.join(self.output_dir, f"loads_{base_name}.csv")
        file_exists = os.path.isfile(filename)

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

        with open(filename, "a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            writer.writerow(
                {
                    "time": self.solver.flow_time,
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

    def log_vlm_loads(self, forces: dict) -> None:
        """
        Append VLM aerodynamic loads to CSV file.

        Args:
            forces: Dictionary containing VLM force/moment coefficients:
                - CL, CD, CC: force coefficients
                - Fx, Fy, Fz: force components
                - Mx, My, Mz: moment components
                - L, D: lift and drag magnitudes
                - q, S_ref: dynamic pressure and reference area
        """
        self._ensure_output_dir()

        base_name = os.path.basename(self.solver.backup_file_name)
        filename = os.path.join(self.output_dir, f"vlm_loads_{base_name}.csv")
        file_exists = os.path.isfile(filename)

        fieldnames = [
            "time",
            "step",
            "CL",
            "CD",
            "CC",
            "Fx",
            "Fy",
            "Fz",
            "Mx",
            "My",
            "Mz",
            "L",
            "D",
            "q",
            "S_ref",
        ]

        with open(filename, "a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            writer.writerow(
                {
                    "time": self.solver.flow_time,
                    "step": self.solver.time_step,
                    "CL": forces.get("CL", 0.0),
                    "CD": forces.get("CD", 0.0),
                    "CC": forces.get("CC", 0.0),
                    "Fx": forces.get("Fx", 0.0),
                    "Fy": forces.get("Fy", 0.0),
                    "Fz": forces.get("Fz", 0.0),
                    "Mx": forces.get("Mx", 0.0),
                    "My": forces.get("My", 0.0),
                    "Mz": forces.get("Mz", 0.0),
                    "L": forces.get("L", 0.0),
                    "D": forces.get("D", 0.0),
                    "q": forces.get("q", 0.0),
                    "S_ref": forces.get("S_ref", 0.0),
                }
            )
