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
    from ..core.solver import VPMSolver

# =========================================================


class SimulationMonitor:
    """
    Handles I/O operations for VPM simulation monitoring.

    Responsibilities:
    - Logging aerodynamic loads to CSV files
    - VLM-specific load logging
    - Triggering periodic checkpoints
    - Flow diagnostics output

    Example:
        >>> monitor = SimulationMonitor(solver)
        >>> # Called each time step
        >>> monitor.log_panel_loads(loads)
        >>> monitor.log_vlm_loads(forces)
    """

    def __init__(self, solver: "VPMSolver", output_dir: str | None = None):
        """
        Initialize simulation monitor.

        Args:
            solver: Reference to the parent VPM solver
            output_dir: Directory for output files (default: ``<case_dir>/samples``)
        """
        self.solver = solver

        if output_dir is None:
            self.output_dir = os.path.join(str(solver.case_dir), "samples")
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
                - force: [force_x, force_y, force_z] aerodynamic force components
                - moment: [moment_x, moment_y, moment_z] moment components
                - thrust, torque, power: scalar values
                - min_pressure_coefficient, max_pressure_coefficient:
                  pressure coefficient extrema (optional)
        """
        self._ensure_output_dir()

        base_name = os.path.basename(self.solver.checkpoint_name)
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
            "min_pressure_coefficient",
            "max_pressure_coefficient",
        ]

        with open(filename, "a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            writer.writerow(
                {
                    "time": self.solver.time,
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

    def log_vlm_loads(self, forces: dict) -> None:
        """
        Append VLM aerodynamic loads to CSV file.

        Args:
            forces: Dictionary containing VLM force/moment coefficients:
                - lift_coefficient, drag_coefficient, side_force_coefficient:
                  force coefficients
                - force_x, force_y, force_z: force components
                - moment_x, moment_y, moment_z: moment components
                - lift, drag: lift and drag magnitudes
                - dynamic_pressure, reference_area: reference quantities
        """
        self._ensure_output_dir()

        base_name = os.path.basename(self.solver.checkpoint_name)
        filename = os.path.join(self.output_dir, f"vlm_loads_{base_name}.csv")
        file_exists = os.path.isfile(filename)

        fieldnames = [
            "time",
            "step",
            "lift_coefficient",
            "drag_coefficient",
            "side_force_coefficient",
            "force_x",
            "force_y",
            "force_z",
            "moment_x",
            "moment_y",
            "moment_z",
            "lift",
            "drag",
            "dynamic_pressure",
            "reference_area",
        ]

        with open(filename, "a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            writer.writerow(
                {
                    "time": self.solver.time,
                    "step": self.solver.step,
                    "lift_coefficient": forces.get("lift_coefficient", 0.0),
                    "drag_coefficient": forces.get("drag_coefficient", 0.0),
                    "side_force_coefficient": forces.get("side_force_coefficient", 0.0),
                    "force_x": forces.get("force_x", 0.0),
                    "force_y": forces.get("force_y", 0.0),
                    "force_z": forces.get("force_z", 0.0),
                    "moment_x": forces.get("moment_x", 0.0),
                    "moment_y": forces.get("moment_y", 0.0),
                    "moment_z": forces.get("moment_z", 0.0),
                    "lift": forces.get("lift", 0.0),
                    "drag": forces.get("drag", 0.0),
                    "dynamic_pressure": forces.get("dynamic_pressure", 0.0),
                    "reference_area": forces.get("reference_area", 0.0),
                }
            )
