"""OpenFOAM-compatible field output and step diagnostics."""

from dataclasses import asdict
import json
import os
from typing import Any

import numpy as np

from ...FVM.fields import field_io


class SolverIO:
    """Unified IO and Diagnostics manager for the FVM Solver.

    Attributes:
        solver (Solver): Reference to the parent FVM solver instance.
        case_dir (str): Working directory for the simulation.
    """

    def __init__(self, solver: Any):
        """Initializes the IO manager.

        Args:
            solver: The FVM solver instance to manage.
        """
        self.solver = solver
        self.case_dir = solver.case_dir

    def write_results(self, time_dir: str | None = None):
        """Writes current fields to disk in OpenFOAM format.

        This method is primarily used for maintaining compatibility with
        post-processing tools that expect standard OpenFOAM structures.

        Args:
            time_dir: Subdirectory name for the snapshot. Defaults to current flow time.
        """
        parallel = getattr(self.solver, "parallel", None)
        if parallel is not None and not parallel.is_root:
            return
        if time_dir is None:
            time_dir = f"{self.solver.flow_time:.5g}"

        output_dir = os.path.join(self.case_dir, time_dir)
        os.makedirs(output_dir, exist_ok=True)

        self.solver.logger.output_info(f"Writing legacy OpenFOAM snapshot to {output_dir}")

        # Prepare fields for IO (include ghost cells via owner copy)
        fields_to_write = self._gather_fields_for_io()

        for field_data in fields_to_write:
            field_io.write_foam_field(
                os.path.join(output_dir, field_data["name"]), self.solver.mesh_data, field_data
            )

    def write_step_diagnostics(self) -> None:
        """Append the accepted step health record as one JSON object."""
        parallel = getattr(self.solver, "parallel", None)
        if parallel is not None and not parallel.is_root:
            return
        record = getattr(self.solver, "last_diagnostics", None)
        if record is None:
            return
        output_dir = os.path.join(self.case_dir, "solution")
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "diagnostics.jsonl"), "a", encoding="utf-8") as stream:
            stream.write(json.dumps(asdict(record), sort_keys=True, allow_nan=False) + "\n")

    def _gather_fields_for_io(self) -> list[dict[str, Any]]:
        """
        Collect and prepare internal solver fields for file-based output.

        Gathers core fields (U, p, phi) and diagnostic fields (Co,
        vorticity, nut) from the solver, extends scalar/vector arrays
        with boundary-face (ghost) values so that OpenFOAM-format
        writers receive a matching number of entries.

        Returns:
            list[dict[str, Any]]: A list of field dictionaries, each
            containing keys ``"name"``, ``"type"`` (e.g. ``"volVectorField"``
            or ``"volScalarField"``), and ``"phi"`` (the extended data array).
        """
        fields = []
        mesh = self.solver.mesh_data
        n_elements = mesh["n_elements"]
        n_interior = mesh["n_interior_faces"]
        owners_b = mesh["owners"][n_interior:]

        def _extend(arr):
            if arr.size == n_elements:
                # Extend scalars
                ext = np.zeros(n_elements + len(owners_b))
                ext[:n_elements] = arr
                ext[n_elements:] = arr[owners_b]
                return ext
            elif arr.ndim == 2 and arr.shape[0] == n_elements:
                # Extend vectors
                ext = np.zeros((n_elements + len(owners_b), arr.shape[1]))
                ext[:n_elements] = arr
                ext[n_elements:] = arr[owners_b]
                return ext
            return arr  # Already extended or unknown

        # Core fields
        if hasattr(self.solver, "U"):
            fields.append({"name": "U", "type": "volVectorField", "phi": self.solver.U})
        if hasattr(self.solver, "p"):
            fields.append({"name": "p", "type": "volScalarField", "phi": self.solver.p})
        if hasattr(self.solver, "phi"):
            fields.append({"name": "phi", "type": "volScalarField", "phi": self.solver.phi})

        # Diagnostics
        algo = self.solver.config.pimple.algorithm.upper()
        if algo in ["PISO", "PIMPLE"]:
            Co = self.solver._courant_field(self.solver._current_dt)
            vort = self.solver._vorticity_field()
            fields.append({"name": "Co", "type": "volScalarField", "phi": _extend(Co)})
            fields.append({"name": "vorticity", "type": "volVectorField", "phi": _extend(vort)})
            self.solver.logger.info(f"Maximum Courant number: {np.max(Co):.3e}")

        if hasattr(self.solver, "nut") and self.solver.nut is not None:
            fields.append(
                {"name": "nut", "type": "volScalarField", "phi": _extend(self.solver.nut)}
            )

        return fields
