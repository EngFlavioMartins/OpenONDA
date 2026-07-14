"""Standardized I/O and Diagnostics for FVM Solver.

Provides tools for writing OpenFOAM-format fields, computing runtime
diagnostics like forces and Courant numbers, and managing snapshot history.

Author: OpenONDA Team
Date: January 2026
"""

import os
import sys
from typing import Any

import numpy as np

from ...FVM.fields import diagnostics, field_io


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

        # Allow output_dir override if runner script wants to redirect data
        base_dir = getattr(self, "output_dir", None) or self.case_dir
        output_dir = os.path.join(base_dir, time_dir)
        os.makedirs(output_dir, exist_ok=True)

        print(f"  Writing legacy OpenFOAM snapshot to {output_dir}")
        sys.stdout.flush()

        # Prepare fields for IO (include ghost cells via owner copy)
        fields_to_write = self._gather_fields_for_io()

        for field_data in fields_to_write:
            field_io.write_foam_field(
                os.path.join(output_dir, field_data["name"]), self.solver.mesh_data, field_data
            )

        # Log forces if a configuration is found
        try:
            self._maybe_log_forces()
        except Exception as e:
            print(f"   (Warning) Force logging failed: {e}")
            sys.stdout.flush()

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
        algo = self.solver.config.solver.algorithm.upper()
        if algo in ["PISO", "PIMPLE"]:
            Co = diagnostics.compute_courant_number(
                self.solver.U,
                self.solver.phi,
                self.solver.dt,
                self.solver.mesh_data,
                self.solver.geo_data,
            )
            vort = diagnostics.compute_vorticity(
                self.solver.U, self.solver.mesh_data, self.solver.geo_data
            )
            fields.append({"name": "Co", "type": "volScalarField", "phi": _extend(Co)})
            fields.append({"name": "vorticity", "type": "volVectorField", "phi": _extend(vort)})
            print(f"    - Max Co: {np.max(Co):.3f}")

        if hasattr(self.solver, "nut") and self.solver.nut is not None:
            fields.append(
                {"name": "nut", "type": "volScalarField", "phi": _extend(self.solver.nut)}
            )

        return fields

    def _maybe_log_forces(self):
        """
        Compute and log surface forces to CSV and stdout.

        Checks for the presence of ``system/forceCoefficients`` or
        ``constant/forceCoefficients`` in the case directory. If found,
        surface forces (pressure + viscous) are computed for patches
        whose names contain ``"cube"`` (or all boundaries if none
        match). Results are appended to ``forces_history.csv`` and a
        summary line is printed to stdout.

        Reference quantities (ref_U, ref_area, ref_length, moment_centre)
        are read from the solver configuration.

        Raises:
            Exception: Propagated from
                :func:`diagnostics.compute_surface_forces` if the force
                evaluation fails. The caller should handle this silently
                (the message is printed as a warning in
                :meth:`write_results`).
        """
        # Detect presence of force configs in legacy locations
        candidates = [
            os.path.join(self.case_dir, "system/forceCoefficients"),
            os.path.join(self.case_dir, "constant/forceCoefficients"),
        ]
        if not any(os.path.exists(c) for c in candidates):
            return

        # Evaluating for all 'cube' or explicitly tagged patches
        patches = [b["name"] for b in self.solver.boundaries if "cube" in b["name"].lower()]
        if not patches:
            patches = [b["name"] for b in self.solver.boundaries]

        # Use initial magnitude for ref_U if not explicit
        U0 = self.solver.config.initial_U
        ref_U = np.linalg.norm(U0) if isinstance(U0, list | np.ndarray) else (U0 or 1.0)

        ref_area = getattr(self.solver.config.solver, "ref_area", 1.0)
        ref_length = getattr(self.solver.config.solver, "ref_length", 1.0)
        moment_centre = getattr(self.solver.config.solver, "moment_centre", [0.0, 0.0, 0.0])

        forces = diagnostics.compute_surface_forces(
            self.solver.U,
            self.solver.p,
            self.solver.config.transport.nu * self.solver.config.transport.density,
            self.solver.config.transport.density,
            self.solver.mesh_data,
            self.solver.geo_data,
            self.solver.boundaries,
            patch_names=patches,
            ref_U=float(ref_U),
            ref_area=float(ref_area),
            ref_length=float(ref_length),
            moment_centre=moment_centre,
        )

        # Append to CSV
        backup_csv = os.path.join(self.case_dir, "forces_history.csv")
        write_header = not os.path.exists(backup_csv)
        with open(backup_csv, "a") as fh:
            import csv

            writer = csv.writer(fh)
            if write_header:
                writer.writerow(["time", "step", "patch", "Fx", "Fy", "Fz", "Cd", "Cl", "Cz", "Cm"])
            for pname, pdata in forces.items():
                F = pdata.get("Ftot", [0, 0, 0])
                C = pdata.get("coeffs", {})
                writer.writerow(
                    [
                        self.solver.flow_time,
                        self.solver.time_step,
                        pname,
                        F[0],
                        F[1],
                        F[2],
                        C.get("Cd"),
                        C.get("Cl"),
                        C.get("Cz"),
                        C.get("Cm"),
                    ]
                )

        # Summary to stdout
        for pname, pdata in forces.items():
            F = pdata.get("Ftot", [0, 0, 0])
            C = pdata.get("coeffs", {})
            print(
                f"   Forces ({pname}): Fx={F[0]:.3f}, Fy={F[1]:.3f}, Fz={F[2]:.3f} | Cd={C.get('Cd', 0):.3f} Cl={C.get('Cl', 0):.3f} Cz={C.get('Cz', 0):.3f} Cm={C.get('Cm', 0):.3f}"
            )
        sys.stdout.flush()
