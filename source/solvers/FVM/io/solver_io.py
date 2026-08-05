"""OpenFOAM-compatible field output and step diagnostics."""

import csv
from dataclasses import asdict
import errno
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from ...FVM.fields import field_io
from .storage import append_line_recoverably


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
        self._diagnostics_write_disabled = False

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
        if record is None or self._diagnostics_write_disabled:
            return
        output_dir = os.path.join(self.case_dir, "solution")
        path = os.path.join(output_dir, "diagnostics.jsonl")
        line = json.dumps(asdict(record), sort_keys=True, allow_nan=False) + "\n"
        try:
            append_line_recoverably(path, line)
        except OSError as error:
            if error.errno != errno.ENOSPC:
                raise
            self._diagnostics_write_disabled = True
            self.solver.logger.warning(
                f"Diagnostics output disabled after disk-full error at {path}; "
                "the accepted simulation will continue and the last complete JSONL record "
                "was preserved"
            )

    def rewind_histories(self, flow_time: float) -> None:
        parallel = getattr(self.solver, "parallel", None)
        if parallel is not None and not parallel.is_root:
            return

        solution = Path(self.case_dir) / "solution"
        self._rewind_csv(solution / "forces_history.csv", flow_time)
        self._rewind_csv(solution / "ibm_forces_history.csv", flow_time)
        self._rewind_jsonl(solution / "diagnostics.jsonl", flow_time)
        self._rewind_jsonl(solution / "performance.jsonl", flow_time)

    @staticmethod
    def _replace(path: Path, lines: list[str]) -> None:
        descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
                stream.writelines(lines)
            os.replace(temporary, path)
        except BaseException:
            if os.path.exists(temporary):
                os.unlink(temporary)
            raise

    @classmethod
    def _rewind_csv(cls, path: Path, flow_time: float) -> None:
        if not path.exists():
            return
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.reader(stream))
        if not rows or "time" not in rows[0]:
            return
        time_column = rows[0].index("time")
        kept = [rows[0]]
        for row in rows[1:]:
            if row and float(row[time_column]) <= flow_time + 1e-12:
                kept.append(row)
        if len(kept) != len(rows):
            cls._replace(path, [",".join(row) + "\n" for row in kept])

    @classmethod
    def _rewind_jsonl(cls, path: Path, flow_time: float) -> None:
        if not path.exists():
            return
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        kept = [line for line in lines if float(json.loads(line)["time"]) <= flow_time + 1e-12]
        if len(kept) != len(lines):
            cls._replace(path, kept)

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
