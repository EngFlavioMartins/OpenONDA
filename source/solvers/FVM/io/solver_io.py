"""Native FVM diagnostics and restart-history maintenance."""

import csv
from dataclasses import asdict
import errno
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from .storage import append_line_recoverably


class SolverIO:
    """Unified IO and Diagnostics manager for the FVM Solver.

    Attributes:
        solver (FVMSolver): Reference to the parent FVM solver instance.
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
            self.solver.logger.warning(f"Diagnostics output disabled: no space left on {path}")

    def rewind_histories(self, time: float) -> None:
        parallel = getattr(self.solver, "parallel", None)
        if parallel is not None and not parallel.is_root:
            return

        samples = Path(self.case_dir) / "samples"
        solution = Path(self.case_dir) / "solution"

        # Every sampler CSV is rewound by its "time" column.
        if samples.is_dir():
            for csv_path in samples.glob("*.csv"):
                self._rewind_csv(csv_path, time)
            # Surface-sampler PVD indices: drop frames past the resume time so
            # a restarted live run or re-run PostProcess does not double-list
            # them (the per-step .vts files stay keyed by their own step).
            for pvd_path in samples.glob("*.pvd"):
                self._rewind_pvd(pvd_path, time)

        self._rewind_jsonl(solution / "diagnostics.jsonl", time)
        self._rewind_jsonl(solution / "performance.jsonl", time)

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
    def _rewind_csv(cls, path: Path, time: float) -> None:
        if not path.exists():
            return
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.reader(stream))
        if not rows:
            return
        time_column = next(
            (i for i, name in enumerate(rows[0]) if name.strip() == "time"),
            None,
        )
        if time_column is None:
            return
        kept = [rows[0]]
        for row in rows[1:]:
            if row and float(row[time_column]) <= time + 1e-12:
                kept.append(row)
        if len(kept) != len(rows):
            cls._replace(path, [",".join(row) + "\n" for row in kept])

    @classmethod
    def _rewind_pvd(cls, path: Path, time: float) -> None:
        if not path.exists():
            return
        import re as _re

        text = path.read_text(encoding="utf-8")
        kept = _re.sub(
            r'<DataSet timestep="([^"]+)"[^>]*?/>',
            lambda m: m.group(0) if float(m.group(1)) <= time + 1e-12 else "",
            text,
        )
        if kept != text:
            cls._replace(path, [kept])

    @classmethod
    def _rewind_jsonl(cls, path: Path, time: float) -> None:
        if not path.exists():
            return
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        kept = [line for line in lines if float(json.loads(line)["time"]) <= time + 1e-12]
        if len(kept) != len(lines):
            cls._replace(path, kept)
