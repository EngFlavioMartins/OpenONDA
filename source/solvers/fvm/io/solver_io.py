"""Native FVM diagnostics and restart-history maintenance."""

import csv
from dataclasses import asdict
import errno
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np

from source.solution_layout import collection_path

from .storage import append_line_recoverably


class SolverIO:
    """Own FVM diagnostics, restart-history, and output reconciliation.

    The numerical solver owns field writers; this adapter owns append-only
    diagnostics and safe history rewinds. It writes only on the root rank in a
    partitioned run and uses temporary files/atomic replacement for rewrites.

    Attributes
    ----------
    solver : FVMSolver
        Parent solver/context.
    case_dir, solution_dir, samples_dir : str or pathlib.Path
        Case-owned output destinations.
    """

    def __init__(self, solver: object) -> None:
        """Bind the I/O manager to an initialized solver.

        Parameters
        ----------
        solver : FVMSolver
            Object exposing case, solution, and sample directories plus a
            logger/parallel context.
        """
        self.solver = solver
        self.case_dir = solver.case_dir
        self.solution_dir = solver.solution_dir
        self.samples_dir = solver.samples_dir
        self._diagnostics_write_disabled = False

    def write_step_diagnostics(self) -> None:
        """Append the accepted-step health record as one JSON object.

        The operation is root-only and recoverable on disk-full: an ENOSPC
        disables later diagnostics writes and emits a warning, while other I/O
        errors propagate.
        """
        parallel = getattr(self.solver, "parallel", None)
        if parallel is not None and not parallel.is_root:
            return
        record = getattr(self.solver, "last_diagnostics", None)
        if record is None or self._diagnostics_write_disabled:
            return
        path = os.path.join(self.solution_dir, "diagnostics.jsonl")
        line = json.dumps(asdict(record), sort_keys=True, allow_nan=False) + "\n"
        try:
            append_line_recoverably(path, line)
        except OSError as error:
            if error.errno != errno.ENOSPC:
                raise
            self._diagnostics_write_disabled = True
            self.solver.logger.warning(
                f"Diagnostic output stopped because the disk is full: {path}"
            )

    def rewind_histories(self, time: float) -> None:
        """Truncate solver-owned histories beyond a restart time.

        Parameters
        ----------
        time : float
            Inclusive accepted physical time in seconds. CSV/PVD/JSONL entries
            after this time are removed; superseded PVD files are copied to a
            ``restart-branches`` directory before replacement.

        Notes
        -----
        Unrecognized files in shared sample directories are preserved. The
        in-memory PVD indexes of active writers are rewound as well.
        """
        parallel = getattr(self.solver, "parallel", None)
        if parallel is not None and not parallel.is_root:
            return

        samples = Path(self.samples_dir)
        solution = Path(self.solution_dir)

        samplers = list(getattr(self.solver, "_samplers", ()) or ())
        samplers.extend(
            sampler
            for sampler in (
                getattr(self.solver, "_default_yplus_sampler", None),
                getattr(self.solver, "_default_ibm_sampler", None),
            )
            if sampler is not None
        )
        owned_stems = {
            str(getattr(sampler, "file_name", None) or sampler.name)
            for sampler in samplers
            if getattr(sampler, "file_name", None) is not None or hasattr(sampler, "name")
        }
        # Every solver-owned sampler CSV is rewound by its "time" column.
        # Unrecognised user files in the shared samples directory are never
        # considered restart products.
        if samples.is_dir():
            for csv_path in samples.glob("*.csv"):
                if csv_path.stem in owned_stems:
                    self._rewind_csv(csv_path, time)
            # Surface-sampler PVD indices: drop frames past the resume time so
            # a restarted live run or re-run PostProcess does not double-list
            # them (the per-step .vts files stay keyed by their own step).
            for pvd_path in samples.glob("*.pvd"):
                if pvd_path.stem in owned_stems:
                    self._rewind_pvd(pvd_path, time)

        self._rewind_jsonl(solution / "diagnostics.jsonl", time)
        self._rewind_jsonl(solution / "performance.jsonl", time)
        self._rewind_pvd(collection_path(solution, "fvm"), time)

        # Reconcile in-memory indexes held by already-created writers.  The
        # on-disk branch remains available for inspection; only the active
        # collection is rewound.
        manager = getattr(self.solver, "pvd_manager", None)
        if manager is not None:
            manager.rewind(time)
        writer = getattr(self.solver, "_buffered_vtk_writer", None)
        if writer is not None:
            writer.rewind(time)
        runtime = getattr(self.solver, "_sample_pvd_entries", None)
        if runtime is not None:
            runtime.clear()

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
    def _replace_csv(cls, path: Path, rows: list[list[str]]) -> None:
        descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerows(rows)
            os.replace(temporary, path)
        except BaseException:
            if os.path.exists(temporary):
                os.unlink(temporary)
            raise

    @staticmethod
    def _archive_superseded(path: Path) -> None:
        """Retain the pre-rewind stream without matching active-output names."""
        branch_root = path.parent / "restart-branches"
        branch_root.mkdir(parents=True, exist_ok=True)
        branch = Path(tempfile.mkdtemp(prefix="before-", dir=branch_root))
        shutil.copy2(path, branch / f"{path.name}.superseded")

    @classmethod
    def _rewind_csv(cls, path: Path, time: float) -> None:
        if not path.exists():
            return
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.reader(stream))
        has_terminal_newline = path.read_bytes().endswith((b"\n", b"\r"))
        if not rows:
            return
        time_column = next(
            (i for i, name in enumerate(rows[0]) if name.strip() == "time"),
            None,
        )
        if time_column is None:
            return
        kept = [rows[0]]
        needs_rewrite = False
        for index, row in enumerate(rows[1:], start=1):
            if len(row) != len(rows[0]):
                if index == len(rows) - 1 and not has_terminal_newline:
                    needs_rewrite = True
                    break
                raise ValueError(f"CSV solver output {path} has an invalid row")
            try:
                row_time = float(row[time_column])
            except ValueError as error:
                raise ValueError(f"CSV solver output {path} has an invalid time") from error
            if not np.isfinite(row_time):
                raise ValueError(f"CSV solver output {path} has a non-finite time")
            if row_time <= time + 1e-12:
                kept.append(row)
                if index == len(rows) - 1 and not has_terminal_newline:
                    needs_rewrite = True
            else:
                needs_rewrite = True
        if needs_rewrite or len(kept) != len(rows):
            cls._archive_superseded(path)
            cls._replace_csv(path, kept)

    @classmethod
    def _rewind_pvd(cls, path: Path, time: float) -> None:
        if not path.exists():
            return
        from xml.etree import ElementTree as XmlElementTree

        from defusedxml import ElementTree as SafeElementTree

        tree = SafeElementTree.parse(path)
        collection = tree.find(".//Collection")
        if collection is None:
            return
        datasets = list(collection)
        kept = []
        future = []
        seen: set[str] = set()
        for dataset in datasets:
            if dataset.tag != "DataSet":
                kept.append(dataset)
                continue
            try:
                dataset_time = float(dataset.attrib["timestep"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"Invalid PVD dataset in {path}") from error
            filename = dataset.attrib.get("file", "")
            if dataset_time > time + 1.0e-12:
                future.append(dataset)
                continue
            if filename in seen:
                continue
            seen.add(filename)
            kept.append(dataset)
        if not future and len(kept) == len(datasets):
            return

        # Preserve the superseded index and its future frame payloads before
        # replay can overwrite the same step-named files.
        branch_root = path.parent / "restart-branches"
        branch_root.mkdir(parents=True, exist_ok=True)
        branch = Path(tempfile.mkdtemp(prefix="before-", dir=branch_root))
        shutil.copy2(path, branch / path.name)
        cls._archive_pvd_frames(path, future, branch)

        collection.clear()
        collection.extend(
            sorted(
                kept, key=lambda item: (float(item.attrib["timestep"]), item.attrib.get("file", ""))
            )
        )
        # defusedxml intentionally exposes parsing primitives only; use the
        # standard library serializer on the already-sanitized element tree.
        xml = XmlElementTree.tostring(tree.getroot(), encoding="unicode") + "\n"
        cls._replace(path, [xml])

    @staticmethod
    def _archive_pvd_frames(path: Path, future: list, branch: Path) -> None:
        """Move superseded VTK payloads with their original relative layout."""
        from defusedxml import ElementTree as SafeElementTree

        seen: set[Path] = set()
        for dataset in future:
            filename = dataset.attrib.get("file", "")
            relative = Path(filename)
            if not filename or relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Unsafe PVD frame path {filename!r} in {path}")
            frame = path.parent / relative
            candidates = [frame]
            if frame.suffix == ".pvtu" and frame.is_file():
                for piece in SafeElementTree.parse(frame).findall(".//Piece"):
                    name = piece.attrib.get("Source", "")
                    piece_path = Path(name)
                    if not name or piece_path.is_absolute() or ".." in piece_path.parts:
                        raise ValueError(f"Unsafe PVTU piece path {name!r} in {frame}")
                    candidates.append(frame.parent / piece_path)
            for source in candidates:
                if source in seen or not source.is_file():
                    continue
                if source.is_symlink():
                    raise ValueError(f"Refusing symlinked PVD frame {source}")
                seen.add(source)
                destination = branch / source.relative_to(path.parent)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(source, destination)

    @classmethod
    def _rewind_jsonl(cls, path: Path, time: float) -> None:
        if not path.exists():
            return
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        kept = []
        needs_rewrite = False
        for index, line in enumerate(lines):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                # A process killed during append can leave one incomplete final
                # line. Preserve it in the branch archive and recover the valid
                # prefix; malformed interior records remain a hard error.
                if index == len(lines) - 1 and not line.endswith(("\n", "\r")):
                    needs_rewrite = True
                    break
                raise ValueError(f"JSONL solver output {path} has an invalid record") from error
            if not isinstance(record, dict):
                raise ValueError(f"JSONL solver output {path} has an invalid record")
            try:
                row_time = float(record["time"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"JSONL solver output {path} has an invalid record") from error
            if not np.isfinite(row_time):
                raise ValueError(f"JSONL solver output {path} has a non-finite time")
            if row_time <= time + 1e-12:
                if not line.endswith(("\n", "\r")):
                    line += "\n"
                    needs_rewrite = True
                kept.append(line)
            else:
                needs_rewrite = True
        if needs_rewrite:
            cls._archive_superseded(path)
            cls._replace(path, kept)
