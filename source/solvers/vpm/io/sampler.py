"""Typed scientific-sampler dispatch and restart-safe output indexes."""

from __future__ import annotations

from collections.abc import Hashable
import csv
from dataclasses import dataclass, field
from enum import StrEnum
import os
from pathlib import Path
import shutil
import tempfile
from tempfile import NamedTemporaryFile
from typing import Protocol, cast, runtime_checkable

import defusedxml.ElementTree as ET  # noqa: N817
import numpy as np

from ..config.artifacts import Samplers
from .logging import Logging
from .sampling import OutputSchedule, resolve_samples_dir, sampler_csv_columns


class SamplerRuntimeSolver(Protocol):
    """Minimal solver surface required by framework-owned sampler dispatch.

    Implementations expose an accepted `step`/`time`, case/output paths, and
    the configured time-step size. `_write_backup()` is called only when the
    accepted-step backup cadence is due; a scientific sampler must not infer
    backup ownership from that hook.
    """

    case: object
    case_dir: Path
    step: int
    time: float
    time_step_size: float

    def _write_backup(self) -> None:
        """Write one numerical backup selected by the configured cadence."""


class OutputEvent(StrEnum):
    """Lifecycle events accepted by the VPM output runtime.

    `INITIAL` and `FINAL` are explicit framework events; `ACCEPTED_STEP` is
    emitted after a committed physical step; `FAILED` is a notification-only
    event and never writes scientific samples.
    """

    INITIAL = "initial"
    ACCEPTED_STEP = "accepted_step"
    FINAL = "final"
    FAILED = "failed"


@dataclass(frozen=True)
class SamplingContext:
    """Immutable accepted-state context passed to a typed sampler.

    Attributes
    ----------
    solver : SamplerRuntimeSolver
        Solver whose state is being written.
    output_directory : pathlib.Path
        Framework-owned destination for this sampler and case.
    step : int
        Accepted-step index associated with the event.
    time : float
        Accepted physical time in seconds.
    event : OutputEvent
        Lifecycle event that selected the sampler.
    continuing_output : bool
        ``True`` when the writer must append to an existing stream from a
        loaded numerical restart or an earlier event in this process. ``False``
        means this is the first event of a fresh run and replaces stale output.

    The context is a snapshot of dispatch metadata; it does not copy particle
    or field arrays and therefore must not be retained as a mutable state view.
    """

    solver: SamplerRuntimeSolver
    output_directory: Path
    step: int
    time: float
    event: OutputEvent
    continuing_output: bool


@runtime_checkable
class Sampler(Protocol):
    """Typed sampler protocol for a framework-owned write operation."""

    schedule: OutputSchedule | None

    def write(self, context: SamplingContext) -> None:
        """Write one sample for ``context``."""


@runtime_checkable
class _VtkSampler(Protocol):
    def save_vtp(
        self, solver: SamplerRuntimeSolver, filepath: Path, time: float | None = None
    ) -> None:
        """Write one VTK/structured-grid snapshot to ``filepath``."""


@runtime_checkable
class _CsvSampler(Protocol):
    def save_csv(
        self, solver: SamplerRuntimeSolver, filepath: Path, time: float | None = None
    ) -> None:
        """Write one CSV snapshot to ``filepath``."""


@runtime_checkable
class _TableSampler(Protocol):
    def sample(self, solver: SamplerRuntimeSolver) -> dict[str, np.ndarray]:
        """Return one-dimensional, equal-length columns for CSV output."""


@dataclass
class _SamplerRuntime:
    """Mutable runtime state separated from immutable sampler configuration."""

    pvd_entries: dict[str, list[tuple[float, str]]] = field(default_factory=dict)
    last_written: dict[Hashable, tuple[int, float]] = field(default_factory=dict)


class OutputManager:
    """Own VPM sampler schedules, output paths, and restart-safe indexes.

    The manager separates immutable sampler configuration from mutable PVD/CSV
    runtime state. It dispatches only after accepted lifecycle events, creates
        the owner-controlled sample destination, writes VTK/CSV files atomically
        where supported, and raises a sampler-specific `RuntimeError` on failed
        scientific output.
    """

    def __init__(self, solver: SamplerRuntimeSolver, samplers: Samplers | None = None) -> None:
        """Bind output dispatch to a solver and its sampler configuration.

        Parameters
        ----------
        solver : SamplerRuntimeSolver
            Live VPM solver exposing case, clock, and backup hooks.
        samplers : Samplers or None
            Optional override; `None` uses ``solver.case.samplers``. The
            configuration is not mutated.
        """
        self.solver = solver
        self.samplers = solver.case.samplers if samplers is None else samplers
        self._runtime = _SamplerRuntime()

    def dispatch(self, event: OutputEvent) -> None:
        """Deliver samplers selected by one lifecycle event.

        `ACCEPTED_STEP` may trigger a numerical backup independently of sampler
        selection. `FAILED` performs no writes. Sampler exceptions are wrapped
        with name/step/time context and propagated to the owning lifecycle.
        """
        if event is OutputEvent.FAILED:
            return
        if event is OutputEvent.ACCEPTED_STEP and self._backup_due():
            # Numerical backups are an output event too, but deliberately
            # does not imply any scientific sampler dispatch.
            self.solver._write_backup()
        for sampler in self._selected(event):
            self._execute_one(sampler, event)

    def write_all(
        self, event: OutputEvent = OutputEvent.INITIAL, *, skip_current: bool = False
    ) -> None:
        """Write every configured sampler once for an explicit manual event.

        Parameters
        ----------
        event : OutputEvent, default=INITIAL
            Metadata event used for the write. `FAILED` is invalid.
        skip_current : bool, default=False
            Skip a sampler already written at the current accepted step/time.
        """
        if event is OutputEvent.FAILED:
            raise ValueError("manual sampler execution cannot use the failed event")
        for sampler in self.samplers.samples:
            if skip_current and self._runtime.last_written.get(self._output_identity(sampler)) == (
                self.solver.step,
                self.solver.time,
            ):
                continue
            self._execute_one(sampler, event)

    def resume(self) -> None:
        """Fill samples interrupted after the native backup was committed."""
        from source.restart import output_has_time

        event = OutputEvent.INITIAL if self.solver.step == 0 else OutputEvent.ACCEPTED_STEP
        directory = resolve_samples_dir(self.solver.case_dir, self.samplers.directory)
        missing = [
            sample
            for sample in self._selected(event)
            if not output_has_time(directory, self._name(sample), self.solver.time)
        ]
        if missing:
            self.solver._refresh_diagnostics_for_output()
            for sample in missing:
                self._execute_one(sample, event)

    def _backup_due(self) -> bool:
        """Return whether the accepted-step backup cadence fires now."""
        interval = self.solver.case.backup.interval_steps
        return interval > 0 and self.solver.step > 0 and self.solver.step % interval == 0

    def any_due(self, step: int, time: float) -> bool:
        """Return whether a scheduled sampler needs the accepted state."""
        return any(self._is_due(sample, step, time) for sample in self.samplers.samples)

    def flow_integrals_due(self, step: int, time: float) -> bool:
        """Return whether due samplers need flow-integral diagnostics."""
        return any(
            bool(getattr(sample, "requires_flow_integrals", False))
            and self._is_due(sample, step, time)
            for sample in self.samplers.samples
        )

    def rewind_histories(self, time: float) -> None:
        """Reconcile VPM scientific output with an accepted restart time.

        Parameters
        ----------
        time : float
            Inclusive accepted physical time in seconds. Rows and VTK
            collection entries after this time are removed before resumed
            output is appended.

        Notes
        -----
        A process can publish a sample after the last atomic numerical
        checkpoint and then be interrupted. Keeping that sample would make a
        resumed writer reject the first repeated event as duplicate or
        nonmonotonic. Only configured sampler streams are owned here;
        unrecognized files in the shared samples directory are preserved.
        Superseded PVD indexes are copied to ``restart-branches`` before the
        active index is replaced. Snapshot payloads remain in place because
        they are immutable and may be useful when inspecting the interrupted
        branch.
        """
        if not np.isfinite(time):
            raise ValueError("restart time must be finite")

        output_directory = resolve_samples_dir(self.solver.case_dir, self.samplers.directory)
        if not output_directory.is_dir():
            self._runtime.pvd_entries.clear()
            self._runtime.last_written.clear()
            return

        owned_names = {self._name(sampler) for sampler in self.samplers.samples}
        if getattr(self.solver, "vlm_solver", None) is not None:
            owned_names.update(("vlm_forces", "vlm_surface_forces"))
            owned_names.update(
                f"vlm_{direction}_{name.replace('/', '_').replace(' ', '_')}"
                for name in getattr(self.solver.vlm_solver, "_surface_sampling", {})
                for direction in ("spanwise", "chordwise")
            )
        for name in owned_names:
            self._rewind_csv(output_directory / f"{name}.csv", time)
            self._rewind_pvd(output_directory, name, time)

        for name, entries in self._runtime.pvd_entries.items():
            self._runtime.pvd_entries[name] = [
                (entry_time, filename)
                for entry_time, filename in entries
                if entry_time <= time + 1.0e-12
            ]
        self._runtime.last_written.clear()

    @classmethod
    def _rewind_csv(cls, filepath: Path, time: float) -> None:
        """Keep configured CSV rows through an inclusive restart time."""
        if not filepath.is_file() or filepath.stat().st_size == 0:
            return
        with filepath.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.reader(stream))
        has_terminal_newline = filepath.read_bytes().endswith((b"\n", b"\r"))
        if not rows:
            return
        try:
            time_column = rows[0].index("time")
        except ValueError:
            raise ValueError(f"CSV sampler output {filepath} has no time column") from None

        kept = [rows[0]]
        needs_rewrite = False
        for index, row in enumerate(rows[1:], start=1):
            if len(row) != len(rows[0]):
                if index == len(rows) - 1 and not has_terminal_newline:
                    needs_rewrite = True
                    break
                raise ValueError(f"CSV sampler output {filepath} has an invalid row")
            try:
                event_time = float(row[time_column])
            except ValueError as exc:
                raise ValueError(f"CSV sampler output {filepath} has an invalid time") from exc
            if not np.isfinite(event_time):
                raise ValueError(f"CSV sampler output {filepath} has a non-finite time")
            if event_time <= time + 1.0e-12:
                kept.append(row)
                if index == len(rows) - 1 and not has_terminal_newline:
                    needs_rewrite = True
            else:
                needs_rewrite = True
        if needs_rewrite or len(kept) != len(rows):
            branch_root = filepath.parent / "restart-branches"
            branch_root.mkdir(parents=True, exist_ok=True)
            branch = Path(tempfile.mkdtemp(prefix="before-", dir=branch_root))
            shutil.copy2(filepath, branch / f"{filepath.name}.superseded")
            cls._replace_csv(filepath, kept[0], kept[1:])

    @classmethod
    def _replace_csv(cls, filepath: Path, header: list[str], rows: list[list[str]]) -> None:
        """Atomically replace a CSV stream after history reconciliation."""
        with NamedTemporaryFile(
            "w", newline="", encoding="utf-8", dir=filepath.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(header)
            writer.writerows(rows)
        try:
            os.replace(temporary, filepath)
        finally:
            temporary.unlink(missing_ok=True)

    @classmethod
    def _rewind_pvd(cls, output_directory: Path, name: str, time: float) -> None:
        """Keep one configured VTK collection through an inclusive time."""
        pvd_path = output_directory / f"{name}.pvd"
        if not pvd_path.is_file():
            return
        entries = cls._read_pvd(output_directory, name)
        kept = [entry for entry in entries if entry[0] <= time + 1.0e-12]
        if len(kept) == len(entries):
            return

        branch_root = output_directory / "restart-branches"
        branch_root.mkdir(parents=True, exist_ok=True)
        branch = Path(tempfile.mkdtemp(prefix="before-", dir=branch_root))
        shutil.copy2(pvd_path, branch / pvd_path.name)
        from xml.etree import ElementTree as XmlElementTree

        from source.solvers.fvm.io.solver_io import SolverIO

        future = [
            XmlElementTree.Element("DataSet", {"timestep": str(t), "file": name})
            for t, name in entries
            if t > time + 1.0e-12
        ]
        SolverIO._archive_pvd_frames(pvd_path, future, branch)
        cls._write_pvd(output_directory, name, kept)

    def _selected(self, event: OutputEvent) -> tuple[object, ...]:
        """Select configured samplers for one lifecycle event."""
        if event is OutputEvent.ACCEPTED_STEP:
            return tuple(
                sample
                for sample in self.samplers.samples
                if self._is_due(sample, self.solver.step, self.solver.time)
            )
        if event is OutputEvent.FINAL:
            return tuple(sample for sample in self.samplers.samples if self._is_final(sample))
        # Initial output is intentionally opt-in: a sampler must explicitly
        # provide an ``initial`` boolean rather than relying on an implicit mode.
        return tuple(
            sample for sample in self.samplers.samples if bool(getattr(sample, "initial", False))
        )

    def _schedule(self, sampler: object) -> OutputSchedule | None:
        """Return an explicitly declared output schedule when present."""
        return cast(OutputSchedule | None, getattr(sampler, "schedule", None))

    def _is_final(self, sampler: object) -> bool:
        schedule = self._schedule(sampler)
        return bool(schedule is not None and schedule.is_final_only)

    def _is_due(self, sampler: object, step: int, time: float) -> bool:
        schedule = self._schedule(sampler)
        return bool(
            schedule is not None
            and not self._is_final(sampler)
            and schedule.is_due(step, time, self.solver.time_step_size)
        )

    def _execute_one(self, sampler: object, event: OutputEvent) -> None:
        """Write one sampler atomically and update its last-written index."""
        identity = self._output_identity(sampler)
        if getattr(self.solver, "_restart_output_time", None) == self.solver.time:
            from source.restart import output_has_time

            directory = resolve_samples_dir(self.solver.case_dir, self.samplers.directory)
            if output_has_time(directory, self._name(sampler), self.solver.time):
                self._runtime.last_written[identity] = (self.solver.step, self.solver.time)
                return
        if event is OutputEvent.FINAL and self._runtime.last_written.get(identity) == (
            self.solver.step,
            self.solver.time,
        ):
            return
        applicable = getattr(sampler, "is_applicable", None)
        if applicable is not None and not applicable(self.solver):
            Logging.info(
                f"Sampler {type(sampler).__name__} skipped because its prerequisite is not met"
            )
            return
        directory = resolve_samples_dir(self.solver.case_dir, self.samplers.directory)
        directory.mkdir(parents=True, exist_ok=True)
        continuing_output = bool(
            getattr(self.solver, "_restart_loaded", False) or identity in self._runtime.last_written
        )
        context = SamplingContext(
            self.solver,
            directory,
            self.solver.step,
            self.solver.time,
            event,
            continuing_output,
        )
        try:
            self._write(sampler, context)
            self._runtime.last_written[identity] = (context.step, context.time)
        except Exception as exc:
            prefix = self._name(sampler)
            raise RuntimeError(
                f"Sampler {prefix!r} failed at step {context.step}, time {context.time}: {exc}"
            ) from exc

    @staticmethod
    def _output_identity(sampler: object) -> Hashable:
        """Allow equivalent scheduled writers to declare one scientific output.

        An explicit identity must include every option affecting the sampled
        values and destination, excluding scheduling. Other samplers retain
        instance identity; a matching filename alone cannot establish equality.
        This index covers successful writes by this manager, never pre-existing
        disk data, which remain subject to the writer's resume checks.
        """
        return cast(Hashable, getattr(sampler, "output_identity", id(sampler)))

    @staticmethod
    def _name(sampler: object) -> str:
        name = getattr(sampler, "file_name", None)
        return str(name) if name else type(sampler).__name__.lower().removesuffix("sampler")

    def _write(self, sampler: object, context: SamplingContext) -> None:
        """Dispatch one sampler through its typed write protocol.

        VTK and CSV snapshots are written through temporary sibling files and
        atomically renamed. Table samplers are appended through a validated
        atomic rewrite so a failed write cannot leave a partial header/row.
        """
        if isinstance(sampler, Sampler):
            sampler.write(context)
            return
        prefix = self._name(sampler)
        if isinstance(sampler, _VtkSampler):
            extension = getattr(sampler, "vtk_extension", ".vts")
            filename = f"{prefix}_{context.step:06d}{extension}"
            final_path = context.output_directory / filename
            if prefix not in self._runtime.pvd_entries:
                self._runtime.pvd_entries[prefix] = (
                    self._read_pvd(context.output_directory, prefix)
                    if context.continuing_output
                    else []
                )
            entries = self._runtime.pvd_entries[prefix]
            temp_path = context.output_directory / f".{filename}.tmp{extension}"
            sampler.save_vtp(context.solver, temp_path, time=context.time)
            os.replace(temp_path, final_path)
            self._append_pvd(entries, context.time, filename)
            self._write_pvd(context.output_directory, prefix, entries)
            return
        if isinstance(sampler, _TableSampler) and getattr(sampler, "csv_time_series", False):
            # A line's save_csv() is a single-snapshot convenience API.
            # Scheduled output must retain all accepted times for averaging
            # and restart continuation, as its explicit flag requests.
            self._append_csv(sampler, context, context.output_directory / f"{prefix}.csv")
            return
        if isinstance(sampler, _CsvSampler):
            final_path = context.output_directory / f"{prefix}.csv"
            temp_path = context.output_directory / f".{prefix}.tmp.csv"
            sampler.save_csv(context.solver, temp_path, time=context.time)
            os.replace(temp_path, final_path)
            return
        if isinstance(sampler, _TableSampler):
            self._append_csv(sampler, context, context.output_directory / f"{prefix}.csv")
            return
        raise TypeError(
            "sampler must implement write(context), save_vtp(), save_csv(), or sample()"
        )

    @staticmethod
    def _append_pvd(entries: list[tuple[float, str]], time: float, filename: str) -> None:
        """Append one unique, monotonically increasing PVD entry."""
        if any(existing_filename == filename for _, existing_filename in entries):
            return
        if entries and time <= entries[-1][0]:
            raise ValueError("PVD event is duplicate or nonmonotonic during resume")
        entries.append((float(time), filename))

    @staticmethod
    def _append_csv(sampler: _TableSampler, context: SamplingContext, filepath: Path) -> None:
        """Validate one table sample and append it to an atomic CSV rewrite."""
        data = sampler.sample(context.solver)
        columns = sampler_csv_columns(sampler)
        missing = [name for name in columns if name not in data]
        if missing:
            raise ValueError(f"Sampler result is missing columns: {', '.join(missing)}")
        lengths = {len(np.asarray(data[name])) for name in columns}
        if len(lengths) != 1:
            raise ValueError("Sampler result columns do not all have the same length")
        header = ["time", "step", *columns]
        existing = (
            OutputManager._read_csv_rows(filepath, header) if context.continuing_output else []
        )
        if existing and float(cast(str, existing[-1][0])) >= context.time:
            raise ValueError("CSV event is duplicate or nonmonotonic during resume")
        rows: list[list[object]] = [
            [context.time, context.step, *values]
            for values in zip(
                *(np.asarray(data[name]).reshape(-1) for name in columns), strict=True
            )
        ]
        OutputManager._atomic_csv(filepath, header, existing + rows)

    @staticmethod
    def _read_csv_rows(filepath: Path, expected_header: list[str]) -> list[list[object]]:
        """Read compatible time-series rows; reject legacy snapshots before append."""
        if not filepath.exists() or filepath.stat().st_size == 0:
            return []
        with filepath.open(newline="", encoding="utf-8") as stream:
            reader = csv.reader(stream)
            if next(reader, None) != expected_header:
                raise ValueError(
                    f"CSV schema mismatch in {filepath.name}: expected a time/step series. "
                    "Preserve the existing file and choose a fresh output directory."
                )
            return [[*row] for row in reader if row]

    @staticmethod
    def _atomic_csv(filepath: Path, header: list[str], rows: list[list[object]]) -> None:
        """Write a complete CSV to a temporary file and atomically replace it."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(
            "w", newline="", encoding="utf-8", dir=filepath.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(header)
            writer.writerows(rows)
        os.replace(temporary, filepath)

    @staticmethod
    def _write_pvd(output_dir: Path, name_prefix: str, entries: list[tuple[float, str]]) -> None:
        """Write a VTK collection index for one sampler prefix."""
        output_dir.mkdir(parents=True, exist_ok=True)
        pvd_path = output_dir / f"{name_prefix}.pvd"
        lines = [
            '<?xml version="1.0"?>',
            '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
            "  <Collection>",
        ]
        lines.extend(
            f'    <DataSet timestep="{time}" file="{filename}"/>' for time, filename in entries
        )
        lines.extend(("  </Collection>", "</VTKFile>"))
        with NamedTemporaryFile("w", encoding="utf-8", dir=output_dir, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write("\n".join(lines))
        os.replace(temporary, pvd_path)

    @staticmethod
    def _read_pvd(output_dir: Path, name_prefix: str) -> list[tuple[float, str]]:
        """Read and validate an existing PVD index for resume-safe output."""
        pvd_path = output_dir / f"{name_prefix}.pvd"
        if not pvd_path.is_file():
            return []
        try:
            root = ET.parse(pvd_path).getroot()
            entries = [
                (float(dataset.attrib["timestep"]), dataset.attrib["file"])
                for dataset in root.findall(".//DataSet")
            ]
        except (ET.ParseError, OSError, ValueError, KeyError) as exc:
            raise ValueError(f"invalid PVD index {pvd_path}") from exc
        if entries != sorted(entries) or len({filename for _, filename in entries}) != len(entries):
            raise ValueError(f"PVD index {pvd_path} is not monotonic and unique")
        return entries
