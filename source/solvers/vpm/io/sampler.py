"""Typed scientific-sampler dispatch and restart-safe output indexes."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from enum import StrEnum
import os
from pathlib import Path
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
        Resolved samples directory for this case.
    step : int
        Accepted-step index associated with the event.
    time : float
        Accepted physical time in seconds.
    event : OutputEvent
        Lifecycle event that selected the sampler.

    The context is a snapshot of dispatch metadata; it does not copy particle
    or field arrays and therefore must not be retained as a mutable state view.
    """

    solver: SamplerRuntimeSolver
    output_directory: Path
    step: int
    time: float
    event: OutputEvent


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
    last_written: dict[int, tuple[int, float]] = field(default_factory=dict)


class OutputManager:
    """Own VPM sampler schedules, output paths, and restart-safe indexes.

    The manager separates immutable sampler configuration from mutable PVD/CSV
    runtime state. It dispatches only after accepted lifecycle events, creates
    the case samples directory, writes VTK/CSV files atomically where supported,
    and raises a sampler-specific `RuntimeError` on failed scientific output.
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
            if skip_current and self._runtime.last_written.get(id(sampler)) == (
                self.solver.step,
                self.solver.time,
            ):
                continue
            self._execute_one(sampler, event)

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
        return bool(
            schedule is not None
            and getattr(schedule, "is_final_only", getattr(schedule, "at_end", False))
        )

    def _is_due(self, sampler: object, step: int, time: float) -> bool:
        schedule = self._schedule(sampler)
        return bool(
            schedule is not None
            and not self._is_final(sampler)
            and schedule.is_due(step, time, self.solver.time_step_size)
        )

    def _execute_one(self, sampler: object, event: OutputEvent) -> None:
        """Write one sampler atomically and update its last-written index."""
        applicable = getattr(sampler, "is_applicable", None)
        if applicable is not None and not applicable(self.solver):
            Logging.info(
                f"component=sampler name={type(sampler).__name__!r} status=skipped "
                "reason=prerequisite_not_met"
            )
            return
        directory = resolve_samples_dir(self.solver.case_dir, self.samplers.directory)
        directory.mkdir(parents=True, exist_ok=True)
        context = SamplingContext(self.solver, directory, self.solver.step, self.solver.time, event)
        try:
            self._write(sampler, context)
            self._runtime.last_written[id(sampler)] = (context.step, context.time)
        except Exception as exc:
            prefix = self._name(sampler)
            raise RuntimeError(
                f"Sampler {prefix!r} failed at step {context.step}, time {context.time}: {exc}"
            ) from exc

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
            temp_path = context.output_directory / f".{filename}.tmp{extension}"
            sampler.save_vtp(context.solver, temp_path, time=context.time)
            os.replace(temp_path, final_path)
            entries = self._runtime.pvd_entries.setdefault(
                prefix, self._read_pvd(context.output_directory, prefix)
            )
            self._append_pvd(entries, context.time, filename)
            self._write_pvd(context.output_directory, prefix, entries)
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
        existing = OutputManager._read_csv_rows(filepath)
        if existing and float(cast(str, existing[-1][0])) >= context.time:
            raise ValueError("CSV event is duplicate or nonmonotonic during resume")
        rows: list[list[object]] = [
            [context.time, context.step, *values]
            for values in zip(
                *(np.asarray(data[name]).reshape(-1) for name in columns), strict=True
            )
        ]
        OutputManager._atomic_csv(filepath, ["time", "step", *columns], existing + rows)

    @staticmethod
    def _read_csv_rows(filepath: Path) -> list[list[object]]:
        """Read existing CSV data rows without interpreting scientific values."""
        if not filepath.exists() or filepath.stat().st_size == 0:
            return []
        with filepath.open(newline="", encoding="utf-8") as stream:
            reader = csv.reader(stream)
            next(reader, None)
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
