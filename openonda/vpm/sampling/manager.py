"""Dispatch of the VPM samplers at output events, with restart-safe series."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from openonda.core.file_output import append_csv_rows, rewind_csv
from openonda.core.output_events import OutputEvent
from openonda.core.restart import output_has_time
from openonda.core.vtk_output import read_collection, rewind_collection, write_collection

from ..io.logging import Logging

if TYPE_CHECKING:
    from ..core.solver import VPMSolver

__all__ = ["SamplerManager", "SamplingContext", "sampler_name"]


@dataclass(frozen=True, slots=True)
class SamplingContext:
    """Accepted state passed to a sampler's ``write`` method.

    Attributes
    ----------
    solver : VPMSolver
        Solver at the sampled state.
    output_directory : pathlib.Path
        Sample directory of the case.
    step : int
        Accepted step.
    time : float
        Solution time in s.
    event : OutputEvent
        Output event that selected the sampler.
    continuing_output : bool
        Append to the existing series (after a restart or an earlier write
        of this run); otherwise the series starts again.
    """

    solver: VPMSolver
    output_directory: Path
    step: int
    time: float
    event: OutputEvent
    continuing_output: bool


@runtime_checkable
class _WritingSampler(Protocol):
    def write(self, context: SamplingContext) -> None:
        """Write one sample."""


@runtime_checkable
class _VtkSampler(Protocol):
    def write_vtk(self, solver: VPMSolver, path: Path, time: float) -> Path:
        """Write one VTK frame."""


@runtime_checkable
class _TableSampler(Protocol):
    def sample(self, solver: VPMSolver) -> dict[str, np.ndarray]:
        """Return equal-length columns named by ``csv_columns``."""


def sampler_name(sampler: object) -> str:
    """Return the output file stem of a sampler."""
    name = getattr(sampler, "file_name", None)
    return str(name) if name else type(sampler).__name__.lower().removesuffix("sampler")


class SamplerManager:
    """Write the configured samplers of a VPM case at each output event.

    A sampler with ``write(context)`` writes its own files. A sampler with
    ``write_vtk(solver, path, time)`` writes one frame per event,
    ``<name>_{step:06d}<vtk_extension>``, listed in ``<name>.pvd``. A sampler
    with ``sample(solver)`` and ``csv_columns`` appends rows
    ``time, *csv_columns`` to ``<name>.csv``.
    """

    def __init__(self, solver: VPMSolver) -> None:
        self.solver = solver
        self.samplers = solver.case.samplers.samples
        self._last_written: dict[int, tuple[int, float]] = {}
        self._written_names: set[str] = set()

    @property
    def directory(self) -> Path:
        """Sample directory of the case."""
        return self.solver.samples_dir

    def owned_names(self) -> set[str]:
        """Return the file stems of every sample series written by this case."""
        names = {sampler_name(sampler) for sampler in self.samplers}
        vlm = self.solver.vlm_solver
        if vlm is not None:
            names.update(("vlm_forces", "vlm_surface_forces"))
            names.update(
                f"vlm_{direction}_{name.replace('/', '_').replace(' ', '_')}"
                for name in vlm._surface_sampling
                for direction in ("spanwise", "chordwise")
            )
        return names

    def dispatch(self, event: OutputEvent) -> None:
        """Write the samplers selected by ``event``; ``failed`` writes nothing."""
        if event is OutputEvent.failed:
            return
        for sampler in self._selected(event):
            self._write_one(sampler, event)

    def write_all(self, event: OutputEvent) -> None:
        """Write every sampler not yet written at the current state."""
        state = (self.solver.step, self.solver.time)
        for sampler in self.samplers:
            if self._last_written.get(id(sampler)) != state:
                self._write_one(sampler, event)

    def flow_integrals_due(self) -> bool:
        """Return whether a due sampler needs the flow integrals."""
        return any(
            getattr(sampler, "requires_flow_integrals", False)
            for sampler in self._selected(OutputEvent.accepted_step)
        )

    def resume(self) -> None:
        """Write the samples of the restored state that an interrupted run missed."""
        event = OutputEvent.initial if self.solver.step == 0 else OutputEvent.accepted_step
        missing = [
            sampler
            for sampler in self._selected(event)
            if not output_has_time(self.directory, sampler_name(sampler), self.solver.time)
        ]
        if missing:
            self.solver._refresh_diagnostics_for_output()
            for sampler in missing:
                self._write_one(sampler, event)

    def rewind(self, time: float) -> None:
        """Remove sample rows and frames after the restored time ``time`` in s."""
        if not np.isfinite(time):
            raise ValueError("restart time must be finite")
        self._last_written.clear()
        self._written_names.clear()
        if not self.directory.is_dir():
            return
        for sampler in self.samplers:
            validate = getattr(sampler, "validate_existing_vtk", None)
            entries = read_collection(self.directory / f"{sampler_name(sampler)}.pvd")
            kept = [file for entry_time, file in entries if entry_time <= time + 1.0e-12]
            if validate is not None and kept:
                validate(self.directory / kept[-1])
        for name in self.owned_names():
            rewind_csv(self.directory / f"{name}.csv", time)
            rewind_collection(self.directory / f"{name}.pvd", time)

    def _selected(self, event: OutputEvent) -> tuple[object, ...]:
        solver = self.solver
        if event is OutputEvent.initial:
            return tuple(s for s in self.samplers if getattr(s, "initial", False))
        if event is OutputEvent.final:
            return tuple(
                s
                for s in self.samplers
                if getattr(getattr(s, "schedule", None), "is_final_only", False)
            )
        return tuple(
            s
            for s in self.samplers
            if getattr(s, "schedule", None) is not None
            and s.schedule.is_due(solver.step, solver.time, solver.time_step_size)
        )

    def _write_one(self, sampler: object, event: OutputEvent) -> None:
        solver = self.solver
        state = (solver.step, solver.time)
        key = id(sampler)
        name = sampler_name(sampler)
        if self._last_written.get(key) == state:
            return
        if (
            solver._restart_loaded
            and solver.time == solver._restart_output_time
            and output_has_time(self.directory, name, solver.time)
        ):
            self._last_written[key] = state
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        context = SamplingContext(
            solver=solver,
            output_directory=self.directory,
            step=solver.step,
            time=solver.time,
            event=event,
            continuing_output=solver._restart_loaded or name in self._written_names,
        )
        try:
            self._write(sampler, name, context)
        except Exception as error:
            raise RuntimeError(
                f"Sampler {name!r} failed at step {context.step}, time {context.time}: {error}"
            ) from error
        self._last_written[key] = state
        self._written_names.add(name)
        Logging.debug(f"Sampler {name} written at step {context.step}")

    def _write(self, sampler: object, name: str, context: SamplingContext) -> None:
        if isinstance(sampler, _WritingSampler):
            sampler.write(context)
        elif isinstance(sampler, _VtkSampler):
            extension = getattr(sampler, "vtk_extension", ".vts")
            file_name = f"{name}_{context.step:06d}{extension}"
            collection = context.output_directory / f"{name}.pvd"
            entries = read_collection(collection) if context.continuing_output else []
            sampler.write_vtk(context.solver, context.output_directory / file_name, context.time)
            write_collection(collection, [*entries, (context.time, file_name)])
        elif isinstance(sampler, _TableSampler):
            columns = tuple(sampler.csv_columns)
            data = sampler.sample(context.solver)
            if set(data) != set(columns):
                raise ValueError("Sampler result does not match its csv_columns")
            path = context.output_directory / f"{name}.csv"
            if not context.continuing_output:
                path.unlink(missing_ok=True)
            arrays = [np.asarray(data[column]).reshape(-1) for column in columns]
            rows = [[context.time, *values] for values in zip(*arrays, strict=True)]
            append_csv_rows(path, ("time", *columns), rows)
        else:
            raise TypeError("sampler must implement write(), write_vtk() or sample()")
