"""Sampler execution for the FVM solver.

:class:`FVMSamplerExecutor` runs after every accepted solver step (and once at
initialisation).  Each sampler decides, through its own
:class:`~.base.SamplingSchedule`, whether it is due; the executor never applies
a global force cadence.

Force, y+ and IBM sampling are MPI-collective in partitioned runs and are never
wrapped in error handling that could let ranks diverge.  Field samplers (line /
surface) gather owned data to root and are therefore **root-only** — non-root
ranks return after the collective sample, they never touch the filesystem, so
there is no PVD race and no ``samples/`` directory creation on non-root ranks.

Sampler failures are strict by default: a missing scientific sample aborts the
step rather than being mistaken for successful output.  Callers may explicitly
choose ``strict=False`` when a best-effort diagnostic is appropriate.
"""

from __future__ import annotations

import os

from . import base
from .forces import ForceSampler, IBMForceSampler, YPlusSampler


class FVMSamplerExecutor:
    """Orchestrates FVM sampler execution for one accepted step."""

    @staticmethod
    def execute(solver, *, strict: bool = True) -> None:
        samplers = list(getattr(solver.setup, "samplers", ()) or ())
        auto_yplus = getattr(solver, "_default_yplus_sampler", None)
        if auto_yplus is not None and auto_yplus not in samplers:
            samplers.append(auto_yplus)
        auto_ibm = getattr(solver, "_default_ibm_sampler", None)
        if auto_ibm is not None and auto_ibm not in samplers:
            samplers.append(auto_ibm)
        if not samplers:
            return

        parallel = solver.parallel
        if not (parallel.is_root or parallel.is_partitioned):
            return

        samples_dir = base.samples_dir(solver.case_dir)
        if parallel.is_root:
            os.makedirs(samples_dir, exist_ok=True)

        for sampler in samplers:
            is_due = getattr(sampler, "is_due", None)
            if is_due is not None and not is_due(
                solver.step, solver.time, solver._current_time_step_size
            ):
                continue
            if isinstance(sampler, ForceSampler):
                forces = sampler.sample(solver)
                solver.last_forces = forces
                if parallel.is_root:
                    sampler.write_csv(solver, samples_dir, forces)
                    solver.logger.force_info(forces)
            elif isinstance(sampler, YPlusSampler):
                stats = sampler.sample(solver)
                if parallel.is_root:
                    solver.last_yplus = stats
                    solver.logger.yplus_info(stats)
                    sampler.write_csv(solver, samples_dir, stats)
            elif isinstance(sampler, IBMForceSampler):
                if not parallel.is_root:
                    continue
                data = sampler.sample(solver)
                sampler.write_csv(solver, samples_dir, data)
                solver.logger.ibm_force_info(sampler.summary(solver, data), data["slip"])
            else:
                FVMSamplerExecutor._write_field_sampler(solver, sampler, samples_dir, strict=strict)

    @staticmethod
    def _write_field_sampler(solver, sampler, samples_dir: str, *, strict: bool = True) -> None:
        if hasattr(sampler, "save_vts"):
            filename = f"{sampler.name}_{solver.step:06d}.vts"
            try:
                data = sampler.save_vts(solver, os.path.join(samples_dir, filename))
            except Exception as exc:
                FVMSamplerExecutor._handle_failure(solver, sampler, exc, strict)
                return
            if not solver.parallel.is_root or data is None:
                return
            entries = getattr(sampler, "_pvd_entries", None)
            if entries is None:
                entries = sampler._pvd_entries = []
            entries.append((solver.time, filename))
            base.write_pvd(samples_dir, sampler.name, entries)
        else:
            try:
                sampler.write_csv(solver, samples_dir)
            except Exception as exc:
                FVMSamplerExecutor._handle_failure(solver, sampler, exc, strict)

    @staticmethod
    def _handle_failure(solver, sampler, exc: Exception, strict: bool) -> None:
        name = getattr(sampler, "file_name", None) or sampler.__class__.__name__
        if strict:
            raise RuntimeError(f"Sampler '{name}' failed: {exc}") from exc
        solver.logger.warning(f"Sampler '{name}' failed: {exc}")
