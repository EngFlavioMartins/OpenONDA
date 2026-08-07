"""Sampler execution for the FVM solver.

:class:`FVMSamplerExecutor` runs after every accepted solver step (and once at
initialisation).  Each sampler decides, through its own
:class:`~.base.SamplingSchedule`, whether it is due; the executor never applies
a global force cadence.

Force and y+ sampling are MPI-collective in partitioned runs and are therefore
never wrapped in error handling that could let ranks diverge.  Field samplers
are root-only, and a failing field sampler must not abort an accepted step —
its error is logged and the step proceeds (legacy samplers without a schedule
are treated as always due).
"""

from __future__ import annotations

import os

from . import base
from .forces import ForceSampler, IBMForceSampler, YPlusSampler


class FVMSamplerExecutor:
    """Orchestrates FVM sampler execution for one accepted step."""

    @staticmethod
    def execute(solver) -> None:
        samplers = list(getattr(solver.config, "samplers", ()) or ())
        auto_yplus = getattr(solver, "_default_yplus_sampler", None)
        if auto_yplus is not None:
            samplers.append(auto_yplus)
        if not samplers:
            return

        parallel = solver.parallel
        if not (parallel.is_root or parallel.is_partitioned):
            return

        samples_dir = base.samples_dir(solver.case_dir)
        if parallel.is_root:
            os.makedirs(samples_dir, exist_ok=True)

        for sampler in samplers:
            if isinstance(sampler, ForceSampler):
                if not sampler.is_due(solver.time_step, solver.flow_time, solver._current_dt):
                    continue
                forces = sampler.sample(solver)
                solver.last_forces = forces
                if parallel.is_root:
                    sampler.write_csv(solver, samples_dir, forces)
                    solver.logger.force_info(forces)
            elif isinstance(sampler, YPlusSampler):
                if not sampler.is_due(solver.time_step, solver.flow_time, solver._current_dt):
                    continue
                stats = sampler.sample(solver)
                if parallel.is_root:
                    solver.last_yplus = stats
                    solver.logger.yplus_info(stats)
                    sampler.write_csv(solver, samples_dir, stats)
            elif isinstance(sampler, IBMForceSampler):
                if not parallel.is_root:
                    continue
                if not sampler.is_due(solver.time_step, solver.flow_time, solver._current_dt):
                    continue
                data = sampler.sample(solver)
                sampler.write_csv(solver, samples_dir, data)
                solver.logger.ibm_force_info(sampler.summary(solver, data), data["slip"])
            else:
                FVMSamplerExecutor._write_field_sampler(solver, sampler, samples_dir)

    @staticmethod
    def _write_field_sampler(solver, sampler, samples_dir: str) -> None:
        is_due = getattr(sampler, "is_due", None)
        if is_due is not None and not is_due(
            solver.time_step, solver.flow_time, solver._current_dt
        ):
            return
        if hasattr(sampler, "save_vts"):
            if not sampler.should_write():
                return
            filename = f"{sampler.name}_{solver.time_step:06d}.vts"
            try:
                sampler.save_vts(solver, os.path.join(samples_dir, filename))
            except Exception as exc:
                name = getattr(sampler, "file_name", None) or sampler.__class__.__name__
                solver.logger.warning(f"Sampler '{name}' failed: {exc}")
                return
            entries = getattr(sampler, "_pvd_entries", None)
            if entries is None:
                entries = sampler._pvd_entries = []
            entries.append((solver.flow_time, filename))
            base.write_pvd(samples_dir, sampler.name, entries)
        else:
            try:
                sampler.write_csv(solver, samples_dir)
            except Exception as exc:
                name = getattr(sampler, "file_name", None) or sampler.__class__.__name__
                solver.logger.warning(f"Sampler '{name}' failed: {exc}")
