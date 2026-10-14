"""Dispatch of the FVM samplers at the output events of a run.

Force and y+ samplers are collective in a partitioned run; field samplers
gather the owned cells to rank 0, and only rank 0 writes files.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from openonda.core.output_events import OutputEvent
from openonda.core.restart import output_has_time
from openonda.core.schedules import FinalOnly
from openonda.core.vtk_output import read_collection, write_collection

from .fields import LineSampler, SurfaceSampler
from .forces import ForceSampler, YPlusSampler

if TYPE_CHECKING:
    from openonda.core.sampling import Sampler

    from ..core.solver import FVMSolver


def _is_due(sampler: Sampler, solver: FVMSolver, event: OutputEvent) -> bool:
    """Return whether ``sampler`` writes at ``event`` for the accepted state."""
    if event is OutputEvent.initial:
        return sampler.initial
    if event is OutputEvent.final:
        return isinstance(sampler.schedule, FinalOnly)
    return sampler.is_due(solver.step, solver.time, solver._accepted_time_step_size)


def execute_samplers(solver: FVMSolver, event: OutputEvent) -> None:
    """Write the samplers that are due at ``event``.

    Every rank must call this function in a partitioned run.
    """
    samplers = [sampler for sampler in solver._samplers if _is_due(sampler, solver, event)]
    parallel = solver.parallel
    if not samplers or not (parallel.is_root or parallel.is_partitioned):
        return
    samples_dir = solver.samples_dir
    if parallel.is_root:
        samples_dir.mkdir(parents=True, exist_ok=True)
    for sampler in samplers:
        if event is OutputEvent.final and solver._restart_loaded:
            # A run restored at its end time already holds the final sample.
            exists = output_has_time(samples_dir, sampler.name, solver.time)
            if parallel.bcast(exists if parallel.is_root else None, root=0):
                continue
        if isinstance(sampler, ForceSampler):
            forces = sampler.sample(solver)
            if parallel.is_root:
                sampler.write_csv(solver, samples_dir, forces)
                solver.logger.forces(forces)
        elif isinstance(sampler, YPlusSampler):
            statistics = sampler.sample(solver)
            if parallel.is_root:
                solver.logger.y_plus(statistics)
                sampler.write_csv(solver, samples_dir, statistics)
        elif isinstance(sampler, SurfaceSampler):
            file_name = f"{sampler.name}_{solver.step:06d}.vts"
            if sampler.write_vtk(solver, samples_dir / file_name):
                collection = samples_dir / f"{sampler.name}.pvd"
                frames = solver._sample_collections.get(sampler.name)
                if frames is None:
                    frames = solver._sample_collections[sampler.name] = read_collection(collection)
                frames.append((solver.time, file_name))
                write_collection(collection, frames)
        elif isinstance(sampler, LineSampler):
            sampler.write_csv(solver, samples_dir)
        else:
            raise TypeError(f"Unsupported FVM sampler {type(sampler).__name__}")
