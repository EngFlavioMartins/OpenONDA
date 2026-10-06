"""Resolve particle advection within an unchanged FVM exchange interval."""

from contextlib import contextmanager


@contextmanager
def particle_advection_substeps(owner, count, measurements):
    """Use native RK integration at smaller intervals without extra remeshing.

    The accepted particle clock, diffusion interval, remeshing frequency and
    FVM boundary histories remain on the original exchange clock. Only the
    Runge--Kutta particle trajectories are refined.
    """
    solver = owner.vpm_solver
    original = solver.stepper._advance_particles
    measurements.update(substeps=count, calls=0, maximum_interval=0.0)

    def advance(interval):
        start = solver.time
        step = interval / count
        try:
            for index in range(count):
                solver.time = start + index * step
                original(step)
        finally:
            solver.time = start
        measurements["calls"] += 1
        measurements["maximum_interval"] = max(measurements["maximum_interval"], step)

    solver.stepper._advance_particles = advance
    try:
        yield
    finally:
        solver.stepper._advance_particles = original
