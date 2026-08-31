"""Revision-keyed cache contract for particle snapshots."""

from __future__ import annotations

from source.solvers.vpm.config.state import cached_particle_property


class _RevisionedParticles:
    def __init__(self) -> None:
        self.state_revision = 0
        self.evaluations = 0

    @cached_particle_property
    def position_cpu(self):
        self.evaluations += 1
        return self.state_revision


def test_particle_snapshot_cache_invalidates_on_source_revision_not_step():
    particles = _RevisionedParticles()

    assert particles.position_cpu() == 0
    assert particles.position_cpu() == 0
    assert particles.evaluations == 1

    # The solver step need not change for a coupled/stabilization mutation.
    particles.state_revision += 1
    assert particles.position_cpu() == 1
    assert particles.evaluations == 2
