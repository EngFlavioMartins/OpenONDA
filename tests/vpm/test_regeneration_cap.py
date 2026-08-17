"""The GBD/DVH regeneration cap must follow the allocation, not a constant.

`MAX_PARTICLES` is a module-level *default* for the particle container. Using it
as the regeneration ceiling pinned the coupled cube case at 490k nodes while its
`VPMSetup` allocated 1.5M, so every VPM step discarded a quarter of the cloud
and the hand-off rebuilt it: pure churn, plus ~1% of Sum|Gamma| lost per step.
"""

from types import SimpleNamespace

import pytest

from source.solvers.VPM.config.constants import MAX_PARTICLES


@pytest.fixture
def regeneration_cap():
    from source.solvers.VPM.physics.diffusion.grid import _GridDiffusionMixin

    return _GridDiffusionMixin._regeneration_cap


def _particles(capacity):
    return SimpleNamespace(capacity=capacity)


def test_cap_follows_a_larger_allocation(regeneration_cap):
    """The regression: a 1.5M allocation must not be capped at the 500k default."""
    cap = regeneration_cap(_particles(1_500_000), n_before=650_000, max_nodes=None)
    assert cap > MAX_PARTICLES
    assert cap == min(3 * 650_000, 1_500_000 - 10_000)


def test_cap_never_exceeds_the_allocation(regeneration_cap):
    for capacity in (60_000, 250_000, 500_000, 2_000_000):
        cap = regeneration_cap(_particles(capacity), n_before=capacity, max_nodes=None)
        assert cap <= capacity - 10_000 or cap == 1


def test_explicit_max_nodes_still_wins(regeneration_cap):
    cap = regeneration_cap(_particles(1_500_000), n_before=650_000, max_nodes=800_000)
    assert cap == 800_000


def test_growth_headroom_is_preserved(regeneration_cap):
    """A small cloud may still triple in one step, as before."""
    assert regeneration_cap(_particles(1_000_000), n_before=10_000, max_nodes=None) == 60_000
    assert regeneration_cap(_particles(1_000_000), n_before=100_000, max_nodes=None) == 300_000


def test_container_without_capacity_falls_back_to_the_default(regeneration_cap):
    cap = regeneration_cap(SimpleNamespace(), n_before=650_000, max_nodes=None)
    assert cap == MAX_PARTICLES - 10_000


def test_cap_is_always_positive(regeneration_cap):
    assert regeneration_cap(_particles(1), n_before=0, max_nodes=None) >= 1


def test_container_exposes_capacity():
    pytest.importorskip("taichi")
    from source.solvers.VPM.particles.container import Particles

    assert Particles.capacity.fget(SimpleNamespace(_max_particles=123_456)) == 123_456
