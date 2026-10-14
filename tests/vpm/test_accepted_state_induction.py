"""An unchanged accepted particle state is evaluated once and reused exactly."""

import numpy as np

from openonda.vpm import ViscousConfig, VPMSolver
from tests.vpm._vpm_case import add_counter_rotating_pair, small_case


def _advance(directory, reuse_induction):
    solver = VPMSolver(small_case(directory, viscous=ViscousConfig.inviscid()))
    solver.stage_rhs.reuse_induction = reuse_induction
    try:
        add_counter_rotating_pair(solver)
        solver.advance()
        evaluations = []
        for step in range(4):
            if step == 2:
                # An external update of the accepted sources forbids reuse.
                strength = solver.particles.vortex_strength_cpu()
                solver.set_particle_properties(vortex_strength=1.1 * strength)
            before = solver.stage_rhs.induction_reuse_statistics.evaluations
            solver.advance()
            evaluations.append(solver.stage_rhs.induction_reuse_statistics.evaluations - before)
        return (
            evaluations,
            solver.particles.position_cpu(use_cache=False),
            solver.particles.vortex_strength_cpu(use_cache=False),
        )
    finally:
        solver.close()


def test_accepted_state_induction_is_reused_bit_for_bit(tmp_path):
    reused, position, strength = _advance(tmp_path / "reused", True)
    evaluated, expected_position, expected_strength = _advance(tmp_path / "evaluated", False)

    # SSPRK3: three stages, plus the post-step state refresh unless the
    # first stage reuses the refreshed state of the previous step.
    assert reused == [3, 3, 4, 3]
    assert evaluated == [4, 4, 4, 4]
    np.testing.assert_array_equal(position, expected_position)
    np.testing.assert_array_equal(strength, expected_strength)
