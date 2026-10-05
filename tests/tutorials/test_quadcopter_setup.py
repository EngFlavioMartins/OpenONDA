"""Physical refinement settings for the quadcopter's small rotor wake."""

from tests._tutorial_helpers import load_tutorial_module


def test_quadcopter_refines_stretched_filaments_without_relaxing_particle_state_limits(monkeypatch):
    setup = load_tutorial_module("vpm/quadcopter")
    rotor = load_tutorial_module("vpm/rotor_flow")
    cases = []
    monkeypatch.setattr(setup, "save_blade", lambda *args, **kwargs: None)
    monkeypatch.setattr(setup, "create_rotor_blade", lambda **kwargs: {})

    class Solver:
        def __init__(self, case):
            cases.append(case)

        def run(self, *, start_from):
            assert start_from == "latest"

    monkeypatch.setattr(setup.vpm, "VPMSolver", Solver)
    setup.run()
    case = cases[0]
    refinement = case.numerics.stabilization.filament_refinement
    assert refinement.interval_steps == 1
    assert (
        refinement.max_vortex_strength_factor
        == (
            rotor.build_case().numerics.stabilization.filament_refinement.max_vortex_strength_factor
        )
        == 2.0
    )
    assert case.numerics.state_limits.lagrangian_cfl.maximum == 1.0
    assert case.numerics.max_n_particles == 500_000
