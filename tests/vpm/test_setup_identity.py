from source.solvers.VPM import VPMSetup, VPMSolver


def test_vpm_solver_owns_its_setup_object():
    setup = VPMSetup(compute_device="CPU", checkpoint_interval_steps=0, logging_interval_steps=0)
    solver = VPMSolver(setup=setup)
    assert solver.setup is setup
    assert not hasattr(solver, "config")
