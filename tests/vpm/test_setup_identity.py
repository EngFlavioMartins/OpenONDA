from source.solvers.VPM import VPMSetup, VPMSolver


def test_vpm_solver_owns_its_setup_object():
    setup = VPMSetup(processing_unit="CPU", backup_frequency=0, logging_frequency=0)
    solver = VPMSolver(setup=setup)
    assert solver.setup is setup
    assert not hasattr(solver, "config")
