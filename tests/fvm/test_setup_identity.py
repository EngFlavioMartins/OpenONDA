"""docs/nomenclature.md: solver constructors store their *Setup object as
``self.setup`` and never keep a parallel ``self.config`` vocabulary."""

import contextlib
import io

from source.solvers.FVM import (
    BoundaryConfig,
    FVMSetup,
    FVMSolver,
    LinearSolverConfig,
    PimpleControl,
    SchemesConfig,
    TimeConfig,
    TransportConfig,
)

from ._structured_mesh import structured_box


def _make_setup() -> FVMSetup:
    return FVMSetup(
        case_name="setup-identity",
        time=TimeConfig(time_step_size=0.1, end_time=1.0, write_interval=999),
        schemes=SchemesConfig(convection_scheme="central"),
        linear=LinearSolverConfig(linear_solver="spsolve"),
        pimple=PimpleControl(n_correctors=2),
        transport=TransportConfig(density=1.0, nu=0.05),
        boundaries=[
            BoundaryConfig.wall(n) for n in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")
        ],
        initial_velocity=[0.0, 0.0, 0.0],
    )


def test_fvm_solver_owns_its_setup_object():
    setup = _make_setup()
    with contextlib.redirect_stdout(io.StringIO()):
        solver = FVMSolver(
            setup, case_dir="/tmp/openonda-setup-identity", mesh_data=structured_box(2, 2, 2)
        )
    assert solver.setup is setup
    assert not hasattr(solver, "config")


def test_fvm_setup_is_not_mutated_during_initialization():
    setup = _make_setup()
    with contextlib.redirect_stdout(io.StringIO()):
        FVMSolver(setup, case_dir="/tmp/openonda-setup-identity", mesh_data=structured_box(2, 2, 2))
    assert setup.samplers == ()
    assert setup.cores == 1
