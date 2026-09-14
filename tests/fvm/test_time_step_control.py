"""Public construction and numerical behavior of FVM time-step control."""

from __future__ import annotations

import contextlib
from dataclasses import FrozenInstanceError, replace
import io

import pytest

from openonda import fvm
from source.solvers.fvm.assemble.time_integration import backward_coefficients
from source.solvers.fvm.core.solver import FVMSolver
from source.solvers.fvm.core.time_step import maximum_courant_time_step_size
from source.solvers.fvm.mesh.cartesian import structured_box


def test_maximum_courant_control_reduces_immediately_and_limits_growth() -> None:
    control = fvm.MaximumCourantTimeStep(maximum=0.9)

    assert maximum_courant_time_step_size(0.01, 1.8, control) == pytest.approx(0.005)
    assert maximum_courant_time_step_size(0.01, 0.45, control) == pytest.approx(0.012)
    assert maximum_courant_time_step_size(0.01, 0.0, control) == pytest.approx(0.012)


def test_maximum_courant_control_applies_the_configured_step_ceiling() -> None:
    control = fvm.MaximumCourantTimeStep(maximum=0.9, maximum_time_step_size=0.011)

    assert maximum_courant_time_step_size(0.01, 0.0, control) == pytest.approx(0.011)


@pytest.mark.parametrize(
    ("kwargs", "error_type"),
    [
        ({"maximum": True}, TypeError),
        ({"maximum": 0.0}, ValueError),
        ({"maximum": float("nan")}, ValueError),
        ({"maximum_time_step_size": -1.0}, ValueError),
    ],
)
def test_maximum_courant_control_rejects_invalid_construction(
    kwargs: dict[str, object], error_type: type[Exception]
) -> None:
    with pytest.raises(error_type):
        fvm.MaximumCourantTimeStep(**kwargs)


def test_time_step_control_round_trips_through_the_fvm_setup(tmp_path) -> None:
    path = tmp_path / "setup.json"
    setup = fvm.FVMSetup(
        case_name="adaptive",
        logging=fvm.LoggingConfig(schedule=fvm.RunSchedule(every_time=0.05)),
        backup=fvm.BackupConfig(
            schedule=fvm.RunSchedule(every_n_steps=8),
            write_at_end=True,
        ),
        time=fvm.TimeConfig(
            time_step_size=0.01,
            output_schedule=fvm.RunSchedule(every_time=0.1),
            adjustment=fvm.MaximumCourantTimeStep(
                maximum=0.7,
                maximum_time_step_size=0.02,
            ),
        ),
    )

    setup.save(str(path))
    restored = fvm.FVMSetup.load(str(path))

    assert restored.time == setup.time
    assert restored.logging.schedule == setup.logging.schedule
    assert restored.backup == setup.backup


def test_time_step_control_is_immutable() -> None:
    control = fvm.MaximumCourantTimeStep(maximum=0.9)
    time = fvm.TimeConfig(time_step_size=0.01, adjustment=control)

    with pytest.raises(FrozenInstanceError):
        control.maximum = 1.0
    with pytest.raises(FrozenInstanceError):
        time.adjustment = None


def test_run_schedule_supports_accepted_steps_and_physical_time() -> None:
    steps = fvm.RunSchedule(every_n_steps=3)
    physical_time = fvm.RunSchedule(every_time=0.1)

    assert steps.is_due(3, 0.07, 0.02)
    assert not steps.is_due(2, 0.05, 0.02)
    assert physical_time.is_due(7, 0.1, 0.03)
    assert physical_time.next_time_after(0.1) == pytest.approx(0.2)
    with pytest.raises(FrozenInstanceError):
        physical_time.every_time = 0.2


def test_variable_step_backward_coefficients_are_quadratic_exact() -> None:
    current_step = 0.2
    previous_step = 0.1
    coefficient_new, coefficient_old, coefficient_older = backward_coefficients(
        current_step, previous_step
    )

    assert backward_coefficients(0.1, 0.1) == pytest.approx((1.5, 2.0, 0.5))
    derivative = (
        coefficient_new * 0.0**2
        - coefficient_old * (-current_step) ** 2
        + coefficient_older * (-current_step - previous_step) ** 2
    ) / current_step
    assert derivative == pytest.approx(0.0, abs=1.0e-14)


def test_adaptive_sampling_does_not_create_the_cube_reference_sliver_step() -> None:
    """Replay the time controller immediately before the saved t=8.4 drag spike."""
    solver = object.__new__(FVMSolver)
    solver.time = 8.389653072879867
    solver.time_step_size = 0.008611637016626665
    solver._time_config = fvm.TimeConfig(
        time_step_size=0.01,
        end_time=30.0,
        adjustment=fvm.MaximumCourantTimeStep(maximum=0.9, maximum_time_step_size=0.04),
    )
    schedule = fvm.RunSchedule(every_time=0.05)
    solver._run_event_schedules = lambda: (schedule,)
    courant_rate = 0.3610601598897632 / solver.time_step_size
    solver._measure_maximum_courant_number = lambda dt: courant_rate * dt
    steps = []
    while solver.time < 8.4 - 1.0e-12:
        dt = solver._select_time_step_size()
        steps.append(dt)
        solver.time += dt

    assert solver.time == pytest.approx(8.4, abs=1.0e-12)
    assert len(steps) == 2
    assert min(steps) > 0.005
    assert max(steps) * courant_rate <= 0.9


def test_adaptive_event_alignment_respects_cfl_and_every_schedule() -> None:
    solver = object.__new__(FVMSolver)
    solver.time = 0.0
    solver.time_step_size = 0.013
    solver._time_config = fvm.TimeConfig(
        time_step_size=0.013,
        end_time=0.3,
        adjustment=fvm.MaximumCourantTimeStep(maximum=0.9),
    )
    schedules = (fvm.RunSchedule(every_time=0.05), fvm.RunSchedule(every_time=0.03))
    solver._run_event_schedules = lambda: schedules
    solver._measure_maximum_courant_number = lambda dt: 50 * dt
    due_times = [[], []]
    previous_step = solver.time_step_size
    while solver.time < solver._time_config.end_time - 1.0e-12:
        dt = solver._select_time_step_size()
        assert dt <= 0.9 / 50 + 1.0e-14
        assert dt <= 1.2 * previous_step + 1.0e-14
        assert dt >= 0.5 * previous_step - 1.0e-14
        solver.time += dt
        for index, schedule in enumerate(schedules):
            if schedule.is_due(0, solver.time, dt):
                due_times[index].append(solver.time)
        previous_step = dt

    assert due_times[0] == pytest.approx([0.05 * k for k in range(1, 7)])
    assert due_times[1] == pytest.approx([0.03 * k for k in range(1, 11)])


def test_adaptive_backward_run_lands_exactly_on_end_time(tmp_path) -> None:
    setup = fvm.FVMSetup(
        case_name="adaptive_backward",
        time=fvm.TimeConfig(
            time_step_size=0.08,
            end_time=0.1,
            output_schedule=fvm.RunSchedule(every_n_steps=100),
            adjustment=fvm.MaximumCourantTimeStep(
                maximum=100.0,
                maximum_time_step_size=0.08,
            ),
        ),
        schemes=fvm.DiscretizationConfig(
            convection_scheme="upwind",
            time_scheme="backward",
        ),
        linear=fvm.LinearSolverConfig(linear_solver="spsolve"),
        pimple=fvm.PimpleControl(n_correctors=2),
        transport=fvm.TransportConfig(density=1.0, kinematic_viscosity=0.02),
        boundaries=[
            fvm.BoundaryConfig.inlet("xmin", [0.5, 0.0, 0.0]),
            fvm.BoundaryConfig.outlet("xmax"),
            fvm.BoundaryConfig.wall("ymin"),
            fvm.BoundaryConfig.wall("ymax"),
            fvm.BoundaryConfig.wall("zmin"),
            fvm.BoundaryConfig.wall("zmax"),
        ],
        initial_velocity=[0.2, 0.0, 0.0],
    )
    with contextlib.redirect_stdout(io.StringIO()):
        solver = FVMSolver(
            setup,
            str(tmp_path),
            mesh_data=structured_box(2, 2, 2),
        )
        solver.auto_write = False
        solver.advance()
        first_time_step_size = solver._accepted_time_step_size
        solver.advance()

    assert first_time_step_size == pytest.approx(0.05)
    assert solver._accepted_time_step_size == pytest.approx(0.05)
    assert solver._previous_time_step_size == pytest.approx(0.05)
    assert solver.time == pytest.approx(0.1)
    assert solver.step == 2
    solver.close()


def test_time_based_backup_is_an_exact_adaptive_step_deadline(tmp_path) -> None:
    setup = fvm.FVMSetup(
        case_name="scheduled_backup",
        logging=fvm.LoggingConfig(schedule=fvm.RunSchedule(every_time=0.03)),
        backup=fvm.BackupConfig(schedule=fvm.RunSchedule(every_time=0.03)),
        time=fvm.TimeConfig(
            time_step_size=0.08,
            end_time=0.1,
            output_schedule=fvm.RunSchedule(every_n_steps=100),
            adjustment=fvm.MaximumCourantTimeStep(
                maximum=100.0,
                maximum_time_step_size=0.08,
            ),
        ),
        schemes=fvm.DiscretizationConfig(convection_scheme="upwind"),
        linear=fvm.LinearSolverConfig(linear_solver="spsolve"),
        pimple=fvm.PimpleControl(n_correctors=2),
        transport=fvm.TransportConfig(density=1.0, kinematic_viscosity=0.02),
        boundaries=[
            fvm.BoundaryConfig.inlet("xmin", [0.5, 0.0, 0.0]),
            fvm.BoundaryConfig.outlet("xmax"),
            fvm.BoundaryConfig.wall("ymin"),
            fvm.BoundaryConfig.wall("ymax"),
            fvm.BoundaryConfig.wall("zmin"),
            fvm.BoundaryConfig.wall("zmax"),
        ],
        initial_velocity=[0.2, 0.0, 0.0],
    )
    with contextlib.redirect_stdout(io.StringIO()):
        solver = FVMSolver(setup, str(tmp_path), mesh_data=structured_box(2, 2, 2))
        solver.advance()

    assert solver.time == pytest.approx(0.03)
    assert solver._accepted_time_step_size == pytest.approx(0.03)
    assert (tmp_path / "solution" / "backup").is_file()
    solver.close()


def test_output_deadline_does_not_generate_a_pressure_impulse(tmp_path) -> None:
    """A 64-cell 3D channel reproduces the pressure jump from a tiny final step."""
    setup = fvm.FVMSetup(
        case_name="event_pressure",
        logging=fvm.LoggingConfig(console=False),
        backup=fvm.BackupConfig(schedule=None),
        time=fvm.TimeConfig(
            time_step_size=0.01,
            end_time=1.0,
            output_schedule=fvm.RunSchedule(every_n_steps=100000),
        ),
        schemes=fvm.DiscretizationConfig(
            convection_scheme="linearUpwind", gradient_scheme="gauss", time_scheme="backward"
        ),
        linear=fvm.LinearSolverConfig(
            linear_solver="spsolve",
            pressure_solver="spsolve",
            pressure_tolerance=1.0e-10,
            momentum_tolerance=1.0e-10,
        ),
        pimple=fvm.PimpleControl(
            n_outer_correctors=2,
            n_correctors=2,
            n_orthogonal_correctors=1,
            velocity_relaxation=0.7,
            pressure_relaxation=0.3,
        ),
        transport=fvm.TransportConfig(density=1.0, kinematic_viscosity=0.001),
        boundaries=[
            fvm.BoundaryConfig.inlet("xmin", [1.0, 0.0, 0.0]),
            fvm.BoundaryConfig.outlet("xmax"),
            fvm.BoundaryConfig.wall("ymin"),
            fvm.BoundaryConfig.wall("ymax"),
            fvm.BoundaryConfig.wall("zmin"),
            fvm.BoundaryConfig.wall("zmax"),
        ],
        initial_velocity=[1.0, 0.0, 0.0],
    )
    with contextlib.redirect_stdout(io.StringIO()):
        solver = FVMSolver(setup, str(tmp_path), mesh_data=structured_box(4, 4, 4))
        solver.auto_write = False
        try:
            for _ in range(20):
                solver.advance()
            end_time = solver.time + 0.010333964419951998 + 1.296270018258383e-5
            solver._time_config = replace(
                setup.time,
                end_time=end_time,
                adjustment=fvm.MaximumCourantTimeStep(
                    maximum=100.0, maximum_time_step_size=0.010333964419951998
                ),
            )
            for _ in range(2):
                solver.advance()
                diagnostic = solver.last_diagnostics
                # The old last-step clipping gives a pressure span of 2.08;
                # balanced steps keep it near the preceding physical span 0.033.
                assert diagnostic.max_kinematic_pressure - diagnostic.min_kinematic_pressure < 0.1
            assert solver.time == pytest.approx(end_time, abs=1.0e-12)
            assert solver.last_diagnostics.max_continuity_error < 1.0e-6
        finally:
            solver.close()
