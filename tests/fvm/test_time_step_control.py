"""Public construction and numerical behavior of FVM time-step control."""

from __future__ import annotations

import contextlib
from dataclasses import FrozenInstanceError, replace
import io
from types import SimpleNamespace

import pytest

from openonda import fvm
from openonda.fvm.assemble.time_integration import backward_coefficients
from openonda.fvm.core.solver import FVMSolver
from openonda.fvm.core.time_step import maximum_courant_time_step_size
from tests.support.fvm_mesh import structured_box


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


def test_time_step_control_is_immutable() -> None:
    control = fvm.MaximumCourantTimeStep(maximum=0.9)
    numerics = fvm.Numerics(time_step_size=0.01, time_step_control=control)

    with pytest.raises(FrozenInstanceError):
        control.maximum = 1.0
    with pytest.raises(FrozenInstanceError):
        numerics.time_step_control = None


def _configuration(time_step_size, end_time, control):
    return SimpleNamespace(
        numerics=fvm.Numerics(time_step_size=time_step_size, time_step_control=control),
        run=fvm.RunPlan(end_time=end_time),
    )


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
    solver._configuration = _configuration(
        0.01, 30.0, fvm.MaximumCourantTimeStep(maximum=0.9, maximum_time_step_size=0.04)
    )
    schedule = fvm.EveryTime(0.05)
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
    solver._configuration = _configuration(0.013, 0.3, fvm.MaximumCourantTimeStep(maximum=0.9))
    schedules = (fvm.EveryTime(0.05), fvm.EveryTime(0.03))
    solver._run_event_schedules = lambda: schedules
    solver._measure_maximum_courant_number = lambda dt: 50 * dt
    due_times = [[], []]
    previous_step = solver.time_step_size
    step = 0
    while solver.time < solver._configuration.run.end_time - 1.0e-12:
        dt = solver._select_time_step_size()
        assert dt <= 0.9 / 50 + 1.0e-14
        assert dt <= 1.2 * previous_step + 1.0e-14
        assert dt >= 0.5 * previous_step - 1.0e-14
        solver.time += dt
        step += 1
        for index, schedule in enumerate(schedules):
            if schedule.is_due(step, solver.time, dt):
                due_times[index].append(solver.time)
        previous_step = dt

    assert due_times[0] == pytest.approx([0.05 * k for k in range(1, 7)])
    assert due_times[1] == pytest.approx([0.03 * k for k in range(1, 11)])


def _channel_case(directory, numerics: fvm.Numerics, **fields) -> fvm.FVMCase:
    return fvm.FVMCase(
        name="channel",
        directory=directory,
        numerics=numerics,
        initial_conditions=fvm.InitialFields(velocity=fields.pop("velocity", (0.2, 0.0, 0.0))),
        run=fields.pop("run", fvm.RunPlan(end_time=0.1)),
        logging=fields.pop("logging", fvm.LoggingConfig(console=False)),
        mesh=fields.pop("mesh", structured_box(2, 2, 2)),
        boundaries=(
            fvm.BoundaryConfig.inlet("xmin", fields.pop("inlet", (0.5, 0.0, 0.0))),
            fvm.BoundaryConfig.outlet("xmax"),
            fvm.BoundaryConfig.wall("ymin"),
            fvm.BoundaryConfig.wall("ymax"),
            fvm.BoundaryConfig.wall("zmin"),
            fvm.BoundaryConfig.wall("zmax"),
        ),
        **fields,
    )


def test_adaptive_backward_run_lands_exactly_on_end_time(tmp_path) -> None:
    numerics = fvm.Numerics(
        time_step_size=0.08,
        time_step_control=fvm.MaximumCourantTimeStep(maximum=100.0, maximum_time_step_size=0.08),
        schemes=fvm.DiscretizationConfig(convection_scheme="upwind", time_scheme="backward"),
        linear=fvm.LinearSolverConfig(linear_solver="spsolve"),
        pimple=fvm.PimpleConfig(n_correctors=2),
        transport=fvm.TransportConfig(density=1.0, kinematic_viscosity=0.02),
    )
    with contextlib.redirect_stdout(io.StringIO()):
        solver = FVMSolver(_channel_case(tmp_path, numerics))
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
    numerics = fvm.Numerics(
        time_step_size=0.08,
        time_step_control=fvm.MaximumCourantTimeStep(maximum=100.0, maximum_time_step_size=0.08),
        schemes=fvm.DiscretizationConfig(convection_scheme="upwind"),
        linear=fvm.LinearSolverConfig(linear_solver="spsolve"),
        pimple=fvm.PimpleConfig(n_correctors=2),
        transport=fvm.TransportConfig(density=1.0, kinematic_viscosity=0.02),
    )
    case = _channel_case(
        tmp_path,
        numerics,
        logging=fvm.LoggingConfig(schedule=fvm.EveryTime(0.03), console=False),
        backup=fvm.Backup(schedule=fvm.EveryTime(0.03)),
    )
    with contextlib.redirect_stdout(io.StringIO()):
        solver = FVMSolver(case)
        solver.advance()

    assert solver.time == pytest.approx(0.03)
    assert solver._accepted_time_step_size == pytest.approx(0.03)
    assert (tmp_path / "solution" / "fvm" / "fvm_000001.npz").is_file()
    solver.close()


def test_output_deadline_does_not_generate_a_pressure_impulse(tmp_path) -> None:
    """A 64-cell 3D channel reproduces the pressure jump from a tiny final step."""
    numerics = fvm.Numerics(
        time_step_size=0.01,
        schemes=fvm.DiscretizationConfig(
            convection_scheme="linear_upwind", gradient_scheme="gauss", time_scheme="backward"
        ),
        linear=fvm.LinearSolverConfig(
            linear_solver="spsolve",
            pressure_solver="spsolve",
            pressure_tolerance=1.0e-10,
            momentum_tolerance=1.0e-10,
        ),
        pimple=fvm.PimpleConfig(
            n_outer_correctors=2,
            n_correctors=2,
            n_nonorthogonal_correctors=1,
            velocity_relaxation=0.7,
            pressure_relaxation=0.3,
        ),
        transport=fvm.TransportConfig(density=1.0, kinematic_viscosity=0.001),
    )
    case = _channel_case(
        tmp_path,
        numerics,
        velocity=(1.0, 0.0, 0.0),
        inlet=(1.0, 0.0, 0.0),
        run=fvm.RunPlan(end_time=1.0),
        mesh=structured_box(4, 4, 4),
    )
    with contextlib.redirect_stdout(io.StringIO()):
        solver = FVMSolver(case)
        solver.auto_write = False
        try:
            for _ in range(20):
                solver.advance()
            end_time = solver.time + 0.010333964419951998 + 1.296270018258383e-5
            solver._configuration = replace(
                solver._configuration,
                numerics=replace(
                    numerics,
                    time_step_control=fvm.MaximumCourantTimeStep(
                        maximum=100.0, maximum_time_step_size=0.010333964419951998
                    ),
                ),
                run=fvm.RunPlan(end_time=end_time),
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
