"""Qualification of the freestream far field on a circular domain boundary."""

from __future__ import annotations

import numpy as np

from openonda import fvm
from tests.support.fvm_mesh import cylinder_o_grid

# Fixed cylinder at Re = 100 inside a circular far field carrying one
# freestream patch all the way round.
DIAMETER = 1.0  # m
FREESTREAM_VELOCITY = 1.0  # m/s
REYNOLDS_NUMBER = 100.0
FINAL_TIME = 90.0  # s
TIME_STEP_SIZE = 0.02  # s
SHEDDING_WINDOW = (70.0, 90.0)  # fully developed shedding [s]


def test_circular_freestream_far_field_keeps_periodic_vortex_shedding(tmp_path, record_property):
    """Wake vortices crossing the far field leave a bounded, periodic lift.

    The far-field circle sits 25 D from the cylinder, so the start-up vortices
    and the developing wake reach it well before the averaging window. The
    shed wake must stay a single periodic Karman street: zero mean lift, the
    Re = 100 lift amplitude and Strouhal number on this coarse grid, and no
    single-step force kicks.
    """
    mesh = cylinder_o_grid(DIAMETER, 50.0 * DIAMETER, 64, 48, 0.01 * DIAMETER)
    case = fvm.FVMCase(
        name="cylinder_freestream",
        directory=tmp_path,
        mesh=mesh,
        numerics=fvm.Numerics(
            time_step_size=TIME_STEP_SIZE,
            transport=fvm.TransportConfig(
                density=1.0,
                kinematic_viscosity=FREESTREAM_VELOCITY * DIAMETER / REYNOLDS_NUMBER,
            ),
            schemes=fvm.DiscretizationConfig(time_scheme="backward"),
            execution=fvm.ComputeConfig(operator_backend="numba"),
        ),
        boundaries=(
            fvm.BoundaryConfig.freestream("farfield", [FREESTREAM_VELOCITY, 0.0, 0.0]),
            fvm.BoundaryConfig.wall("cylinder"),
            fvm.BoundaryConfig.empty("front"),
            fvm.BoundaryConfig.empty("back"),
        ),
        # A small cross-flow component breaks the symmetry of the wake.
        initial_conditions=fvm.InitialFields(velocity=(FREESTREAM_VELOCITY, 0.05, 0.0)),
        run=fvm.RunPlan(
            end_time=FINAL_TIME,
            initial_output=False,
            final_output=False,
        ),
        samplers=fvm.Samplers(
            (
                fvm.ForceSampler(
                    ["cylinder"],
                    reference_velocity=FREESTREAM_VELOCITY,
                    reference_area=DIAMETER,
                    reference_length=DIAMETER,
                    schedule=fvm.EverySteps(1),
                ),
            )
        ),
    )
    with fvm.FVMSolver(case) as solver:
        solver.auto_write = False
        solver.run(start_from="initial")
    history = np.genfromtxt(
        tmp_path / "samples" / "forces_history.csv",
        delimiter=",",
        names=True,
        usecols=("time", "drag_coefficient", "lift_coefficient"),
    )
    time, drag, lift = history["time"], history["drag_coefficient"], history["lift_coefficient"]

    # No force kicks after the impulsive start: the second difference of a
    # resolved shedding signal is far below one hundredth.
    after_start = time[1:-1] >= 5.0
    largest_kick = float(np.max(np.abs(np.diff(lift, 2))[after_start]))

    window = (time >= SHEDDING_WINDOW[0]) & (time <= SHEDDING_WINDOW[1])
    t, cl = time[window], lift[window]
    rising = np.flatnonzero((cl[:-1] < 0.0) & (cl[1:] >= 0.0))
    crossings = t[rising] - cl[rising] * (t[rising + 1] - t[rising]) / (cl[rising + 1] - cl[rising])
    periods = np.diff(crossings)
    strouhal = DIAMETER / (FREESTREAM_VELOCITY * float(np.mean(periods)))
    period_spread = float(np.std(periods) / np.mean(periods))
    lift_amplitude = 0.5 * float(np.ptp(cl))
    mean_lift = float(np.mean(cl))
    mean_drag = float(np.mean(drag[window]))
    for name, value in (
        ("largest_lift_kick", largest_kick),
        ("strouhal_number", strouhal),
        ("period_spread", period_spread),
        ("lift_amplitude", lift_amplitude),
        ("mean_lift", mean_lift),
        ("mean_drag", mean_drag),
    ):
        record_property(name, value)

    assert largest_kick < 0.02
    assert len(periods) >= 2
    assert period_spread < 0.01
    assert 0.14 < strouhal < 0.18
    assert 0.25 < lift_amplitude < 0.38
    assert abs(mean_lift) < 0.03
    assert 1.2 < mean_drag < 1.45
