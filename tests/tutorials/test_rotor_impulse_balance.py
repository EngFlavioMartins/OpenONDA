"""Native-clock rotor momentum accounting must not hide numerical sources."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from openonda.tutorial_runner import load_case_module

rotor = load_case_module(Path(__file__).parents[2] / "tutorials/vpm/06_rotor_flow_PENDING", "assets._common")


def histories():
    clock = np.arange(1, 9) * 0.1
    force = pd.DataFrame({"time": clock})
    flow = pd.DataFrame({"time": clock[::2]})
    for axis in "xyz":
        # KJ=2t; pressure is an interval load d(t^2)/dt with a backward difference.
        force[f"unsteady_force_{axis}"] = 2 * clock - 0.1 if axis == "x" else 0.0
        force[f"force_{axis}"] = 4 * clock - 0.1 if axis == "x" else 0.0
        # rho=2: -rho I = 2t^2 plus a separate, deliberately unphysical source 3t.
        flow[f"coupled_linear_impulse_{axis}"] = (
            -(2 * flow.time**2 + 3 * flow.time) / 2 if axis == "x" else 0.0
        )
        flow[f"pedrizzetti_cumulative_linear_impulse_transfer_{axis}"] = (
            -3 * flow.time / 2 if axis == "x" else 0.0
        )
    return force, flow


def evaluate(force, flow):
    return rotor.impulse_history(
        force,
        flow,
        density=2.0,
        time_step_size=0.1,
        flow_interval_steps=2,
        start_time=0.25,
        end_time=0.75,
    )


def test_pressure_uses_accepted_intervals_and_raw_balance_retains_relaxation():
    force, flow = histories()
    time, load, fluid, relaxation = evaluate(force, flow)
    expected = 2 * (time**2 - time[0] ** 2)
    np.testing.assert_allclose(load[:, 0], expected, atol=1e-14)
    np.testing.assert_allclose(fluid[:, 0], expected + 3 * (time - time[0]), atol=1e-14)
    np.testing.assert_allclose(relaxation[:, 0], 3 * (time - time[0]), atol=1e-14)
    assert fluid[-1, 0] > 2 * load[-1, 0]


def test_manufactured_ledger_applies_density_once_to_per_density_impulse():
    """Native impulse [m^4/s] must be density-scaled before comparing [N s]."""
    clock = np.array([0.1, 0.2, 0.3])
    force = pd.DataFrame({"time": clock})
    flow = pd.DataFrame({"time": clock})
    for axis in "xyz":
        force[f"unsteady_force_{axis}"] = 0.0
        force[f"force_{axis}"] = 4.0 if axis == "x" else 0.0
        # rho=2 and F=4 N imply I_per_density = -2*(t-t0) m^4/s.
        flow[f"coupled_linear_impulse_{axis}"] = -2.0 * (clock - clock[0]) if axis == "x" else 0.0
        flow[f"pedrizzetti_cumulative_linear_impulse_transfer_{axis}"] = 0.0

    time, load, fluid, relaxation = rotor.impulse_history(
        force,
        flow,
        density=2.0,
        time_step_size=0.1,
        flow_interval_steps=1,
        start_time=0.1,
        end_time=0.3,
    )
    np.testing.assert_allclose(time, clock)
    np.testing.assert_allclose(load[:, 0], [0.0, 0.4, 0.8])
    np.testing.assert_allclose(fluid[:, 0], load[:, 0])
    np.testing.assert_allclose(relaxation, 0.0)


def test_manufactured_ledger_separates_kj_trapezoid_and_backward_pressure():
    """The accepted-clock ledger integrates KJ trapezoidally and pressure by interval."""
    clock = np.array([0.1, 0.2, 0.3])
    dt = 0.1
    kj = clock**2
    potential = 5.0 * clock**3
    pressure = np.array(
        [0.0, (potential[1] - potential[0]) / dt, (potential[2] - potential[1]) / dt]
    )
    force = pd.DataFrame({"time": clock})
    flow = pd.DataFrame({"time": clock})
    for axis in "xyz":
        force[f"unsteady_force_{axis}"] = pressure if axis == "x" else 0.0
        force[f"force_{axis}"] = (kj + pressure) if axis == "x" else 0.0
        flow[f"coupled_linear_impulse_{axis}"] = 0.0
        flow[f"pedrizzetti_cumulative_linear_impulse_transfer_{axis}"] = 0.0
    # Supply the exact cumulative fluid impulse for the two intervals.
    kj_integral = np.array(
        [0.0, 0.5 * (kj[0] + kj[1]) * dt, 0.5 * (kj[0] + kj[1]) * dt + 0.5 * (kj[1] + kj[2]) * dt]
    )
    pressure_integral = np.array([0.0, pressure[1] * dt, pressure[1] * dt + pressure[2] * dt])
    flow["coupled_linear_impulse_x"] = -(kj_integral + pressure_integral)

    time, load, fluid, _ = rotor.impulse_history(
        force,
        flow,
        density=1.0,
        time_step_size=dt,
        flow_interval_steps=1,
        start_time=0.1,
        end_time=0.3,
    )
    expected_load = kj_integral + pressure_integral
    np.testing.assert_allclose(time, clock)
    np.testing.assert_allclose(load[:, 0], expected_load)
    np.testing.assert_allclose(fluid[:, 0], expected_load)


@pytest.mark.parametrize("defect", ["sparse", "missing", "nonfinite", "duplicate", "off_clock"])
def test_incomplete_or_incompatible_native_data_cannot_qualify(defect):
    force, flow = histories()
    if defect == "sparse":
        force = force.iloc[::2]
    elif defect == "missing":
        flow = flow.drop(columns="coupled_linear_impulse_x")
    elif defect == "nonfinite":
        flow.loc[1, "pedrizzetti_cumulative_linear_impulse_transfer_x"] = np.nan
    elif defect == "duplicate":
        force.loc[1, "time"] = force.loc[0, "time"]
    else:
        flow.time += 0.002
    with pytest.raises(ValueError):
        evaluate(force, flow)


@pytest.mark.parametrize("defect", ["gap", "truncated"])
def test_flow_window_must_cover_the_configured_interval(defect):
    force, flow = histories()
    flow = flow.drop(index=1) if defect == "gap" else flow.iloc[:2]
    with pytest.raises(ValueError):
        evaluate(force, flow)
