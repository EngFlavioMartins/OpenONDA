"""Physical sampling cadence does not depend on a destination's divisors."""

from pathlib import Path

import numpy as np
import pytest

from openonda.tutorial_runner import load_case_module
from tests.coupler.test_cylinder_support_execution import CASE


def _sampling(exchange_dt=0.04):
    return load_case_module(CASE).sampling_plan(
        0.2,
        exchange_dt=exchange_dt,
        fvm_time_step=0.01,
    )


@pytest.mark.parametrize("end_time", [100.0, 100.04])
def test_2500_and_prime_2501_exchanges_keep_same_physical_cadence(end_time):
    schedule = _sampling()
    assert schedule.sample_steps == 5
    assert schedule.fvm_sample_steps == 20
    module = load_case_module(CASE)
    flow, particles, _, _ = module.build_case(end_time=end_time)
    assert flow.samplers[0].schedule.every_n_steps * flow.time.time_step_size == pytest.approx(0.04)
    assert particles.samplers.samples[0].schedule.interval == 1


def test_off_cadence_horizon_covers_registered_statistics_without_extrapolation(tmp_path):
    """Synthetic periodic loading tests consumer coverage, not CFD accuracy."""
    horizon = 100.04
    schedule = _sampling()
    accepted_steps = round(horizon / 0.04)
    steps = np.arange(schedule.sample_steps, accepted_steps + 1, schedule.sample_steps)
    time = steps * 0.04
    assert time[-1] == 100
    assert time[-1] < horizon
    history = tmp_path / "synthetic_forces.csv"
    np.savetxt(
        history,
        np.column_stack(
            (time, np.full_like(time, 1.3), np.sin(2 * np.pi * 0.2 * time), np.zeros_like(time))
        ),
        delimiter=",",
        header="time,drag_coefficient,lift_coefficient,side_force_coefficient",
        comments="",
    )
    post = load_case_module(
        Path(__file__).resolve().parents[2] / "tests/support/cylinder", "postprocess_grid_study"
    )
    result = post.force_statistics(history, 40, 100)
    assert result["mean_drag"] == pytest.approx(1.3)
    assert result["strouhal"] == pytest.approx(0.2, abs=0.001)
    with pytest.raises(ValueError, match="does not cover"):
        post.force_statistics(history, 40, horizon)
