"""NACA force history resolves every accepted FVM time step."""

import csv
from types import SimpleNamespace

import numpy as np
import pytest

from openonda.fvm import IBMForceSampler
from tests._tutorial_helpers import load_tutorial_module


def test_naca_force_history_keeps_step_cadence_without_dense_fields(tmp_path) -> None:
    case = load_tutorial_module("coupled_fvm_vpm/naca4412_flow")
    (force,) = (sample for sample in case.FVM_SAMPLERS if isinstance(sample, IBMForceSampler))
    assert all(
        force.is_due(step, step * case.FVM_TIME_STEP_SIZE, case.FVM_TIME_STEP_SIZE)
        for step in range(1, round(case.END_TIME / case.FVM_TIME_STEP_SIZE) + 1)
    )
    assert all(
        not sample.is_due(1, case.FVM_TIME_STEP_SIZE, case.FVM_TIME_STEP_SIZE)
        for sample in case.FVM_SAMPLERS
        if sample is not force
    )

    data = {"forces": {"airfoil": (0.1, 0.2, 0.0)}, "slip_error": 0.01}
    for step in (1, 2):
        context = SimpleNamespace(
            setup=case.FVM_SETUP,
            time=step * case.FVM_TIME_STEP_SIZE,
            step=step,
            _accepted_time_step_size=case.FVM_TIME_STEP_SIZE,
        )
        force.write_csv(context, str(tmp_path), data)
    with (tmp_path / "ibm_forces_history.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [int(row["step"]) for row in rows] == [1, 2]


def test_naca_transfer_preserves_finite_body_and_les_physics():
    case = load_tutorial_module("coupled_fvm_vpm/naca4412_flow")
    flow, particles, exchange = case.FVM_SETUP, case.VPM_CASE.numerics, case.COUPLER_SETUP
    assert case.NACA_CODE == "4412" and case.ALPHA_DEG == 10.0
    assert case.CHORD == 1.0 and case.SPAN == 5.0 and case.REYNOLDS == 1000.0
    assert flow.transport.kinematic_viscosity == particles.viscous.kinematic_viscosity
    assert particles.viscous.kinematic_viscosity == pytest.approx(0.001)
    assert flow.time.time_step_size == 0.01 and particles.time_step_size == 0.04
    assert flow.time.end_time == 12.0 and case.VPM_CASE.run.steps == 300
    np.testing.assert_allclose(flow.initial_velocity, particles.freestream_velocity)
    assert particles.turbulence.smagorinsky_coefficient == flow.turbulence.smagorinsky_coefficient
    assert particles.viscous.scheme == "GBD"
    assert particles.viscous.particle_spacing == pytest.approx(0.04)
    assert particles.viscous.gbd_grid_spacing == pytest.approx(0.04)
    assert particles.viscous.gbd_threshold_mode == "absolute"
    assert particles.viscous.gbd_threshold == pytest.approx(0.01 * 0.04**3)
    exchange.validate_transfer_region_box(case.FVM_BOX)
    box = np.asarray(exchange.transfer_region_bounds)
    core_lower, core_upper = (
        box[::2] + exchange.eta_blend_width,
        box[1::2] - exchange.eta_blend_width,
    )
    body_lower = np.r_[case.AIRFOIL_VERTICES.min(axis=0), -case.SPAN / 2]
    body_upper = np.r_[case.AIRFOIL_VERTICES.max(axis=0), case.SPAN / 2]
    assert np.all(body_lower > core_lower) and np.all(body_upper < core_upper)
    assert exchange.eta_blend_width == pytest.approx(0.24)
    assert exchange.vpm_only_width == pytest.approx(0.08)
