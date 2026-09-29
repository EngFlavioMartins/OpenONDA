"""NACA force history resolves every accepted FVM time step."""

import csv
from types import SimpleNamespace

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
