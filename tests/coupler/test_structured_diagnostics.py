"""Structured backend evidence must survive accepted-step reporting intact."""

import json
import logging
from types import SimpleNamespace

import numpy as np
import pytest

from source.coupler import reporting


def _coupler(tail, directory=None):
    backups = []
    owner = SimpleNamespace(
        _last_transfer_result=None,
        _last_vpm_boundary_condition_flux_diagnostics=dict.fromkeys(
            (
                "raw_mismatch",
                "raw_relative",
                "acceptance_limit",
                "applied_correction",
                "corrected_mismatch",
            ),
            0.0,
        ),
        vorticity_transfer=None,
        vpm_solver=SimpleNamespace(
            physics=SimpleNamespace(induction=SimpleNamespace(last_tail=tail))
        ),
        n_fvm_substeps=5,
        _is_master=True,
        _log_stop_step=276,
        _step_transfer_stats={},
        fvm_solver=SimpleNamespace(step=1380, time=11.04, max_courant_number=0.59),
        vpm_time_step_size=0.04,
        setup=SimpleNamespace(backup_interval_steps=1),
        solution_dir=directory,
        coupling_diagnostics=[],
        save_backup=lambda path, **kwargs: backups.append((path, kwargs)),
        _last_interface_iteration_diagnostics={
            "sweeps": np.int64(3),
            "accepted_sweep": 3,
            "converged": np.bool_(True),
            "residuals": [
                {
                    "normal_residual_rms": np.float64(6.562e-6),
                    "gradient_residual_rms": 7.499e-6,
                    "acceleration_alpha": None,
                }
            ],
        },
    )
    return owner, backups


def _nested_tail():
    return {
        "shell": np.int64(128),
        "relative": np.float64(2.5e-5),
        "seconds": 0.125,
        "declined_blocks": [
            {
                "shell": 16,
                "target_start": 8192,
                "pair_capacity": np.int32(524288),
                "passes_seconds": {"traversal": np.float32(0.125)},
                "remaining": np.array([1, 2], dtype=np.int64),
                "optional": None,
                "rebuilt_targets": np.bool_(True),
                "reason": "bounded scratch",
                "empty": [],
            }
        ],
    }


def test_nested_decline_evidence_is_copied_without_loss():
    tail = _nested_tail()
    coupler, _ = _coupler(tail)
    record = reporting.compute_diagnostics(coupler, None)["last_induction_image_call"]
    encoded = json.loads(json.dumps(record, allow_nan=False))
    assert encoded == {
        "shell": 128,
        "relative": 2.5e-5,
        "seconds": 0.125,
        "declined_blocks": [
            {
                "shell": 16,
                "target_start": 8192,
                "pair_capacity": 524288,
                "passes_seconds": {"traversal": 0.125},
                "remaining": [1, 2],
                "optional": None,
                "rebuilt_targets": True,
                "reason": "bounded scratch",
                "empty": [],
            }
        ],
    }
    tail["declined_blocks"][0]["remaining"][0] = 99
    tail["declined_blocks"].clear()
    assert record["declined_blocks"][0]["remaining"] == [1, 2]


@pytest.mark.parametrize(
    "tail", [{"shell": 16, "relative": 2.5e-5}, {"shell": 128, "declined_blocks": []}]
)
def test_scalar_records_and_empty_decline_lists_remain_supported(tail):
    coupler, _ = _coupler(tail)
    assert reporting.compute_diagnostics(coupler, None)["last_induction_image_call"] == tail


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), np.float32(-np.inf)])
def test_nonfinite_nested_diagnostic_has_precise_error_path(bad):
    tail = _nested_tail()
    tail["declined_blocks"][0]["passes_seconds"]["traversal"] = bad
    coupler, _ = _coupler(tail)
    with pytest.raises(FloatingPointError, match=r"slip-slab.*declined_blocks.*\[0\].*traversal"):
        reporting.compute_diagnostics(coupler, None)


@pytest.mark.parametrize("value", [complex(1, 2), object(), {1: "would collide", "1": "distinct"}])
def test_unsupported_values_are_not_silently_stringified(value):
    with pytest.raises(TypeError):
        reporting._diagnostic_json_value(value)


def test_cycles_rejected_but_shared_noncyclic_values_are_preserved():
    child = [np.float64(1)]
    assert reporting._diagnostic_json_value([child, child]) == [[1.0], [1.0]]
    child.append(child)
    with pytest.raises(ValueError, match="cyclic"):
        reporting._diagnostic_json_value(child)


@pytest.mark.parametrize("tail", [_nested_tail(), {"shell": 128, "declined_blocks": []}])
def test_complete_record_step_with_structured_tail_reaches_backup_and_jsonl(tmp_path, tail):
    coupler, backups = _coupler(tail, tmp_path)
    reporting.record_step(
        coupler,
        276,
        11.04,
        (0.02, 0.01, 0.03, 0.04),
        None,
        logger=logging.getLogger("test.structured_reporting"),
        state_checks_and_sampling_seconds=0.01,
    )
    rows = (tmp_path / "coupler_diagnostics.jsonl").read_text().splitlines()
    assert len(rows) == len(coupler.coupling_diagnostics) == len(backups) == 1
    row = json.loads(rows[0])
    assert row == coupler.coupling_diagnostics[0]
    assert row["step"] == 276 and row["time"] == 11.04
    assert row["last_induction_image_call"] == reporting._diagnostic_json_value(tail)
    assert row["interface_iteration"]["converged"] is True
    assert row["backup_phase"]["status"] == "complete"
    assert backups[0][1] == {"coupling_step": 276}


def test_nonfinite_nested_interface_evidence_is_rejected_before_backup(tmp_path):
    coupler, backups = _coupler({"declined_blocks": []}, tmp_path)
    coupler._last_interface_iteration_diagnostics["residuals"][0]["normal_residual_rms"] = np.nan
    with pytest.raises(FloatingPointError, match="normal_residual_rms"):
        reporting.record_step(
            coupler,
            276,
            11.04,
            (0.02, 0.01, 0.03, 0.04),
            None,
            logger=logging.getLogger("test.structured_reporting"),
        )
    assert not backups and not coupler.coupling_diagnostics
    assert not (tmp_path / "coupler_diagnostics.jsonl").exists()
