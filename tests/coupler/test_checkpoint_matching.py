"""Comparison requires matching clocks and numerical configuration."""

from copy import deepcopy

import pytest

from tests.coupler.test_checkpoint_comparison_files import comparison


def _state():
    config = {"vpm": {"time_step_size": 0.04, "particle_kernel": "GAUSSIAN"}}
    return {
        "checkpoint_info": {
            "config": config,
            "config_sha256": comparison.mapping_digest(config),
            "coupling_step": 12,
            "fvm_step": 60,
            "vpm_step": 12,
            "n_fvm_substeps": 5,
            "time": 0.48,
        }
    }


def test_matching_checkpoints_are_comparable():
    control = _state()
    assert comparison.validate_matching_checkpoints(control, deepcopy(control)) is None


@pytest.mark.parametrize(
    "field", ["coupling_step", "fvm_step", "vpm_step", "n_fvm_substeps", "time"]
)
def test_different_clocks_are_rejected(field):
    control, candidate = _state(), _state()
    candidate["checkpoint_info"][field] += 1
    with pytest.raises(ValueError, match="Unmatched"):
        comparison.validate_matching_checkpoints(control, candidate)


def test_numerical_changes_are_rejected_even_with_valid_digests():
    control, candidate = _state(), _state()
    candidate["checkpoint_info"]["config"]["vpm"]["time_step_size"] = 0.08
    candidate["checkpoint_info"]["config_sha256"] = comparison.mapping_digest(
        candidate["checkpoint_info"]["config"]
    )
    with pytest.raises(ValueError, match="Unmatched.*config"):
        comparison.validate_matching_checkpoints(control, candidate)


def test_invalid_digest_is_rejected():
    control, candidate = _state(), _state()
    candidate["checkpoint_info"]["config_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="digest mismatch"):
        comparison.validate_matching_checkpoints(control, candidate)
