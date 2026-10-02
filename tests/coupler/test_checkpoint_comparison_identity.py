"""Comparison requires matching clocks and numerical configuration."""

from copy import deepcopy

import pytest

from tests.coupler.test_checkpoint_comparison_asset import comparison


def _state():
    config = {"vpm": {"time_step_size": 0.04, "particle_kernel": "GAUSSIAN"}}
    return {
        "manifest": {
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
    assert comparison.admit_comparison_identity(control, deepcopy(control)) is None


@pytest.mark.parametrize(
    "field", ["coupling_step", "fvm_step", "vpm_step", "n_fvm_substeps", "time"]
)
def test_different_clocks_are_rejected(field):
    control, candidate = _state(), _state()
    candidate["manifest"][field] += 1
    with pytest.raises(ValueError, match="Unmatched"):
        comparison.admit_comparison_identity(control, candidate)


def test_numerical_changes_are_rejected_even_with_valid_digests():
    control, candidate = _state(), _state()
    candidate["manifest"]["config"]["vpm"]["time_step_size"] = 0.08
    candidate["manifest"]["config_sha256"] = comparison.mapping_digest(
        candidate["manifest"]["config"]
    )
    with pytest.raises(ValueError, match="Unmatched.*config"):
        comparison.admit_comparison_identity(control, candidate)


def test_invalid_digest_is_rejected():
    control, candidate = _state(), _state()
    candidate["manifest"]["config_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="digest mismatch"):
        comparison.admit_comparison_identity(control, candidate)
