"""Explicit restart permissions use exact paths, never subtree grants."""

from copy import deepcopy

import pytest

from source.solvers.vpm.config.restart_changes import (
    MISSING_CONFIGURATION_VALUE as MISSING,
)
from source.solvers.vpm.config.restart_changes import (
    admit_configuration_changes,
)


def test_default_strict_and_exact_scalar_permission():
    old = {"induction": {"method": "old", "theta": .1}}
    new = {"induction": {"method": "new", "theta": .1}}
    with pytest.raises(ValueError, match="mismatch at induction.method"):
        admit_configuration_changes(new, old)
    original = deepcopy((old, new))
    changes = admit_configuration_changes(new, old, allowed_config_differences={"induction.method"})
    assert changes == ({"path": "induction.method", "stored": {"present": True, "value": "old"},
                        "current": {"present": True, "value": "new"}},)
    assert (old, new) == original


@pytest.mark.parametrize("path", ["induction", "induction.*", "induction.method.*", "unknown", "induction.theta"])
def test_parent_wildcard_unknown_and_unchanged_permissions_reject(path):
    with pytest.raises(ValueError):
        admit_configuration_changes({"induction": {"method": "new", "theta": .1}},
                                    {"induction": {"method": "old", "theta": .1}},
                                    allowed_config_differences={path})


@pytest.mark.parametrize("old", [None, MISSING])
def test_new_policy_requires_exact_whole_value_and_cannot_hide_extra_controls(old):
    source = {"induction": {}} if old is MISSING else {"induction": {"policy": old}}
    policy = {"order": 10, "spacing_ratio": .25}
    target = {"induction": {"policy": policy}}
    path = "induction.policy"
    with pytest.raises(ValueError, match="requires exact stored/current"):
        admit_configuration_changes(target, source, allowed_config_differences={path})
    expectations = {path: (old, deepcopy(policy))}
    evidence = admit_configuration_changes(target, source, allowed_config_differences={path},
                                          expected_config_differences=expectations)
    policy["tail_tolerance"] = .1
    with pytest.raises(ValueError, match="expectation mismatch"):
        admit_configuration_changes(target, source, allowed_config_differences={path},
                                    expected_config_differences=expectations)
    assert "tail_tolerance" not in evidence[0]["current"]["value"]


def test_missing_is_not_null_and_removal_needs_expectation():
    path = "induction.policy"
    source, target = {"induction": {"policy": None}}, {"induction": {}}
    with pytest.raises(ValueError, match="expectation mismatch"):
        admit_configuration_changes(target, source, allowed_config_differences={path},
                                    expected_config_differences={path: (None, None)})
    result = admit_configuration_changes(target, source, allowed_config_differences={path},
                                         expected_config_differences={path: (None, MISSING)})
    assert result[0]["current"] == {"present": False}


def test_list_scalar_and_structural_transitions_are_distinct():
    old, new = {"a": [1, 2]}, {"a": [1, 3]}
    assert admit_configuration_changes(new, old, allowed_config_differences={"a[1]"})
    with pytest.raises(ValueError, match="requires exact"):
        admit_configuration_changes({"a": [1, 2, 3]}, old, allowed_config_differences={"a"})
    assert admit_configuration_changes({"a": [1, 2, 3]}, old, allowed_config_differences={"a"},
                                       expected_config_differences={"a": ([1, 2], [1, 2, 3])})


def test_time_step_stays_on_its_separate_explicit_route():
    old, new = {"time_step_size": .1}, {"time_step_size": .05}
    with pytest.raises(ValueError, match="explicit time_step_size"):
        admit_configuration_changes(new, old, allowed_config_differences={"time_step_size"})
    assert admit_configuration_changes(new, old, allow_time_step_size_mismatch=True) == ()


@pytest.mark.parametrize("paths,expectations", [
    ("a", None), (["a", "a"], None), ([None], None),
    (["a"], {"b": (1, 2)}), (["a"], {"a": (1,)}),
    (["a"], {"a": (float("nan"), 2)}), (["a"], {"a": (True, 2)}),
])
def test_malformed_or_inexact_expectations_reject(paths, expectations):
    with pytest.raises((TypeError, ValueError)):
        admit_configuration_changes({"a": 2}, {"a": 1}, allowed_config_differences=paths,
                                    expected_config_differences=expectations)


def test_capacity_operational_compatibility_is_unchanged_but_not_a_permission():
    assert admit_configuration_changes({"max_n_particles": 128}, {"max_n_particles": 64}) == ()
    with pytest.raises(ValueError, match="not an exact changed path"):
        admit_configuration_changes({"max_n_particles": 128}, {"max_n_particles": 64},
                                    allowed_config_differences={"max_n_particles"})
