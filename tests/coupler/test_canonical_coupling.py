"""The public coupling API admits one algorithm and retains accuracy controls."""

import pytest

from source.coupler import CouplerSetup


@pytest.mark.parametrize(
    "controls",
    [
        {"interface_iterations": 0},
        {"interface_normal_tolerance": 0},
        {"interface_gradient_tolerance": float("nan")},
        {"eta_blend_width": 0.1, "vpm_only_width": 0.1},
        {"transfer_amplification_cap": 0.9},
        {"transfer_discretization_error_limit": 0},
    ],
)
def test_accuracy_and_resolution_controls_remain_validated(controls):
    with pytest.raises(ValueError):
        CouplerSetup(**controls)
