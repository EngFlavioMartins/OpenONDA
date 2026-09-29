"""Public numerical configuration exposes particle capacity, not buffer sizing."""

import inspect

import pytest

from source.solvers.vpm.config.case import Numerics
from source.solvers.vpm.config.viscous import ViscousConfig
from source.solvers.vpm.runtime.backend import initialize_taichi_backend


@pytest.mark.parametrize(
    "name",
    ["device_memory_fraction", "memory_buffer_bytes", "rss_limit_bytes", "soft_limit_bytes"],
)
def test_numerics_rejects_memory_sizing_controls(name):
    with pytest.raises(TypeError, match=name):
        Numerics(**{name: 1})


def test_backend_memory_sizing_is_private():
    parameters = inspect.signature(initialize_taichi_backend).parameters
    assert "device_memory_fraction" not in parameters
    assert "minimum_pool_bytes" not in parameters
    assert parameters["_minimum_pool_bytes"].kind is inspect.Parameter.KEYWORD_ONLY


@pytest.mark.parametrize("name", ["gbd_max_nodes", "dvh_max_nodes"])
def test_viscous_config_rejects_population_buffer_controls(name):
    with pytest.raises(TypeError, match=name):
        ViscousConfig(**{name: 64})


@pytest.mark.parametrize("factory", [ViscousConfig.gbd, ViscousConfig.dvh])
def test_diffusion_factories_reject_extra_population_caps(factory):
    with pytest.raises(TypeError, match="max_nodes"):
        factory(max_nodes=64)


@pytest.mark.parametrize(
    "name", ["late_interval_steps", "late_start_step", "late_absolute_only", "end_step"]
)
def test_filament_refinement_rejects_incident_schedules(name):
    from source.solvers.vpm.config.filament_refinement import FilamentRefinementConfig

    with pytest.raises(TypeError, match=name):
        FilamentRefinementConfig(**{name: 1})


@pytest.mark.parametrize(
    "name",
    [
        "regularization_max_events",
        "selective_eddy_viscosity_feedback_gain",
        "selective_eddy_viscosity_feedback_interval_steps",
        "selective_eddy_viscosity_feedback_growth_limit",
        "selective_eddy_viscosity_max_coefficient",
    ],
)
def test_stabilization_rejects_incident_feedback_controls(name):
    from source.solvers.vpm.config.stabilization import StabilizationConfig

    with pytest.raises(TypeError, match=name):
        StabilizationConfig(**{name: 1})
