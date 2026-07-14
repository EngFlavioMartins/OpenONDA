"""Fast-fail validation of scheme / turbulence-model names in the config."""

import pytest

from source.solvers.FVM.config.types import SolverParams, TurbulenceConfig
from source.solvers.FVM.schemes import (
    validate_boundary_conditions,
    validate_solver_params,
    validate_turbulence,
)


def test_default_and_factory_params_are_valid():
    validate_solver_params(SolverParams())
    validate_solver_params(SolverParams.pimple())
    validate_solver_params(SolverParams.simple())


@pytest.mark.parametrize(
    "field,bad",
    [
        ("convection_scheme", "quik"),  # typo of QUICK
        ("convection_scheme", "LimitedLin"),  # not a real name
        ("time_scheme", "rk4"),  # unsupported
        ("gradient_scheme", "greenGauss"),  # wrong name
    ],
)
def test_invalid_scheme_names_rejected(field, bad):
    sp = SolverParams()
    setattr(sp, field, bad)
    with pytest.raises(ValueError, match="not recognised"):
        validate_solver_params(sp)


def test_known_scheme_aliases_accepted():
    for cs in ("upwind", "central", "deferred", "LUST", "limitedLinear", "vanLeer", "MUSCL"):
        sp = SolverParams()
        sp.convection_scheme = cs
        validate_solver_params(sp)
    for ts in ("euler_implicit", "backward", "bdf2"):
        sp = SolverParams()
        sp.time_scheme = ts
        validate_solver_params(sp)


def test_adaptive_bdf2_rejected_until_variable_step_weights_exist():
    from source.solvers.FVM import TimeConfig

    sp = SolverParams.pimple(time_scheme="backward")
    time = TimeConfig.transient(dt=0.1, duration=1.0)
    time.adjust_timestep = True
    with pytest.raises(ValueError, match="adaptive time stepping with BDF2"):
        validate_solver_params(sp, time)


@pytest.mark.parametrize(
    "field,bad",
    [("pressure_tol", 0.0), ("momentum_tol", -1.0), ("pressure_maxiter", 0)],
)
def test_invalid_linear_solver_controls_rejected(field, bad):
    sp = SolverParams()
    setattr(sp, field, bad)
    with pytest.raises(ValueError, match=field):
        validate_solver_params(sp)


def test_invalid_linear_failure_policy_rejected():
    sp = SolverParams(linear_failure_policy="ignore")
    with pytest.raises(ValueError, match="linear_failure_policy"):
        validate_solver_params(sp)


def test_unsupported_boundary_condition_rejected():
    boundaries = [
        {"name": "inlet", "bc_type_U": "fixedValue", "bc_type_p": "zeroGradient"},
        {"name": "outlet", "bc_type_U": "zeroGradient", "bc_type_p": "waveTransmissive"},
    ]
    with pytest.raises(ValueError, match="waveTransmissive"):
        validate_boundary_conditions(boundaries)


def test_turbulence_models():
    for cfg in (
        TurbulenceConfig.none(),
        TurbulenceConfig.smagorinsky(),
        TurbulenceConfig.wale(),
        TurbulenceConfig.sigma(),
        TurbulenceConfig.dynamic_smagorinsky(),
    ):
        validate_turbulence(cfg)
    with pytest.raises(ValueError, match="Unknown turbulence model"):
        validate_turbulence(TurbulenceConfig(model="kEpsilon"))
