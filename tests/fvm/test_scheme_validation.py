"""Fast-fail validation of scheme / turbulence-model names in the config."""

import pytest

from source.solvers.FVM.config.types import SolverParams, TurbulenceConfig
from source.solvers.FVM.schemes import validate_solver_params, validate_turbulence


def test_default_and_factory_params_are_valid():
    validate_solver_params(SolverParams())
    validate_solver_params(SolverParams.pimple())
    validate_solver_params(SolverParams.simple())


@pytest.mark.parametrize(
    "field,bad",
    [
        ("convection_scheme", "quik"),       # typo of QUICK
        ("convection_scheme", "LimitedLin"),  # not a real name
        ("time_scheme", "rk4"),               # unsupported
        ("gradient_scheme", "greenGauss"),    # wrong name
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
