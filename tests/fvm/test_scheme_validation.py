"""Fast-fail validation of scheme / turbulence-model names in the config."""

from types import SimpleNamespace

import numpy as np
import pytest

from source.solvers.FVM.config.types import (
    LinearSolverConfig,
    PimpleControl,
    SchemesConfig,
    TurbulenceConfig,
)
from source.solvers.FVM.schemes import (
    validate_boundary_conditions,
    validate_solver_params,
    validate_turbulence,
)
from source.solvers.FVM.schemes.boundaries import BOUNDARIES, BoundaryStrategy
from source.solvers.FVM.solve.simple_solver import update_scalar_boundaries


def _params(**overrides):
    """Merged solver-parameter namespace, as the solver feeds
    ``validate_solver_params`` (union of the grouped configs' fields)."""
    flat: dict = {}
    for group in (SchemesConfig(), LinearSolverConfig(), PimpleControl()):
        flat.update(vars(group))
    flat.update(overrides)
    return SimpleNamespace(**flat)


def _params_from(groups):
    flat: dict = {}
    for group in (*groups,):
        flat.update(vars(group))
    return SimpleNamespace(**flat)


def test_default_and_factory_params_are_valid():
    validate_solver_params(_params())
    validate_solver_params(_params_from((SchemesConfig(), LinearSolverConfig(), PimpleControl())))
    validate_solver_params(
        _params_from((SchemesConfig(), LinearSolverConfig(), PimpleControl(algorithm="SIMPLE")))
    )


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
    with pytest.raises(ValueError, match="not recognised"):
        validate_solver_params(_params(**{field: bad}))


def test_known_scheme_aliases_accepted():
    for cs in ("upwind", "central", "deferred", "LUST", "limitedLinear", "vanLeer", "MUSCL"):
        validate_solver_params(_params(convection_scheme=cs))
    for ts in ("euler_implicit", "backward", "bdf2"):
        validate_solver_params(_params(time_scheme=ts))


def test_adaptive_bdf2_rejected_until_variable_step_weights_exist():
    from source.solvers.FVM import TimeConfig

    params = _params_from(
        (SchemesConfig(time_scheme="backward"), LinearSolverConfig(), PimpleControl())
    )
    time = TimeConfig.transient(dt=0.1, duration=1.0)
    time.adjust_timestep = True
    with pytest.raises(ValueError, match="adaptive time stepping with BDF2"):
        validate_solver_params(params, time)


@pytest.mark.parametrize(
    "field,bad",
    [
        ("pressure_tol", 0.0),
        ("momentum_tol", -1.0),
        ("pressure_rel_tol", -0.1),
        ("momentum_final_rel_tol", 1.1),
        ("pressure_maxiter", 0),
    ],
)
def test_invalid_linear_solver_controls_rejected(field, bad):
    with pytest.raises(ValueError, match=field):
        validate_solver_params(_params(**{field: bad}))


def test_invalid_linear_failure_policy_rejected():
    with pytest.raises(ValueError, match="linear_failure_policy"):
        validate_solver_params(_params(linear_failure_policy="ignore"))


def test_invalid_pressure_nullspace_policy_rejected():
    with pytest.raises(ValueError, match="pressure_nullspace_policy"):
        validate_solver_params(_params(pressure_nullspace_policy="pin_or_guess"))


def test_unsupported_boundary_condition_rejected():
    boundaries = [
        {"name": "inlet", "bc_type_U": "fixedValue", "bc_type_p": "zeroGradient"},
        {"name": "outlet", "bc_type_U": "zeroGradient", "bc_type_p": "waveTransmissive"},
    ]
    with pytest.raises(ValueError, match="waveTransmissive"):
        validate_boundary_conditions(boundaries)


def test_registry_resolves_every_claimed_operator_to_a_strategy():
    operators = ("gradient", "convection", "diffusion", "pressure", "flux", "ghost", "diagnostics")
    for field in ("U", "p"):
        for name in BOUNDARIES.names_for(field):
            for operator in operators:
                assert isinstance(BOUNDARIES.strategy(name, field, operator), BoundaryStrategy)


def test_direct_pressure_update_cannot_fall_back_for_unknown_bc(hand_built_3d_mesh):
    mesh = hand_built_3d_mesh
    boundaries = [dict(patch, bc_type_p="typoGradient") for patch in mesh["boundary"]]
    n_total = mesh["n_elements"] + mesh["n_faces"] - mesh["n_interior_faces"]
    with pytest.raises(ValueError, match="typoGradient.*ghost for p"):
        update_scalar_boundaries(np.zeros(n_total), mesh, boundaries, field_name="p")


def test_turbulence_models():
    for cfg in (
        TurbulenceConfig.none(),
        TurbulenceConfig.smagorinsky(),
        TurbulenceConfig.openfoam_smagorinsky(),
        TurbulenceConfig.wale(),
        TurbulenceConfig.sigma(),
        TurbulenceConfig.dynamic_smagorinsky(),
    ):
        validate_turbulence(cfg)
    with pytest.raises(ValueError, match="Unknown turbulence model"):
        validate_turbulence(TurbulenceConfig(model="kEpsilon"))
    with pytest.raises(ValueError, match="Ck"):
        validate_turbulence(TurbulenceConfig.openfoam_smagorinsky(Ck=-0.1))
    with pytest.raises(ValueError, match="Ce"):
        validate_turbulence(TurbulenceConfig.openfoam_smagorinsky(Ce=0.0))
