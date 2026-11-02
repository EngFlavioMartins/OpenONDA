"""The selective eddy viscosity follows the VPM turbulence model unless set explicitly."""

from __future__ import annotations

import pytest

from openonda import vpm
from openonda.vpm.config.stabilization import AUTOMATIC_SELECTIVE_EDDY_VISCOSITY

BOUNDS = (0.0, 1.0, 0.0, 1.0, 0.0, 1.0)


def _coefficient(**numerics) -> float:
    return vpm.Numerics(**numerics).stabilization.selective_eddy_viscosity_coefficient


@pytest.mark.parametrize(
    "turbulence",
    [vpm.TurbulenceConfig.smagorinsky(), vpm.TurbulenceConfig.equilibrium_smagorinsky()],
)
def test_les_wakes_get_the_selective_eddy_viscosity_by_default(turbulence) -> None:
    assert _coefficient(turbulence=turbulence) == AUTOMATIC_SELECTIVE_EDDY_VISCOSITY
    stabilization = vpm.StabilizationConfig.bounded_domain(BOUNDS)
    assert (
        _coefficient(turbulence=turbulence, stabilization=stabilization)
        == AUTOMATIC_SELECTIVE_EDDY_VISCOSITY
    )


@pytest.mark.parametrize(
    "turbulence", [vpm.TurbulenceConfig.dns(), vpm.TurbulenceConfig.inviscid()]
)
def test_non_les_models_keep_no_selective_eddy_viscosity(turbulence) -> None:
    stabilization = vpm.StabilizationConfig.bounded_domain(BOUNDS)
    assert _coefficient(turbulence=turbulence) == 0.0
    assert _coefficient(turbulence=turbulence, stabilization=stabilization) == 0.0


def test_explicit_and_disabled_settings_are_respected() -> None:
    les = vpm.TurbulenceConfig.smagorinsky()
    assert _coefficient(turbulence=les, stabilization=vpm.StabilizationConfig.disabled()) == 0.0
    explicit = vpm.StabilizationConfig(selective_eddy_viscosity_coefficient=0.2)
    assert _coefficient(turbulence=les, stabilization=explicit) == 0.2
    with pytest.raises(ValueError):
        vpm.StabilizationConfig(selective_eddy_viscosity_coefficient=-1.0)
