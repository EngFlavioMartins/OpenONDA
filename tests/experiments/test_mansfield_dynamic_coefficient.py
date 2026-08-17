"""Regression tests for the Mansfield dynamic-coefficient procedure.

Context
-------
Until 2026-08-17, `stage_8a_particle_functional_gate.mansfield_dynamic_coefficient`
built the Germano/Leonard term as

    ell = convection - convection_test - stretching + stretching_test

which (a) omitted the test filter T on the two base terms and (b) carried the
opposite overall sign to the project's own SGS convention in
`exact_sgs_for_filter`.  The estimator therefore returned C_r^2 = -0.00667 on
the audited AGARD operating point.  Because the Mansfield eddy diffusivity must
be non-negative, the clip set C_r = 0, the closure supplied no SGS transfer, and
*every* predeclared check in Stage 8A failed as a downstream consequence.

The decisive invariant tested here: the Leonard term is not a free choice.  It
is identically the exact SGS source of the test filter acting on the resolved
field, so it must equal `exact_sgs_for_filter(grid, u_resolved, T)["g"]` to
machine precision.  That reference path is independent of the dynamic procedure
and was already validated in Stage 7A, where the two-filter decomposition closed
to 3.1e-15.

These tests are deliberately written against the research scripts under
`scripts/experiments/`, not the production solver.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "experiments"))

from stage_4a_formulation import SpectralGrid, nonlinear_parts  # noqa: E402
from stage_6a_composite_filter_gate import PHASES  # noqa: E402
from stage_7a_composite_sgs_audit import (  # noqa: E402
    apply_symbol,
    exact_sgs_for_filter,
    particle_symbol,
)
from stage_8a_particle_functional_gate import (  # noqa: E402
    divergence_relative,
    gaussian_energy_width_ratio,
    leonard_term,
    mansfield_dynamic_coefficient,
    mansfield_torque,
    openonda_current_torque,
    strain_magnitude,
)

AGARD = ROOT / "docs/dns/agard_hom02/CB128_9.bin"
SIGMA_OVER_H = 2.5
LES_N = 32


# ── helpers ──────────────────────────────────────────────────────────────────


def synthetic_field(n: int = 32, seed: int = 20260817) -> tuple[SpectralGrid, np.ndarray]:
    """Smooth, solenoidal, zero-mean periodic velocity field on [0, 2 pi)^3."""
    grid = SpectralGrid(n)
    rng = np.random.default_rng(seed)
    spectral = rng.standard_normal((3, n, n, n)) + 1j * rng.standard_normal((3, n, n, n))
    # Decay high wavenumbers so the field is smooth and well resolved.
    spectral *= np.exp(-grid.k2 / 8.0)
    # Solenoidal projection.
    k = np.stack([grid.kx, grid.ky, grid.kz])
    k2 = np.where(grid.k2 == 0.0, 1.0, grid.k2)
    spectral -= k * np.sum(k * spectral, axis=0) / k2
    spectral[:, 0, 0, 0] = 0.0
    velocity = grid.ifft(spectral)
    return grid, velocity / float(np.sqrt(np.mean(velocity**2)))


def legacy_leonard_term(grid, velocity, vorticity, test_filter):
    """The pre-fix expression, reproduced verbatim for regression purposes."""
    convection, stretching = nonlinear_parts(grid, velocity, vorticity)
    convection_test, stretching_test = nonlinear_parts(
        grid, test_filter(velocity), test_filter(vorticity)
    )
    return convection - convection_test - stretching + stretching_test


def relative_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)) / np.sqrt(np.mean(b**2)))


def gaussian_test_filter(grid: SpectralGrid, delta: float):
    def apply(field: np.ndarray) -> np.ndarray:
        return grid.gaussian(field, delta)

    return apply


def agard_resolved_state():
    """Particle-filtered resolved state at the audited operating point."""
    from stage_4a_formulation import load_agard

    velocity = load_agard(AGARD)
    grid = SpectralGrid(velocity.shape[-1])
    h = 2.0 * np.pi / LES_N
    sigma = SIGMA_OVER_H * h
    symbol = particle_symbol(grid, h, sigma, PHASES)
    exact = exact_sgs_for_filter(grid, velocity, lambda f: apply_symbol(grid, f, symbol))
    return grid, h, sigma, exact


requires_agard = pytest.mark.skipif(
    not AGARD.exists(), reason="AGARD 128^3 reference field is not present"
)


# ── the decisive invariant ───────────────────────────────────────────────────


def test_leonard_term_equals_exact_sgs_of_test_filter():
    """L must be the exact SGS source of T acting on the resolved field.

    Cross-checked against `exact_sgs_for_filter`, an independent code path.
    """
    grid, velocity = synthetic_field()
    vorticity = grid.curl(velocity)
    test_filter = gaussian_test_filter(grid, 0.35)

    computed = leonard_term(grid, velocity, vorticity, test_filter)
    reference = exact_sgs_for_filter(grid, velocity, test_filter)["g"]

    assert relative_error(computed, reference) < 1.0e-12


def test_legacy_leonard_term_violates_the_identity():
    """Guard against reverting the fix for apparent consistency.

    The robust, field-independent statement is that the legacy expression is
    wrong by order unity.  Its *sign* behaviour is field dependent -- see
    `test_legacy_expression_inverts_the_coefficient_sign_on_agard` for the
    specific inversion that produced the Stage 8A failure.
    """
    grid, velocity = synthetic_field()
    vorticity = grid.curl(velocity)
    test_filter = gaussian_test_filter(grid, 0.35)

    legacy = legacy_leonard_term(grid, velocity, vorticity, test_filter)
    reference = exact_sgs_for_filter(grid, velocity, test_filter)["g"]

    assert relative_error(legacy, reference) > 0.5


# ── estimator properties ─────────────────────────────────────────────────────


def test_coefficient_squared_is_invariant_under_velocity_rescaling():
    """L and M both scale as u^2, so C_r^2 must be amplitude independent."""
    grid, velocity = synthetic_field()
    vorticity = grid.curl(velocity)
    width = 0.4

    base, _ = mansfield_dynamic_coefficient(grid, velocity, vorticity, width, 0.2)
    scaled, _ = mansfield_dynamic_coefficient(grid, 7.5 * velocity, 7.5 * vorticity, width, 0.2)
    assert base == pytest.approx(scaled, rel=1.0e-9)


def test_scale_separation_diagnostic_is_reported():
    """The estimator must expose how much box the test filter occupies."""
    grid, velocity = synthetic_field()
    vorticity = grid.curl(velocity)
    _, diagnostics = mansfield_dynamic_coefficient(grid, velocity, vorticity, 0.4, 0.2)

    assert "test_filter_width_over_box" in diagnostics
    assert diagnostics["test_filter_width_over_box"] == pytest.approx(2.0 * 0.4 / (2.0 * math.pi))


# ── behaviour on the audited operating point ─────────────────────────────────


@requires_agard
def test_dynamic_coefficient_is_admissible_on_agard():
    """The regression that the bug produced: C_r^2 < 0 for a dissipative model."""
    grid, _, sigma, exact = agard_resolved_state()
    width = gaussian_energy_width_ratio() * sigma

    coefficient, diagnostics = mansfield_dynamic_coefficient(
        grid, exact["u"], exact["w"], width, sigma
    )

    assert diagnostics["coefficient_squared_raw"] > 0.0, (
        "negative C_r^2 is inadmissible for a non-negative eddy diffusivity; "
        "this was the pre-fix failure mode"
    )
    assert 0.01 < coefficient < 0.5, f"C_r={coefficient} outside any plausible band"


@requires_agard
def test_legacy_expression_inverts_the_coefficient_sign_on_agard():
    """The exact regression: legacy gives C_r^2 < 0, corrected gives C_r^2 > 0.

    This is the failure that zeroed the closure and cascaded into every
    predeclared Stage 8A check.
    """
    grid, _, sigma, exact = agard_resolved_state()
    width = gaussian_energy_width_ratio() * sigma
    test_filter = gaussian_test_filter(grid, 2.0 * sigma)

    base = -(width**2) * grid.curl(
        strain_magnitude(grid, exact["u"])[None, ...] * grid.curl(exact["w"])
    )
    test_width = 2.0 * width
    velocity_test, vorticity_test = test_filter(exact["u"]), test_filter(exact["w"])
    test_basis = -(test_width**2) * grid.curl(
        strain_magnitude(grid, velocity_test)[None, ...] * grid.curl(vorticity_test)
    )
    m = test_filter(base) - test_basis
    denominator = float(np.mean(np.sum(m * m, axis=0)))

    def coefficient_squared(ell):
        return float(np.mean(np.sum(ell * m, axis=0))) / denominator

    legacy = coefficient_squared(legacy_leonard_term(grid, exact["u"], exact["w"], test_filter))
    corrected = coefficient_squared(leonard_term(grid, exact["u"], exact["w"], test_filter))

    assert legacy < 0.0, "legacy expression should reproduce the historical failure"
    assert corrected > 0.0, "corrected expression must be admissible"


@requires_agard
def test_dynamic_procedure_still_underpredicts_transfer():
    """Documents the *remaining*, separate problem after the bug fix.

    The corrected coefficient is admissible but roughly three times too small,
    because the test filter's equivalent width is a large fraction of the box
    and the Germano identity has no self-similar range to work in.  This test
    pins that finding so it is not mistaken for a fresh regression.
    """
    grid, _, sigma, exact = agard_resolved_state()
    width = gaussian_energy_width_ratio() * sigma

    coefficient, diagnostics = mansfield_dynamic_coefficient(
        grid, exact["u"], exact["w"], width, sigma
    )
    torque, _ = mansfield_torque(grid, exact["u"], exact["w"], coefficient, width)

    exact_transfer = float(np.mean(np.sum(exact["w"] * exact["g"], axis=0)))
    model_transfer = float(np.mean(np.sum(exact["w"] * torque, axis=0)))

    assert exact_transfer < 0.0
    assert 0.0 < model_transfer / exact_transfer < 0.5
    # No scale separation: the test filter spans a large fraction of the box.
    assert diagnostics["test_filter_width_over_box"] > 0.4


# ── operator form ────────────────────────────────────────────────────────────


@requires_agard
def test_mansfield_operator_is_solenoidal():
    """curl-curl form cannot inject divergence into the vorticity field."""
    grid, _, sigma, exact = agard_resolved_state()
    width = gaussian_energy_width_ratio() * sigma
    torque, _ = mansfield_torque(grid, exact["u"], exact["w"], 0.1367, width)
    assert divergence_relative(grid, torque) < 1.0e-12


@requires_agard
def test_current_openonda_operator_is_not_solenoidal():
    """Known production defect, pinned so the contrast stays visible.

    `nu_t * laplacian(omega)` with spatially varying nu_t does not preserve
    div(omega) = 0.  Fixing this is tracked separately from the LES question.
    """
    grid, h, _, exact = agard_resolved_state()
    torque, _ = openonda_current_torque(grid, exact["u"], exact["w"], cs=0.17, h=h)
    assert divergence_relative(grid, torque) > 0.1
