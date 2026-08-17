#!/usr/bin/env python3
"""Physics gate for an established particle-filter functional closure.

DIAD is intentionally absent.  This script treats the Gaussian particle core
and M4' particle/grid transfer as the LES filter, computes its exact SGS torque
on the AGARD 128^3 field, and evaluates Mansfield's vorticity eddy-diffusivity
operator by transfer and spectral budgets rather than pointwise correlation.

The script also evaluates the continuous operator currently approximated by
OpenONDA's GBD+Smagorinsky path.  That comparison is diagnostic only: the
production code is not modified here.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/openonda_stage8a_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/openonda_stage8a_cache")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from stage_4a_formulation import (  # noqa: E402
    SpectralGrid,
    load_agard,
    metrics,
    nonlinear_parts,
    norm,
    shell_transfer,
)
from stage_6a_composite_filter_gate import PHASES  # noqa: E402
from stage_7a_composite_sgs_audit import (  # noqa: E402
    apply_symbol,
    exact_sgs_for_filter,
    particle_symbol,
)

INK = "#20252a"
BLUE = "#286f9b"
GOLD = "#d9973b"
RED = "#b54a4a"
GREY = "#8a99a8"
GRID = "#d8dde2"


def strain_magnitude(grid: SpectralGrid, velocity: np.ndarray) -> np.ndarray:
    gradient = grid.gradient(velocity)
    strain = 0.5 * (gradient + np.swapaxes(gradient, 0, 1))
    return np.sqrt(2.0 * np.sum(strain * strain, axis=(0, 1)))


def laplacian(grid: SpectralGrid, field: np.ndarray) -> np.ndarray:
    return grid.ifft(-grid.k2 * grid.fft(field))


def mansfield_torque(
    grid: SpectralGrid,
    velocity: np.ndarray,
    vorticity: np.ndarray,
    coefficient: float,
    filter_width: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return g=-curl(nu_t curl(omega)) and the non-negative nu_t field."""
    nu_t = (coefficient * filter_width) ** 2 * strain_magnitude(grid, velocity)
    torque = -grid.curl(nu_t[None, ...] * grid.curl(vorticity))
    return torque, nu_t


def leonard_term(
    grid: SpectralGrid,
    velocity: np.ndarray,
    vorticity: np.ndarray,
    test_filter: object,
) -> np.ndarray:
    """Resolved (Leonard) SGS source of the test filter on the resolved field.

    This is the left-hand side of the Germano identity.  It is *identically*
    the exact SGS source that `exact_sgs_for_filter` returns for the test
    filter acting on the resolved field, which is what
    `tests/experiments/test_mansfield_dynamic_coefficient.py` asserts.

    Every term must carry the same composition of the test filter T as the
    model difference M = T(B) - B(T); omitting T on the two base terms leaves
    the unfiltered content (I - T)N(u) in the estimator and corrupts the sign.
    """
    convection, stretching = nonlinear_parts(grid, velocity, vorticity)
    convection_test, stretching_test = nonlinear_parts(
        grid, test_filter(velocity), test_filter(vorticity)
    )
    return -test_filter(convection) + convection_test + test_filter(stretching) - stretching_test


def mansfield_dynamic_coefficient(
    grid: SpectralGrid,
    velocity: np.ndarray,
    vorticity: np.ndarray,
    filter_width: float,
    particle_sigma: float,
    test_ratio: float = 2.0,
) -> tuple[float, dict[str, float]]:
    """Evaluate Mansfield Eqs. (22)-(31) with global spatial averaging.

    The Germano identity requires the resolved (Leonard) term and the model
    difference to be built with the *same* composition of the test filter T:

        L = -[T(u.grad w) - (Tu).grad(Tw)] + [T(w.grad u) - (Tw).grad(Tu)]
        M =  T(B) - B(T)                                with  B = -D^2 curl(|S| curl w)
        L = C_r^2 M   ->   C_r^2 = <L.M> / <M.M>

    The sign convention for L must match `exact_sgs_for_filter`, which defines
    g = -F(convection) + convection(F u) + F(stretching) - stretching(F u).

    Before 2026-08-17 this function omitted T on the two base terms and carried
    the opposite overall sign, i.e. `ell = convection - convection_test
    - stretching + stretching_test`.  That leaves the unfiltered high-wavenumber
    part (I - T)N(u) in L, which is large and uncorrelated with M, and it
    inverted the sign of <L.M>.  The result was C_r^2 < 0 for a purely
    dissipative model, which the non-negativity clip then zeroed, silently
    disabling the closure and failing every downstream gate.  See
    `tests/test_mansfield_dynamic_coefficient.py`.
    """
    test_sigma = test_ratio * particle_sigma

    def test_filter(field):
        return grid.gaussian(field, test_sigma)

    velocity_test = test_filter(velocity)
    vorticity_test = test_filter(vorticity)
    ell = leonard_term(grid, velocity, vorticity, test_filter)

    base_basis = -(filter_width**2) * grid.curl(
        strain_magnitude(grid, velocity)[None, ...] * grid.curl(vorticity)
    )
    test_width = test_ratio * filter_width
    test_basis = -(test_width**2) * grid.curl(
        strain_magnitude(grid, velocity_test)[None, ...] * grid.curl(vorticity_test)
    )
    m = test_filter(base_basis) - test_basis
    numerator = float(np.mean(np.sum(ell * m, axis=0)))
    denominator = float(np.mean(np.sum(m * m, axis=0)))
    coefficient_squared_raw = numerator / denominator if denominator > 0.0 else 0.0
    coefficient = math.sqrt(max(0.0, coefficient_squared_raw))

    # Scale-separation diagnostic.  The Germano identity assumes the test filter
    # sits in a self-similar range above the base filter.  If the test filter's
    # energy-equivalent width is a significant fraction of the periodic box,
    # the identity has no range to work in and C_r is not trustworthy even when
    # it is positive.
    box = 2.0 * math.pi
    return coefficient, {
        "coefficient_squared_raw": coefficient_squared_raw,
        "numerator": numerator,
        "denominator": denominator,
        "test_filter_ratio": test_ratio,
        "base_filter_width_over_box": filter_width / box,
        "test_filter_width_over_box": test_width / box,
        "leonard_term_convention": (
            "L = -T(conv) + conv(Tu) + T(stre) - stre(Tu); matches "
            "exact_sgs_for_filter and the test filter used to build M"
        ),
    }


def openonda_current_torque(
    grid: SpectralGrid,
    velocity: np.ndarray,
    vorticity: np.ndarray,
    cs: float,
    h: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Continuous analogue of current GBD: nu_t(x) times component Laplacian."""
    nu_t = (cs * h) ** 2 * strain_magnitude(grid, velocity)
    return nu_t[None, ...] * laplacian(grid, vorticity), nu_t


def divergence_relative(grid: SpectralGrid, vector: np.ndarray) -> float:
    divergence = sum(grid.derivative(vector[i], i) for i in range(3))
    gradient_scale = norm(grid.gradient(vector))
    return norm(divergence) / max(gradient_scale, np.finfo(float).tiny)


def gaussian_energy_width_ratio() -> float:
    """Box-filter width / sigma from equal kernel L2 energy in three dimensions."""
    return float((6.0 * 2.0**1.5 * math.sqrt(math.pi)) ** (1.0 / 3.0))


def mansfield_gaussian_coefficient(skewness: float = -0.4) -> float:
    """Mansfield Appendix-A estimate specialized to exp(-sigma^2 k^2/4)."""
    c = gaussian_energy_width_ratio()
    # With E(k)~k^-5/3 and |G|^2=exp(-sigma^2 k^2/2), I_2/I_4=3 sigma^2/4.
    spectral_ratio = 0.75 / c**2
    return float(math.sqrt(35.0 / (15.0**1.5) * abs(skewness) * spectral_ratio))


def transfer(vorticity: np.ndarray, torque: np.ndarray) -> float:
    return float(np.mean(np.sum(vorticity * torque, axis=0)))


def evaluate(
    agard_path: Path,
    les_n: int = 32,
    sigma_over_h: float = 2.5,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    velocity = load_agard(agard_path)
    grid = SpectralGrid(velocity.shape[-1])
    h = 2.0 * np.pi / les_n
    sigma = sigma_over_h * h
    p_symbol = particle_symbol(grid, h, sigma, PHASES)

    def particle_filter(field):
        return apply_symbol(grid, field, p_symbol)

    exact = exact_sgs_for_filter(grid, velocity, particle_filter)

    width_ratio = gaussian_energy_width_ratio()
    particle_width = width_ratio * sigma
    cr_paper = 0.12
    cr_gaussian = mansfield_gaussian_coefficient()

    current, current_nu_t = openonda_current_torque(grid, exact["u"], exact["w"], cs=0.17, h=h)
    mansfield_paper, paper_nu_t = mansfield_torque(
        grid, exact["u"], exact["w"], cr_paper, particle_width
    )
    mansfield_gaussian, gaussian_nu_t = mansfield_torque(
        grid, exact["u"], exact["w"], cr_gaussian, particle_width
    )
    cr_dynamic, dynamic_diagnostics = mansfield_dynamic_coefficient(
        grid, exact["u"], exact["w"], particle_width, sigma
    )
    mansfield_dynamic, dynamic_nu_t = mansfield_torque(
        grid, exact["u"], exact["w"], cr_dynamic, particle_width
    )

    torques = {
        "openonda_current": current,
        "mansfield_paper_coefficient": mansfield_paper,
        "mansfield_gaussian_adjusted": mansfield_gaussian,
        "mansfield_dynamic": mansfield_dynamic,
    }
    viscosities = {
        "openonda_current": current_nu_t,
        "mansfield_paper_coefficient": paper_nu_t,
        "mansfield_gaussian_adjusted": gaussian_nu_t,
        "mansfield_dynamic": dynamic_nu_t,
    }
    exact_transfer = transfer(exact["w"], exact["g"])
    model_records: dict[str, object] = {}
    for name, torque in torques.items():
        record = metrics(grid, exact["w"], exact["g"], torque)
        record.update(
            {
                "enstrophy_transfer": transfer(exact["w"], torque),
                "divergence_relative": divergence_relative(grid, torque),
                "nu_t_mean": float(np.mean(viscosities[name])),
                "nu_t_max": float(np.max(viscosities[name])),
            }
        )
        if not np.isfinite(record["correlation"]):
            record["correlation"] = None
        model_records[name] = record

    model_records["mansfield_dynamic"]["dynamic_coefficient"] = cr_dynamic
    model_records["mansfield_dynamic"]["dynamic_diagnostics"] = dynamic_diagnostics
    selected = model_records["mansfield_dynamic"]
    checks = {
        "reference_has_forward_mean_transfer": bool(exact_transfer < 0.0),
        "mansfield_operator_is_solenoidal": bool(selected["divergence_relative"] < 1.0e-12),
        "mansfield_operator_is_mean_dissipative": bool(selected["enstrophy_transfer"] < 0.0),
        "mansfield_transfer_within_50_percent": bool(0.5 <= selected["transfer_ratio"] <= 1.5),
        "mansfield_shell_transfer_error_below_50_percent": bool(selected["shell_error"] <= 0.5),
    }
    result: dict[str, object] = {
        "stage": "8A — particle-filter functional closure physics gate",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "source_field": str(agard_path.relative_to(ROOT)),
        "configuration": {
            "dns_n": grid.n,
            "nominal_les_n": les_n,
            "sigma_over_h": sigma_over_h,
            "explicit_auxiliary_filter": "none",
            "particle_filter": "Gaussian core plus M4' P2M/M2P symbol",
            "particle_filter_energy_equivalent_width_over_h": particle_width / h,
            "paper_coefficient_for_third_order_gaussian": cr_paper,
            "appendix_A_coefficient_for_openonda_gaussian": cr_gaussian,
            "dynamic_coefficient_from_equation_31": cr_dynamic,
            "coefficient_skewness_assumption": -0.4,
        },
        "exact_particle_filter_enstrophy_transfer": exact_transfer,
        "models": model_records,
        "predeclared_checks": checks,
        "gate_basis": (
            "Functional closure is judged by transfer sign, magnitude, and shell "
            "distribution; pointwise correlation and L2 are reported but not gated."
        ),
        "implementation_warning": (
            "OpenONDA current uses Delta=h and nu_t Laplacian(omega); Mansfield "
            "uses the particle-filter width and -curl(nu_t curl(omega))."
        ),
    }
    arrays = {
        "exact": shell_transfer(grid, exact["w"], exact["g"]),
        **{name: shell_transfer(grid, exact["w"], torque) for name, torque in torques.items()},
    }
    return result, arrays


def plot(result: dict[str, object], arrays: dict[str, np.ndarray], output: Path) -> None:
    models = result["models"]
    order = (
        "openonda_current",
        "mansfield_paper_coefficient",
        "mansfield_gaussian_adjusted",
        "mansfield_dynamic",
    )
    labels = (
        "current\nOpenONDA",
        "Mansfield\n$C_r=0.12$",
        "Mansfield\nfilter-adjusted",
        "Mansfield\ndynamic",
    )
    colors = (GREY, GOLD, BLUE, RED)
    fig, axes = plt.subplots(1, 3, figsize=(14.1, 4.3), constrained_layout=True)

    exact_transfer = result["exact_particle_filter_enstrophy_transfer"]
    k = np.arange(len(arrays["exact"]))
    scale = max(abs(exact_transfer), np.finfo(float).tiny)
    axes[0].plot(
        k, arrays["exact"] / scale, color=INK, linewidth=2.2, label="exact particle-filter SGS"
    )
    for name, label, color in zip(order, labels, colors, strict=True):
        axes[0].plot(
            k, arrays[name] / scale, color=color, linestyle="--", label=label.replace("\n", " ")
        )
    axes[0].axhline(0.0, color=GREY, linewidth=0.8)
    axes[0].set_xlim(0, 24)
    axes[0].set_xlabel("wavenumber shell $k$")
    axes[0].set_ylabel("shell transfer / exact mean transfer")
    axes[0].set_title("Scale-by-scale enstrophy transfer")
    axes[0].legend(frameon=False, fontsize=7)

    x = np.arange(len(order))
    ratios = [models[name]["transfer_ratio"] for name in order]
    shell_errors = [models[name]["shell_error"] for name in order]
    axes[1].bar(x - 0.18, ratios, 0.36, color=BLUE, label="transfer ratio (1 ideal)")
    axes[1].bar(x + 0.18, shell_errors, 0.36, color=RED, label="shell error (0 ideal)")
    axes[1].axhline(1.0, color=INK, linestyle="--", linewidth=1.0)
    axes[1].axhline(0.5, color=GOLD, linestyle=":", linewidth=1.2, label="declared limits")
    axes[1].axhline(1.5, color=GOLD, linestyle=":", linewidth=1.2)
    axes[1].set_xticks(x, labels)
    axes[1].set_title("Functional-model gate quantities")
    axes[1].legend(frameon=False, fontsize=7)

    divergence = [models[name]["divergence_relative"] for name in order]
    axes[2].semilogy(
        x, np.maximum(divergence, 1e-18), "o-", color=BLUE, label="relative divergence"
    )
    axes[2].axhline(1e-12, color=GOLD, linestyle=":", label=r"$10^{-12}$ gate")
    axes[2].set_xticks(x, labels)
    axes[2].set_ylabel("relative divergence of modeled torque")
    axes[2].set_title("Vorticity compatibility")
    axes[2].legend(frameon=False, fontsize=8)

    for axis in axes:
        axis.grid(color=GRID, linewidth=0.6, alpha=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(r"Particle-filter closure gate: $\sigma/h=2.5$, no auxiliary LES filter")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--agard",
        type=Path,
        default=ROOT / "docs/dns/agard_hom02/CB128_9.bin",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "scripts/experiments/stage_8a_particle_functional_results.json",
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=ROOT / "docs/figures/vpm_les/stage_8a_particle_functional_gate.png",
    )
    args = parser.parse_args()
    result, arrays = evaluate(args.agard)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    plot(result, arrays, args.figure)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
