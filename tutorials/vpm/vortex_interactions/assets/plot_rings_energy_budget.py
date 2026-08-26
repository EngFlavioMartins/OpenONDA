#!/usr/bin/env python3
"""Energy-budget audit for vortex-ring interaction tests.

The target identity for an unbounded viscous incompressible flow is

    dE/dt = -kinematic_viscosity * integral(|omega|^2 dV)

for the total_kinetic_energy and total_enstrophy conventions used by the VPM diagnostics.  Some
texts define total_enstrophy as 0.5*integral(|omega|^2), which is the same statement
written as dE/dt = -2*kinematic_viscosity*Enstrophy. For LES, ``effective_viscosity`` varies in space and the
    solver exports the exact weighted quadratic viscous_kinetic_energy_rate as ``viscous_kinetic_energy_rate``;
replacing it by a mean viscosity times total_enstrophy is not generally valid.
The stabilized cases additionally export the exact cumulative total_kinetic_energy transfer
of conservative regularization. Its interval derivative is added to the
continuous viscous kinetic-energy rate before comparing with the measured total kinetic-energy decay.

It reads the flow-integral CSV written by the VPM sampler.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _common import (
    RING_CIRCULATION,
    RING_RADIUS,
    REFERENCE_TIME,
    build_arg_parser,
    case_style,
    compact_case_legend_handles,
    discover_cases,
    figure_size,
    load_theme,
    mark_every,
    read_integrals,
    reference_fill_style,
    save_fig,
    secondary_line_style,
)


def read_budget(case_dir: Path) -> tuple[pd.DataFrame | None, str]:
    """Return a monotone sampler-derived time series."""
    df = read_integrals(case_dir)
    if df is None:
        return None, ""
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["time", "total_kinetic_energy"])
    if len(df) < 2:
        return None, ""
    return df.reset_index(drop=True), "sampler"


def interval_derivative(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return backward slopes that are consistent with every sampled interval."""
    derivative = np.full_like(y, np.nan, dtype=float)
    if len(t) >= 2:
        derivative[1:] = np.diff(y) / np.diff(t)
    return derivative


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)))) if len(x) else np.nan


def balance_metrics(
    t: np.ndarray,
    total_kinetic_energy: np.ndarray,
    kinetic_energy_rate: np.ndarray,
    viscous_kinetic_energy_rate: np.ndarray,
) -> dict[str, float]:
    """Compare each sampled total_kinetic_energy increment with its interval-averaged viscous_kinetic_energy_rate."""
    time_step_size = np.diff(t)
    interval_rate = kinetic_energy_rate[1:]
    interval_viscous_kinetic_energy_rate = viscous_kinetic_energy_rate[1:]
    valid = (
        np.isfinite(interval_rate)
        & np.isfinite(interval_viscous_kinetic_energy_rate)
        & np.isfinite(time_step_size)
        & (time_step_size > 0.0)
    )
    if valid.sum() < 2:
        return {
            "point_rel_l2": np.nan,
            "point_bias": np.nan,
            "integrated_ratio": np.nan,
            "integrated_residual_E0": np.nan,
            "fit_slope": np.nan,
            "fit_r2": np.nan,
        }

    dv = interval_rate[valid]
    sv = interval_viscous_kinetic_energy_rate[valid]
    dtv = time_step_size[valid]
    diff = dv - sv
    denom = rms(sv)
    point_rel_l2 = rms(diff) / denom if denom > 0.0 else np.nan
    point_bias = float(np.mean(diff) / (np.mean(np.abs(sv)) + 1e-30))

    dE_total = float(np.sum(dv * dtv))
    sink_int = float(np.sum(sv * dtv))
    residual_int = dE_total - sink_int
    integrated_ratio = dE_total / sink_int if abs(sink_int) > 1e-30 else np.nan

    cumulative = np.cumsum(sv * dtv)
    delta_e = np.cumsum(dv * dtv)
    if np.std(cumulative) > 0.0:
        slope, intercept = np.polyfit(cumulative, delta_e, 1)
        pred = slope * cumulative + intercept
        ss_res = float(np.sum((delta_e - pred) ** 2))
        ss_tot = float(np.sum((delta_e - np.mean(delta_e)) ** 2))
        fit_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else np.nan
    else:
        slope, fit_r2 = np.nan, np.nan

    e0 = float(total_kinetic_energy[0])
    return {
        "point_rel_l2": point_rel_l2,
        "point_bias": point_bias,
        "integrated_ratio": integrated_ratio,
        "integrated_residual_E0": residual_int / e0 if abs(e0) > 1e-30 else np.nan,
        "fit_slope": float(slope),
        "fit_r2": fit_r2,
    }


def summarize_case(case_dir: Path) -> tuple[pd.DataFrame | None, dict | None]:
    df, source = read_budget(case_dir)
    if df is None:
        return None, None

    t = df["time"].to_numpy(float)
    total_kinetic_energy = df["total_kinetic_energy"].to_numpy(float)
    kinetic_energy_rate_interval = interval_derivative(t, total_kinetic_energy)

    sink_nu = (
        df["viscous_kinetic_energy_rate"].to_numpy(float)
        if "viscous_kinetic_energy_rate" in df.columns
        else np.full_like(t, np.nan)
    )
    filter_transfer = (
        df["regularization_cumulative_total_kinetic_energy_transfer"].to_numpy(float)
        if "regularization_cumulative_total_kinetic_energy_transfer" in df.columns
        else np.zeros_like(t)
    )
    filter_rate = interval_derivative(t, filter_transfer)
    filter_rate[0] = 0.0
    modeled_rate = np.full_like(t, np.nan)
    modeled_rate[1:] = 0.5 * (sink_nu[:-1] + sink_nu[1:]) + filter_rate[1:]
    total_enstrophy = (
        df["total_enstrophy"].to_numpy(float)
        if "total_enstrophy" in df.columns
        else np.full_like(t, np.nan)
    )

    e0 = float(total_kinetic_energy[0]) if abs(float(total_kinetic_energy[0])) > 1e-30 else 1.0

    spurious_rate = np.clip(kinetic_energy_rate_interval, 0.0, None)
    spurious_rate[~np.isfinite(spurious_rate)] = 0.0
    cum_spurious = np.zeros_like(t)
    if len(t) > 1:
        inc = 0.5 * (spurious_rate[1:] + spurious_rate[:-1]) * np.diff(t)
        cum_spurious[1:] = np.cumsum(inc)

    out = df.copy()
    out.insert(0, "case", case_dir.name)
    out["source"] = source
    out["nondimensional_time"] = t / REFERENCE_TIME
    out["kinetic_energy_rate_interval"] = kinetic_energy_rate_interval
    out["regularization_total_kinetic_energy_rate"] = filter_rate
    out["modeled_energy_rate"] = modeled_rate
    out["residual_vs_modeled_energy_rate"] = kinetic_energy_rate_interval - modeled_rate
    out["cum_spurious_E0"] = cum_spurious / e0

    modeled_viscous_metrics = balance_metrics(
        t,
        total_kinetic_energy,
        kinetic_energy_rate_interval,
        modeled_rate,
    )

    valid = (
        np.isfinite(kinetic_energy_rate_interval)
        & np.isfinite(total_enstrophy)
        & (total_enstrophy > 0.0)
    )
    fitted_c = np.nan
    if valid.sum() >= 2:
        # Best positive factor c in dE/dt ~= -c * total_enstrophy.
        fitted_c = -float(
            np.dot(kinetic_energy_rate_interval[valid], total_enstrophy[valid])
        ) / float(np.dot(total_enstrophy[valid], total_enstrophy[valid]))

    e0 = float(total_kinetic_energy[0])

    # Conserved net vortex strength and impulses. Closed rings may have a
    # nearly zero vector sum, so normalize by physical ring scales rather than
    # by a cancellation-prone initial norm alone.
    net_vortex_strength_columns = [f"net_vortex_strength_{axis}" for axis in "xyz"]
    if all(column in df.columns for column in net_vortex_strength_columns):
        net_vortex_strength = df[net_vortex_strength_columns].to_numpy(float)
        net_vortex_strength_drift_relative = float(
            np.linalg.norm(net_vortex_strength - net_vortex_strength[0], axis=1).max()
            / max(RING_CIRCULATION, 1e-30)
        )
    else:
        net_vortex_strength_drift_relative = np.nan

    linear_impulse_columns = [f"linear_impulse_{axis}" for axis in "xyz"]
    if all(column in df.columns for column in linear_impulse_columns):
        linear_impulse = df[linear_impulse_columns].to_numpy(float)
        impulse_scale = max(
            float(np.linalg.norm(linear_impulse[0])),
            RING_CIRCULATION * RING_RADIUS**2,
        )
        linear_impulse_drift_relative = float(
            np.linalg.norm(linear_impulse - linear_impulse[0], axis=1).max() / impulse_scale
        )
    else:
        linear_impulse_drift_relative = np.nan

    angular_impulse_columns = [f"angular_impulse_{axis}" for axis in "xyz"]
    if all(column in df.columns for column in angular_impulse_columns):
        angular_impulse = df[angular_impulse_columns].to_numpy(float)
        angular_scale = max(
            float(np.linalg.norm(angular_impulse[0])),
            RING_CIRCULATION * RING_RADIUS**3,
        )
        angular_impulse_drift_relative = float(
            np.linalg.norm(angular_impulse - angular_impulse[0], axis=1).max() / angular_scale
        )
    else:
        angular_impulse_drift_relative = np.nan

    n_particles_total = (
        df["n_particles_total"].to_numpy(float)
        if "n_particles_total" in df.columns
        else np.array([np.nan])
    )
    valid_kinetic_energy_rate = kinetic_energy_rate_interval[
        np.isfinite(kinetic_energy_rate_interval)
    ]
    summary = {
        "case": case_dir.name,
        "source": source,
        "n_samples": len(df),
        "mean_sample_time_step_size": float(np.mean(np.diff(t))) if len(t) > 1 else np.nan,
        "final_time": float(t[-1]),
        "total_kinetic_energy_ratio": (
            float(total_kinetic_energy[-1] / e0) if abs(e0) > 1e-30 else np.nan
        ),
        "total_kinetic_energy_change_relative": (
            float((total_kinetic_energy[-1] - total_kinetic_energy[0]) / e0)
            if abs(e0) > 1e-30
            else np.nan
        ),
        "best_c_for_minus_c_enstrophy": fitted_c,
        # --- objective verdict: dE/dt ≤ 0 and physics preserved -------------
        "fraction_positive_kinetic_energy_rate": (
            float(np.mean(valid_kinetic_energy_rate > 0.0))
            if valid_kinetic_energy_rate.size
            else np.nan
        ),
        "max_positive_kinetic_energy_rate_normalized": (
            float(valid_kinetic_energy_rate.max() / e0)
            if valid_kinetic_energy_rate.size and abs(e0) > 1e-30
            else np.nan
        ),
        "positive_kinetic_energy_injection_normalized": (
            float(cum_spurious[-1] / e0) if abs(e0) > 1e-30 else np.nan
        ),
        "net_vortex_strength_drift_relative": net_vortex_strength_drift_relative,
        "linear_impulse_drift_relative": linear_impulse_drift_relative,
        "angular_impulse_drift_relative": angular_impulse_drift_relative,
        "n_particles_ratio": (
            float(n_particles_total[-1] / n_particles_total[0])
            if n_particles_total.size and n_particles_total[0] > 0
            else np.nan
        ),
    }
    for key, value in modeled_viscous_metrics.items():
        summary[f"modeled_viscous_{key}"] = value
    if (
        "vortex_strength_magnitude_sum" in df.columns
        and df["vortex_strength_magnitude_sum"].iloc[0] != 0.0
    ):
        summary["vortex_strength_magnitude_sum_ratio"] = float(
            df["vortex_strength_magnitude_sum"].iloc[-1]
            / df["vortex_strength_magnitude_sum"].iloc[0]
        )
    else:
        summary["vortex_strength_magnitude_sum_ratio"] = np.nan

    return out, summary


def make_figure(
    timeseries: pd.DataFrame,
    summary: pd.DataFrame,
    figures_dir: Path,
    dpi: int,
    figure_format: str = "png",
) -> None:
    load_theme()
    fig, (ax_rate, ax_spur) = plt.subplots(2, 1, figsize=figure_size("wide_stacked"), sharex=True)
    rate_values: list[np.ndarray] = []
    spurious_values: list[np.ndarray] = []

    for case, df in timeseries.groupby("case", sort=True):
        st = case_style(case)
        t = df["nondimensional_time"].to_numpy(float)
        e0 = float(df["total_kinetic_energy"].iloc[0])
        scale = abs(e0) if abs(e0) > 1e-30 else 1.0
        common = dict(
            color=st["color"],
            linestyle=st["linestyle"],
            lw=st["linewidth"],
            marker=st["marker"],
            ms=st["markersize"],
            markevery=mark_every(),
            mew=st["markeredgewidth"],
        )

        rate = df["kinetic_energy_rate_interval"].to_numpy(float) / scale
        rate_values.append(rate)
        ax_rate.plot(t, rate, label=st["label"], **common)
        if "modeled_energy_rate" in df:
            ax_rate.plot(
                t,
                df["modeled_energy_rate"] / scale,
                color=st["color"],
                **secondary_line_style(),
            )

        # Cumulative spurious total_kinetic_energy: monotone, stays at 0 for a run that never
        # violates dE/dt ≤ 0.  This is the primary pass/fail signal.
        spur = df["cum_spurious_E0"].to_numpy(float)
        spurious_values.append(spur)
        ax_spur.plot(t, spur, **common)

    def padded_limits(values: list[np.ndarray]) -> tuple[float, float]:
        finite = np.concatenate([array[np.isfinite(array)] for array in values])
        lower = min(-0.05, float(finite.min()))
        upper = max(0.05, float(finite.max()))
        padding = 0.06 * (upper - lower)
        return lower - padding, upper + padding

    # Energy may only decay: dE/dt > 0 (shaded) is spurious injection.
    ax_rate.axhspan(0, 2.0, **reference_fill_style())
    ax_rate.axhline(0.0, color="0.55", linestyle=":", linewidth=0.8)
    ax_rate.set_ylabel(r"$E_0^{-1}\,dE/dt$")
    ax_rate.set_title(r"Energy budget — target $dE/dt \leq 0$ everywhere")
    ax_rate.set_ylim(padded_limits(rate_values))

    ax_spur.set_xlabel(r"Normalized time, $t\Gamma_0/R_0^2$")
    ax_spur.set_ylabel(r"$E_0^{-1}\!\int_0^t\max(dE/dt,0)\,dt$")
    upper = max(1e-3, float(np.nanmax([np.nanmax(v) for v in spurious_values])))
    ax_spur.set_ylim(-0.02 * upper, 1.06 * upper)

    fig.legend(
        handles=compact_case_legend_handles(),
        ncol=5,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
    )
    save_fig(
        fig,
        figures_dir / "rings_energy_budget.png",
        dpi=dpi,
        figure_format=figure_format,
        tight_rect=(0.0, 0.16, 1.0, 1.0),
    )


def main() -> None:
    parser = build_arg_parser("Audit dE/dt against viscous and filter dissipation.")
    args = parser.parse_args()

    solution_dir = Path(args.solution_dir)
    figures_dir = Path("figures")

    all_series: list[pd.DataFrame] = []
    rows: list[dict] = []
    for case_dir in discover_cases(solution_dir):
        ts, summary = summarize_case(case_dir)
        if ts is None or summary is None:
            continue
        all_series.append(ts)
        rows.append(summary)

    if not rows:
        raise SystemExit(f"No total_kinetic_energy diagnostics found under {solution_dir}")

    timeseries = pd.concat(all_series, ignore_index=True)
    summary = pd.DataFrame(rows).sort_values("case")

    make_figure(timeseries, summary, figures_dir, args.dpi, args.format)

    # Objective verdict: a method succeeds only if it never injects total_kinetic_energy and
    # preserves net vortex strength and both impulses at every recorded state. Ranked
    # worst-violation first.
    verdict = summary.copy()
    verdict["PASS"] = (
        (verdict["fraction_positive_kinetic_energy_rate"] <= 0.0)
        & (verdict["net_vortex_strength_drift_relative"].abs() <= 1e-3)
        & (verdict["linear_impulse_drift_relative"].abs() <= 1e-2)
        & (verdict["angular_impulse_drift_relative"].abs() <= 1e-2)
    )
    cols = [
        "case",
        "PASS",
        "fraction_positive_kinetic_energy_rate",
        "max_positive_kinetic_energy_rate_normalized",
        "positive_kinetic_energy_injection_normalized",
        "net_vortex_strength_drift_relative",
        "linear_impulse_drift_relative",
        "angular_impulse_drift_relative",
        "total_kinetic_energy_ratio",
        "n_particles_ratio",
    ]
    verdict = verdict.sort_values(
        ["fraction_positive_kinetic_energy_rate", "positive_kinetic_energy_injection_normalized"],
        ascending=False,
    )
    print("\n=== OBJECTIVE VERDICT (kinetic_energy_rate ≤ 0 always, physics preserved) ===")
    print(verdict[cols].to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    print(
        "\nPASS requires: fraction_positive_kinetic_energy_rate=0, "
        "|net_vortex_strength_drift_relative|≤1e-3, "
        "and the maximum observed drift of both impulses≤1e-2.\n"
        "fraction_positive_kinetic_energy_rate = samples with kinetic_energy_rate>0;\n"
        "positive_kinetic_energy_injection_normalized = ∫max(rate,0)dt / E0."
    )
    failed = verdict[~verdict["PASS"]]
    if not failed.empty:
        names = ", ".join(failed["case"].astype(str))
        raise SystemExit(f"Energy/conservation verdict failed: {names}")


if __name__ == "__main__":
    main()
