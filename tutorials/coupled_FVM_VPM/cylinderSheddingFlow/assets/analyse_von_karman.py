#!/usr/bin/env python3
"""Quantitative Von-Karman-instability report for the cylinder shedding experiment.

Reads the sampled probe and force histories of the fully meshed reference and
the coupled hybrid, extracts the linear growth rate, initial amplitude,
saturated frequency, onset threshold, and onset-time offset, and applies the
seed-amplitude acceptance criteria.  Writes ``solution/analysis_summary.json``.

Exit codes:
    0  analysis completed (the verdict is in the report/JSON)
    2  required sampler output is missing or empty
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _plotutil as util  # noqa: E402
from _vonkarman import Series, analyse_series, compare  # noqa: E402

SUMMARY_PATH = util.SOLUTION / "analysis_summary.json"


def _load_series(source: str) -> Series:
    probe = util.load_probe(source)
    forces = util.load_forces(source)
    if probe is None or forces is None:
        missing = [name for name, data in (("probe", probe), ("forces", forces)) if data is None]
        raise SystemExit(
            f"ERROR: {util.label(source)} sampler output missing: {missing}. Run the case first."
        )
    return Series(
        label=util.label(source),
        t_q=probe["t"],
        q=probe["q"],
        t_cd=forces["t"],
        cd=forces["Cd"],
        t_cl=forces["t"],
        cl=forces["Cl"],
    )


def _fmt(value, spec: str = ".4g") -> str:
    return "NaN" if value is None or not np.isfinite(value) else f"{value:{spec}}"


def report(reference: Series, hybrid: Series) -> dict:
    ref = analyse_series(reference)
    hyb = analyse_series(hybrid)
    seed = util.run_constants()["seed_amplitude"]
    comp = compare(ref, hyb, seed=bool(seed > 0.0))

    lines = [
        "\n===== VON KARMAN INSTABILITY REPORT =====",
        f"  Case: infinite cylinder, Re=150  |  seed amplitude eps={seed:g}",
        "",
        f"  {'Quantity':<28}{'Reference FVM':>18}{'Coupled hybrid':>18}",
        f"  {'sigma [1/s]':<28}{_fmt(ref.growth['sigma']):>18}{_fmt(hyb.growth['sigma']):>18}",
        f"  {'A0 (u_y/U)':<28}{_fmt(ref.growth['A0'], '.3e'):>18}{_fmt(hyb.growth['A0'], '.3e'):>18}",
        f"  {'growth fit R^2':<28}{_fmt(ref.growth['r2']):>18}{_fmt(hyb.growth['r2']):>18}",
        f"  {'growth fit interval [s]':<28}{_fmt(ref.growth['t_fit_start'])}..{_fmt(ref.growth['t_fit_end']):<10}{_fmt(hyb.growth['t_fit_start'])}..{_fmt(hyb.growth['t_fit_end']):<10}",
        f"  {'growth-interval f [Hz]':<28}{_fmt(ref.f_growth):>18}{_fmt(hyb.f_growth):>18}",
        f"  {'saturated St [-D/U]':<28}{_fmt(ref.st):>18}{_fmt(hyb.st):>18}",
        f"  {'t* onset (A>=A*) [s]':<28}{_fmt(ref.t_onset):>18}{_fmt(hyb.t_onset):>18}",
        f"  {'mean Cd (saturated)':<28}{_fmt(ref.saturated['mean_Cd']):>18}{_fmt(hyb.saturated['mean_Cd']):>18}",
        f"  {'RMS Cl (saturated)':<28}{_fmt(ref.saturated['rms_Cl']):>18}{_fmt(hyb.saturated['rms_Cl']):>18}",
        f"  {'RMS q (saturated)':<28}{_fmt(ref.saturated['rms_q']):>18}{_fmt(hyb.saturated['rms_q']):>18}",
        f"  {'envelope CV (growth)':<28}{_fmt(ref.modulation):>18}{_fmt(hyb.modulation):>18}",
        "",
        "  --- onset-shift prediction ---",
        f"  predicted  dt = (1/sigma) ln(A0,hyb/A0,ref) = {_fmt(comp.dt_pred)} s",
        f"  measured   dt = t*_ref - t*_hyb           = {_fmt(comp.dt_meas)} s",
        f"  shedding period T = 1/St = {_fmt(comp.shedding_period)} s "
        f"(tolerance {_fmt(0.25 * comp.shedding_period)} s)",
        f"  cross-correlation in saturation: r = {_fmt(comp.correlation['correlation'])} "
        f"at shift {_fmt(comp.correlation['shift'])} s",
        "",
        "  --- acceptance metrics ---",
        f"  |d sigma|/sigma = {_fmt(comp.metrics['rel_sigma'], '.2%')}   (criterion < 15%)",
        f"  |d St|/St       = {_fmt(comp.metrics['rel_st'], '.2%')}   (criterion < 5%)",
        f"  A0,hyb / A0,ref = {_fmt(comp.metrics['a0_ratio'], '.3g')}   (criterion > 1)",
        f"  onset error     = {_fmt(comp.metrics['onset_error_periods'], '.2f')} shedding periods (criterion < 0.25)",
        "",
        f"  VERDICT: {comp.verdict.upper()}",
    ]
    if comp.flags:
        lines.append("  flags:")
        lines.extend(f"    - {flag}" for flag in comp.flags)

    summary = {
        "seed_amplitude": seed,
        "reference": {
            "label": ref.label,
            "t_onset": ref.t_onset,
            "growth": ref.growth,
            "f_growth": ref.f_growth,
            "st": ref.st,
            "saturated": ref.saturated,
        },
        "hybrid": {
            "label": hyb.label,
            "t_onset": hyb.t_onset,
            "growth": hyb.growth,
            "f_growth": hyb.f_growth,
            "st": hyb.st,
            "saturated": hyb.saturated,
        },
        "comparison": {
            "dt_pred": comp.dt_pred,
            "dt_meas": comp.dt_meas,
            "shedding_period": comp.shedding_period,
            "correlation": comp.correlation,
            "metrics": comp.metrics,
            "verdict": comp.verdict,
            "flags": comp.flags,
        },
        "a_star": 0.25,
        "noise_floor": 1e-3,
    }
    util.SOLUTION.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines.append(f"\n  wrote {SUMMARY_PATH.relative_to(util.CASE_DIR)}")
    print("\n".join(lines))
    return summary


def main() -> None:
    reference = _load_series("reference")
    hybrid = _load_series("fvm")
    report(reference, hybrid)


if __name__ == "__main__":
    main()
