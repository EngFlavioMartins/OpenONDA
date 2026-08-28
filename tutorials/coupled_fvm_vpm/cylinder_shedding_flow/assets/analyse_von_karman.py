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
        normalized_transverse_velocity_time=probe["t"],
        normalized_transverse_velocity=probe["normalized_transverse_velocity"],
        drag_coefficient_time=forces["t"],
        drag_coefficient=forces["drag_coefficient"],
        lift_coefficient_time=forces["t"],
        lift_coefficient=forces["lift_coefficient"],
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
        f"  Case: 4D spanwise cylinder segment, Re=150  |  seed amplitude eps={seed:g}",
        "",
        f"  {'Quantity':<28}{'Reference FVM':>18}{'Coupled hybrid':>18}",
        f"  {'sigma [1/s]':<28}{_fmt(ref.growth['growth_rate']):>18}{_fmt(hyb.growth['growth_rate']):>18}",
        f"  {'initial_amplitude (u_y/U)':<28}{_fmt(ref.growth['initial_amplitude'], '.3e'):>18}{_fmt(hyb.growth['initial_amplitude'], '.3e'):>18}",
        f"  {'growth fit R^2':<28}{_fmt(ref.growth['coefficient_of_determination']):>18}{_fmt(hyb.growth['coefficient_of_determination']):>18}",
        f"  {'growth fit interval [s]':<28}{_fmt(ref.growth['fit_start_time'])}..{_fmt(ref.growth['fit_end_time']):<10}{_fmt(hyb.growth['fit_start_time'])}..{_fmt(hyb.growth['fit_end_time']):<10}",
        f"  {'growth-interval f [Hz]':<28}{_fmt(ref.growth_frequency):>18}{_fmt(hyb.growth_frequency):>18}",
        f"  {'saturated St [-D/U]':<28}{_fmt(ref.strouhal_number):>18}{_fmt(hyb.strouhal_number):>18}",
        f"  {'t* onset (A>=A*) [s]':<28}{_fmt(ref.onset_time):>18}{_fmt(hyb.onset_time):>18}",
        f"  {'mean drag coefficient (saturated)':<28}{_fmt(ref.saturated['mean_drag_coefficient']):>18}{_fmt(hyb.saturated['mean_drag_coefficient']):>18}",
        f"  {'RMS lift coefficient (saturated)':<28}{_fmt(ref.saturated['rms_lift_coefficient']):>18}{_fmt(hyb.saturated['rms_lift_coefficient']):>18}",
        f"  {'RMS q (saturated)':<28}{_fmt(ref.saturated['rms_normalized_transverse_velocity']):>18}{_fmt(hyb.saturated['rms_normalized_transverse_velocity']):>18}",
        f"  {'envelope CV (growth)':<28}{_fmt(ref.modulation):>18}{_fmt(hyb.modulation):>18}",
        "",
        "  --- onset-shift prediction ---",
        f"  predicted  dt = (1/sigma) ln(initial_amplitude,hyb/initial_amplitude,ref) = {_fmt(comp.predicted_onset_time_shift)} s",
        f"  measured   dt = t*_ref - t*_hyb           = {_fmt(comp.measured_onset_time_shift)} s",
        f"  shedding period T = 1/St = {_fmt(comp.shedding_period)} s "
        f"(tolerance {_fmt(0.25 * comp.shedding_period)} s)",
        f"  cross-correlation in saturation: r = {_fmt(comp.correlation['correlation'])} "
        f"at shift {_fmt(comp.correlation['shift'])} s",
        "",
        "  --- acceptance metrics ---",
        f"  |d sigma|/sigma = {_fmt(comp.metrics['relative_growth_rate_difference'], '.2%')}   (criterion < 15%)",
        f"  |d St|/St       = {_fmt(comp.metrics['relative_strouhal_number_difference'], '.2%')}   (criterion < 5%)",
        f"  initial_amplitude,hyb / initial_amplitude,ref = {_fmt(comp.metrics['initial_amplitude_ratio'], '.3g')}   (criterion > 1)",
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
            "onset_time": ref.onset_time,
            "growth": ref.growth,
            "growth_frequency": ref.growth_frequency,
            "strouhal_number": ref.strouhal_number,
            "saturated": ref.saturated,
        },
        "hybrid": {
            "label": hyb.label,
            "onset_time": hyb.onset_time,
            "growth": hyb.growth,
            "growth_frequency": hyb.growth_frequency,
            "strouhal_number": hyb.strouhal_number,
            "saturated": hyb.saturated,
        },
        "comparison": {
            "predicted_onset_time_shift": comp.predicted_onset_time_shift,
            "measured_onset_time_shift": comp.measured_onset_time_shift,
            "shedding_period": comp.shedding_period,
            "correlation": comp.correlation,
            "metrics": comp.metrics,
            "verdict": comp.verdict,
            "flags": comp.flags,
        },
        "onset_amplitude_threshold": 0.25,
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
