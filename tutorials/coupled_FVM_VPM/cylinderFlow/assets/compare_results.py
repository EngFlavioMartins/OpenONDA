#!/usr/bin/env python3
"""Write quantitative reference-versus-hybrid cylinder comparison metrics."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _plotutil as util  # noqa: E402


def _relative_error(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return 100.0 * (left - right) / np.maximum(np.abs(right), 1.0e-8)


def _profile_rms(source: str, name: str, time: float, reference: dict) -> float:
    frame = util.load_line(source, name, time)
    if frame is None:
        return float("nan")
    coordinate = "y" if name.startswith("section") else "x"
    x = np.asarray(frame[coordinate], dtype=float)
    ref_x = np.asarray(reference[coordinate], dtype=float)
    order = np.argsort(ref_x)
    ref_u = np.interp(x, ref_x[order], np.asarray(reference["Ux"])[order])
    valid = (x >= np.min(ref_x)) & (x <= np.max(ref_x))
    if name == "centerline":
        valid &= np.abs(x) > 0.55
    return float(100.0 * np.sqrt(np.mean((np.asarray(frame["Ux"])[valid] - ref_u[valid]) ** 2)))


def main() -> None:
    reference = util.load_forces("reference")
    hybrid = util.load_forces("fvm")
    if reference is None or hybrid is None:
        raise SystemExit("Both force histories are required for comparison.")
    start = max(float(reference["time"][0]), float(hybrid["time"][0]))
    end = min(float(reference["time"][-1]), float(hybrid["time"][-1]))
    time = np.linspace(start, end, max(2, int(round((end - start) / 0.1)) + 1))
    cd_ref = np.interp(time, reference["time"], reference["Cd"])
    cl_ref = np.interp(time, reference["time"], reference["Cl"])
    cd_hybrid = np.interp(time, hybrid["time"], hybrid["Cd"])
    cl_hybrid = np.interp(time, hybrid["time"], hybrid["Cl"])

    comparison_split = util.SHEDDING_START if util.SHEDDING_START < end else 0.5 * (start + end)
    pre = time < comparison_split
    post = ~pre
    cd_error = _relative_error(cd_hybrid, cd_ref)
    ref_metrics = util.settled_force_metrics(reference, comparison_split)
    hybrid_metrics = util.settled_force_metrics(hybrid, comparison_split)

    def percent_difference(key: str) -> float:
        baseline = abs(float(ref_metrics[key]))
        if not np.isfinite(baseline) or baseline < 1.0e-12:
            return float("nan")
        return 100.0 * (float(hybrid_metrics[key]) - float(ref_metrics[key])) / baseline

    force_metrics = {
        "comparison_split_time": comparison_split,
        "pre_shedding_cd_mape_percent": float(np.mean(np.abs(cd_error[pre])))
        if np.any(pre)
        else float("nan"),
        "post_shedding_cd_mape_percent": float(np.mean(np.abs(cd_error[post])))
        if np.any(post)
        else float("nan"),
        "post_shedding_cl_rms_difference": float(
            np.sqrt(np.mean((cl_hybrid[post] - cl_ref[post]) ** 2))
        )
        if np.any(post)
        else float("nan"),
        "settled_mean_cd_difference_percent": percent_difference("mean_cd"),
        "settled_cl_rms_difference_percent": percent_difference("rms_cl"),
        "strouhal_difference_percent": percent_difference("strouhal"),
        "reference": ref_metrics,
        "hybrid": hybrid_metrics,
    }

    available = util.common_times(
        util.line_times("reference", "centerline"),
        util.line_times("fvm", "centerline"),
        util.line_times("vpm", "centerline"),
    )
    profile_metrics: dict[str, dict] = {}
    for sample_time in util.plot_times(available):
        row = {}
        for name in ("centerline", "offaxis_y075", "section_x100", "section_x200"):
            ref = util.load_line("reference", name, float(sample_time))
            if ref is None:
                continue
            row[name] = {
                "hybrid_fvm_rms_percent": _profile_rms("fvm", name, float(sample_time), ref),
                "hybrid_vpm_rms_percent": _profile_rms("vpm", name, float(sample_time), ref),
            }
        profile_metrics[f"{sample_time:.1f}"] = row

    path = util.write_metrics({"forces": force_metrics, "velocity_profiles": profile_metrics})
    print(f"  wrote {path.relative_to(util.CASE_DIR)}")
    print(
        f"  Cd MAPE: pre-shedding={force_metrics['pre_shedding_cd_mape_percent']:.3f}%, "
        f"post-shedding={force_metrics['post_shedding_cd_mape_percent']:.3f}%"
    )


if __name__ == "__main__":
    main()
