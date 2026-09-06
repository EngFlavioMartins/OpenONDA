"""Fail-closed cycle statistics and a spatial/temporal/iterative error budget."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .postprocess import statistics, time_mean, richardson_gci
from ..case_definition import CONTROL_DOMAINS, REFINEMENT_RATIO

METRICS = ("mean_cd", "cd_rms", "cd_peak_to_peak", "cl_rms", "cl_amplitude", "strouhal")


def read_history(path):
    with Path(path).open() as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) < 4 or any(row.get("patch", "cylinder") != "cylinder" for row in rows):
        raise ValueError("Missing cylinder-only force history")
    history = {
        key: np.array([float(row[key]) for row in rows])
        for key in ("time", "drag_coefficient", "lift_coefficient")
    }
    if not all(np.isfinite(value).all() for value in history.values()) or np.any(
        np.diff(history["time"]) <= 0
    ):
        raise ValueError("Nonfinite values or non-increasing force times (restart seam)")
    return history


def window(history, start, end):
    time = history["time"]
    if start < time[0] - 1e-9 or end > time[-1] + 1e-9 or end <= start:
        raise ValueError("Incomplete statistics interval")
    selected = np.r_[start, time[(time > start) & (time < end)], end]
    return {
        "time": selected,
        **{
            key: np.interp(selected, time, values)
            for key, values in history.items()
            if key != "time"
        },
    }


def basic_statistics(history):
    t, cd, cl = (history[key] for key in ("time", "drag_coefficient", "lift_coefficient"))
    mean_cd, mean_cl = time_mean(t, cd), time_mean(t, cl)
    return {
        "mean_cd": mean_cd,
        "mean_cl": mean_cl,
        "cd_rms": float(np.sqrt(time_mean(t, (cd - mean_cd) ** 2))),
        "cl_rms": float(np.sqrt(time_mean(t, (cl - mean_cl) ** 2))),
        "cd_peak_to_peak": float(np.ptp(cd)),
        "cl_amplitude": float(np.ptp(cl) / 2),
    }


def cycle_statistics(history, config):
    """Use the last N complete cycles after discard, with four cycle blocks."""
    end = config["end_time"]
    if history["time"][-1] < end - config["force_interval"] * 1.01:
        raise ValueError("Run has not reached the configured end time")
    h = window(history, config["discard_time"], min(end, history["time"][-1]))
    time, lift = h["time"], h["lift_coefficient"]
    if np.max(np.diff(time)) > 1.5 * config["force_interval"]:
        raise ValueError("Force sampling has gaps")
    centred = lift - time_mean(time, lift)
    if np.ptp(centred) < 1e-5:
        raise ValueError("No resolved shedding: lift signal is absent or too small")
    indices = np.flatnonzero((centred[:-1] <= 0) & (centred[1:] > 0))
    crossings = (
        time[indices] - centred[indices] * np.diff(time)[indices] / np.diff(centred)[indices]
    )
    minimum = int(config["minimum_cycles"])
    if len(crossings) < minimum + 1:
        raise ValueError(
            f"Only {max(0, len(crossings) - 1)} complete settled cycles; require {minimum}"
        )
    crossings = crossings[-minimum - 1 :]
    periods = np.diff(crossings)
    if min(periods) / np.max(np.diff(time)) < 100:
        raise ValueError("Fewer than 100 force samples per shedding cycle")
    if np.std(periods) / np.mean(periods) > 0.05:
        raise ValueError("Shedding periods are not stable within 5%")
    selected = window(h, crossings[0], crossings[-1])
    result: dict[str, Any] = dict(statistics(selected, crossings[0], crossings[-1]))
    cycle_rows = [basic_statistics(window(h, a, b)) for a, b in zip(crossings[:-1], crossings[1:])]
    # Amplitudes are averages over cycles, not maxima that grow with record length.
    for key in ("cd_peak_to_peak", "cl_amplitude"):
        result[key] = float(np.mean([row[key] for row in cycle_rows]))
    result["strouhal"] = float(1 / np.mean(periods))  # D/U = 1 in this frozen case.
    block_indices = np.array_split(np.arange(minimum), 4)
    blocks = []
    for indices in block_indices:
        block = window(h, crossings[indices[0]], crossings[indices[-1] + 1])
        row = basic_statistics(block)
        row["strouhal"] = float(len(indices) / (block["time"][-1] - block["time"][0]))
        for key in ("cd_peak_to_peak", "cl_amplitude"):
            row[key] = float(np.mean([cycle_rows[index][key] for index in indices]))
        blocks.append(row)
    uncertainty, drift = {}, {}
    for key in (*METRICS, "mean_cl"):
        values = np.array([row[key] for row in blocks])
        # Engineering 95%-style block interval (t_3=3.182), inflated for positive
        # adjacent-block correlation; four blocks cannot certify independence.
        rho = 0.0
        if np.std(values[:-1]) > 1e-14 and np.std(values[1:]) > 1e-14:
            rho = float(np.clip(np.corrcoef(values[:-1], values[1:])[0, 1], 0.0, 0.8))
        uncertainty[key] = float(
            3.182 * np.std(values, ddof=1) / 2 * np.sqrt((1 + rho) / (1 - rho))
        )
        drift[key] = float(abs(np.mean(values[:2]) - np.mean(values[2:])))
    # Decimate the exact same cycle window to detect force-sampling sensitivity.
    decimated = {
        key: value[np.unique(np.r_[np.arange(0, len(value), 2), len(value) - 1])]
        for key, value in h.items()
    }
    decimated_rows = [
        basic_statistics(window(decimated, a, b)) for a, b in zip(crossings[:-1], crossings[1:])
    ]
    coarse_stats = statistics(
        window(decimated, crossings[0], crossings[-1]), crossings[0], crossings[-1]
    )
    for key in ("cd_peak_to_peak", "cl_amplitude"):
        coarse_stats[key] = float(np.mean([row[key] for row in decimated_rows]))
    for key in (*METRICS, "mean_cl"):
        uncertainty[key] += abs(result[key] - coarse_stats[key])
    # Spectral cross-check only: finite-window resolution is not a precision St estimate.
    uniform = np.linspace(crossings[0], crossings[-1], len(selected["time"]))
    signal = np.asarray(
        np.interp(uniform, selected["time"], selected["lift_coefficient"]), dtype=float
    )
    power = abs(np.fft.rfft((signal - signal.mean()) * np.hanning(len(signal)))) ** 2
    frequencies = np.fft.rfftfreq(len(signal), uniform[1] - uniform[0])
    peak = float(frequencies[1 + np.argmax(power[1:])])
    reasons = []
    if abs(peak - result["strouhal"]) > 2 / (crossings[-1] - crossings[0]):
        reasons.append("Zero-crossing and spectral shedding frequencies disagree")
    for key in METRICS:
        allowance = abs(result[key]) * config["tolerance_percent"][key] / 100
        if allowance <= 1e-12:
            reasons.append(f"{key}: near-zero metric needs an explicit absolute tolerance")
        elif max(drift[key], uncertainty[key]) > 0.2 * allowance:
            reasons.append(f"{key}: sampling uncertainty or cycle-block drift exceeds budget")
    if (
        max(drift["mean_cl"], uncertainty["mean_cl"], abs(result["mean_cl"]))
        > config["mean_cl_absolute_tolerance"]
    ):
        reasons.append("Mean Cl bias/drift/uncertainty exceeds absolute tolerance")
    result.update(
        {
            "cycles": minimum,
            "window": {"start": float(crossings[0]), "end": float(crossings[-1])},
            "sampling_uncertainty": uncertainty,
            "block_drift": drift,
            "blocks": blocks,
            "spectral_strouhal": peak,
            "qualified": not reasons,
            "reasons": reasons,
        }
    )
    return result


def convergence_verdict(records, config):
    """Select medium or fine only when all eleven runs support an error budget."""
    spatial = [records[name] for name in ("coarse", "medium", "fine")]
    convergence = {
        key: richardson_gci(spatial, key, config["tolerance_percent"][key]) for key in METRICS
    }
    candidates = {}
    for name in ("medium", "fine"):
        base, half, quarter, tight = (
            records[key] for key in (name, name + "_dt2", name + "_dt4", name + "_tight")
        )
        budgets = {}
        for key in METRICS:
            fit = convergence[key]
            time_fit = richardson_gci(
                [
                    {"case": "dt", "dx": 4.0, key: base[key]},
                    {"case": "dt/2", "dx": 2.0, key: half[key]},
                    {"case": "dt/4", "dx": 1.0, key: quarter[key]},
                ],
                key,
                config["tolerance_percent"][key],
                refinement_ratio=2.0,
            )
            if time_fit["observed_order"] is not None and time_fit["observed_order"] > 0:
                # Error of the BASE time step used for spatial comparisons.
                temporal = 1.25 * abs(base[key] - time_fit["richardson_extrapolated_value"])
            else:
                # No fitted time order: conservatively retain the observed range;
                # only allow roundoff-equal time values, never oscillatory fits.
                temporal = max(abs(base[key] - half[key]), abs(base[key] - quarter[key]))
            temporal += base["sampling_uncertainty"][key] + quarter["sampling_uncertainty"][key]
            iterative = abs(tight[key] - base[key]) + tight["sampling_uncertainty"][key]
            spatial_error = None
            if fit["observed_order"] is not None and 0 < fit["observed_order"] <= 2.5:
                spatial_error = 1.25 * abs(base[key] - fit["richardson_extrapolated_value"])
            sample = base["sampling_uncertainty"][key]
            physical = {
                control: abs(records[control][key] - records["fine"][key])
                + records[control]["sampling_uncertainty"][key]
                + records["fine"]["sampling_uncertainty"][key]
                for control in CONTROL_DOMAINS
            }
            allowance = abs(base[key]) * config["tolerance_percent"][key] / 100
            signal = min(
                abs(spatial[0][key] - spatial[1][key]), abs(spatial[1][key] - spatial[2][key])
            )
            noise = max(row["sampling_uncertainty"][key] for row in spatial)
            time_ok = time_fit["status"] == "differences_unresolved" or (
                time_fit["observed_order"] is not None and 0 < time_fit["observed_order"] <= 2.5
            )
            total = (
                None
                if spatial_error is None
                else spatial_error + temporal + iterative + sample + sum(physical.values())
            )
            passed = (
                all(row["qualified"] for row in records.values())
                and allowance > 1e-12
                and spatial_error is not None
                and signal > 2 * noise
                and time_ok
                and spatial_error <= 0.4 * allowance
                and temporal <= 0.2 * allowance
                and iterative <= 0.2 * allowance
                and sample <= 0.2 * allowance
                and total <= allowance
                and all(value <= 0.2 * allowance for value in physical.values())
            )
            budgets[key] = {
                "spatial_gci_absolute": spatial_error,
                "temporal_absolute": temporal,
                "iterative_absolute": iterative,
                "sampling_absolute": sample,
                "domain_absolute": physical["fine_domain"],
                "span_absolute": physical["fine_span"],
                "total_estimate": total,
                "allowance": allowance,
                "temporal_fit": time_fit,
                "spatial_difference_resolved": signal > 2 * noise,
                "passed": bool(passed),
                "checks": {
                    "all_runs_stationary": all(row["qualified"] for row in records.values()),
                    "spatial_fit_usable": spatial_error is not None and signal > 2 * noise,
                    "temporal_fit_usable": bool(time_ok),
                    "spatial_budget": spatial_error is not None
                    and spatial_error <= 0.4 * allowance,
                    "temporal_budget": temporal <= 0.2 * allowance,
                    "iterative_budget": iterative <= 0.2 * allowance,
                    "sampling_budget": sample <= 0.2 * allowance,
                    "domain_sensitivity": physical["fine_domain"] <= 0.2 * allowance,
                    "span_sensitivity": physical["fine_span"] <= 0.2 * allowance,
                },
            }
        mean_cl_ok = all(
            abs(row["mean_cl"] - base["mean_cl"]) + row["sampling_uncertainty"]["mean_cl"]
            <= config["mean_cl_absolute_tolerance"]
            for row in (
                half,
                quarter,
                tight,
                records["fine"],
                *(records[key] for key in CONTROL_DOMAINS),
            )
        )
        candidates[name] = {
            "passed": all(row["passed"] for row in budgets.values()) and mean_cl_ok,
            "mean_cl_consistent": mean_cl_ok,
            "metrics": budgets,
        }
    selected = next((name for name in ("medium", "fine") if candidates[name]["passed"]), None)
    return convergence, candidates, selected


def report_campaign(output, failure=None):
    from ..study import write_json, sha256, verify_mesh, force_digest

    output = Path(output)
    campaign = json.loads((output / "campaign.json").read_text())
    config = campaign["config"]
    records, reasons, mesh_records = {}, [], {}
    if failure:
        reasons.append(failure)
    for grid in ("coarse", "medium", "fine", *CONTROL_DOMAINS):
        directory = output / "meshes" / grid
        try:
            verify_mesh(directory, grid)
            independent = json.loads((directory / "independent_check.json").read_text())
            if (
                not independent["passed"]
                or independent["mesh_sha256"] != sha256(directory / "mesh.npz")
                or independent["log_sha256"] != sha256(directory / "checkMesh.log")
            ):
                raise ValueError("Missing/mismatched independent quality acceptance")
            data = json.loads((directory / "mesh_report.json").read_text())
            generation = data["mesh_generation"]
            mesh_records[grid] = {
                "identity": data["mesh_identity"],
                "archive_sha256": sha256(directory / "mesh.npz"),
                "cells": int(data["counts"]["cells"]),
                "h": float(generation["resolved_background_cell_size"]),
                "wall": float(generation["resolved_surface_patch_sizes"]["cylinder"]),
            }
            del data
        except (OSError, ValueError, KeyError) as exc:
            reasons.append(f"{grid} mesh: {exc}")
    if all(name in mesh_records for name in ("coarse", "medium", "fine")):
        if len({mesh_records[name]["identity"] for name in ("coarse", "medium", "fine")}) != 3:
            reasons.append("Duplicate mesh topology/coordinates")
        for key in ("h", "wall"):
            values = np.array([mesh_records[name][key] for name in ("coarse", "medium", "fine")])
            if not np.allclose(values[:-1] / values[1:], REFINEMENT_RATIO, rtol=1e-9, atol=0):
                reasons.append(
                    f"Realized {key} sizes do not form the registered r={REFINEMENT_RATIO} family"
                )
    for spec in campaign["runs"]:
        name, grid = spec["name"], spec["grid"]
        try:
            directory = output / "runs" / name
            progress = json.loads((directory / "progress.json").read_text())
            if (
                not progress["completed"]
                or progress["time"] < config["end_time"] - 1e-8
                or progress["spec"] != spec
            ):
                raise ValueError("Run incomplete or different numerical configuration")
            if (
                grid not in mesh_records
                or progress["mesh_sha256"] != mesh_records[grid]["archive_sha256"]
            ):
                raise ValueError("Run/mesh provenance mismatch")
            history_path = output / "samples" / name / "forces_history.csv"
            if progress["force_sha256"] != force_digest(history_path):
                raise ValueError("Force history checksum mismatch")
            health = progress["health"]
            if (
                not health["all_finite"]
                or not health["all_linear_converged"]
                or health["max_courant"] > 0.9
                or health["max_continuity"] > 1e-4
                or health["max_residual"] > 1e-4
            ):
                raise ValueError(
                    "Run health exceeds study limits (Courant .9, continuity/residual 1e-4)"
                )
            row = cycle_statistics(read_history(history_path), config)
            row.update(
                {
                    "case": name,
                    "dx": spec["dx"],
                    "effective_h": mesh_records[grid]["h"],
                    "mesh_cells": mesh_records[grid]["cells"],
                    "dt": spec["dt"],
                    "health": health,
                }
            )
            records[name] = row
            reasons.extend(f"{name}: {reason}" for reason in row["reasons"])
        except (OSError, ValueError, KeyError) as exc:
            reasons.append(f"{name}: {exc}")
    convergence, candidates, selected = {}, {}, None
    if len(records) == len(campaign["runs"]):
        convergence, candidates, selected = convergence_verdict(records, config)
        if selected is None:
            reasons.append("No mesh meets every spatial/temporal/iterative/sampling error budget")
    if reasons:
        selected = None
    report = {
        "schema_version": 2,
        "status": "passed" if selected else "inconclusive",
        "grid_independent": selected is not None,
        "selected_mesh": selected,
        "reasons": reasons,
        "config": config,
        "production_cases": ["coarse", "medium", "fine"],
        "cases": list(records.values()),
        "meshes": mesh_records,
        "grid_convergence": convergence,
        "candidate_meshes": candidates,
        "scope": "Grid independence of listed force statistics for this fixed domain, geometry and model; not physical validation. Coarse is a convergence anchor, not a qualified selection.",
        "uncertainty_note": "Conservative engineering estimates; block intervals are not rigorous confidence bounds or independent proof of the asymptotic range.",
    }
    write_json(output / "grid_study.json", report)
    lines = [
        "# Cylinder mesh convergence",
        "",
        f"Status: **{report['status']}**. Selected mesh: **{selected or 'none'}**.",
        "",
        report["scope"],
        "",
        "| Run | Cells | mean Cd | Cd RMS | mean Cl | Cl RMS | Cl amplitude | St | Cycles |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in records.items():
        lines.append(
            f"| {name} | {row['mesh_cells']} | {row['mean_cd']:.6g} | {row['cd_rms']:.6g} | {row['mean_cl']:.6g} | {row['cl_rms']:.6g} | {row['cl_amplitude']:.6g} | {row['strouhal']:.6g} | {row['cycles']} |"
        )
    if candidates:
        lines += [
            "",
            "## Candidate decisions",
            "",
            "| Mesh | Metric | Decision | Unmet checks |",
            "|---|---|---|---|",
        ]
        for name, candidate in candidates.items():
            for metric, budget in candidate["metrics"].items():
                unmet = ", ".join(key for key, passed in budget["checks"].items() if not passed)
                lines.append(
                    f"| {name} | {metric} | {'pass' if budget['passed'] else 'inconclusive'} | {unmet or 'none'} |"
                )
    lines += ["", "## Qualification / next action", ""] + (
        [f"- {reason}" for reason in reasons]
        or [f"Use {selected} at the registered dt for these observables and tolerances."]
    )
    lines += [
        "",
        "See grid_study.json for per-metric budgets and figures/ for plots.",
        "",
        report["uncertainty_note"],
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    return report
