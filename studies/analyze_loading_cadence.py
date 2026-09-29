#!/usr/bin/env python3
"""Measure loading-history reconstruction error without modifying source data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.integrate import trapezoid


def load_history(path, metadata, panel):
    """Read one numeric history into bounded arrays, through its accepted clock."""
    maximum_step = int(metadata["state"]["step"])
    accepted_time = float(metadata["state"]["time"])
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    with path.open(newline="") as stream:
        reader = csv.reader(stream)
        names = next(reader)
        step_index, time_index = names.index("step"), names.index("time")
        columns = (
            ["panel_circulation", "panel_force_x", "panel_force_y", "panel_force_z"]
            if panel
            else ["force_x", "force_z", "moment_x", "moment_y", "power"]
        )
        indices = [names.index(name) for name in columns]
        keys = [names.index("station_id"), names.index("chord_index")] if panel else []
        first = next(reader)
        first_step = int(first[step_index])
        initial = [first]
        for row in reader:
            if int(row[step_index]) != first_step:
                pending = row
                break
            initial.append(row)
        else:
            raise ValueError(f"{path}: fewer than two saved times")
        identities = {tuple(row[i] for i in keys): n for n, row in enumerate(initial)}
        if len(identities) != len(initial):
            raise ValueError(f"{path}: duplicate panel identities")
        values = np.full((maximum_step + 1, len(identities), len(columns)), np.nan)
        times = np.full(maximum_step + 1, np.nan)
        excluded = 0
        for row in itertools.chain(initial, [pending], reader):
            step, time = int(row[step_index]), float(row[time_index])
            if step > maximum_step or time > accepted_time + 1e-9:
                excluded += 1
                continue
            identity = identities[tuple(row[i] for i in keys)]
            if np.isfinite(values[step, identity, 0]):
                raise ValueError(f"{path}: repeated accepted panel/step")
            values[step, identity] = [float(row[i]) for i in indices]
            times[step] = time
        keep = np.isfinite(times)
        values, times = values[keep], times[keep]
        if not np.isfinite(values).all():
            raise ValueError(f"{path}: missing or nonfinite panel samples")
        return (
            times,
            values,
            columns,
            {
                "file": path.name,
                "sha256": digest.hexdigest(),
                "accepted_times": len(times),
                "panel_count": len(identities),
                "first_time_s": float(times[0]),
                "last_time_s": float(times[-1]),
                "excluded_unaccepted_rows": excluded,
                "numeric_array_bytes": values.nbytes,
            },
        )


def evaluate(times, values, columns, factor):
    dt = float(np.median(np.diff(times)))
    if not np.allclose(np.diff(times), dt, rtol=1e-5, atol=1e-9):
        raise ValueError("cadence screening requires uniform accepted source samples")
    keep = np.unique(np.r_[np.arange(0, len(times), factor), len(times) - 1])
    nyquist = 1 / (2 * dt * factor)
    results = {}
    for column_index, column in enumerate(columns):
        channels = values[:, :, column_index]
        family = "force" if "force_" in column else "moment" if "moment_" in column else column
        family_indices = [
            index
            for index, name in enumerate(columns)
            if (family == "force" and "force_" in name)
            or (family == "moment" and "moment_" in name)
            or name == family
        ]
        reference_scale = max(
            float(np.max(np.sqrt(np.mean(values[:, :, index] ** 2, axis=0))))
            for index in family_indices
        )
        errors, peaks, integrals, spectral = [], [], [], []
        meaningful = 0
        for signal in channels.T:
            scale = float(np.sqrt(np.mean(signal**2)))
            if scale < max(reference_scale * 0.01, 1e-12):
                continue
            meaningful += 1
            reconstruction = np.interp(times, times[keep], signal[keep])
            errors.append(float(np.sqrt(np.mean((signal - reconstruction) ** 2)) / scale))
            peaks.append(
                float(
                    abs(np.max(np.abs(signal)) - np.max(np.abs(signal[keep])))
                    / max(np.max(np.abs(signal)), 1e-30)
                )
            )
            integral = trapezoid(signal, times)
            integrals.append(
                float(
                    abs(trapezoid(signal[keep], times[keep]) - integral)
                    / max(trapezoid(np.abs(signal), times), 1e-30)
                )
            )
            detrended = signal - np.polyval(np.polyfit(times, signal, 1), times)
            energy = np.abs(np.fft.rfft(detrended * np.hanning(len(times)))) ** 2
            frequencies = np.fft.rfftfreq(len(times), dt)
            spectral.append(float(energy[frequencies > nyquist].sum() / max(energy.sum(), 1e-30)))
        results[column] = {
            "significant_channels": meaningful,
            "ignored_near_zero_channels": channels.shape[1] - meaningful,
            "significance_floor_rms": max(reference_scale * 0.01, 1e-12),
            "max_relative_rms_reconstruction_error": max(errors, default=0),
            "p95_relative_rms_reconstruction_error": float(np.percentile(errors, 95))
            if errors
            else 0,
            "max_relative_absolute_peak_error": max(peaks, default=0),
            "max_integral_error_over_absolute_integral": max(integrals, default=0),
            "max_detrended_energy_above_candidate_nyquist": max(spectral, default=0),
        }
    worst = {
        metric: max(item[metric] for item in results.values())
        for metric in (
            "max_relative_rms_reconstruction_error",
            "max_relative_absolute_peak_error",
            "max_integral_error_over_absolute_integral",
            "max_detrended_energy_above_candidate_nyquist",
        )
    }
    passed = all(
        value <= limit
        for value, limit in zip(worst.values(), [0.01, 0.01, 0.005, 0.001], strict=True)
    )
    return {
        "factor": factor,
        "interval_s": dt * factor,
        "nyquist_hz": nyquist,
        "retained_time_count": len(keep),
        "columns": results,
        "worst": worst,
        "screening_decision": "passes_available_history_only" if passed else "rejected",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive_root", type=Path, help="restored root containing vpm case folders")
    parser.add_argument("--output", type=Path, default=Path("studies/loading_cadence_screen.json"))
    args = parser.parse_args()
    report = {
        "method": "linear reconstruction; endpoint retained; linear-detrended Hann periodogram",
        "limits": {
            "relative_rms": 0.01,
            "relative_peak": 0.01,
            "absolute_integral_normalization": 0.005,
            "high_frequency_energy": 0.001,
        },
        "qualification": "Finite nonstationary partial histories; no final-run cadence certification.",
        "significance": "Exclude channels below 1% of maximum RMS in the same unit family (force, moment, circulation, power). Symmetry noise is not a physical cadence requirement.",
        "cases": {},
    }
    for name, case_folder, frequency, blades, factors in (
        ("delta", "05_delta_wing", 1.0, 1, [2, 5, 10]),
        ("rotor", "06_rotor_flow", (7 * 7 / 6) / (2 * np.pi), 3, [2, 4]),
    ):
        folder = args.archive_root / "vpm" / case_folder
        metadata = json.loads((folder / "solution/vpm_metadata.json").read_text())
        samples = folder / "samples" / ("delta_wing" if name == "delta" else "rotor")
        item = {
            "accepted_clock_s": metadata["state"]["time"],
            "fundamental_hz": frequency,
            "blade_passage_hz": frequency * blades,
            "histories": [],
        }
        for path in [samples / "vlm_forces.csv", *sorted(samples.glob("vlm_chordwise_*.csv"))]:
            times, values, columns, evidence = load_history(
                path, metadata, "chordwise" in path.name
            )
            evidence["physical_cycles_covered"] = (times[-1] - times[0]) * frequency
            evidence["candidates"] = [
                evaluate(times, values, columns, factor) for factor in factors
            ]
            item["histories"].append(evidence)
            print(name, path.name, len(times), values.shape[1], flush=True)
            del values
        report["cases"][name] = item
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
