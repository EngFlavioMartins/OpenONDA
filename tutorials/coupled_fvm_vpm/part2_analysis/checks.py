#!/usr/bin/env python3
"""Checks of the Part II analysis, run before any result is analysed.

1. A synthetic signal with a known mean, three harmonics and noise is recovered
   within 0.5 %, and a 2 % change of one amplitude is reported as a difference.
2. Comparing a run with itself gives zero differences.
3. A force equal to the body velocity has a phase of 90 degrees relative to the
   displacement and does positive work per cycle.

Writes ``checks.json`` next to this file and exits non-zero when a check fails.
"""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import signals  # noqa: E402
from summary import compare, statistic, write_json  # noqa: E402

FREQUENCY = 0.2  # Hz, the pitch frequency of case 05
MEAN = 0.35
AMPLITUDES = (1.2, 0.3, 0.08)  # harmonics 1, 2, 3
PHASES = (25.0, -60.0, 140.0)  # degrees relative to sin(2 pi f t)
NOISE = 0.01  # standard deviation of white noise
SAMPLE_STEP = 0.025  # s, the force interval of case 05
N_CYCLES = 6


def synthetic(amplitudes=AMPLITUDES, seed=11):
    time = np.arange(0.0, (N_CYCLES + 2) / FREQUENCY, SAMPLE_STEP)
    signal = np.full_like(time, MEAN)
    for k, (amplitude, phase) in enumerate(zip(amplitudes, PHASES, strict=True), start=1):
        signal += amplitude * np.sin(2.0 * np.pi * k * FREQUENCY * time + np.radians(phase))
    signal += np.random.default_rng(seed).normal(0.0, NOISE, len(time))
    return time, signal


def summarise(time, signal):
    window = signals.motion_window(1.0 / FREQUENCY, 1.0 / FREQUENCY, time[-1])
    cycles = signals.cycle_statistics(time, signal, window)
    mean, amplitudes, phases, _ = signals.harmonics(time, signal, window, FREQUENCY)
    # Per-cycle harmonic amplitudes give the standard errors of the window fit.
    per_cycle = np.array(
        [
            signals.harmonics(time, signal, signals.Window(window.edges[i : i + 2]), FREQUENCY)[1]
            for i in range(window.n_cycles)
        ]
    )
    summary = {"mean": statistic(mean, signals.standard_error(cycles["mean"]))}
    for k in range(3):
        summary[f"harmonic_{k + 1}_amplitude"] = statistic(
            amplitudes[k], signals.standard_error(per_cycle[:, k])
        )
        summary[f"harmonic_{k + 1}_phase"] = statistic(phases[k], unit="deg")
    return window, summary


def check_synthetic_signal():
    time, signal = synthetic()
    window, summary = summarise(time, signal)
    expected = {"mean": MEAN} | {
        f"harmonic_{k + 1}_amplitude": amplitude for k, amplitude in enumerate(AMPLITUDES)
    }
    errors = {
        name: abs(summary[name]["value"] - value) / abs(value) for name, value in expected.items()
    }
    recovered = all(error <= 0.005 for error in errors.values())
    changed_amplitudes = (AMPLITUDES[0], AMPLITUDES[1] * 1.02, AMPLITUDES[2])
    _, changed = summarise(*synthetic(changed_amplitudes))
    rows = compare(summary, changed)
    row = rows["harmonic_2_amplitude"]
    ratio = abs(row["difference"]) / row["combined_standard_error"]
    detected = ratio > 3.0 and abs(row["difference"] / row["first"] - 0.02) < 0.005
    return {
        "window": window.describe(),
        "relative_errors": errors,
        "recovered_within_0.5_percent": recovered,
        "two_percent_change": row,
        "difference_over_combined_standard_error": ratio,
        "reported_as_difference": detected,
        "passed": bool(recovered and detected),
    }


def check_run_against_itself():
    time, signal = synthetic()
    _, summary = summarise(time, signal)
    rows = compare(summary, summary)
    zero = all(row["difference"] == 0.0 for row in rows.values())
    return {"statistics": len(rows), "all_differences_zero": zero, "passed": bool(zero)}


def check_velocity_force():
    omega = 2.0 * np.pi * FREQUENCY
    time = np.arange(0.0, (N_CYCLES + 1) / FREQUENCY, SAMPLE_STEP)
    displacement = 0.2 * np.sin(omega * time)
    velocity = 0.2 * omega * np.cos(omega * time)
    force = velocity.copy()
    window = signals.motion_window(1.0 / FREQUENCY, 0.0, time[-1])
    _, _, force_phase, _ = signals.harmonics(time, force, window, FREQUENCY, count=1)
    _, _, displacement_phase, _ = signals.harmonics(time, displacement, window, FREQUENCY, count=1)
    phase = float((force_phase[0] - displacement_phase[0] + 180.0) % 360.0 - 180.0)
    work = signals.work_per_cycle(time, force, velocity, window)
    frequency, cross = signals.cross_spectrum(time, displacement, force, window)
    peak = int(np.argmax(np.abs(cross)))
    # The phase of the cross-spectrum S_xF is that of F relative to x.
    spectral_phase = float(np.degrees(np.angle(cross[peak])))
    passed = abs(phase - 90.0) < 0.5 and np.all(work > 0.0) and abs(abs(spectral_phase) - 90.0) < 2.0
    return {
        "phase_from_harmonics_deg": phase,
        "phase_from_cross_spectrum_deg": spectral_phase,
        "work_per_cycle_J": work.tolist(),
        "passed": bool(passed),
    }


if __name__ == "__main__":
    results = {
        "synthetic_signal": check_synthetic_signal(),
        "run_against_itself": check_run_against_itself(),
        "force_equal_to_velocity": check_velocity_force(),
    }
    write_json(Path(__file__).resolve().parent / "checks.json", results)
    for name, result in results.items():
        print(f"{name}: {'passed' if result['passed'] else 'FAILED'}")
    sys.exit(0 if all(result["passed"] for result in results.values()) else 1)
