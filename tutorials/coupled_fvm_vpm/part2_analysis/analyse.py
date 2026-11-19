#!/usr/bin/env python3
"""Analyse one Part II run and write ``analysis/<run>/`` next to its case.

python analyse.py <run> [--samples DIR] [--comparison DIR] [--solution DIR] [--end-time T]

Runs: S1 (case 05 against its reference), S2_B1 and S2_B2 (case 04 against its
references), S3 (case 06 against the VLM rotor), S4a (case 07, both cylinders
fixed, against its reference) and S4b (case 07, the downstream cylinder heaving;
no comparison). Every number comes from a named column of a delivered sample or
diagnostics file; ``manifest.json`` lists the files and their hashes.

Files written: manifest.json, summary.json, cycles.csv, phase_average.csv,
harmonics.csv, spectra.csv, wake_profiles.csv, circulation_budget.csv,
diagnostics_summary.json, diagnostics_by_phase.csv and, when the run has a
comparison, compare_<run>_vs_<comparison>.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import signals  # noqa: E402
from summary import compare, statistic, write_json  # noqa: E402

TUTORIALS = Path(__file__).resolve().parents[1]
REPOSITORY = TUTORIALS.parents[1]


# Run definitions


@dataclass
class Body:
    """One body of a run: its force file and the force columns that are analysed."""

    name: str
    forces: str  # sample file of the coupled run, relative to the samples directory
    comparison_forces: str | None = None
    patch: str | None = None  # rows of this patch only, for files with several patches
    motion: str | None = None
    comparison_motion: str | None = None


@dataclass
class Line:
    """A wake line: its sample file in the coupled run and in the comparison."""

    name: str
    coupled: str
    comparison: str | None


@dataclass
class Run:
    name: str
    case: str
    kind: str  # "pitching", "elastic", "rotor", "fixed", "heaving"
    samples: str
    solution: str
    comparison_name: str | None
    comparison_samples: str | None
    bodies: list[Body]
    lines: list[Line] = field(default_factory=list)
    frequency: float | None = None  # Hz, prescribed motion
    end_time: float = 0.0
    window_periods: float | None = None  # statistics over the last periods
    start_after: float | None = None  # or over whole cycles after this time
    length: float = 1.0  # m, body length for Strouhal numbers and positions
    speed: float = 1.0  # m/s, freestream
    span: float = 1.0  # m, depth of the slab
    section_area: float = 0.0  # m^2, for the circulation of a turning body
    natural_period: float | None = None  # s, elastic cylinder


def naca_section_area(thickness: float, chord: float = 1.0, cut: float = 1.0) -> float:
    """Area of a closed-trailing-edge NACA four-digit section up to the chord fraction ``cut``."""
    x = np.linspace(0.0, cut, 20001)
    half = (
        5.0
        * thickness
        * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1036 * x**4)
    )
    return float(2.0 * np.trapezoid(half, x) * chord**2)


def runs() -> dict[str, Run]:
    c04 = "coupled_fvm_vpm/04_oscillating_cylinder"
    definitions = {
        "S1": Run(
            name="S1",
            case="coupled_fvm_vpm/05_pitching_airfoil",
            kind="pitching",
            samples="samples",
            solution="solution",
            comparison_name="reference_flow",
            comparison_samples="reference_flow/samples",
            bodies=[
                Body(
                    "airfoil",
                    "forces_history.csv",
                    "forces_history.csv",
                    motion="motion_history.csv",
                    comparison_motion="motion_history.csv",
                )
            ],
            lines=[
                Line("x2", "vpm_transverse_x2.csv", "transverse_x2.csv"),
                Line("x4", "vpm_transverse_x4.csv", "transverse_x4.csv"),
            ],
            frequency=0.2,
            end_time=50.0,
            window_periods=6,
            span=0.025,
            section_area=naca_section_area(0.12),
        ),
        "S3": Run(
            name="S3",
            case="coupled_fvm_vpm/06_rotor_flow",
            kind="rotor",
            samples="samples",
            solution="solution",
            comparison_name="vlm_rotor",
            comparison_samples="../../vpm/06_rotor_flow/samples/rotor",
            bodies=[
                Body(f"blade_{index}", "blade_forces.csv", None, patch=f"blade_{index}")
                for index in range(3)
            ],
            lines=[Line(f"wake_{d}D", f"wake_{d}D.csv", f"wake_{d}D.csv") for d in (0, 1, 2)]
            + [
                Line(f"streamwise_r{s}", f"streamwise_r{s}.csv", f"streamwise_r{s}.csv")
                for s in ("025", "065")
            ],
            frequency=7.0 * 7.0 / 6.0 / (2.0 * math.pi),
            end_time=15.0,
            window_periods=6,
            length=12.0,
            speed=7.0,
        ),
        "S4a": Run(
            name="S4a",
            case="coupled_fvm_vpm/07_tandem_cylinders",
            kind="fixed",
            samples="samples/fixed",
            solution="solution/fixed",
            comparison_name="reference_flow",
            comparison_samples="reference_flow/samples/fixed",
            bodies=[
                Body(name, f"{name}/forces_history.csv", f"forces_{name}.csv")
                for name in ("upstream", "downstream")
            ],
            lines=[
                Line(
                    f"{name}_x{d}D", f"{name}_transverse_x{d}D.csv", f"{name}_transverse_x{d}D.csv"
                )
                for name in ("upstream", "downstream")
                for d in (2, 4)
            ],
            end_time=200.0,
            start_after=100.0,
            span=0.04,
        ),
        "S4b": Run(
            name="S4b",
            case="coupled_fvm_vpm/07_tandem_cylinders",
            kind="heaving",
            samples="samples/heaving",
            solution="solution/heaving",
            comparison_name=None,
            comparison_samples=None,
            bodies=[
                Body("upstream", "upstream/forces_history.csv"),
                Body(
                    "downstream",
                    "downstream/forces_history.csv",
                    motion="downstream/motion_history.csv",
                ),
            ],
            lines=[
                Line(f"{name}_x{d}D", f"{name}_transverse_x{d}D.csv", None)
                for name in ("upstream", "downstream")
                for d in (2, 4)
            ],
            end_time=200.0,
            start_after=100.0,
            span=0.04,
        ),
    }
    for label, (reynolds, reduced_velocity) in {"B1": (6400, 7.68), "B2": (8300, 9.96)}.items():
        variant = f"re_{reynolds:g}_vstar_{reduced_velocity:g}"
        definitions[f"S2_{label}"] = Run(
            name=f"S2_{label}",
            case=c04,
            kind="elastic",
            samples=f"samples/{variant}",
            solution=f"solution/{variant}",
            comparison_name="reference_flow",
            comparison_samples=f"reference_flow/samples/{variant}",
            bodies=[
                Body(
                    "cylinder",
                    "forces_history.csv",
                    "forces_history.csv",
                    motion="motion_history.csv",
                    comparison_motion="motion_history.csv",
                )
            ],
            lines=[
                Line(f"x{d}D", f"vpm_transverse_x{d}D.csv", f"transverse_x{d}D.csv") for d in (2, 4)
            ],
            natural_period=reduced_velocity,
            end_time=25 * reduced_velocity,
            window_periods=10,
            span=0.04,
        )
    return definitions


# Reading


def accepted(frame: pd.DataFrame) -> pd.DataFrame:
    """Rows of the accepted history: a time written again after a restart replaces the earlier row."""
    keys = ["time"] + (["patch"] if "patch" in frame else [])
    return frame.drop_duplicates(keys, keep="last").sort_values(keys).reset_index(drop=True)


def read_table(path: Path, patch: str | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if patch is not None and "patch" in frame:
        frame = frame[frame["patch"] == patch]
    return accepted(frame)


def read_line(path: Path) -> dict[str, np.ndarray]:
    """Line or plane samples as arrays ``(n_times, n_points)`` per column."""
    frame = pd.read_csv(path)
    frame = frame.drop_duplicates(["time", "position_x", "position_y", "position_z"], keep="last")
    times = np.sort(frame["time"].unique())
    first = frame[frame["time"] == times[0]]
    n_points = len(first)
    frame = frame.sort_values(["time"], kind="stable")
    counts = frame.groupby("time").size()
    complete = counts[counts == n_points].index.to_numpy()
    frame = frame[frame["time"].isin(complete)]
    result = {"time": complete}
    for column in frame.columns:
        if column == "time" or frame[column].dtype == object:
            continue
        result[column] = frame[column].to_numpy().reshape(len(complete), n_points)
    return result


def read_jsonl(path: Path) -> list[dict]:
    rows = {}
    with path.open() as stream:
        for line in stream:
            line = line.strip()
            if line:
                record = json.loads(line)
                rows[float(record.get("time", len(rows)))] = record  # a restart rewrites a time
    return [rows[key] for key in sorted(rows)]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# Windows and periods


def statistics_window(run: Run, time, reference, motion_frequency) -> signals.Window:
    """Whole cycles of the statistics window of ``run``."""
    end = min(run.end_time, float(time[-1]))
    if run.kind in ("pitching", "rotor", "heaving") and motion_frequency:
        period = 1.0 / motion_frequency
        start = end - run.window_periods * period if run.window_periods else run.start_after
        return signals.motion_window(period, start - 1.0e-9, end + 1.0e-9)
    if run.window_periods:
        period = run.natural_period
        start = end - run.window_periods * period
    else:
        start = run.start_after
    window = signals.upcrossing_window(time, reference, start)
    return window


# Analysis of one body


FORCE_COLUMNS = (
    "total_force_x",
    "total_force_y",
    "pressure_force_x",
    "pressure_force_y",
    "viscous_force_x",
    "viscous_force_y",
    "moment_z",
    "drag_coefficient",
    "lift_coefficient",
    "pitching_moment_coefficient",
)
ROTOR_COLUMNS = (
    "total_force_x",
    "total_force_y",
    "total_force_z",
    "moment_x",
    "moment_y",
    "moment_z",
    "rotational_power",
)


def signal_rows(
    prefix, time, values, window, frequency, summary, cycles, phases, harmonic_rows, spectra
):
    """Statistics of one signal into the summary and the tables."""
    stats = signals.window_statistics(time, values, window)
    summary[f"{prefix}.mean"] = statistic(stats["mean"], stats["mean_standard_error"])
    summary[f"{prefix}.std"] = statistic(stats["std"], stats["std_standard_error"])
    summary[f"{prefix}.cycle_max"] = statistic(
        stats["cycle_max_mean"], stats["cycle_max_standard_error"]
    )
    summary[f"{prefix}.cycle_min"] = statistic(
        stats["cycle_min_mean"], stats["cycle_min_standard_error"]
    )
    per_cycle = signals.cycle_statistics(time, values, window)
    for index in range(window.n_cycles):
        cycles.append(
            {
                "signal": prefix,
                "cycle": index,
                "start": window.edges[index],
                "end": window.edges[index + 1],
                **{name: float(per_cycle[name][index]) for name in per_cycle},
            }
        )
    average = signals.phase_average(time, values, window)
    for b, centre in enumerate(average["phase_centre"]):
        phases.append(
            {
                "signal": prefix,
                "phase": centre,
                "mean": average["mean"][b],
                "cycle_std": average["cycle_std"][b],
            }
        )
    if frequency:
        mean, amplitudes, angles, residual = signals.harmonics(time, values, window, frequency)
        per_cycle_amplitudes = np.array(
            [
                signals.harmonics(time, values, signals.Window(window.edges[i : i + 2]), frequency)[
                    1
                ]
                for i in range(window.n_cycles)
            ]
        )
        for k, (amplitude, angle) in enumerate(zip(amplitudes, angles, strict=True), start=1):
            error = signals.standard_error(per_cycle_amplitudes[:, k - 1])
            harmonic_rows.append(
                {
                    "signal": prefix,
                    "harmonic": k,
                    "frequency": k * frequency,
                    "amplitude": amplitude,
                    "amplitude_standard_error": error,
                    "phase_deg": angle,
                    "fit_residual_rms": residual,
                }
            )
            summary[f"{prefix}.harmonic_{k}_amplitude"] = statistic(amplitude, error)
            summary[f"{prefix}.harmonic_{k}_phase"] = statistic(angle, unit="deg")
    frequency_axis, psd, resolution = signals.welch_spectrum(time, values, window)
    peak = signals.peak_frequency(frequency_axis, psd, minimum=0.5 * resolution)
    summary[f"{prefix}.spectral_peak_frequency"] = statistic(peak, resolution / 2.0, "Hz")
    for f, p in zip(frequency_axis, psd, strict=True):
        spectra.append({"signal": prefix, "frequency": f, "psd": p, "resolution": resolution})
    return stats


def analyse_side(run: Run, side: str, samples: Path, tables: dict, files: list[Path]) -> dict:
    """Statistics of one side (coupled or comparison) of a run; returns its summary."""
    summary: dict = {}
    cycles, phases, harmonic_rows, spectra = (
        tables[side][name] for name in ("cycles", "phases", "harmonics", "spectra")
    )
    data = {}
    for body in run.bodies:
        name = body.forces if side == "coupled" else body.comparison_forces
        if name is None:
            continue
        path = samples / name
        files.append(path)
        forces = read_table(path, body.patch)
        motion = None
        motion_name = body.motion if side == "coupled" else body.comparison_motion
        if motion_name:
            files.append(samples / motion_name)
            motion = read_table(samples / motion_name)
        data[body.name] = (forces, motion)

    # The cycle reference: prescribed motion, cross-flow displacement or lift
    motion_frequency = run.frequency
    first_forces, first_motion = next(iter(data.values()))
    if run.kind == "elastic":
        reference_time = first_motion["time"].to_numpy()
        reference = first_motion["displacement_y"].to_numpy()
    elif run.kind == "fixed":
        downstream = data.get("downstream", next(iter(data.values())))[0]
        reference_time = downstream["time"].to_numpy()
        reference = downstream["lift_coefficient"].to_numpy()
    else:
        reference_time = first_forces["time"].to_numpy()
        reference = None
    window = statistics_window(run, reference_time, reference, motion_frequency)
    summary["window"] = window.describe()

    response_frequency = None
    if run.kind == "elastic":
        f, psd, resolution = signals.welch_spectrum(reference_time, reference, window)
        response_frequency = signals.peak_frequency(f, psd, minimum=0.5 * resolution)
        summary["frequency_ratio_y"] = statistic(
            response_frequency * run.natural_period, resolution * run.natural_period / 2
        )
    if run.kind == "fixed":
        f, psd, resolution = signals.welch_spectrum(reference_time, reference, window)
        response_frequency = signals.peak_frequency(f, psd, minimum=0.5 * resolution)
    harmonic_frequency = motion_frequency or response_frequency

    columns = ROTOR_COLUMNS if run.kind == "rotor" else FORCE_COLUMNS
    for body_name, (forces, motion) in data.items():
        time = forces["time"].to_numpy()
        for column in columns:
            if column in forces:
                signal_rows(
                    f"{body_name}.{column}",
                    time,
                    forces[column].to_numpy(),
                    window,
                    harmonic_frequency,
                    summary,
                    cycles,
                    phases,
                    harmonic_rows,
                    spectra,
                )
        if run.kind == "fixed" and "lift_coefficient" in forces:
            f, psd, resolution = signals.welch_spectrum(
                time, forces["lift_coefficient"].to_numpy(), window
            )
            peak = signals.peak_frequency(f, psd, minimum=0.5 * resolution)
            summary[f"{body_name}.strouhal_number"] = statistic(
                peak * run.length / run.speed, resolution * run.length / run.speed / 2
            )
        if motion is not None:
            mt = motion["time"].to_numpy()
            if run.kind == "pitching":
                rate = motion["angular_velocity_z"].to_numpy()
                angle = np.concatenate(
                    ([0.0], np.cumsum(0.5 * (rate[1:] + rate[:-1]) * np.diff(mt)))
                )
                signal_rows(
                    f"{body_name}.pitch_angle_deg",
                    mt,
                    np.degrees(angle),
                    window,
                    harmonic_frequency,
                    summary,
                    cycles,
                    phases,
                    harmonic_rows,
                    spectra,
                )
                work = signals.work_per_cycle(
                    mt, np.interp(mt, time, forces["moment_z"].to_numpy()), rate, window
                )
            else:
                for axis in ("x", "y"):
                    values = motion[f"displacement_{axis}"].to_numpy() / run.length
                    signal_rows(
                        f"{body_name}.displacement_{axis}_over_D",
                        mt,
                        values,
                        window,
                        harmonic_frequency,
                        summary,
                        cycles,
                        phases,
                        harmonic_rows,
                        spectra,
                    )
                    if run.kind == "elastic":
                        cycle_peaks = signals.cycle_statistics(mt, values, window)["max"]
                        for key, value in signals.peak_amplitudes(mt, values, window).items():
                            summary[f"{body_name}.amplitude_{axis}_over_D.{key}"] = statistic(value)
                        low, high = signals.block_bootstrap_interval(cycle_peaks - values.mean())
                        summary[f"{body_name}.amplitude_{axis}_over_D.cycle_peak_bootstrap_95"] = {
                            "low": low,
                            "high": high,
                            "unit": "1",
                        }
                velocity = np.column_stack([motion["velocity_x"], motion["velocity_y"]])
                force = np.column_stack(
                    [
                        np.interp(mt, time, forces["total_force_x"]),
                        np.interp(mt, time, forces["total_force_y"]),
                    ]
                )
                work = signals.work_per_cycle(mt, force, velocity, window)
                # Lift relative to the cross-flow displacement at the response frequency
                frequency_axis, cross = signals.cross_spectrum(
                    mt,
                    motion["displacement_y"].to_numpy(),
                    np.interp(mt, time, forces["lift_coefficient"]),
                    window,
                )
                peak = int(np.argmax(np.abs(cross[1:]))) + 1
                summary[f"{body_name}.lift_displacement_phase_deg"] = statistic(
                    float(np.degrees(np.angle(cross[peak]))), unit="deg"
                )
            summary[f"{body_name}.work_per_cycle"] = statistic(
                work.mean(), signals.standard_error(work), "J"
            )
            for index, value in enumerate(work):
                cycles.append(
                    {
                        "signal": f"{body_name}.work",
                        "cycle": index,
                        "start": window.edges[index],
                        "end": window.edges[index + 1],
                        "mean": float(value),
                    }
                )
    if run.kind == "rotor":
        total = None
        for body_name, (forces, _) in data.items():
            values = forces.set_index("time")[["total_force_x", "rotational_power"]]
            total = values if total is None else total.add(values, fill_value=0.0)
        reference_force = 0.5 * 1.225 * run.speed**2 * math.pi * (run.length / 2) ** 2
        time = total.index.to_numpy()
        signal_rows(
            "rotor.thrust_coefficient",
            time,
            total["total_force_x"].to_numpy() / reference_force,
            window,
            harmonic_frequency,
            summary,
            cycles,
            phases,
            harmonic_rows,
            spectra,
        )
        signal_rows(
            "rotor.power_coefficient",
            time,
            total["rotational_power"].to_numpy() / (reference_force * run.speed),
            window,
            harmonic_frequency,
            summary,
            cycles,
            phases,
            harmonic_rows,
            spectra,
        )
    if run.kind == "fixed" and {"upstream", "downstream"} <= set(data):
        up, down = data["upstream"][0], data["downstream"][0]
        t = down["time"].to_numpy()
        frequency_axis, cross = signals.cross_spectrum(
            t,
            np.interp(t, up["time"], up["lift_coefficient"]),
            down["lift_coefficient"].to_numpy(),
            window,
        )
        peak = int(np.argmax(np.abs(cross[1:]))) + 1
        summary["lift_phase_downstream_vs_upstream_deg"] = statistic(
            float(np.degrees(np.angle(cross[peak]))), unit="deg"
        )
    summary["_window"] = window
    summary["_frequency"] = harmonic_frequency
    return summary


def wake_profiles(
    run: Run, side: str, samples: Path, window: signals.Window, rows: list, files: list[Path]
) -> dict:
    """Mean and fluctuation of velocity and vorticity along each line over the window."""
    summary = {}
    for line in run.lines:
        name = line.coupled if side == "coupled" else line.comparison
        if name is None or not (samples / name).exists():
            continue
        files.append(samples / name)
        data = read_line(samples / name)
        inside = (data["time"] >= window.start) & (data["time"] <= window.end)
        if inside.sum() < 2:
            continue
        for column in (
            "velocity_x",
            "velocity_y",
            "velocity_z",
            "vorticity_x",
            "vorticity_y",
            "vorticity_z",
        ):
            if column not in data:
                continue
            values = data[column][inside]
            mean, rms = values.mean(axis=0), values.std(axis=0)
            for point in range(values.shape[1]):
                rows.append(
                    {
                        "side": side,
                        "line": line.name,
                        "quantity": column,
                        "x": data["position_x"][0, point],
                        "y": data["position_y"][0, point],
                        "z": data["position_z"][0, point],
                        "mean": mean[point],
                        "rms": rms[point],
                        "samples": int(inside.sum()),
                    }
                )
            summary[f"wake.{line.name}.{column}.mean_of_profile"] = statistic(float(mean.mean()))
            summary[f"wake.{line.name}.{column}.max_rms"] = statistic(float(rms.max()))
    return summary


# Coupling diagnostics


def numeric_leaves(record, prefix=""):
    for key, value in record.items():
        if isinstance(value, dict):
            yield from numeric_leaves(value, f"{prefix}{key}.")
        elif isinstance(value, bool):
            yield f"{prefix}{key}", float(value)
        elif isinstance(value, int | float) and value is not None:
            yield f"{prefix}{key}", float(value)


def domain_records(record: dict) -> dict[str, dict]:
    """Per-domain parts of one interval record; one domain is flattened at the top level."""
    if "domains" in record:
        return dict(record["domains"])
    return {"domain": record}


def coupling_iterations(record: dict) -> tuple[float, float]:
    rows = record.get("coupling_iteration") or []
    if isinstance(rows, dict):
        rows = rows.get("rows", [])
    if not rows:
        return 1.0, float("nan")
    return float(len(rows)), float(rows[-1].get("scaled_residual", float("nan")))


def diagnostics(run: Run, solution: Path, window: signals.Window, frequency, files: list[Path]):
    path = solution / "coupler_diagnostics.jsonl"
    files.append(path)
    records = read_jsonl(path)
    values: dict[str, dict[str, list[float]]] = {}
    times = []
    for record in records:
        times.append(float(record["time"]))
        count, residual = coupling_iterations(record)
        for domain, part in domain_records(record).items():
            store = values.setdefault(domain, {})
            for key, value in numeric_leaves(part):
                store.setdefault(key, []).append(value)
            store.setdefault("coupling_iterations", []).append(count)
            store.setdefault("final_scaled_residual", []).append(residual)
        for key, value in numeric_leaves(record.get("particle_set", {})):
            values.setdefault("particle_set", {}).setdefault(key, []).append(value)
    times = np.asarray(times)
    summary = {
        domain: {
            key: signals.percentiles(np.asarray(series))
            for key, series in store.items()
            if len(series) == len(times)
        }
        for domain, store in values.items()
    }
    by_phase = []
    if run.kind in ("pitching", "elastic", "rotor", "heaving"):
        phase = signals.phase_of(times, window)
        bins = np.floor(phase * signals.DIAGNOSTIC_PHASE_BINS)
        for domain, store in values.items():
            for key, series in store.items():
                series = np.asarray(series)
                if len(series) != len(times):
                    continue
                for b in range(signals.DIAGNOSTIC_PHASE_BINS):
                    chosen = series[bins == b]
                    if len(chosen):
                        stats = signals.percentiles(chosen)
                        by_phase.append(
                            {
                                "domain": domain,
                                "quantity": key,
                                "phase_bin": b,
                                "phase_centre": (b + 0.5) / signals.DIAGNOSTIC_PHASE_BINS,
                                **stats,
                            }
                        )
    return records, summary, by_phase


def circulation_budget(
    run: Run, records: list[dict], samples: Path, files: list[Path]
) -> list[dict]:
    """Net circulation of the particle set and the FVM regions per interval, against its expected value.

    The expected fluid circulation is zero for fixed and translating bodies and
    -2 Omega A for a section of area A turning at Omega. Differences are
    normalised by the particle set's total absolute strength.
    """
    rate = None
    if run.kind == "pitching":
        motion = read_table(samples / run.bodies[0].motion)
        rate = (motion["time"].to_numpy(), motion["angular_velocity_z"].to_numpy())
    rows = []
    for record in records:
        particles = record.get("particle_set", {})
        net = particles.get("vortex_strength_net_z")
        absolute = particles.get("vortex_strength_l1")
        if net is None or not absolute:
            continue
        fvm_total = 0.0
        for part in domain_records(record).values():
            fvm_total += float(
                part.get("transfer", {}).get("fvm_budget_total_vortex_strength_z", 0.0)
            )
        time = float(record["time"])
        expected = 0.0
        if rate is not None:
            expected = -2.0 * float(np.interp(time, *rate)) * run.section_area * run.span
        rows.append(
            {
                "time": time,
                "particle_net_strength_z": net,
                "particle_strength_l1": absolute,
                "fvm_net_strength_z": fvm_total,
                "expected_net_strength_z": expected,
                "particle_residual_over_l1": (net - expected) / absolute,
                "particle_and_fvm_residual_over_l1": (net + fvm_total - expected) / absolute,
            }
        )
    return rows


def cost(solution: Path, records: list[dict]) -> dict:
    phases: dict[str, float] = {}
    for record in records:
        for key, value in numeric_leaves(record.get("timing_seconds", {})):
            phases[key] = phases.get(key, 0.0) + value
    return {"intervals": len(records), "seconds_by_phase": phases}


# Driver


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(REPOSITORY), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def public(summary: dict) -> dict:
    return {key: value for key, value in summary.items() if not key.startswith("_")}


def analyse(
    run: Run, case_dir: Path, samples: Path, comparison: Path | None, solution: Path
) -> Path:
    out = case_dir / "analysis" / run.name
    out.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []
    tables = {
        side: {name: [] for name in ("cycles", "phases", "harmonics", "spectra")}
        for side in ("coupled", "comparison")
    }
    coupled = analyse_side(run, "coupled", samples, tables, files)
    window, frequency = coupled["_window"], coupled["_frequency"]
    profiles: list[dict] = []
    coupled.update(wake_profiles(run, "coupled", samples, window, profiles, files))
    records, diagnostics_summary, by_phase = diagnostics(run, solution, window, frequency, files)
    budget = circulation_budget(run, records, samples, files)
    if budget:
        residual = np.array([row["particle_and_fvm_residual_over_l1"] for row in budget])
        coupled["circulation_residual_over_l1"] = {
            "unit": "1",
            **signals.percentiles(np.abs(residual)),
        }
    summary = {"run": run.name, "coupled": public(coupled), "cost": cost(solution, records)}

    comparison_summary = None
    if comparison is not None and run.comparison_name:
        comparison_summary = analyse_side(run, "comparison", comparison, tables, files)
        comparison_summary.update(
            wake_profiles(
                run, "comparison", comparison, comparison_summary["_window"], profiles, files
            )
        )
        summary["comparison"] = public(comparison_summary)
        rows = compare(public(coupled), public(comparison_summary))
        write_json(
            out / f"compare_{run.name}_vs_{run.comparison_name}.json",
            {
                "first": run.comparison_name,
                "second": run.name,
                "difference": "second - first",
                "rows": compare(public(comparison_summary), public(coupled)),
            },
        )
        del rows

    def table(name):
        frames = []
        for side in ("coupled", "comparison"):
            if tables[side][name]:
                frame = pd.DataFrame(tables[side][name])
                frame.insert(0, "side", side)
                frames.append(frame)
        return pd.concat(frames) if frames else pd.DataFrame()

    table("cycles").to_csv(out / "cycles.csv", index=False)
    table("phases").to_csv(out / "phase_average.csv", index=False)
    table("harmonics").to_csv(out / "harmonics.csv", index=False)
    table("spectra").to_csv(out / "spectra.csv", index=False)
    pd.DataFrame(profiles).to_csv(out / "wake_profiles.csv", index=False)
    pd.DataFrame(budget).to_csv(out / "circulation_budget.csv", index=False)
    write_json(out / "diagnostics_summary.json", diagnostics_summary)
    pd.DataFrame(by_phase).to_csv(out / "diagnostics_by_phase.csv", index=False)
    write_json(out / "summary.json", summary)
    unique = sorted({path.resolve() for path in files if path.exists()})
    write_json(
        out / "manifest.json",
        {
            "run": run.name,
            "case": run.case,
            "commit_of_analysis": git_commit(),
            "window": window.describe(),
            "harmonic_frequency_hz": frequency,
            "inputs": [
                {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}
                for path in unique
            ],
            "outputs": sorted(path.name for path in out.iterdir()),
        },
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("run", choices=sorted(runs()))
    parser.add_argument("--case-dir", type=Path, help="case directory (default: the tutorial)")
    parser.add_argument("--end-time", type=float, help="analyse a run that ended earlier")
    parser.add_argument(
        "--window-periods", type=float, help="statistics over the last periods (pre-runs)"
    )
    parser.add_argument("--start-after", type=float, help="statistics after this time (pre-runs)")
    arguments = parser.parse_args()
    run = runs()[arguments.run]
    if arguments.end_time is not None:
        run.end_time = arguments.end_time
    if arguments.window_periods is not None:
        run.window_periods = arguments.window_periods
    if arguments.start_after is not None:
        run.start_after = arguments.start_after
    case_dir = (arguments.case_dir or TUTORIALS / run.case).resolve()
    comparison = case_dir / run.comparison_samples if run.comparison_samples else None
    if comparison is not None and not comparison.exists():
        comparison = None
    out = analyse(run, case_dir, case_dir / run.samples, comparison, case_dir / run.solution)
    print(out)


if __name__ == "__main__":
    main()
