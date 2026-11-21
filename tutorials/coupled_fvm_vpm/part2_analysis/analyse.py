#!/usr/bin/env python3
"""Analyse one Part II run and write ``analysis/<run>/`` next to its case.

python analyse.py <run> --case-dir DIR [--comparison-case-dir DIR]
    [--record DIR ...] [--comparison-record DIR ...] [--frequency F]
    [--end-time T] [--window-periods N] [--start-after T]

Runs: S1 (case 05 against its reference), S2_B1 and S2_B2 (case 04 against its
references), S3 (case 06 against the VLM rotor), S4a (case 07, both cylinders
fixed, against its reference) and S4b (case 07, the downstream cylinder heaving
at ``--frequency``; no comparison). ``--case-dir`` is the case folder of the
coupled run; ``--comparison-case-dir`` the same case folder in the export where
the comparison ran. ``--record`` names the launch folders of a run (the
``run_record.json`` and ``run.log`` of its 2 % pre-run and its continuation).

Every number comes from a named column of a delivered sample or diagnostics
file; ``manifest.json`` lists the files and their hashes, and holds the commit,
command line, configuration, mesh, hardware, peak GPU memory, wall time, end
state and warnings of each run.

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
import os
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
    comparison_patch: str | None = None  # rows of the comparison file, when named otherwise
    # comparison column -> coupled column, when the comparison names its forces otherwise
    comparison_columns: dict[str, str] | None = None


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
    comparison_solution: str | None = None  # relative to the comparison case directory
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
            comparison_solution="reference_flow/solution",
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
            comparison_solution="../../vpm/06_rotor_flow/solution/rotor",
            # Blade forces of the FVM patches against the VLM surfaces of the same names; the
            # wake planes of both runs are saved as surfaces (.vts) and the lines common to
            # both are analysed
            bodies=[
                Body(
                    f"blade_{index}",
                    "blade_forces.csv",
                    "vlm_surface_forces.csv",
                    patch=f"blade_{index}",
                    comparison_patch=f"blade_{index}",
                    comparison_columns={
                        "force_x": "total_force_x",
                        "force_y": "total_force_y",
                        "force_z": "total_force_z",
                    },
                )
                for index in range(3)
            ],
            lines=[
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
            comparison_solution="reference_flow/solution/fixed",
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
            comparison_solution=f"reference_flow/solution/{variant}",
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
    """Rows of a sample file; ``patch`` selects one patch (FVM) or surface (VLM) of the file."""
    frame = pd.read_csv(path)
    if patch is not None:
        for column in ("patch", "surface"):
            if column in frame:
                frame = frame[frame[column] == patch].drop(columns=column)
                break
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
        if run.window_periods:
            # The last whole periods (from t = 0) that end by the end time
            last_edge = math.floor(end / period + 1.0e-9) * period
            start = last_edge - run.window_periods * period
        else:
            start = run.start_after
        return signals.motion_window(period, start - 1.0e-9, end + 1.0e-9)
    if run.window_periods:
        period = run.natural_period
        start = end - run.window_periods * period
    else:
        start = run.start_after
    return signals.upcrossing_window(time, reference, start, end + 1.0e-9)


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


def bootstrap_field(per_cycle_values, bootstrap: bool) -> dict:
    """``{"bootstrap_95": [low, high]}`` of the mean of per-cycle values when asked (S2), else ``{}``."""
    if not bootstrap:
        return {}
    return {"bootstrap_95": list(signals.block_bootstrap_interval(per_cycle_values))}


def signal_rows(
    prefix,
    time,
    values,
    window,
    frequency,
    summary,
    cycles,
    phases,
    harmonic_rows,
    spectra,
    bootstrap=False,
):
    """Statistics of one signal into the summary and the tables.

    ``bootstrap`` adds a block-bootstrap 95 % interval over cycles to every
    statistic that has per-cycle values (the S2 runs).
    """
    stats = signals.window_statistics(time, values, window)
    unit = "deg" if prefix.endswith("_deg") else "1"
    per_cycle = signals.cycle_statistics(time, values, window)
    summary[f"{prefix}.mean"] = statistic(
        stats["mean"],
        stats["mean_standard_error"],
        unit,
        **bootstrap_field(per_cycle["mean"], bootstrap),
    )
    summary[f"{prefix}.std"] = statistic(
        stats["std"],
        stats["std_standard_error"],
        unit,
        **bootstrap_field(per_cycle["std"], bootstrap),
    )
    summary[f"{prefix}.cycle_max"] = statistic(
        stats["cycle_max_mean"],
        stats["cycle_max_standard_error"],
        unit,
        **bootstrap_field(per_cycle["max"], bootstrap),
    )
    summary[f"{prefix}.cycle_min"] = statistic(
        stats["cycle_min_mean"],
        stats["cycle_min_standard_error"],
        unit,
        **bootstrap_field(per_cycle["min"], bootstrap),
    )
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
            summary[f"{prefix}.harmonic_{k}_amplitude"] = statistic(
                amplitude,
                error,
                unit,
                **bootstrap_field(per_cycle_amplitudes[:, k - 1], bootstrap),
            )
            summary[f"{prefix}.harmonic_{k}_phase"] = statistic(angle, unit="deg")
    frequency_axis, psd, resolution = signals.welch_spectrum(time, values, window)
    peak = signals.peak_frequency(frequency_axis, psd, minimum=0.5 * resolution)
    # A spectral peak has no cycle-scatter standard error; its resolution is stated instead
    summary[f"{prefix}.spectral_peak_frequency"] = statistic(
        peak, unit="Hz", frequency_resolution=resolution
    )
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
        if side == "coupled":
            forces = read_table(path, body.patch)
        else:
            forces = read_table(path, body.comparison_patch or body.patch)
            if body.comparison_columns:
                forces = forces.rename(columns=body.comparison_columns)
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
    bootstrap = run.kind == "elastic"  # block-bootstrap intervals over cycles for S2

    response_frequency = None
    if run.kind == "elastic":
        f, psd, resolution = signals.welch_spectrum(reference_time, reference, window)
        response_frequency = signals.peak_frequency(f, psd, minimum=0.5 * resolution)
        # Dominant frequency of each displacement over the natural frequency: spectral peaks
        # (with their resolution) and, across the stream, 1 / mean period of the window's cycles
        summary["frequency_ratio_y"] = statistic(
            response_frequency * run.natural_period,
            frequency_resolution_ratio=resolution * run.natural_period,
        )
        cycle_f, cycle_error = signals.cycle_frequency(window)
        summary["frequency_ratio_y_from_cycles"] = statistic(
            cycle_f * run.natural_period, cycle_error * run.natural_period
        )
        f, psd, resolution = signals.welch_spectrum(
            reference_time, first_motion["displacement_x"].to_numpy(), window
        )
        summary["frequency_ratio_x"] = statistic(
            signals.peak_frequency(f, psd, minimum=0.5 * resolution) * run.natural_period,
            frequency_resolution_ratio=resolution * run.natural_period,
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
                    bootstrap=bootstrap,
                )
        if run.kind in ("fixed", "heaving") and "lift_coefficient" in forces:
            lift = forces["lift_coefficient"].to_numpy()
            f, psd, resolution = signals.welch_spectrum(time, lift, window)
            peak = signals.peak_frequency(f, psd, minimum=0.5 * resolution)
            scale = run.length / run.speed
            summary[f"{body_name}.strouhal_number"] = statistic(
                peak * scale, frequency_resolution_strouhal=resolution * scale
            )
            # 1 / mean period between the lift up-crossings of this body, over the same span as
            # the window: after its start time, up to its end
            own_start = run.start_after if run.start_after is not None else window.start
            own_end = min(run.end_time, float(time[-1])) + 1.0e-9  # the data span of the window
            own = signals.upcrossing_window(time, lift, own_start, own_end)
            cycle_f, cycle_error = signals.cycle_frequency(own)
            summary[f"{body_name}.strouhal_number_from_cycles"] = statistic(
                cycle_f * scale, cycle_error * scale, cycles=own.n_cycles
            )
        if motion is not None:
            mt = motion["time"].to_numpy()
            if run.kind == "pitching":
                rate = motion["angular_velocity_z"].to_numpy()
                # Nose up positive: case 05 turns about -z
                angle = -np.concatenate(
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
                    bootstrap=bootstrap,
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
                        bootstrap=bootstrap,
                    )
                    if run.kind == "elastic":
                        cycle_peaks = signals.cycle_statistics(mt, values, window)["max"]
                        amplitudes = signals.peak_amplitudes(mt, values, window)
                        for key, (value, error) in amplitudes.items():
                            summary[f"{body_name}.amplitude_{axis}_over_D.{key}"] = statistic(
                                value, error
                            )
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
                work.mean(), signals.standard_error(work), "J", **bootstrap_field(work, bootstrap)
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
        # Azimuth of the rotor at the window start, from its angular speed since t = 0
        # (both rotors spin up over their first revolution)
        if side == "coupled" and (samples / "motion_history.csv").exists():
            files.append(samples / "motion_history.csv")
            motion = read_table(samples / "motion_history.csv")
            spin = motion[["time", "angular_velocity_x"]].to_numpy()
        else:
            forces = next(iter(data.values()))[0]
            spin = (
                forces[["time", "angular_velocity_x"]].to_numpy()
                if ("angular_velocity_x" in forces)
                else None
            )
        if spin is not None:
            turned = np.concatenate(
                ([0.0], np.cumsum(0.5 * (spin[1:, 1] + spin[:-1, 1]) * np.diff(spin[:, 0])))
            )
            azimuth = float(np.degrees(np.abs(np.interp(window.start, spin[:, 0], turned))))
            summary["rotor.azimuth_at_window_start_deg"] = statistic(azimuth % 360.0, unit="deg")
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
            bootstrap=bootstrap,
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
            bootstrap=bootstrap,
        )
    if run.kind in ("fixed", "heaving") and {"upstream", "downstream"} <= set(data):
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
    """Per-domain parts of one interval record; one domain is flattened at the top level.

    With several domains, the records shared by all of them (particle set,
    coupling iterations, timing, particle-solver corrections) form ``shared``.
    """
    if "domains" in record:
        parts = dict(record["domains"])
        parts["shared"] = {key: value for key, value in record.items() if key != "domains"}
        return parts
    return {"domain": record}


RESIDUALS = ("normal_velocity_residual", "normal_gradient_residual", "scaled_residual")


def coupling_state(record: dict) -> dict[str, dict[str, float]]:
    """Coupling iterations of the interval and the residuals of its last iteration, per domain.

    ``coupling_iteration`` holds ``iterations`` and one ``residuals`` row per
    iteration; with several domains each row holds the residuals below ``domains``.
    """
    state = record.get("coupling_iteration") or {}
    rows = state.get("residuals") or []
    count = float(state.get("iterations", len(rows)))
    last = rows[-1] if rows else {}
    per_domain = last.get("domains") if "domains" in last else {"domain": last}
    return {
        name: {
            "coupling_iterations": count,
            **{f"final_{key}": float(values.get(key, float("nan"))) for key in RESIDUALS},
        }
        for name, values in per_domain.items()
    }


def diagnostics(run: Run, solution: Path, window: signals.Window, frequency, files: list[Path]):
    path = solution / "coupler_diagnostics.jsonl"
    files.append(path)
    records = read_jsonl(path)
    values: dict[str, dict[str, list[float]]] = {}
    times = []
    for record in records:
        times.append(float(record["time"]))
        state = coupling_state(record)
        for domain, part in domain_records(record).items():
            store = values.setdefault(domain, {})
            for key, value in numeric_leaves(part):
                store.setdefault(key, []).append(value)
            for key, value in state.get(domain, {}).items():
                store.setdefault(key, []).append(value)
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


def particle_set(record: dict) -> dict:
    """The particle set after the interval's transfers: ``particle_set`` of the record.

    Records written before the coupler had that field hold the same values in
    the ``transfer`` record of their single domain, with a ``particle_set_`` prefix.
    """
    if "particle_set" in record:
        return record["particle_set"]
    transfer = record.get("transfer", {})
    if "particle_set_vortex_strength_l1" not in transfer:
        return {}
    return {
        "vortex_strength_net_z": transfer.get("particle_set_vortex_strength_net_z"),
        "vortex_strength_l1": transfer.get("particle_set_vortex_strength_l1"),
        "fringe_removed_vortex_strength_z": transfer.get("fringe_removed_vortex_strength_z", 0.0),
        "particle_count": transfer.get("n_particles_after"),
    }


def circulation_budget(
    run: Run, records: list[dict], samples: Path, files: list[Path]
) -> list[dict]:
    """Net circulation of the particle set per interval, against its expected value.

    The particle circulation is the net strength of the particle set plus the
    strength the outflow fringe has removed since the start (a running total in
    the record). The expected fluid circulation is zero for fixed and
    translating bodies and -2 Omega A for a section of area A turning at Omega,
    times the slab depth. Differences are normalised by the particle set's total
    absolute strength. The variant with the net strength still held by the FVM
    regions added is listed beside it.
    """
    rate = None
    if run.kind == "pitching":
        motion = read_table(samples / run.bodies[0].motion)
        rate = (motion["time"].to_numpy(), motion["angular_velocity_z"].to_numpy())
    rows = []
    for record in records:
        particles = particle_set(record)
        net = particles.get("vortex_strength_net_z")
        absolute = particles.get("vortex_strength_l1")
        if net is None or not absolute:
            continue
        fringe = float(particles.get("fringe_removed_vortex_strength_z") or 0.0)
        fvm_total = 0.0
        for part in domain_records(record).values():
            fvm_total += float(
                part.get("transfer", {}).get("fvm_budget_total_vortex_strength_z", 0.0)
            )
        time = float(record["time"])
        expected = 0.0
        if rate is not None:
            expected = -2.0 * float(np.interp(time, *rate)) * run.section_area * run.span
        circulation = net + fringe
        rows.append(
            {
                "time": time,
                "particle_net_strength_z": net,
                "fringe_removed_strength_z": fringe,
                "particle_circulation_z": circulation,
                "particle_strength_l1": absolute,
                "fvm_net_strength_z": fvm_total,
                "expected_net_strength_z": expected,
                "particle_residual_over_l1": (circulation - expected) / absolute,
                "particle_and_fvm_residual_over_l1": (circulation + fvm_total - expected)
                / absolute,
            }
        )
    return rows


def covered_flow_time(times: np.ndarray) -> float:
    """Flow time covered by records at the ends of equal intervals: last - first + one interval."""
    times = np.unique(np.asarray(times, dtype=np.float64))
    if len(times) < 2:
        return float("nan")
    return float(times[-1] - times[0] + np.median(np.diff(times)))


def metadata_files(solution: Path, name: str) -> dict[str, Path]:
    """``name`` (e.g. ``fvm_metadata.json``) of the run: one file, or one per domain folder."""
    if (solution / name).exists():
        return {"domain": solution / name}
    return {path.parent.name: path for path in sorted(solution.glob(f"*/{name}"))}


def cell_counts(solution: Path, files: list[Path]) -> dict[str, int]:
    counts = {}
    for domain, path in metadata_files(solution, "fvm_metadata.json").items():
        files.append(path)
        counts[domain] = int(json.loads(path.read_text())["mesh"]["n_cells"])
    return counts


def per_flow_time(seconds: dict[str, float], flow_time: float) -> dict[str, float]:
    return {key: value / flow_time for key, value in seconds.items()}


def coupled_cost(solution: Path, records: list[dict], files: list[Path]) -> dict:
    """Wall time by phase from the per-interval ``timing_seconds`` of the coupler records.

    A restarted interval is counted once (its last record); the wall time of
    the whole run, restarts included, is in the manifest.
    """
    phases: dict[str, float] = {}
    domains: dict[str, dict[str, float]] = {}
    counts = []
    for record in records:
        for key, value in numeric_leaves(record.get("timing_seconds", {})):
            phases[key] = phases.get(key, 0.0) + value
        for name, part in (record.get("domains") or {}).items():
            store = domains.setdefault(name, {})
            for key, value in numeric_leaves(part.get("domain_timing_seconds", {})):
                store[key] = store.get(key, 0.0) + value
        count = particle_set(record).get("particle_count")
        if count is not None:
            counts.append(float(count))
    flow_time = covered_flow_time([record["time"] for record in records])
    return {
        "source": "coupler_diagnostics.jsonl timing_seconds",
        "intervals": len(records),
        "flow_time_s": flow_time,
        "seconds_by_phase": phases,
        "seconds_per_unit_flow_time_by_phase": per_flow_time(phases, flow_time),
        "seconds_per_unit_flow_time_by_domain_phase": {
            name: per_flow_time(store, flow_time) for name, store in domains.items()
        },
        "fvm_cells": cell_counts(solution, files),
        "mean_particle_count": float(np.mean(counts)) if counts else float("nan"),
    }


def fvm_only_cost(solution: Path, files: list[Path]) -> dict:
    """Wall time by section from ``performance.jsonl`` of an FVM-only run (one row per step).

    A step written again after a restart is counted once (its last row).
    """
    path = solution / "performance.jsonl"
    files.append(path)
    rows: dict[int, dict] = {}
    with path.open() as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                rows[int(row["step"])] = row
    sections: dict[str, float] = {}
    total = 0.0
    for row in rows.values():
        total += float(row["step_seconds"]["max"])
        for name, section in row.get("sections", {}).items():
            sections[name] = sections.get(name, 0.0) + float(section["seconds"]["max"])
    times = [float(row["time"]) for row in rows.values()]
    flow_time = covered_flow_time(times)
    return {
        "source": "performance.jsonl step_seconds and sections (maximum over ranks)",
        "steps": len(rows),
        "flow_time_s": flow_time,
        "step_seconds": total,
        "seconds_per_unit_flow_time": total / flow_time,
        "seconds_per_unit_flow_time_by_section": per_flow_time(sections, flow_time),
        "fvm_cells": cell_counts(solution, files),
        "mean_particle_count": 0.0,
    }


def vpm_only_cost(solution: Path, samples: Path, files: list[Path]) -> dict:
    """Wall time from the ``Evolution wall time`` of every step in ``vpm.log`` of a VPM-only run."""
    path = solution / "vpm.log"
    files.append(path)
    seconds, times = {}, {}
    current = None
    with path.open() as stream:
        for line in stream:
            if line.lstrip().startswith("FLOW TIME"):
                current = float(line.split()[2])
            elif "Evolution wall time" in line and current is not None:
                seconds[current] = float(line.split()[3])  # a restarted step counts once
    times = np.array(sorted(seconds))
    flow_time = covered_flow_time(times)
    counts = float("nan")
    integrals = samples / "flow_integrals.csv"
    if integrals.exists():
        files.append(integrals)
        counts = float(read_table(integrals)["n_particles"].mean())
    total = float(sum(seconds.values()))
    return {
        "source": "vpm.log Evolution wall time per step",
        "steps": len(seconds),
        "flow_time_s": flow_time,
        "evolution_seconds": total,
        "seconds_per_unit_flow_time": total / flow_time,
        "fvm_cells": {},
        "mean_particle_count": counts,
    }


def comparison_cost(solution: Path, samples: Path, files: list[Path]) -> dict:
    if (solution / "performance.jsonl").exists():
        return fvm_only_cost(solution, files)
    if (solution / "vpm.log").exists():
        return vpm_only_cost(solution, samples, files)
    return {"source": None, "note": f"no timing file in {solution}"}


# Run records and manifest


def warning_lines(paths: list[Path], limit: int = 50) -> dict:
    """Lines that contain ``warning`` (any case) in the run's logs: count and distinct messages."""
    count, distinct = 0, {}
    for path in paths:
        if not path.exists():
            continue
        with path.open(errors="replace") as stream:
            for line in stream:
                if "warning" in line.lower():
                    count += 1
                    text = line.strip()
                    distinct[text] = distinct.get(text, 0) + 1
    ordered = sorted(distinct.items(), key=lambda item: -item[1])
    return {
        "lines": count,
        "distinct": len(distinct),
        "messages": [{"text": text, "count": n} for text, n in ordered[:limit]],
    }


def hardware() -> dict:
    """CPU model and GPU names of this machine, where the runs were made."""
    cpu = "unknown"
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    try:
        gpus = (
            subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                check=True,
            )
            .stdout.strip()
            .splitlines()
        )
    except (OSError, subprocess.CalledProcessError):
        gpus = []
    return {"cpu": cpu, "logical_cpus": os.cpu_count(), "gpus": gpus}


def case_folder(solution: Path) -> Path:
    """The folder of the run's ``setup.py``: the nearest parent of ``solution`` that has one."""
    for folder in solution.resolve().parents:
        if (folder / "setup.py").exists():
            return folder
    return solution.parent


def mesh_files(solution: Path) -> list[dict]:
    """Mesh files of the run (``assets/*_mesh.npz`` of its case folder) with their hashes."""
    rows = []
    for path in sorted((case_folder(solution) / "assets").glob("*_mesh.npz")):
        rows.append({"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size})
    return rows


def run_side(solution: Path, records: list[Path], files: list[Path]) -> dict:
    """What the manifest says about one side: commit, command, configuration, mesh, cost, state.

    ``records`` are the wrapper folders of the run's launches (the 2 % pre-run,
    then the continuation), each with ``run_record.json`` and ``run.log``.
    """
    launches = []
    logs = []
    for folder in records:
        path = folder / "run_record.json"
        files.append(path)
        record = json.loads(path.read_text())
        root = Path(record.get("case_dir", "."))
        while not (root / "openonda").is_dir() and root != root.parent:
            root = root.parent
        launches.append(
            {
                **record,
                "command_line": (
                    f"PYTHONPATH={root} python setup.py {record.get('arguments', '')}".strip()
                ),
                "working_directory": record.get("case_dir"),
                "record": str(folder),
            }
        )
        logs.append(folder / "run.log")
    configuration, state = {}, {}
    for name in ("fvm_metadata.json", "vpm_metadata.json", "coupler_metadata.json"):
        for domain, path in metadata_files(solution, name).items():
            files.append(path)
            data = json.loads(path.read_text())
            key = name.removesuffix("_metadata.json") + ("" if domain == "domain" else f".{domain}")
            configuration[key] = data.get("configuration")
            state[key] = {
                "run_status": data.get("run_status"),
                "state": data.get("state"),
                "mesh": {k: v for k, v in (data.get("mesh") or {}).items() if k.startswith("n_")},
            }
    logs += sorted(solution.glob("*.log")) + sorted(solution.glob("*/*.log"))
    commits = sorted({launch.get("commit") for launch in launches})
    return {
        "commit": commits[0] if len(commits) == 1 else commits,
        "launches": launches,
        "wall_seconds": float(sum(launch.get("wall_seconds", 0.0) for launch in launches)),
        "peak_gpu_memory_mib": max(
            (launch.get("peak_gpu_memory_mib", 0) for launch in launches), default=None
        ),
        "end_state": state,
        "exit_codes": [launch.get("exit_code") for launch in launches],
        "stopped_for_gpu_memory": any(launch.get("stopped_for_gpu_memory") for launch in launches),
        "configuration": configuration,
        "mesh_files": mesh_files(solution),
        "warnings": warning_lines(logs),
    }


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
    run: Run,
    case_dir: Path,
    samples: Path,
    comparison: Path | None,
    solution: Path,
    comparison_solution: Path | None = None,
    launch_records: list[Path] = (),
    comparison_launch_records: list[Path] = (),
    command: str = "",
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
        inside = [row for row in budget if window.start <= row["time"] <= window.end]
        for key in ("particle_residual_over_l1", "particle_and_fvm_residual_over_l1"):
            residual = np.array([row[key] for row in inside])
            coupled[f"circulation_budget.{key}"] = {
                "unit": "1",
                "over": "window",
                **signals.percentiles(np.abs(residual)),
            }
    summary = {
        "run": run.name,
        "coupled": public(coupled),
        "cost": {"coupled": coupled_cost(solution, records, files)},
    }

    comparison_summary = None
    if comparison is not None and run.comparison_name:
        comparison_summary = analyse_side(run, "comparison", comparison, tables, files)
        comparison_summary.update(
            wake_profiles(
                run, "comparison", comparison, comparison_summary["_window"], profiles, files
            )
        )
        summary["comparison"] = public(comparison_summary)
        if comparison_solution is not None:
            summary["cost"]["comparison"] = comparison_cost(comparison_solution, comparison, files)
        write_json(
            out / f"compare_{run.name}_vs_{run.comparison_name}.json",
            {
                "first": run.comparison_name,
                "second": run.name,
                "difference": "second - first",
                "rows": compare(public(comparison_summary), public(coupled)),
            },
        )

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
    sides = {"coupled": run_side(solution, list(launch_records), files)}
    if comparison_solution is not None and comparison_solution.exists():
        sides["comparison"] = run_side(comparison_solution, list(comparison_launch_records), files)
    for side in sides.values():
        side["fvm_cells"] = (
            summary["cost"]
            .get("coupled" if side is sides["coupled"] else "comparison", {})
            .get("fvm_cells")
        )
    unique = sorted({path.resolve() for path in files if path.exists()})
    write_json(
        out / "manifest.json",
        {
            "run": run.name,
            "case": run.case,
            "analysis_command_line": command,
            "commit_of_analysis": git_commit(),
            "hardware": hardware(),
            "window": window.describe(),
            "harmonic_frequency_hz": frequency,
            "runs": sides,
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
    parser.add_argument(
        "--comparison-case-dir",
        type=Path,
        help="the case directory in the export of the comparison run (default: --case-dir)",
    )
    parser.add_argument(
        "--record",
        type=Path,
        action="append",
        default=[],
        help="launch record folder of the coupled run (run_record.json, run.log); repeat per launch",
    )
    parser.add_argument(
        "--comparison-record",
        type=Path,
        action="append",
        default=[],
        help="launch record folder of the comparison run; repeat per launch",
    )
    parser.add_argument(
        "--frequency", type=float, help="prescribed motion frequency in Hz (S4b: the heave)"
    )
    arguments = parser.parse_args()
    run = runs()[arguments.run]
    if arguments.frequency is not None:
        run.frequency = arguments.frequency
    if run.kind == "heaving" and not run.frequency:
        parser.error("S4b needs --frequency, the heave frequency of the run")
    if arguments.end_time is not None:
        run.end_time = arguments.end_time
    if arguments.window_periods is not None:
        run.window_periods = arguments.window_periods
    if arguments.start_after is not None:
        run.start_after = arguments.start_after
    case_dir = (arguments.case_dir or TUTORIALS / run.case).resolve()
    comparison_dir = (arguments.comparison_case_dir or case_dir).resolve()
    comparison = comparison_dir / run.comparison_samples if run.comparison_samples else None
    if comparison is not None and not comparison.exists():
        comparison = None
    comparison_solution = (
        comparison_dir / run.comparison_solution
        if comparison is not None and run.comparison_solution
        else None
    )
    out = analyse(
        run,
        case_dir,
        case_dir / run.samples,
        comparison,
        case_dir / run.solution,
        comparison_solution,
        arguments.record,
        arguments.comparison_record,
        " ".join(sys.argv),
    )
    print(out)


if __name__ == "__main__":
    main()
