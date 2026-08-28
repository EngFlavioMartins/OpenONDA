"""Recompute force, shedding, and velocity errors for the coupled benchmark.

The fully meshed reference is the authority. Force statistics use the final
30 convective units, beginning no earlier than t=30. Velocity errors are
normalized by U_inf and are evaluated on every coincident sampled frame in
that window. The authority-stitched centreline uses FVM values inside the
coupled FVM box and VPM values downstream of it.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.signal import periodogram

CASE_DIR = Path(__file__).resolve().parents[1]
COUPLED_DIR = CASE_DIR
REFERENCE_DIR = CASE_DIR / "reference_flow"


def _table(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = np.atleast_1d(
        np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    )
    if rows.size == 0:
        raise ValueError(f"Empty sampler table: {path}")
    table = {name: np.asarray(rows[name]) for name in rows.dtype.names}
    if "step" in table:
        reset = np.flatnonzero(np.diff(np.asarray(table["step"], dtype=float)) < 0.0)
        if reset.size:
            start = int(reset[-1] + 1)
            table = {name: values[start:] for name, values in table.items()}
    return table


def _peak_frequency(time: np.ndarray, values: np.ndarray) -> float:
    dt = float(np.median(np.diff(time)))
    nfft = max(4096, 8 * int(2 ** np.ceil(np.log2(len(time)))))
    frequency, power = periodogram(values - np.mean(values), fs=1.0 / dt, nfft=nfft)
    selected = np.flatnonzero((frequency >= 0.10) & (frequency <= 0.30))
    peak = int(selected[np.argmax(power[selected])])
    if 0 < peak < len(power) - 1:
        y0, y1, y2 = np.log(np.maximum(power[peak - 1 : peak + 2], 1.0e-300))
        denominator = y0 - 2.0 * y1 + y2
        correction = 0.0 if denominator == 0.0 else 0.5 * (y0 - y2) / denominator
        return float(frequency[peak] + correction * (frequency[1] - frequency[0]))
    return float(frequency[peak])


def _harmonic(time: np.ndarray, values: np.ndarray, frequency: float) -> tuple[float, float]:
    phase = 2.0 * np.pi * frequency * time
    design = np.column_stack((np.ones_like(time), np.sin(phase), np.cos(phase)))
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    amplitude = float(np.hypot(coefficients[1], coefficients[2]))
    angle = float(np.arctan2(coefficients[2], coefficients[1]))
    return amplitude, angle


def _force_metrics(table: dict[str, np.ndarray], start: float) -> dict[str, float | int]:
    time = np.asarray(table["time"], dtype=float)
    selected = time >= start - 1.0e-10
    if np.count_nonzero(selected) < 64:
        raise ValueError("fewer than 64 force samples in the saturated window")
    time = time[selected]
    drag = np.asarray(table["drag_coefficient"], dtype=float)[selected]
    lift = np.asarray(table["lift_coefficient"], dtype=float)[selected]
    frequency = _peak_frequency(time, lift)
    amplitude, phase = _harmonic(time, lift, frequency)
    return {
        "window_start": float(time.min()),
        "window_end": float(time.max()),
        "sample_count": int(len(time)),
        "mean_cd": float(np.mean(drag)),
        "cd_rms": float(np.std(drag)),
        "cl_rms": float(np.std(lift)),
        "strouhal": frequency,
        "cl_first_harmonic": amplitude,
        "cl_harmonic_phase_rad": phase,
    }


def _relative(actual: float, reference: float) -> float:
    return abs(actual - reference) / max(abs(reference), 1.0e-30)


def _frame(table: dict[str, np.ndarray], time: float) -> dict[str, np.ndarray] | None:
    available = np.unique(np.asarray(table["time"], dtype=float))
    if available.size == 0:
        return None
    picked = float(available[np.argmin(np.abs(available - time))])
    if not np.isclose(picked, time, rtol=0.0, atol=1.0e-8):
        return None
    selected = np.isclose(table["time"], picked, rtol=0.0, atol=1.0e-10)
    return {name: values[selected] for name, values in table.items()}


def _profile_error(
    candidate: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
    *,
    coordinate: str,
    exclude_cylinder: bool,
) -> tuple[dict[str, float | int], np.ndarray]:
    ref_coordinate = np.asarray(reference[coordinate], dtype=float)
    ref_velocity = np.column_stack(
        [np.asarray(reference[f"velocity_{axis}"], dtype=float) for axis in "xyz"]
    )
    reference_valid = np.isfinite(ref_coordinate) & np.all(np.isfinite(ref_velocity), axis=1)
    ref_coordinate = ref_coordinate[reference_valid]
    ref_velocity = ref_velocity[reference_valid]
    order = np.argsort(ref_coordinate)
    ref_coordinate = ref_coordinate[order]
    ref_velocity = ref_velocity[order]

    candidate_coordinate = np.asarray(candidate[coordinate], dtype=float)
    candidate_velocity = np.column_stack(
        [np.asarray(candidate[f"velocity_{axis}"], dtype=float) for axis in "xyz"]
    )
    valid = np.isfinite(candidate_coordinate) & np.all(np.isfinite(candidate_velocity), axis=1)
    valid &= (candidate_coordinate >= ref_coordinate.min()) & (
        candidate_coordinate <= ref_coordinate.max()
    )
    if exclude_cylinder:
        valid &= np.abs(candidate_coordinate) > 0.5 + 1.0e-10
    coordinate_values = candidate_coordinate[valid]
    velocity_values = candidate_velocity[valid]
    interpolated = np.column_stack(
        [np.interp(coordinate_values, ref_coordinate, ref_velocity[:, axis]) for axis in range(3)]
    )
    component_error = velocity_values - interpolated
    vector_error = np.linalg.norm(component_error, axis=1)
    if vector_error.size == 0:
        raise ValueError("profile comparison contains no coincident fluid points")
    return (
        {
            "point_count": int(len(vector_error)),
            "mean_l1_over_u_inf": float(np.mean(vector_error)),
            "p95_over_u_inf": float(np.quantile(vector_error, 0.95)),
            "max_over_u_inf": float(np.max(vector_error)),
            "mean_abs_ux_over_u_inf": float(np.mean(np.abs(component_error[:, 0]))),
            "mean_abs_uy_over_u_inf": float(np.mean(np.abs(component_error[:, 1]))),
            "mean_abs_uz_over_u_inf": float(np.mean(np.abs(component_error[:, 2]))),
        },
        vector_error,
    )


def _stitched_centreline(
    fvm_frame: dict[str, np.ndarray], vpm_frame: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    fvm_x = np.asarray(fvm_frame["position_x"], dtype=float)
    finite = fvm_x[np.isfinite(fvm_x)]
    if finite.size == 0:
        raise ValueError("FVM centreline has no finite positions")
    vpm_x = np.asarray(vpm_frame["position_x"], dtype=float)
    use_vpm = (vpm_x < finite.min() - 1.0e-10) | (vpm_x > finite.max() + 1.0e-10)
    names = ("position_x", "position_y", "position_z", "velocity_x", "velocity_y", "velocity_z")
    return {
        name: np.concatenate((np.asarray(fvm_frame[name]), np.asarray(vpm_frame[name])[use_vpm]))
        for name in names
    }


def _profile_metrics(start: float) -> tuple[dict, list[dict]]:
    reference_samples = REFERENCE_DIR / "samples"
    coupled_samples = COUPLED_DIR / "samples"
    profiles = {
        "centreline": "position_x",
        "transverse_x1": "position_y",
        "transverse_x2": "position_y",
        "transverse_x4": "position_y",
    }
    summary: dict[str, dict] = {}
    records: list[dict] = []
    for name, coordinate in profiles.items():
        tables = {
            "reference": _table(reference_samples / f"{name}.csv"),
            "fvm": _table(coupled_samples / f"fvm_{name}.csv"),
            "vpm": _table(coupled_samples / f"vpm_{name}.csv"),
        }
        common = [
            float(time)
            for time in np.unique(np.asarray(tables["reference"]["time"], dtype=float))
            if time >= start - 1.0e-10
            and all(
                np.any(
                    np.isclose(
                        np.asarray(tables[source]["time"], dtype=float),
                        time,
                        rtol=0.0,
                        atol=1.0e-8,
                    )
                )
                for source in ("fvm", "vpm")
            )
        ]
        if not common:
            raise ValueError(f"No coincident saturated frames for {name}")
        accumulated: dict[str, list[np.ndarray]] = {"fvm": [], "vpm": [], "stitched": []}
        for time in common:
            frames = {source: _frame(table, time) for source, table in tables.items()}
            if any(frame is None for frame in frames.values()):
                continue
            reference_frame = frames["reference"]
            assert reference_frame is not None
            candidates = {"fvm": frames["fvm"], "vpm": frames["vpm"]}
            if name == "centreline":
                assert frames["fvm"] is not None and frames["vpm"] is not None
                candidates["stitched"] = _stitched_centreline(frames["fvm"], frames["vpm"])
            else:
                candidates["stitched"] = frames["fvm"]
            for source, candidate in candidates.items():
                assert candidate is not None
                metrics, errors = _profile_error(
                    candidate,
                    reference_frame,
                    coordinate=coordinate,
                    exclude_cylinder=name == "centreline",
                )
                accumulated[source].append(errors)
                records.append({"profile": name, "source": source, "time": time, **metrics})

        summary[name] = {}
        for source, blocks in accumulated.items():
            all_errors = np.concatenate(blocks)
            source_records = [
                record
                for record in records
                if record["profile"] == name and record["source"] == source
            ]
            summary[name][source] = {
                "frame_count": len(source_records),
                "point_count": int(len(all_errors)),
                "mean_l1_over_u_inf": float(np.mean(all_errors)),
                "p95_over_u_inf": float(np.quantile(all_errors, 0.95)),
                "max_over_u_inf": float(np.max(all_errors)),
                "mean_abs_ux_over_u_inf": float(
                    np.mean([record["mean_abs_ux_over_u_inf"] for record in source_records])
                ),
                "mean_abs_uy_over_u_inf": float(
                    np.mean([record["mean_abs_uy_over_u_inf"] for record in source_records])
                ),
                "mean_abs_uz_over_u_inf": float(
                    np.mean([record["mean_abs_uz_over_u_inf"] for record in source_records])
                ),
            }
    return summary, records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="write an incomplete smoke summary instead of requiring t>=30",
    )
    arguments = parser.parse_args()

    coupled_force = _table(COUPLED_DIR / "samples" / "forces_history.csv")
    reference_force = _table(REFERENCE_DIR / "samples" / "forces_history.csv")
    end = min(
        float(np.max(coupled_force["time"])), float(np.max(reference_force["time"]))
    )
    output_path = COUPLED_DIR / "solution" / "coupled_benchmark_summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if end < 30.0:
        payload = {
            "schema": "openonda-cylinder-coupled-analysis/1",
            "status": "incomplete",
            "available_end_time": end,
            "required_saturated_start": 30.0,
        }
        output_path.write_text(json.dumps(payload, indent=2) + "\n")
        if arguments.allow_incomplete:
            print(json.dumps(payload, indent=2))
            return
        raise SystemExit("Coupled benchmark has not reached the t=30 saturation window")

    start = max(30.0, end - 30.0)
    reference = _force_metrics(reference_force, start)
    coupled = _force_metrics(coupled_force, start)
    force_errors = {
        name: _relative(float(coupled[name]), float(reference[name]))
        for name in ("mean_cd", "cl_rms", "strouhal", "cl_first_harmonic")
    }
    phase_error = float(
        np.angle(
            np.exp(
                1j
                * (
                    float(coupled["cl_harmonic_phase_rad"])
                    - float(reference["cl_harmonic_phase_rad"])
                )
            )
        )
        / (2.0 * np.pi)
    )
    velocity, velocity_records = _profile_metrics(start)
    authority = {
        "centreline": velocity["centreline"]["stitched"],
        "transverse_x1": velocity["transverse_x1"]["fvm"],
        "transverse_x2": velocity["transverse_x2"]["fvm"],
        "transverse_x4": velocity["transverse_x4"]["fvm"],
    }
    limits = {
        "force_relative": 0.01,
        "strouhal_relative": 0.01,
        "velocity_mean_l1_over_u_inf": 0.01,
        "velocity_p95_over_u_inf": 0.03,
    }
    force_pass = all(
        value
        < (
            limits["strouhal_relative"]
            if name == "strouhal"
            else limits["force_relative"]
        )
        for name, value in force_errors.items()
    )
    velocity_pass = all(
        metrics["mean_l1_over_u_inf"] < limits["velocity_mean_l1_over_u_inf"]
        and metrics["p95_over_u_inf"] < limits["velocity_p95_over_u_inf"]
        for metrics in authority.values()
    )
    payload = {
        "schema": "openonda-cylinder-coupled-analysis/1",
        "status": "pass" if force_pass and velocity_pass else "fail",
        "saturated_window": {"start": start, "end": end},
        "limits": limits,
        "reference_force_metrics": reference,
        "coupled_force_metrics": coupled,
        "force_relative_errors": force_errors,
        "lift_phase_error_cycles": phase_error,
        "velocity_profile_errors": velocity,
        "authority_velocity_errors": authority,
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    frame_path = COUPLED_DIR / "solution" / "coupled_velocity_frame_errors.csv"
    with frame_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(velocity_records[0]))
        writer.writeheader()
        writer.writerows(velocity_records)

    print(f"COUPLED BENCHMARK: {payload['status'].upper()}")
    print(
        "  force errors: "
        + ", ".join(f"{name}={value:.3%}" for name, value in force_errors.items())
    )
    print(
        "  authority velocity mean/p95: "
        + ", ".join(
            f"{name}={metrics['mean_l1_over_u_inf']:.3%}/"
            f"{metrics['p95_over_u_inf']:.3%}"
            for name, metrics in authority.items()
        )
    )
    print(f"  wrote {output_path}")
    print(f"  wrote {frame_path}")
    if payload["status"] != "pass":
        raise SystemExit("COUPLED SUB-1% BENCHMARK GATE FAILED")


if __name__ == "__main__":
    main()
